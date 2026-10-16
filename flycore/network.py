"""事件驱动 LIF 脉冲网络内核。

工程纪律（对应构思文档"第一道门：实现可信"）：
  * 每一步都记录 模拟时间 / 现实时间 之比，绝不在繁忙时悄悄落后于实时。
  * 单步脉冲上限被触发时，超出的脉冲计入 `dropped_spikes`，**不做静默传播**。
  * 数值失控（NaN/Inf）与饱和步数单独计数，不藏进平均值。
  * 全部动力学参数都是人为建模选择，不是果蝇实测生理值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp

from .connectome import NeuronGraph
from .schemas import DynamicsConfig
from .util import Telemetry, now_ms


@dataclass
class StepOutcome:
    spikes: np.ndarray      # bool (n,)
    fired: int
    dropped: int
    saturated: bool
    nan_events: int


class SpikingNetwork:
    """电流型 LIF 网络，稀疏 CSR 突触传播，向量化 numpy 实现。"""

    def __init__(self, graph: NeuronGraph, cfg: DynamicsConfig | None = None, seed: int = 0):
        self.graph = graph
        self.n = graph.n
        self.cfg = cfg or DynamicsConfig()
        self.seed = seed
        self._seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self.W: sp.csr_matrix = (graph.propagation_matrix() * self.cfg.synaptic_scale).astype(np.float32)
        self._syn_decay = float(np.exp(-self.cfg.dt_ms / max(1e-6, self.cfg.tau_syn_ms)))
        self._ext_scale = float(1.0 - self._syn_decay)   # 使注入值等于 i_syn 的稳态值
        self._refractory_steps = max(1, int(round(self.cfg.refractory_ms / self.cfg.dt_ms)))
        self.telemetry = Telemetry()
        self.spike_counts = np.zeros(self.n, dtype=np.int64)
        self._ext_current = np.zeros(self.n, dtype=np.float32)
        self.reset()

    # ---------------------------------------------------------- 生命周期
    def reset(self, reseed: bool = False, clear_counters: bool = False) -> None:
        """回到初始状态。

        `reseed=True` 必须显式要求：噪声来自一个**持久** Generator，
        只重置膜电位的话，两次从"同一初始状态"出发会得到不同噪声序列。
        "逐候选评估"正是靠这一个开关才能成立——同一次决策里，
        每个候选看到的是**逐位相同**的初始状态，唯一差别只有观测。
        """
        cfg = self.cfg
        self.v = np.full(self.n, cfg.v_rest_mv, dtype=np.float32)
        self.i_syn = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.last_spikes = np.zeros(self.n, dtype=bool)
        if reseed:
            self._rng = np.random.default_rng(self._seed)
        if clear_counters:
            self.spike_counts = np.zeros(self.n, dtype=np.int64)

    def get_state(self) -> dict[str, np.ndarray]:
        """检查点：脑内状态必须完整保存，否则恢复的只是部分系统。"""
        return {
            "v": self.v.copy(),
            "i_syn": self.i_syn.copy(),
            "refractory": self.refractory.copy(),
            "last_spikes": self.last_spikes.copy(),
            "spike_counts": self.spike_counts.copy(),
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        for k in ("v", "i_syn", "refractory", "last_spikes", "spike_counts"):
            arr = np.asarray(state[k])
            if arr.shape != (self.n,):
                raise ValueError(f"检查点字段 {k} 形状不匹配：{arr.shape} != ({self.n},)")
            setattr(self, k, arr.copy())

    # ---------------------------------------------------------- 电流注入
    def add_current(self, current: np.ndarray) -> None:
        """叠加一步外部电流。多次调用累加，便于 传感/回忆/同伴 三路分别注入与归因。

        单位含义：稳态等效突触电流 —— 持续注入 c 时 i_syn 的稳态值即为 c。
        """
        if current is None:
            return
        c = np.asarray(current, dtype=np.float32)
        if c.shape[0] != self.n:
            raise ValueError(f"注入电流维度 {c.shape[0]} != 神经元数 {self.n}")
        self._ext_current += c

    def clear_current(self) -> None:
        self._ext_current.fill(0.0)

    # ---------------------------------------------------------- 单步
    def step(self) -> StepOutcome:
        cfg = self.cfg
        t0 = now_ms()
        if self.refractory.any():
            self.v[self.refractory > 0] = cfg.v_reset_mv
            self.refractory -= 1

        self.i_syn = self.i_syn * self._syn_decay
        if self.last_spikes.any():
            self.i_syn += np.asarray(self.W @ self.last_spikes, dtype=np.float32).ravel()
        ext = self._ext_current
        if cfg.background_current:
            ext = ext + cfg.background_current
        self.i_syn += ext * self._ext_scale
        if cfg.noise_sigma:
            self.i_syn += (self._rng.standard_normal(self.n).astype(np.float32)
                           * cfg.noise_sigma * self._ext_scale)

        dv = (cfg.dt_ms / max(1e-6, cfg.tau_m_ms)) * (-(self.v - cfg.v_rest_mv) + cfg.r_in_mohm * self.i_syn)
        self.v += dv.astype(np.float32)
        self.clear_current()

        nan_events = int(np.count_nonzero(~np.isfinite(self.v)))
        if nan_events:
            self.telemetry.nan_events += nan_events
            bad = ~np.isfinite(self.v)
            self.v[bad] = cfg.v_rest_mv
            self.i_syn[bad] = 0.0

        saturated = bool(np.any(np.abs(self.v) > cfg.v_clip_mv))
        if saturated:
            np.clip(self.v, -cfg.v_clip_mv, cfg.v_clip_mv, out=self.v)

        spikes = self.v >= cfg.v_thresh_mv
        fired = int(np.count_nonzero(spikes))
        dropped = 0
        cap = cfg.max_spikes_per_step
        if cap and fired > cap:
            idx = np.flatnonzero(spikes)[:cap]
            keep = np.zeros(self.n, dtype=bool)
            keep[idx] = True
            dropped = fired - cap
            spikes = keep

        if spikes.any():
            self.v[spikes] = cfg.v_reset_mv
            self.refractory[spikes] = self._refractory_steps
            np.add.at(self.spike_counts, np.flatnonzero(spikes), 1)

        self.last_spikes = spikes
        self.telemetry.observe(
            sim_ms=cfg.dt_ms, wall_ms=max(1e-6, now_ms() - t0),
            spikes=int(np.count_nonzero(spikes)), dropped=dropped,
            nan_events=nan_events, saturated=int(saturated),
        )
        return StepOutcome(spikes=spikes, fired=int(np.count_nonzero(spikes)), dropped=dropped,
                           saturated=saturated, nan_events=nan_events)

    # ---------------------------------------------------------- 批量运行
    def simulate(self, steps: int, current_fn: Callable[[int], np.ndarray | None] | None = None,
                 on_step: Callable[[int, StepOutcome], None] | None = None) -> np.ndarray:
        """跑 steps 步；current_fn(t) 每步返回注入电流（或 None）。返回脉冲计数矩阵的按步增量。"""
        counts = np.zeros((steps, self.n), dtype=np.int32)
        for t in range(steps):
            if current_fn is not None:
                self.add_current(current_fn(t))
            out = self.step()
            counts[t] = out.spikes
            if on_step is not None:
                on_step(t, out)
        return counts

    # ---------------------------------------------------------- 读出
    def membrane_summary(self) -> dict[str, float]:
        """L1 即时工作记忆的压缩表示。"""
        v = self.v
        return {
            "v_mean": float(v.mean()),
            "v_std": float(v.std()),
            "v_min": float(v.min()),
            "v_max": float(v.max()),
            "i_syn_abs_mean": float(np.abs(self.i_syn).mean()),
            "refractory_fraction": float(np.count_nonzero(self.refractory > 0) / self.n),
            "active_fraction": float(np.count_nonzero(self.last_spikes) / self.n),
        }

    def health(self) -> dict[str, object]:
        return {
            "neurons": self.n,
            "edges": self.graph.edge_count(),
            "connectome": self.graph.meta.model_dump(),
            **self.telemetry.as_dict(self.n),
            "bounded": bool(np.all(np.isfinite(self.v))),
        }


class SpikeCounter:
    """按组统计脉冲计数窗口，用于读出与消息编码。

    与 SpikingNetwork 解耦：网络只负责产生脉冲，谁去读、读多宽由这里决定。
    """

    def __init__(self, n: int, window_steps: int = 20):
        self.n = n
        self.window = max(1, int(window_steps))
        self.buf = np.zeros((self.window, n), dtype=np.int32)
        self.cursor = 0

    def push(self, spikes: np.ndarray) -> None:
        self.buf[self.cursor] = spikes
        self.cursor = (self.cursor + 1) % self.window

    def counts(self, ids: np.ndarray) -> np.ndarray:
        if ids.size == 0:
            return np.zeros(0, dtype=np.float32)
        return self.buf[:, ids].sum(axis=0).astype(np.float32)

    def rate(self, ids: np.ndarray) -> np.ndarray:
        return self.counts(ids) / float(self.window)

    def reset(self) -> None:
        self.buf.fill(0)
        self.cursor = 0
