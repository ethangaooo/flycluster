"""对照组模型（全部使用与脑模型相同的 I/O 接口、记忆协议、消息口径与奖励规则）。

| 模型 | 回答的问题 |
|---|---|
| BrainEngine（见 services/brain） | 真实连接子图能否完成任务 |
| ReservioirModel | 小型随机储备池能否用更少结构做同样的事 |
| GRUModel | 普通时序模型能否用更低的成本做到 |
| ClassicalModel | 是否根本不需要神经网络 |
| NoBrainModel | 实际解题能力是否来自外围代码 |

公平性纪律：
  * 所有模型通过 `NodeCommon` 走完全相同的检索 / 记录 / 提交 / 消息路径。
  * 输入信息量与脑模型三条通道一致（见 `external_features`）。
  * 只有读出器可训练，学习预算与更新规则完全一致，且上报可训练参数量。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from collections import deque

from flycore.coding import MessageCodec, ReadoutHead
from flycore.schemas import MSG_VALUES, N_ACTIONS, BrainNodeConfig, BrainStepRequest, BrainStepResponse
from flycore.util import blas_quiet, rng_for
from services.brain.hub_client import MemoryBackend
from services.shared.node_common import NodeCommon

ACTION_NAMES_LOCAL = ("上", "下", "左", "右", "停")
RESERVOIR_UNITS = 512
GRU_HIDDEN = 64
NOBRAIN_HIDDEN = 128


class NeuralControlBase(NodeCommon):
    """带固定随机内部状态 + 可训练读出器的对照模型基类。"""

    model_name = "control"
    hidden = 128
    seed_tag = "control"

    def __init__(self, cfg: BrainNodeConfig, hub: MemoryBackend | None = None,
                 steps_per_tick: int = 20, local_root: str | Path = "data/local",
                 k_local: int = 4, k_central: int = 6, readout_lr: float = 0.05):
        self.init_common(cfg, hub, local_root, k_local, k_central)
        self.steps_per_tick = int(steps_per_tick)
        rng = rng_for(self.seed_tag, cfg.seed, cfg.node_id)
        in_dim = 105 + MSG_VALUES + 24      # 与 external_features 一致
        self.in_dim = in_dim
        self.W_in = (rng.standard_normal((self.hidden, in_dim)) / np.sqrt(in_dim)).astype(np.float32)
        self.readout = ReadoutHead(self.hidden, self.seed_tag, cfg.seed, readout_lr)
        self._pending: tuple[np.ndarray, int] | None = None
        self.rng = rng
        self.activations = np.zeros(self.hidden, dtype=np.float32)

    # 子类实现
    def _forward(self, feats: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def comm_values(self) -> np.ndarray:
        """消息通道 = 内部状态按固定映射压到 8 个数值。"""
        a = np.abs(self.activations)
        if not hasattr(self, "_comm_proj"):
            self._comm_proj = rng_for("comm-proj", self.seed_tag).standard_normal(
                (MSG_VALUES, self.hidden)).astype(np.float32) / np.sqrt(self.hidden)
        # 与 _forward 一致地静音：某些 BLAS 路径会对零值输入误报
        # "divide by zero / overflow encountered in matmul"，而结果其实是有限的。
        # 这类假告警如果不管，真异常出现时就没人注意了。
        with blas_quiet():
            v = self._comm_proj @ a
        return np.tanh(v).astype(np.float32)

    def step(self, req: BrainStepRequest) -> BrainStepResponse:
        obs, cue, recalled, peer, ctx = self.prepare(req)
        feats = self.external_features(obs, recalled, peer)
        self.activations = self._forward(feats)
        scores = self.readout.scores(self.activations, 1)

        reward = float(ctx.get("last_reward", 0.0) or 0.0)
        if self.cfg.learning_mode == "plastic" and self._pending is not None:
            prev_act, prev_action = self._pending
            self.readout.reward_update(prev_act, prev_action, reward, 1)
        action = self.readout.select_action(scores, self.rng, self.cfg.exploration_epsilon)
        self._pending = (self.activations.copy(), action)

        tags = {"sensor_neurons": int(self.in_dim), "recall_neurons": int(len(recalled)),
                "peer_neurons": int(len(peer)), "recalled_records": len(recalled),
                "peer_messages": len(req.peer_messages)}
        return self.finish(node_id=self.node_id, tick=req.tick, action=action, scores=scores,
                           obs=obs, cue=cue, recalled=recalled, ctx=ctx,
                           realtime_ratio=1.0, source_tags=tags,
                           extra_payload={"model": self.model_name})

    def health(self) -> dict[str, Any]:
        return {"backend": self.model_name, "ok": True, "hidden": self.hidden,
                # 对照模型的"读出维度"就是内部宽度；不填会让报告里出现无意义的 0
                "output_dim": int(self.hidden),
                "trainable_params": int(self.readout.W_base.size),
                "plasticity_norm": self.readout.plasticity_norm(),
                "activations": {"mean": float(self.activations.mean()),
                                "std": float(self.activations.std())},
                **self.common_health()}


class ReservoirModel(NeuralControlBase):
    """漏积分 tanh 储备池（Echo State Network）。递归矩阵按目标谱半径缩放。"""

    model_name = "reservoir"
    hidden = RESERVOIR_UNITS
    seed_tag = "reservoir"

    def __init__(self, *a, spectral_radius: float = 0.95, leak: float = 0.3, **kw):
        super().__init__(*a, **kw)
        rng = rng_for("reservoir-recurrent", self.cfg.seed)
        W = rng.standard_normal((self.hidden, self.hidden)).astype(np.float32)
        sr = float(np.max(np.abs(np.linalg.eigvals(W.astype(np.float64))))) or 1.0
        self.W_r = (W * (spectral_radius / sr)).astype(np.float32)
        self.leak = float(leak)

    def _forward(self, feats: np.ndarray) -> np.ndarray:
        with blas_quiet():
            pre = self.W_r @ self.activations + self.W_in @ feats
        out = ((1 - self.leak) * self.activations + self.leak * np.tanh(pre)).astype(np.float32)
        return out if np.isfinite(out).all() else self.activations


class GRUModel(NeuralControlBase):
    """单层 GRU，权重固定随机初始化，只有读出器可训练（与脑模型同预算口径）。"""

    model_name = "gru"
    hidden = GRU_HIDDEN
    seed_tag = "gru"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        rng = rng_for("gru-params", self.cfg.seed)
        s = 1.0 / np.sqrt(self.hidden)
        self.h = np.zeros(self.hidden, dtype=np.float32)
        # 输入门 z、重置门 r、候选门 n，各含 W_in/W_h
        self.p = {k: (rng.standard_normal((3 * self.hidden, d)) * s).astype(np.float32)
                  for k, d in (("in", self.in_dim), ("hh", self.hidden))}

    def _forward(self, feats: np.ndarray) -> np.ndarray:
        h = self.hidden
        Wi, Wh = self.p["in"], self.p["hh"]
        with blas_quiet():
            zi = Wi[:h] @ feats + Wh[:h] @ self.h
            ri = Wi[h:2 * h] @ feats + Wh[h:2 * h] @ self.h
            ni = Wi[2 * h:] @ feats + Wh[2 * h:] @ self.h
        z = 1.0 / (1.0 + np.exp(-zi))
        r = 1.0 / (1.0 + np.exp(-ri))
        n = np.tanh(ni + r * 0.0)
        new_h = ((1 - z) * self.h + z * n).astype(np.float32)
        if np.isfinite(new_h).all():
            self.h = new_h
        return self.h


class ClassicalModel(NodeCommon):
    """经典局部探索与协调策略：不需要神经网络。

    规则（前沿式探索，刻意避免"最近未访问格"导致的原地震荡）：
      1. 候选方向 = 视野内可通行、无同伴、且不在最近 3 步足迹里的邻格；
      2. 用每个候选邻格周围 3×3 的未访问格数作为"前方还有多少新地方"的打分；
      3. 打分并列时，用检索到的相似经验打破并列，否则随机。
    它回答的问题是：**是否根本不需要神经网络**。
    """

    model_name = "classical"
    seed_tag = "classical"
    RECENT_STEPS = 3
    # 邻居消息里同一方向的平均强度达到该阈值即视为"扎堆"
    CROWD_THRESHOLD = 0.5

    def __init__(self, cfg: BrainNodeConfig, hub: MemoryBackend | None = None,
                 steps_per_tick: int = 20, local_root: str | Path = "data/local",
                 k_local: int = 4, k_central: int = 6, readout_lr: float = 0.05):
        self.init_common(cfg, hub, local_root, k_local, k_central)
        self.rng = rng_for("classical", cfg.seed, cfg.node_id)
        self._search_mode = True
        self._last_action = 4
        self._trail: deque[tuple[int, int]] = deque(maxlen=self.RECENT_STEPS)

    def set_search_mode(self, on: bool) -> None:
        """关闭时退化为"只避障不找新地方"的随机游走，用于区分探索规则与经验的贡献。"""
        self._search_mode = bool(on)

    def comm_values(self) -> np.ndarray:
        """广播自己上一步的方向（one-hot）。

        必须真的带信息：若这里返回全零，`classical_comm` 与 `classical` 会逐位相同，
        "通信对经典策略有没有用"这个问题就变成了构造上的必然，而不是实验结果。
        """
        v = np.zeros(MSG_VALUES, dtype=np.float32)
        if 0 <= self._last_action < N_ACTIONS:
            v[self._last_action] = 1.0
        return v

    def step(self, req: BrainStepRequest) -> BrainStepResponse:
        obs, cue, recalled, peer, ctx = self.prepare(req)
        action, reason = self._decide(obs, recalled, ctx, peer)
        self._last_action = action
        tags = {"sensor_neurons": int(obs.size), "recall_neurons": int(len(recalled)),
                "peer_neurons": int(len(peer)), "recalled_records": len(recalled),
                "peer_messages": len(req.peer_messages)}
        return self.finish(node_id=self.node_id, tick=req.tick, action=action,
                           scores=np.zeros(N_ACTIONS, dtype=np.float32), obs=obs, cue=cue,
                           recalled=recalled, ctx=ctx, realtime_ratio=1.0, source_tags=tags,
                           extra_payload={"model": "classical", "decision_rule": reason})

    def _decide(self, obs: np.ndarray, recalled: list[dict],
                ctx: dict[str, Any], peer_values: list[float] | None = None) -> tuple[int, str]:
        from flycore.schemas import VIEW_CHANNELS, VIEW_SIZE
        view = obs[: VIEW_SIZE * VIEW_SIZE * VIEW_CHANNELS].reshape(VIEW_SIZE, VIEW_SIZE, VIEW_CHANNELS)
        r = VIEW_SIZE // 2
        walls = view[:, :, 0] > 0.5
        peers = view[:, :, 2] > 0.5
        visited = view[:, :, 3] > 0.5
        deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]   # 上 下 左 右

        cell = ctx.get("self_cell") or [None, None]
        if ctx.get("last_outcome") is not None and cell[0] is not None:
            self._trail.append((int(cell[0]), int(cell[1])))

        candidates: list[tuple[int, float]] = []
        for a, (dx, dy) in enumerate(deltas):
            cx, cy = r + dx, r + dy
            if walls[cy, cx] or peers[cy, cx]:
                continue
            # 最近足迹去重：让方向保持，而不是原地上下震荡
            trail_hit = False
            if cell[0] is not None:
                wx = int(cell[0]) + dx
                wy = int(cell[1]) + dy
                trail_hit = (wx, wy) in self._trail
            if self._search_mode:
                y0, y1 = max(0, cy - 1), min(VIEW_SIZE, cy + 2)
                x0, x1 = max(0, cx - 1), min(VIEW_SIZE, cx + 2)
                patch = ~visited[y0:y1, x0:x1] & ~walls[y0:y1, x0:x1]
                score = float(patch.sum())
            else:
                score = 0.0
            if trail_hit:
                score -= 5.0
            candidates.append((a, score))

        if not candidates:
            self._trail.clear()
            return 4, "被阻挡，原地不动"
        best_score = max(s for _, s in candidates)
        best = [a for a, s in candidates if s == best_score]

        # 协同规则：若足够多的邻居正朝同一方向走，就换一个方向，避免扎堆重复探索。
        # 这条规则只使用经通信通道送来的信息，因此 classical 与 classical_comm 的差异
        # 才是"直接通信对经典策略有没有用"的实验结论。
        crowd = ""
        if peer_values:
            pv = np.asarray(peer_values, dtype=np.float32)[:N_ACTIONS]
            if pv.size == N_ACTIONS and float(pv.max()) >= self.CROWD_THRESHOLD:
                avoid = int(np.argmax(pv))
                alt = [a for a in best if a != avoid]
                if alt:
                    best = alt
                    crowd = f"（避让同伴集中方向 {ACTION_NAMES_LOCAL[avoid]}）"
        if len(best) == 1:
            return best[0], f"唯一最优（前沿分 {best_score:.0f}）{crowd}"

        if recalled:
            votes: dict[int, float] = {}
            for rec in recalled:
                act = (rec.get("payload") or {}).get("conclusion")
                if act in best:
                    votes[int(act)] = votes.get(int(act), 0.0) + \
                        float(rec.get("similarity", 0.0)) * float(rec.get("confidence", 0.5) or 0.5)
            if votes:
                return max(votes.items(), key=lambda kv: kv[1])[0], "经验打破并列"
        return int(best[int(self.rng.integers(0, len(best)))]), "随机打破并列"

    def health(self) -> dict[str, Any]:
        return {"backend": self.model_name, "ok": True, "search_mode": self._search_mode,
                "trainable_params": 0, **self.common_health()}


class NoBrainModel(NeuralControlBase):
    """移除脑模型，只保留相同的接口与读出器。

    观测经固定随机映射后直接驱动读出器。它回答的问题是：
    **我们观察到的解题能力，有多少其实来自外围代码而不是那段结构？**
    """

    model_name = "noop"
    hidden = NOBRAIN_HIDDEN
    seed_tag = "noop"

    def _forward(self, feats: np.ndarray) -> np.ndarray:
        with blas_quiet():
            pre = self.W_in @ feats
        out = np.tanh(pre).astype(np.float32)
        return out if np.isfinite(out).all() else self.activations


MODEL_REGISTRY = {
    "reservoir": ReservoirModel,
    "gru": GRUModel,
    "classical": ClassicalModel,
    "noop": NoBrainModel,
}


def build_control(kind: str, cfg: BrainNodeConfig, hub: MemoryBackend | None,
                  steps_per_tick: int, local_root: str | Path) -> NodeCommon:
    if kind not in MODEL_REGISTRY:
        raise ValueError(f"未知对照模型 {kind}；可选 {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[kind]
    return cls(cfg, hub=hub, steps_per_tick=steps_per_tick, local_root=local_root)
