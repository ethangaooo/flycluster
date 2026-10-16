"""单个果蝇脑节点的运行引擎。

一个 env tick 对应 `steps_per_tick` 毫秒的神经仿真，读出窗口与之一致，
因此"动作"与"消息"都由同一段仿真时间读出，口径统一。

闭环（构思文档 §四）：
  当前观测与内部状态 → 检索相关经验 → 经记忆适配器编码为记忆输入
  → 改变脑内计算 → 产生行动与消息 → 获得反馈 → 更新记忆或学习参数

被省略的一步就是被跳过的一步：数据库里有记录，不等于脑模型用上了它。

记忆/消息/记录三件事由 `NodeCommon` 实现，与全部对照组共用同一份代码——
否则性能差异无法归因到中间那段结构。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from flycore.coding import BrainIO
from flycore.connectome import NeuronGraph
from flycore.network import SpikeCounter, SpikingNetwork
from flycore.schemas import MSG_VALUES, BrainNodeConfig, BrainStepRequest, BrainStepResponse
from flycore.util import rng_for
from services.shared.node_common import NodeCommon

from .hub_client import MemoryBackend


class BrainEngine(NodeCommon):
    def __init__(self, graph: NeuronGraph, cfg: BrainNodeConfig, hub: MemoryBackend | None = None,
                 steps_per_tick: int = 20, local_root: str | Path = "data/local",
                 k_local: int = 4, k_central: int = 6, record_cue_dim: int = 32):
        self.init_common(cfg, hub, local_root, k_local, k_central, record_cue_dim,
                         local_memory_enabled=True,
                         central_memory_enabled=cfg.enable_memory_input)
        self.graph = graph
        self.steps_per_tick = int(steps_per_tick)

        self.net = SpikingNetwork(graph, cfg.dynamics, seed=cfg.seed)
        self.io = BrainIO.build(graph, seed=cfg.seed, sensory_gain=cfg.sensory_gain,
                               memory_gain=cfg.memory_gain, peer_gain=cfg.peer_gain)
        if not cfg.enable_peer_input:
            self.io.peer.enabled = False
        if not cfg.enable_memory_input:
            self.io.memory.enabled = False
        self.counter = SpikeCounter(graph.n, window_steps=self.steps_per_tick)
        self.rng = rng_for("brain-rng", cfg.seed, cfg.node_id)
        self._pending: tuple[np.ndarray, int] | None = None

    # ---------------------------------------------------------------- 主循环
    def step(self, req: BrainStepRequest) -> BrainStepResponse:
        obs, cue, recalled, peer, ctx = self.prepare(req)

        current, tags = self.io.compose_current(obs, recalled, peer, self.graph.n)
        spikes_total = 0
        dropped_total = 0
        for _ in range(self.steps_per_tick):
            self.net.add_current(current)
            out = self.net.step()
            self.counter.push(out.spikes)
            spikes_total += out.fired
            dropped_total += out.dropped
            if self.trace is not None:
                # 只记"哪些神经元在放电"的汇总，不记整个 891 维数组——
                # 逐步记录是给界面看"思考过程"用的，不该把内存吃满。
                fired = np.flatnonzero(out.spikes)
                self.trace.append({
                    "spikes": int(out.fired),
                    "firing": [int(x) for x in fired[:600]],
                    "groups": {g: int(np.isin(fired, getattr(self.io.groups, g)).sum())
                               for g in ("sensor", "central", "memory", "peer",
                                         "message_out", "action_out")
                               if getattr(self.io.groups, g, None) is not None},
                })

        counts_action = self.counter.counts(self.io.groups.action_out)
        scores = self.io.readout.scores(counts_action, self.steps_per_tick)

        reward = float(ctx.get("last_reward", 0.0) or 0.0)
        if self.cfg.learning_mode == "plastic" and self._pending is not None:
            prev_counts, prev_action = self._pending
            self.io.readout.reward_update(prev_counts, prev_action, reward, self.steps_per_tick)
        action = self.io.readout.select_action(scores, self.rng, self.cfg.exploration_epsilon)
        self._pending = (counts_action.copy(), action)

        return self.finish(
            node_id=self.node_id, tick=req.tick, action=action, scores=scores,
            obs=obs, cue=cue, recalled=recalled, ctx=ctx,
            spikes=spikes_total, dropped=dropped_total,
            realtime_ratio=self.net.telemetry.realtime_ratio,
            source_tags=tags.as_dict()
            | {"recalled_records": tags.recalled_records, "peer_messages": len(req.peer_messages)},
            telemetry=self.net.telemetry.as_dict(self.graph.n),
            extra_payload={"model": "brain", "connectome": self.graph.meta.name},
        )

    # ---------------------------------------------------------------- 消息通道
    def comm_values(self) -> np.ndarray:
        """消息输出组在读出窗口内的放电率 → 8 个量化数值（脑模型输出）。"""
        counts = self.counter.buf.sum(axis=0).astype(np.float32)
        c = self.io.comm_counts(counts)
        w = max(1.0, float(self.steps_per_tick))
        vals = np.empty(MSG_VALUES, dtype=np.float32)
        per = max(1, c.size // MSG_VALUES)
        for i in range(MSG_VALUES):
            vals[i] = float(c[i * per: (i + 1) * per].sum()) / (w * per)
        return (2.0 * np.clip(vals, 0.0, 1.0) - 1.0).astype(np.float32)

    # ---------------------------------------------------------------- 状态
    def checkpoint(self) -> dict[str, Any]:
        """完整检查点：脑内状态 + 学习参数 + 连接组校验和。缺任一项，恢复的只是部分系统。"""
        return {
            "node_id": self.node_id, "tick": self.tick,
            "net_state": self.net.get_state(),
            "readout": self.io.readout.state(),
            "graph_checksum": self.graph.meta.checksum,
            "local_memory": self.local.stats(),
            "working": self.working.summary(),
            "counters": self.counters(),
        }

    def restore(self, ckpt: dict[str, Any]) -> None:
        if ckpt.get("graph_checksum") not in (None, self.graph.meta.checksum):
            raise ValueError("检查点的连接组校验和不匹配，拒绝恢复")
        self.net.set_state(ckpt["net_state"])
        self.io.readout.set_state(ckpt["readout"])
        self.tick = int(ckpt.get("tick", 0))
        self.counter.reset()

    trace: list[dict[str, Any]] | None = None
    """设为一个 list 即开启逐步记录（界面上画"思考过程"用）。默认关闭：
    常规实验每步都记会把内存吃满，而且并没有人去看。"""

    def reset_for_evaluation(self) -> None:
        """回到可复现的初始状态，供"逐个候选评估"使用。

        与 `reset_run_stats` 的区别：这里连**网络状态与噪声序列**一起重置。
        棋类桥接层每次决策要对几十个候选各跑一次网络，必须保证它们
        从逐位相同的状态出发——否则"只有候选不同"这个前提不成立，
        评估热力图就变成了噪声图。**读出器权重不重置**：同一个脑评估所有候选，
        这正是我们要看的东西。
        """
        self.net.reset(reseed=True, clear_counters=True)
        self.net.telemetry.reset()
        self.counter.reset()
        self.local_hits = 0
        self.central_hits = 0
        self.tick = 0

    def apply_reward(self, reward: float) -> bool:
        """**立即**把奖励施加到读出器上，返回是否真的更新了。

        原先奖励是靠 `run_context["last_reward"]` 在**下一步**消费的。
        这对棋局终局是致命的：赢了之后没有下一步了，±1 的胜负奖励
        **永远不会被用上**——果蝇因此从来不知道"什么叫赢"。
        """
        if self.io is None or self._pending is None or not reward:
            return False
        prev_counts, prev_action = self._pending
        self.io.readout.reward_update(prev_counts, prev_action, float(reward),
                                     self.steps_per_tick)
        return True

    def reset_run_stats(self) -> None:
        """清掉上一次运行留下的累计量。

        `Telemetry` 是累积累加器，而容器里的引擎是**常驻单例**——不重置的话，
        报告里"本次运行的仿真步数/脉冲数/实时比"其实是**从容器启动至今**的累计值。
        实测过：同一容器连续跑三次，报告分别显示 64000 / 128000 / 192000 步，
        而每次真实步数都是 64000。这类错误不报错，但让"可信度同报"直接失效。
        """
        self.net.telemetry.reset()
        self.counter.reset()
        self.local_hits = 0
        self.central_hits = 0
        self.tick = 0

    def counters(self) -> dict[str, Any]:
        return {**super().counters(), "plasticity_norm": self.io.readout.plasticity_norm()}

    def health(self) -> dict[str, Any]:
        return {
            "backend": "brain", "ok": True,
            "connectome": self.graph.meta.model_dump(),
            "groups": {k: int(v.size) for k, v in
                       {"sensor": self.io.groups.sensor, "action_out": self.io.groups.action_out,
                        "comm": self.io.groups.comm, "memory": self.io.groups.memory,
                        "peer": self.io.groups.peer}.items()},
            "simulation": self.net.telemetry.as_dict(self.graph.n),
            "dynamics": self.cfg.dynamics.model_dump(),
            "trainable_params": int(self.io.readout.W_base.size),
            "membrane": self.net.membrane_summary(),
            **self.common_health(),
        }
