"""对照实验协议：主对照六组 + 实验 A/B/C 的组定义。

所有组的差别只有三个轴：中间结构（连接组 / 重连 / 储备池 / GRU / 经典 / 无脑）、
同伴通信（开/关）、记忆通道配置。观测、动作集、消息权限、奖励规则、读出器
预算全部一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flycore.schemas import RunConfig
from services.env.world import TaskSpec

# 主对照：结构 × 通信 的 2×2 核心，加上三类普通方法与"无脑"下限
GROUP_SPECS: dict[str, dict] = {
    "real_comm":      {"backend": "brain",     "comm": True,  "label": "真实连接子图 + 通信"},
    "real_nocomm":    {"backend": "brain",     "comm": False, "label": "真实连接子图 − 通信"},
    "rewired_comm":   {"backend": "rewired",   "comm": True,  "label": "保度重连 + 通信"},
    "rewired_nocomm": {"backend": "rewired",   "comm": False, "label": "保度重连 − 通信"},
    "reservoir_comm": {"backend": "reservoir", "comm": True,  "label": "随机储备池 + 通信"},
    "reservoir_nocomm": {"backend": "reservoir", "comm": False, "label": "随机储备池 − 通信"},
    "gru_comm":       {"backend": "gru",       "comm": True,  "label": "小型 GRU + 通信"},
    "gru_nocomm":     {"backend": "gru",       "comm": False, "label": "小型 GRU − 通信"},
    "classical":      {"backend": "classical", "comm": False, "label": "经典局部探索策略"},
    "classical_comm": {"backend": "classical", "comm": True,  "label": "经典策略 + 通信"},
    "noop":           {"backend": "noop",      "comm": False, "label": "移除脑模型，仅留接口与读出"},
}

DELTA_KEYS = ("real_comm", "real_nocomm", "rewired_comm", "rewired_nocomm")


@dataclass
class ProtocolConfig:
    task: TaskSpec = field(default_factory=TaskSpec)
    connectome: str = "synthetic-2000"
    ticks: int = 120
    agents: int = 4
    memory_scope: str = "run"
    learning_mode: str = "plastic"
    message_kind: str = "learned_compressed"
    comm_latency_ticks: int = 1
    comm_loss: float = 0.1
    # 感知失效走 TaskSpec.perception_fraction（环境属性），不放这里
    steps_per_tick: int = 20
    memory_channel: bool = False
    central_planner: bool = False
    tenant: str = "default"


def task_for(proto: ProtocolConfig, seed: int, task_overrides: dict | None = None) -> TaskSpec:
    """构造任务实例。`task_overrides` 最后合并，因此可以覆盖任何环境参数（含感知失效）。"""
    t = proto.task
    kwargs: dict = dict(
        task_id=f"{t.task_id or 'grid'}-s{seed}",
        grid=t.grid, n_agents=proto.agents, n_targets=t.n_targets, max_ticks=proto.ticks,
        obstacle_fraction=t.obstacle_fraction, perception_fraction=t.perception_fraction,
        seed=seed, env_version=t.env_version, visibility_regions=t.visibility_regions,
    )
    kwargs.update(task_overrides or {})
    return TaskSpec(**kwargs)


def run_config_for(group: str, proto: ProtocolConfig, seed: int, *,
                   memory_scope: str | None = None, enable_memory: bool | None = None,
                   enable_comm: bool | None = None, central_planner: bool | None = None,
                   learning_mode: str | None = None, run_id: str | None = None,
                   run_overrides: dict | None = None,
                   tag_overrides: dict | None = None) -> RunConfig:
    """构造一次运行的配置。`run_overrides` 最后合并，因此可以覆盖上面任何一项。"""
    spec = GROUP_SPECS[group]
    kwargs: dict = dict(
        run_id=run_id or f"{group}-s{seed}",
        task_id=f"grid{proto.task.grid}-a{proto.agents}-t{proto.task.n_targets}-s{seed}",
        seed=seed,
        backend=spec["backend"],
        connectome=proto.connectome,
        agents=proto.agents,
        max_ticks=proto.ticks,
        memory_scope=(memory_scope or proto.memory_scope),
        learning_mode=(learning_mode or proto.learning_mode),
        memory_version="head",
        enable_peer_messages=(spec["comm"] if enable_comm is None else enable_comm),
        enable_memory_channel=(proto.memory_channel if enable_memory is None else enable_memory),
        enable_central_planner=(proto.central_planner if central_planner is None else central_planner),
        comm_latency_ticks=proto.comm_latency_ticks,
        comm_loss=proto.comm_loss,
        message_kind=proto.message_kind,
    )
    kwargs["tags"] = {"tenant": proto.tenant, "group": group, "protocol": "main",
                      **(tag_overrides or {})}
    kwargs.update(run_overrides or {})
    return RunConfig(**kwargs)
