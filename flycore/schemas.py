"""跨服务共享数据契约（Pydantic v2）。

命名纪律：凡是"人为建模选择"的字段必须能表达出来，报告里才能如实标注。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- 观测空间常量
VIEW_SIZE = 5                       # 局部观测 5x5
VIEW_CHANNELS = 4                   # wall / target / peer / visited
POSE_DIM = 4                        # x, y, sin(heading), cos(heading)
AUX_DIM = 1                         # 已走步数占比
OBS_DIM = VIEW_SIZE * VIEW_SIZE * VIEW_CHANNELS + POSE_DIM + AUX_DIM  # = 105

GRID_SIZE = 32
N_ACTIONS = 5                       # 上/下/左/右/停
MSG_VALUES = 8                      # 每条消息 8 个 16 位数值
MSG_INT16_FULL = 32767.0

NodeKind = Literal["excitatory", "inhibitory"]
GraphSource = Literal["synthetic", "flywire", "flychip", "male-cns", "imported"]
LearningMode = Literal["readonly", "accumulate", "plastic"]
MemoryScope = Literal["run", "session", "project", "tenant"]
MemoryVersion = Literal["pinned", "head"]
RecordKind = Literal["observation", "inference", "alert", "skill"]
BackendKind = Literal["brain", "rewired", "reservoir", "gru", "classical", "noop"]


class ConnectomeMeta(BaseModel):
    """连接组元数据。许可与口径必须显式登记，不得留白。"""

    name: str
    source: GraphSource
    license: str = "UNKNOWN"
    license_url: str | None = None
    neuron_count: int
    edge_count: int = Field(description="聚合后的连接边数，不等于突触数")
    synapse_count: int | None = Field(default=None, description="原始数据突触数，未知则留空")
    inhibitory_fraction: float
    mean_indegree: float
    mean_outdegree: float
    largest_scc_fraction: float = Field(description="最大强连通分量占比，作为递归性的代理指标")
    modularity: float
    io_reachable: bool = Field(description="输入神经元集合能否到达输出神经元集合")
    input_count: int
    output_count: int
    includes_optic_lobe: bool = False
    dynamics_modeled: bool = Field(
        default=True, description="恒为 True：神经元动力学一律为人为建模，非实测"
    )
    checksum: str = ""
    created_at: str = ""
    derived_from: str | None = None
    rewire_swaps: int = 0
    notes: str = ""


class DynamicsConfig(BaseModel):
    """LIF 动力学参数。全部为工程选择，不是果蝇实测生理参数。

    电流标度说明：`*_gain` 与 `background_current` 的单位是"稳态等效突触电流"，
    即持续注入该电流时 i_syn 的稳态值约为该数值。取 1 时，静息膜电位下的上升率
    约为 dt/tau_m mV/步，因此典型可用区间在 10~60 之间。
    """

    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    v_rest_mv: float = -65.0
    v_reset_mv: float = -70.0
    v_thresh_mv: float = -50.0
    refractory_ms: float = 2.0
    r_in_mohm: float = 1.0
    synaptic_scale: float = Field(
        default=0.35,
        description="全局突触增益。缩放整张 W，用于把网络从饱和区拉到输入敏感区。人为建模选择",
    )
    background_current: float = Field(
        default=14.0, description="全局背景电流，维持自发活动。人为建模选择，非实测"
    )
    noise_sigma: float = Field(
        default=6.0, description="逐神经元高斯噪声电流标准差，提供随机性来源"
    )
    max_spikes_per_step: int = Field(
        default=0, description="单步允许传播的脉冲上限；0 表示不限。超出部分计入 dropped_spikes"
    )
    v_clip_mv: float = 200.0


class BrainNodeConfig(BaseModel):
    node_id: str
    connectome: str
    backend: BackendKind = "brain"
    seed: int = 0
    learning_mode: LearningMode = "accumulate"
    memory_scope: MemoryScope = "run"
    memory_version: MemoryVersion = "head"
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    enable_peer_input: bool = True
    enable_memory_input: bool = True
    sensory_gain: float = 30.0
    memory_gain: float = 24.0
    peer_gain: float = 24.0
    readout_window_ms: float = 20.0
    exploration_epsilon: float = Field(
        default=0.2,
        description=("决策时的随机探索比例。全部神经模型共享同一取值——没有探索，"
                     "未训练的读出器会长期锁死在同一个动作上，任务不可能完成，"
                     "实验会退化成「所有组都是零分」。"),
    )


class ObservationPacket(BaseModel):
    agent_id: int
    tick: int
    vector: list[float] = Field(description=f"长度 {OBS_DIM} 的归一化向量")
    perception_valid: bool = True
    note: str = ""


class MessagePacket(BaseModel):
    src_agent: int
    dst_agent: int
    tick: int
    issued_tick: int
    values: list[int] = Field(description=f"长度 {MSG_VALUES} 的 int16 数值")
    kind: Literal["shared_observation", "learned_compressed"] = "learned_compressed"

    def payload_bytes(self) -> int:
        return len(self.values) * 2


class BrainStepRequest(BaseModel):
    node_id: str
    tick: int
    observation: ObservationPacket
    peer_messages: list[MessagePacket] = Field(default_factory=list)
    recalled: list[dict[str, Any]] = Field(
        default_factory=list, description="中心/本地检索返回的经验，经适配器编码后进入神经计算"
    )
    run_context: dict[str, Any] = Field(default_factory=dict)


class BrainStepResponse(BaseModel):
    node_id: str
    tick: int
    action: int
    action_scores: list[float]
    outgoing_message: list[int] = Field(default_factory=list)
    spikes_emitted: int = 0
    dropped_spikes: int = 0
    realtime_ratio: float = 0.0
    source_tags: dict[str, int] = Field(
        default_factory=dict, description="本步电流注入中 传感/回忆/同伴 三类来源的贡献计数"
    )
    telemetry: dict[str, Any] = Field(default_factory=dict)


class ExperienceRecord(BaseModel):
    """一条经验。origin_event_id 是防回声的锚点：同源转发不得计为新样本。"""

    record_id: str
    origin_event_id: str
    producer_node: str
    kind: RecordKind = "observation"
    run_id: str = ""
    task_id: str = ""
    episode: int = 0
    tick: int = 0
    env_version: str = ""
    schema_version: str = "1"
    scope: MemoryScope = "run"
    obs_vector: list[float] = Field(default_factory=list, description="用于相似检索的线索向量")
    payload: dict[str, Any] = Field(default_factory=dict)
    conditions: dict[str, Any] = Field(default_factory=dict)
    parents: list[str] = Field(default_factory=list, description="来源记录 id，用于构建来源链")
    derived_from_origin: str | None = None
    confidence: float = 0.5
    observed_at: str = ""
    valid_from: str = ""
    valid_to: str | None = None


class MemorySubmitResult(BaseModel):
    accepted: bool
    record_id: str
    is_echo: bool = False
    echo_of: str | None = None
    evidence_count_delta: int = 0
    reason: str = ""


class MemoryQuery(BaseModel):
    cue_vector: list[float] = Field(default_factory=list)
    task_id: str | None = None
    scope: MemoryScope | None = None
    run_id: str | None = Field(
        default=None,
        description="scope=run 时强制匹配的运行 id。范围隔离不得默认跨运行共享",
    )
    tenant: str | None = None
    kinds: list[RecordKind] | None = None
    k: int = 8
    as_of_version: str | None = Field(default=None, description="固定记忆快照；None 表示读 head")
    producer_node: str | None = None


class MemoryQueryResult(BaseModel):
    version: str
    retrieved: list[dict[str, Any]]
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    query_cost_bytes: int = Field(
        default=0, description="检索返回的字节数。中心记忆本身是一条通信通道，必须计入预算"
    )


class RunConfig(BaseModel):
    run_id: str = ""
    task_id: str = ""
    seed: int = 0
    backend: BackendKind = "brain"
    connectome: str = "synthetic-2000"
    agents: int = Field(default=4, ge=1, le=16)
    max_ticks: int = 120
    memory_scope: MemoryScope = "run"
    learning_mode: LearningMode = "accumulate"
    memory_version: MemoryVersion = "head"
    enable_peer_messages: bool = True
    enable_memory_channel: bool = True
    enable_central_planner: bool = Field(
        default=False,
        description="中心额外提供规划（而非仅检索）。开启时必须把能力归因于组合系统",
    )
    # 感知失效属于**环境属性**，用 TaskSpec.perception_fraction 表达。
    # 这里不重复声明：两个来源必然漂移，而且是静默漂移——
    # 早期版本就留过一个声明了却从未被 runner 读取的 perception_dropout 字段。
    comm_latency_ticks: int = 0
    comm_loss: float = Field(default=0.0, ge=0.0, le=1.0)
    message_rate_hz: float = 5.0
    message_kind: Literal["shared_observation", "learned_compressed"] = "learned_compressed"
    k_local: int = Field(default=4, description="本地经验检索条数；0 表示关闭本地检索")
    k_central: int = Field(default=6, description="中心经验检索条数")
    exploration_epsilon: float = Field(default=0.2, description="决策时的随机探索比例")
    callback_url: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class ScoreBreakdown(BaseModel):
    targets_found: int = 0
    targets_total: int = 0
    collisions: int = 0
    redundant_steps: int = 0
    coverage: float = 0.0
    steps_used: int = 0
    score: float = 0.0


class GateResult(BaseModel):
    gate: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class RunReport(BaseModel):
    run_id: str
    task_id: str
    backend: BackendKind
    seed: int
    config: dict[str, Any]
    score: ScoreBreakdown
    per_agent: list[dict[str, Any]] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    comm_accounting: dict[str, Any] = Field(default_factory=dict)
    memory_accounting: dict[str, Any] = Field(default_factory=dict)
    gates: list[GateResult] = Field(default_factory=list)
    attribution: dict[str, str] = Field(
        default_factory=dict, description="结论归属：哪些能力来自脑模型，哪些来自外围代码"
    )
    env_snapshot: dict[str, Any] | None = Field(
        default=None, description="运行结束时的环境快照，供控制台直接渲染"
    )
    trainable_params: int = Field(
        default=0, description="可训练参数量，用于核对各对照是否处于同预算"
    )
    output_dim: int = Field(default=0, description="读出维度（各模型内部宽度不同，需如实披露）")
    finished_at: str = ""
