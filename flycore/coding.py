"""编解码层：感知编码、动作读出、消息编解码、记忆适配器。

这一层是整个系统"能被人为解释的部分"，也是归因的关键：
  * 感官编码器把观测投影成输入组电流 —— 外围代码的贡献。
  * 读出器把读出组脉冲计数映射成动作 —— 外围代码的贡献（可训练）。
  * 记忆适配器把检索到的经验编码成电流 —— 外围代码的贡献。
  * 三者之间的**连接组结构**才是候选的"脑内计算"贡献。

因此所有对照组必须共享完全相同的这一层与观测/消息权限，只更换中间的结构。
三路电流分别注入互不相交的神经元组（传感/回忆/同伴），消融时可以逐路关掉，
并且每一步都上报三路的贡献计数，避免把外围能力记在脑结构账上。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .connectome import NeuronGraph
from .schemas import MSG_INT16_FULL, MSG_VALUES, N_ACTIONS, OBS_DIM
from .util import blas_quiet, rng_for

CUE_DIM = 32
MSG_SUBGROUPS = MSG_VALUES
MSG_NEURONS_PER_SUBGROUP = 4
MEM_GROUP_SIZE = 24
PEER_GROUP_SIZE = 24
FEATURES_PER_INPUT = 12


@dataclass
class SourceTags:
    """本步电流注入的来源构成。报告里必须和性能一起出现。"""

    sensor_neurons: int = 0
    recall_neurons: int = 0
    peer_neurons: int = 0
    recalled_records: int = 0
    peer_messages: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "sensor_neurons": self.sensor_neurons,
            "recall_neurons": self.recall_neurons,
            "peer_neurons": self.peer_neurons,
            "recalled_records": self.recalled_records,
            "peer_messages": self.peer_messages,
        }


def _pick_excitatory(graph: NeuronGraph, exclude: set[int], size: int, tag: str) -> np.ndarray:
    pool = np.flatnonzero(~graph.inhibitory)
    pool = pool[~np.isin(pool, list(exclude))]
    if pool.size < size:
        raise ValueError(f"{tag} 组需要 {size} 个兴奋性神经元，可用仅 {pool.size}")
    rng = rng_for("group", tag, graph.meta.checksum)
    return np.sort(rng.choice(pool, size=size, replace=False)).astype(np.int32)


@dataclass
class NeuronGroups:
    """四类功能组：感觉输入、动作读出、消息输出、记忆输入、同伴输入。"""

    sensor: np.ndarray
    action_out: np.ndarray
    comm: np.ndarray            # 形状 (MSG_SUBGROUPS * MSG_NEURONS_PER_SUBGROUP,)
    memory: np.ndarray
    peer: np.ndarray

    @staticmethod
    def build(graph: NeuronGraph) -> "NeuronGroups":
        sensor = graph.input_ids.astype(np.int32)
        action_out = graph.output_ids.astype(np.int32)
        used = set(sensor.tolist()) | set(action_out.tolist())
        comm = _pick_excitatory(graph, used, MSG_SUBGROUPS * MSG_NEURONS_PER_SUBGROUP, "comm")
        used |= set(comm.tolist())
        memory = _pick_excitatory(graph, used, MEM_GROUP_SIZE, "memory")
        used |= set(memory.tolist())
        peer = _pick_excitatory(graph, used, PEER_GROUP_SIZE, "peer")
        return NeuronGroups(sensor, action_out, comm, memory, peer)

    def comm_subgroup(self, i: int) -> np.ndarray:
        a = i * MSG_NEURONS_PER_SUBGROUP
        return self.comm[a: a + MSG_NEURONS_PER_SUBGROUP]


class RandomProjector:
    """固定稀疏随机投影：把 in_dim 维外部信号注入到 m 个神经元。

    每行权重按 1/sqrt(fan_in) 归一（行 L2 范数约为 1），输入信号在投影前做去均值，
    因此"某特征出现"与"某特征缺失"分别产生正、负电流，而不是单极性的稀疏响应。

    权重在初始化后冻结，不参与学习——它属于外围接口，不属于脑内可塑性。
    """

    def __init__(self, target_ids: np.ndarray, in_dim: int, fan_in: int, seed_tag: str,
                 checksum: str = "", gain: float = 1.0):
        self.ids = target_ids.astype(np.int32)
        m = self.ids.size
        fan_in = max(1, min(fan_in, in_dim))
        rng = rng_for("proj", seed_tag, checksum, m, in_dim)
        rows = np.repeat(np.arange(m), fan_in)
        cols = np.empty(m * fan_in, dtype=np.int32)
        vals = np.empty(m * fan_in, dtype=np.float32)
        scale = 1.0 / float(np.sqrt(fan_in))
        for i in range(m):
            c = rng.choice(in_dim, size=fan_in, replace=False)
            cols[i * fan_in: (i + 1) * fan_in] = c
            vals[i * fan_in: (i + 1) * fan_in] = (
                rng.choice([-1.0, 1.0], size=fan_in).astype(np.float32) * scale
            )
        self.proj = sparse.csr_matrix((vals, (rows, cols)), shape=(m, in_dim))
        self.gain = float(gain)
        self.in_dim = int(in_dim)

    def project(self, signal: np.ndarray) -> np.ndarray:
        if signal.size != self.in_dim:
            raise ValueError(f"投影输入维度 {signal.size} != {self.in_dim}")
        x = np.asarray(signal, dtype=np.float32)
        with blas_quiet():
            return (self.proj @ x).astype(np.float32) * self.gain


class SensoryEncoder:
    """观测向量 → 感觉输入组电流。投影前对观测去均值，使特征出现/缺失产生正负电流。"""

    def __init__(self, groups: NeuronGroups, checksum: str = "", gain: float = 60.0):
        self.proj = RandomProjector(groups.sensor, OBS_DIM, FEATURES_PER_INPUT, "sensor", checksum, gain)
        self.enabled = True

    def encode(self, obs: np.ndarray) -> tuple[np.ndarray, int]:
        if not self.enabled:
            return np.zeros(0, dtype=np.float32), 0
        x = np.asarray(obs, dtype=np.float32)
        return self.proj.project(x - x.mean()), 1


class ReadoutHead:
    """读出组脉冲计数 → 动作分数。

    W = W_base + ΔW。`plastic` 模式下只更新 ΔW（外围读出器），
    因此任何性能提升只能被表述为"外围读出器学到了东西"，
    不能表述为"脑内学习"。要声称脑内学习，必须另行更新脑内可塑性参数并通过对应测试。
    """

    def __init__(self, n_out: int, checksum: str = "", seed: int = 0, lr: float = 0.05,
                 max_delta_ratio: float = 0.5):
        rng = rng_for("readout", seed, checksum)
        m = max(1, int(n_out))
        self.n_out = m
        self.W_base = (rng.standard_normal((N_ACTIONS, m)) / np.sqrt(m)).astype(np.float32)
        self.delta_W = np.zeros_like(self.W_base)
        self.bias = np.zeros(N_ACTIONS, dtype=np.float32)
        self.lr = float(lr)
        self.max_delta = float(max_delta_ratio) * float(np.abs(self.W_base).max())
        self.updates = 0

    @property
    def W(self) -> np.ndarray:
        return self.W_base + self.delta_W

    def scores(self, counts: np.ndarray, window: int) -> np.ndarray:
        rate = np.asarray(counts, dtype=np.float32) / max(1.0, float(window))
        return (self.W @ rate + self.bias).astype(np.float32)

    def select_action(self, scores: np.ndarray, rng: np.random.Generator | None = None,
                      epsilon: float = 0.0) -> int:
        if epsilon > 0 and rng is not None and rng.random() < epsilon:
            return int(rng.integers(0, scores.size))
        return int(np.argmax(scores))

    def reward_update(self, counts: np.ndarray, action: int, reward: float, window: int) -> None:
        """奖励调制的 Hebbian 更新，只作用于被选中动作那一行。"""
        rate = np.asarray(counts, dtype=np.float32) / max(1.0, float(window))
        self.delta_W[action] += self.lr * float(reward) * rate
        np.clip(self.delta_W, -self.max_delta, self.max_delta, out=self.delta_W)
        self.updates += 1

    def plasticity_norm(self) -> float:
        return float(np.abs(self.delta_W).sum())

    def state(self) -> dict[str, object]:
        return {"W_base": self.W_base, "delta_W": self.delta_W, "bias": self.bias,
                "updates": self.updates, "plasticity_norm": self.plasticity_norm()}

    def set_state(self, st: dict[str, object]) -> None:
        self.W_base = np.asarray(st["W_base"], dtype=np.float32).copy()
        self.delta_W = np.asarray(st["delta_W"], dtype=np.float32).copy()
        self.bias = np.asarray(st["bias"], dtype=np.float32).copy()
        self.updates = int(st.get("updates", 0))


class MessageCodec:
    """8 × int16 消息的量化编解码与带宽核算。

    带宽口径（不含报头、重传、多邻居重复传输）：
        n_senders × 8 × 2 字节 × 频率(Hz)
    """

    def __init__(self, n_values: int = MSG_VALUES):
        self.n_values = int(n_values)

    def encode(self, values: np.ndarray) -> list[int]:
        v = np.nan_to_num(np.asarray(values, dtype=np.float32).ravel(), nan=0.0, posinf=1.0, neginf=-1.0)
        if v.size != self.n_values:
            raise ValueError(f"消息维度 {v.size} != {self.n_values}")
        return (np.clip(v, -1.0, 1.0) * MSG_INT16_FULL).astype(np.int32).tolist()

    def decode(self, values: list[int]) -> np.ndarray:
        v = np.asarray(values, dtype=np.float32)
        if v.size != self.n_values:
            raise ValueError(f"消息维度 {v.size} != {self.n_values}")
        return v / MSG_INT16_FULL

    @staticmethod
    def raw_bytes_per_second(n_senders: int, rate_hz: float, n_values: int = MSG_VALUES) -> int:
        return int(n_senders * n_values * 2 * rate_hz)


class MessageEncoder:
    """消息输出组脉冲 → 8 个量化数值。"""

    def __init__(self, groups: NeuronGroups):
        self.groups = groups
        self.codec = MessageCodec()

    def encode(self, counts: np.ndarray, window: int) -> list[int]:
        """counts 为整条 comm 组的脉冲计数，按子组求率后量化。"""
        c = np.asarray(counts, dtype=np.float32)
        w = max(1.0, float(window))
        vals = np.empty(MSG_SUBGROUPS, dtype=np.float32)
        for i in range(MSG_SUBGROUPS):
            a = i * MSG_NEURONS_PER_SUBGROUP
            vals[i] = float(c[a: a + MSG_NEURONS_PER_SUBGROUP].sum()) / (w * MSG_NEURONS_PER_SUBGROUP)
        return self.codec.encode(2.0 * vals - 1.0)


class CueEncoder:
    """观测 → 检索线索向量。固定投影 + L2 归一化，与连接组无关、无学习参数。

    脑模型与所有对照组必须共用**同一个**线索函数，否则记忆检索的差异会
    混进模型差异里，实验就不再是单一变量对照。
    """

    def __init__(self, dim: int = CUE_DIM):
        self.dim = int(dim)
        # 与 obs 保持同一 dtype：混用 float64 权重 @ float32 输入会让部分 BLAS
        # 后端误报 overflow / divide-by-zero，污染日志。
        self._proj = (rng_for("cue", "fixed").standard_normal((self.dim, OBS_DIM))
                      .astype(np.float32) / np.float32(np.sqrt(OBS_DIM)))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with blas_quiet():
            v = self._proj @ np.asarray(obs, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)


class MemoryAdapter:
    """检索到的经验 → 记忆输入组电流。

    这是构思文档里"记忆怎样真正进入神经计算"的接口：
       检索相关经验 → 编码为记忆输入 → 改变后续脑内计算
    数据库里存在一条记录，不代表脑模型能用它；必须经过这里。

    研究模式下**不**返回"这一步该往哪走"的最终指令，也不在内部跑规划器，
    否则必须把该规划器单独纳入贡献评估。
    """

    def __init__(self, groups: NeuronGroups, checksum: str = "", gain: float = 24.0,
                 feature_dim: int = 24):
        self.proj = RandomProjector(groups.memory, feature_dim, 12, "memory", checksum, gain)
        self.feature_dim = int(feature_dim)
        self.enabled = True
        self.cue_encoder = CueEncoder()

    def cue(self, obs: np.ndarray) -> np.ndarray:
        return self.cue_encoder(obs)

    def encode(self, records: list[dict]) -> tuple[np.ndarray, int]:
        """把 k 条经验压成一个 feature_dim 向量。空检索返回零电流（不是随机噪声）。"""
        if not self.enabled or not records:
            return np.zeros(self.proj.in_dim, dtype=np.float32), 0
        feats = np.zeros(self.feature_dim, dtype=np.float32)
        used = 0
        for r in records[: self.feature_dim]:
            cue = r.get("cue_vector") or []
            conf = float(r.get("confidence", 0.5) or 0.5)
            if not cue:
                continue
            c = np.asarray(cue, dtype=np.float32)
            m = min(c.size, CUE_DIM)
            feats[used % self.feature_dim] += float(c[:m].mean()) * conf
            used += 1
        if used == 0:
            return np.zeros(self.proj.in_dim, dtype=np.float32), 0
        feats /= max(1, used)
        return self.proj.project(feats), used


class PeerInputAdapter:
    """其他脑的消息 → 同伴输入组电流。"""

    def __init__(self, groups: NeuronGroups, checksum: str = "", gain: float = 24.0):
        self.proj = RandomProjector(groups.peer, MSG_VALUES, 6, "peer", checksum, gain)
        self.codec = MessageCodec()
        self.enabled = True

    def encode(self, values: list[float]) -> tuple[np.ndarray, int]:
        if not self.enabled or not values:
            return np.zeros(self.proj.in_dim, dtype=np.float32), 0
        v = np.asarray(values[: MSG_VALUES], dtype=np.float32)
        if v.size < MSG_VALUES:
            v = np.pad(v, (0, MSG_VALUES - v.size))
        return self.proj.project(np.clip(v, -1.0, 1.0)), 1


@dataclass
class BrainIO:
    """一个脑节点完整的 I/O 组合。所有对照组共享本类，只换中间结构。"""

    groups: NeuronGroups
    sensory: SensoryEncoder
    readout: ReadoutHead
    message: MessageEncoder
    memory: MemoryAdapter
    peer: PeerInputAdapter

    def compose_current(self, obs: np.ndarray, recalled: list[dict], peer_values: list[float],
                        n: int) -> tuple[np.ndarray, SourceTags]:
        """把三路外部信号分别编码成电流，注入互不相交的神经元组。

        三路分离的意义：消融时可以单独关掉任一路（传感/回忆/同伴），
        并且每步都能上报三路的贡献计数，避免把外围能力归因到脑结构上。
        """
        cur = np.zeros(n, dtype=np.float32)
        tags = SourceTags()

        sens, n_sens = self.sensory.encode(obs)
        if n_sens:
            cur[self.groups.sensor] += sens
            tags.sensor_neurons = int(self.groups.sensor.size)

        mem, n_mem = self.memory.encode(recalled)
        if n_mem:
            cur[self.groups.memory] += mem
            tags.recall_neurons = int(self.groups.memory.size)
            tags.recalled_records = int(n_mem)

        peer, n_peer = self.peer.encode(peer_values)
        if n_peer:
            cur[self.groups.peer] += peer
            tags.peer_neurons = int(self.groups.peer.size)
            tags.peer_messages = int(n_peer)

        return cur, tags

    def readout_scores(self, spike_counts: np.ndarray, window: int) -> np.ndarray:
        counts = np.asarray(spike_counts, dtype=np.float32)
        return self.readout.scores(counts[: self.groups.action_out.size], window)

    def comm_counts(self, spike_counts: np.ndarray) -> np.ndarray:
        return np.asarray(spike_counts, dtype=np.float32)[: self.groups.comm.size]

    @staticmethod
    def build(graph: NeuronGraph, seed: int = 0, sensory_gain: float = 30.0,
              memory_gain: float = 24.0, peer_gain: float = 24.0, readout_lr: float = 0.05) -> "BrainIO":
        groups = NeuronGroups.build(graph)
        cs = graph.meta.checksum
        return BrainIO(
            groups=groups,
            sensory=SensoryEncoder(groups, cs, sensory_gain),
            readout=ReadoutHead(groups.action_out.size, cs, seed, readout_lr),
            message=MessageEncoder(groups),
            memory=MemoryAdapter(groups, cs, memory_gain),
            peer=PeerInputAdapter(groups, cs, peer_gain),
        )
