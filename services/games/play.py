"""让果蝇集群下棋：桥接层 + 对局管理。

## 设计要点

**1. 不改造脑引擎。** `BrainEngine` 的观测维度（105）与动作数（5）是硬编码的，
动它就会改变已验证的实验结论。所以桥接层这么做：

    对每个候选着法各编码一份 105 维观测 → 送进同一个脑实例跑 20 步仿真
    → 取 5 个动作分 → 经一个固定随机投影压成"倾向分" → 取最高者落子

**2. 每个候选必须从逐位相同的初始状态出发。** 否则不同候选之间除了观测还有噪声差异，
评估热力图就成了噪声图。这靠 `reset_for_evaluation()`（连随机数发生器的序列一起重置）。
这也是本项目"对照公平性"纪律在演示里的同一应用。

**3. 能力归因必须写清楚。** 棋盘编码、候选剪枝、动作分→倾向分的投影、胜负判定、
电脑对手的搜索——**全部是外围代码**。脑只提供一个脉冲动力系统。
页面上如实标注，否则就是把外围能力记在脑结构账上。

**4. 顺带提供一个诚实的小实验。** 飞蝇一方可选不同"结构"（真实连接组 / 保度重连 /
随机储备池 / 纯随机基线 / 外围启发式基线）。跑一串对局就能看出：
**任何结构是否显著强于"随机合法走子"**。这正是项目方法论在演示上的应用——
如果都不强于随机，那也是一个有价值的负面结果，而不是需要掩饰的事。
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from flycore.connectome import get_graph
from flycore.schemas import (N_ACTIONS, OBS_DIM, BrainNodeConfig, BrainStepRequest,
                             BrainStepResponse)
from flycore.util import rng_for
from services.games.base import FLY, PC, Move, Outcome, check_observe

# --------------------------------------------------------------------- 棋种注册

def _registry() -> dict[str, Any]:
    from services.games.checkers import Checkers
    from services.games.gomoku import Gomoku
    from services.games.xiangqi import Xiangqi

    return {
        "gomoku": {"cls": Gomoku, "label": "五子棋", "desc": "15×15，五子连珠即胜"},
        "xiangqi": {"cls": Xiangqi, "label": "中国象棋", "desc": "9×10，七种棋子完整走法"},
        "checkers": {"cls": Checkers, "label": "跳棋", "desc": "121 洞六角星，跳跃链"},
    }


REGISTRY: dict[str, Any] = {}


def games_catalog() -> list[dict[str, Any]]:
    global REGISTRY
    if not REGISTRY:
        REGISTRY = _registry()
    return [{"key": k, "label": v["label"], "desc": v["desc"]} for k, v in REGISTRY.items()]


def make_game(key: str, seed: int = 0):
    """构造棋局并把它的随机数发生器播种。

    三个棋种的构造函数签名不一致（有的收 seed、有的不收），
    而且不收 seed 的那个默认用的是**未播种**的 `random.Random()`——
    那样 AI 的随机性不可复现，"跑 20 局比胜率"这类实验就没法重跑。
    所以这里统一处理：先按最小签名构造，再给它身上的 RNG 播种。
    """
    global REGISTRY
    if not REGISTRY:
        REGISTRY = _registry()
    if key not in REGISTRY:
        raise KeyError(f"未知棋种 {key}")
    cls = REGISTRY[key]["cls"]
    try:
        g = cls(seed=seed)
    except TypeError:
        g = cls()
    for attr in ("rng", "_rng"):
        r = getattr(g, attr, None)
        if r is not None and hasattr(r, "seed"):
            r.seed(seed)
            break
    return g


# --------------------------------------------------------------------- 飞蝇侧结构

FLY_STRUCTURES: dict[str, dict[str, str]] = {
    "brain": {"label": "真实连接组", "kind": "brain",
              "note": "受检验对象：果蝇连接组派生的脉冲网络"},
    "rewired": {"label": "保度重连对照", "kind": "brain",
                "note": "同度序列的重连图——用来区分「结构」与「规模」的贡献"},
    "reservoir": {"label": "随机储备池", "kind": "control",
                  "note": "同样规模但无连接组结构的普通时序网络"},
    "random": {"label": "纯随机基线", "kind": "baseline",
               "note": "在合法着法里均匀随机——判断结构有没有用的下界"},
    "heuristic": {"label": "外围启发式基线", "kind": "baseline",
                  "note": "直接用棋类自身的启发式选点，不经过脑——上界参考"},
}


@dataclass
class Decision:
    """一次决策的完整记录，含"思考过程"所需的一切。"""

    move: Move
    move_label: str
    structure: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    chosen_index: int = -1
    chosen_probe: dict[str, Any] | None = None
    elapsed_ms: float = 0.0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "move": self.move, "move_label": self.move_label,
            "structure": self.structure, "candidates": self.candidates,
            "chosen_index": self.chosen_index, "chosen_probe": self.chosen_probe,
            "elapsed_ms": round(self.elapsed_ms, 2), "note": self.note,
        }


def _probe_summary(probe: list[dict[str, Any]]) -> dict[str, Any]:
    spikes = [p["spikes"] for p in probe]
    groups = [p["groups"] for p in probe]
    keys = sorted({k for g in groups for k in g})
    return {
        "steps": len(probe),
        "spikes_total": int(sum(spikes)),
        "spikes_per_step": spikes,
        "group_totals": {k: int(sum(g.get(k, 0) for g in groups)) for k in keys},
        "group_per_step": {k: [int(g.get(k, 0)) for g in groups] for k in keys},
    }


class FlyGamePlayer:
    """用脑实例为棋局选着法。

    `structure` 决定用哪种中间结构（或退化为纯基线，不建脑实例）。
    """

    # 每个候选给多长的神经仿真窗口。用主任务同口径的 20 步即可。
    #
    # 曾经把它调到 120，是因为用**脉冲计数**当读出源时 20 步完全没区分度。
    # 但后来查明那是读出源选错了：读出组只有 48 个神经元、20 步里放电约 5 次，
    # 每神经元 0~2 次，**整数计数量化太粗**，24 个候选里大半拿到逐位相同的向量。
    # 换成膜电位向量后，20 步就已经 12/12 全部可分辨：
    #
    #   读出源        20 步            120 步
    #   脉冲计数      1/12 有差异      2/12 有差异
    #   膜电位        12/12 有差异     12/12 有差异
    #
    # 所以窗口回到与主任务一致的 20，顺带快 6 倍。
    GAME_STEPS_DEFAULT = 20

    def __init__(self, structure: str = "brain", connectome: str = "male-cns-gateway",
                 seed: int = 0, steps_per_tick: int | None = None, epsilon: float = 0.05,
                 connectome_dir: str | Path | None = None, local_root: str | Path | None = None,
                 learning: bool = False, use_goal_info: bool = True):
        if structure not in FLY_STRUCTURES:
            raise KeyError(f"未知结构 {structure}")
        self.structure = structure
        self.spec_meta = FLY_STRUCTURES[structure]
        self.connectome = connectome
        self.seed = int(seed)
        self.steps_per_tick = int(steps_per_tick or self.GAME_STEPS_DEFAULT)
        self.epsilon = float(epsilon)
        # 奖励调制的读出器学习。开启后"飞蝇会越下越好"——但提升只能归因于**外围读出器**，
        # 不能表述为"脑内学习"（项目里已有同样的纪律）。
        self.learning = bool(learning)
        # **规则与目标信息开关**。关掉它就把观测最后 4 维（"这步能赢/会不会输"）清零，
        # 等于回到"果蝇只知道棋子怎么摆、不知道什么叫赢"的状态。
        # 留这个开关是为了能做消融——不然"它是不是真靠这条信息选棋"只是猜测。
        self.use_goal_info = bool(use_goal_info)
        self.lr = 0.02
        self.max_delta_ratio = 1.0
        self.pending_reward = 0.0
        self.reward_history: list[float] = []
        self.rng = np.random.default_rng(seed + 977)
        self.engine: Any | None = None
        self.graph_name: str | None = None
        self._proj: np.ndarray | None = None

        # 连接组目录必须走环境变量：容器里 data/ 是可写卷、不含 .npz，
        # 只读的 data/ 挂在 /connectomes。硬编码 "data" 会在容器里直接找不到图。
        self.connectome_dir = str(connectome_dir or os.environ.get(
            "FLYCLUSTER_CONNECTOME_DIR", "data"))
        kind = self.spec_meta["kind"]
        if kind == "brain":
            self._build_brain(self.connectome_dir, local_root)
        elif kind == "control":
            self._build_control(local_root)
        # baseline 不需要引擎

    # ---------------------------------------------------------------- 构建
    def _build_brain(self, connectome_dir: str | Path, local_root: str | Path | None) -> None:
        from services.brain.engine import BrainEngine

        spec = self.connectome
        if self.structure == "rewired" and not spec.endswith("-rewired"):
            spec = f"{spec}-rewired"
        graph = get_graph(spec, data_dir=connectome_dir)
        self.graph_name = graph.meta.name
        cfg = BrainNodeConfig(
            node_id="game-fly", connectome=spec, seed=self.seed,
            learning_mode=("plastic" if self.learning else "readonly"),
            memory_scope="run",
            enable_peer_input=False,            # 单机下棋，没有同伴；开着只会引入无关变量
            enable_memory_input=False,          # 同上：记忆会让不同候选之间的差异来源不唯一
        )
        root = Path(local_root or (Path(connectome_dir).parent / "local" / "game"))
        self.engine = BrainEngine(graph, cfg, hub=None,
                                 steps_per_tick=self.steps_per_tick, local_root=root)
        # **必须关掉记忆的读写**，两个原因：
        #   1. 正确性：每步都会往 JSONL 追写一条记录，于是后面的候选看到的是前面候选
        #      留下的状态，"只有观测不同"这个前提被破坏，热力图会带上顺序效应。
        #   2. 性能：实测每步一次磁盘追写 × 16 个候选 × 20 步 = 320 次写，
        #      单步决策从毫秒级劣化到 3.8~7.5 秒，页面根本没法交互。
        self.engine.local_memory_enabled = False
        self.engine.central_memory_enabled = False
        self._make_proj(graph.meta.checksum)

    def _build_control(self, local_root: str | Path | None) -> None:
        from services.orchestrator.backends import build_control

        graph = get_graph(self.connectome, data_dir=self.connectome_dir)
        cfg = BrainNodeConfig(node_id="game-fly", connectome=self.connectome, seed=self.seed,
                              learning_mode="readonly", memory_scope="run",
                              enable_peer_input=False, enable_memory_input=False)
        root = Path(local_root or "data/local/game")
        self.engine = build_control("reservoir", cfg, None, self.steps_per_tick, root)
        self.engine.set_message_kind("learned_compressed")
        self.graph_name = f"reservoir({graph.n})"
        self._make_proj("reservoir")

    def _make_proj(self, checksum: str) -> None:
        """读出组活动 → 候选倾向分 的**固定随机投影**。

        用 5 个动作分做投影时**所有候选的动作分完全相同**（只差小数点后第 4 位）；
        换成读出组脉冲计数后仍然大半相同（量化太粗）。最终改用**膜电位向量**，
        见 `_readout_source()` 里的实测对比表。

        权重是固定随机值（不是学出来的），这是**外围读出方式的选择**——
        脑只负责提供动力学，读出怎么读是外围的事。边界写在注释里。
        """
        g = rng_for("game-projection", self.seed, checksum)
        n_out = 1
        if self.engine is not None:
            probe = self._readout_source()
            if probe is not None:
                n_out = int(probe.size)
        w = g.standard_normal(max(1, n_out)).astype(np.float32)
        self._proj = w / max(1e-9, float(np.linalg.norm(w)))

        # **可学习的候选读出器**。
        #
        # 这是必须的，不是锦上添花：奖励更新的是脑内读出器的 5 个动作行，
        # 而候选倾向分是用膜电位投影算的 —— 学习更新的东西和真正决定落子的东西
        # 不是同一个，奖励对选点**毫无影响**。
        # 所以把候选打分本身做成可学习的外围读出器（与项目里 ReadoutHead 同一族机制：
        # 奖励调制的 Hebbian，只更新外围权重）。
        self._w_base = self._proj.copy()
        self._w_delta = np.zeros_like(self._w_base)
        self._last_state: np.ndarray | None = None
        self._updates = 0

    def _obs(self, game, move) -> list[float]:
        """统一的观测入口：按开关决定是否保留"规则与目标"槽位。"""
        from services.games.base import GOAL_SLOTS

        v = list(game.observe(move))
        if not self.use_goal_info:
            v[OBS_DIM - GOAL_SLOTS:] = [0.0] * GOAL_SLOTS
        return v

    def _state_of(self, obs: list[float]) -> np.ndarray:
        """跑一次评估并返回读出向量（供打分与奖励更新共用同一份表征）。"""
        _, _, counts = self._evaluate(obs)
        return counts

    def _readout_source(self) -> np.ndarray | None:
        """候选倾向分从哪个向量算出来。

        - 脑/重连：**网络膜电位向量（减均值）**。这是连续量，分辨率足够。
          一开始用的是读出组的脉冲计数，结果 24 个候选里大半拿到**逐位相同**的向量 ——
          读出组只有 48 个神经元、20 步里放电约 5 次，整数计数的量化太粗。
          实测对比（12 个不同观测，看有几个拿到不同取值）：

              | 读出源 | 20 步 | 120 步 |
              | 脉冲计数 | 1/12 | 2/12 |
              | 膜电位 | **12/12** | 12/12 |

          用膜电位是**外围读出方式的选择**，不是脑结构的能力；脑只是照常提供动力学。
          减均值是为了让投影捕捉**空间模式**，否则整体电位水平会把差异淹没。
        - 对照模型（储备池等）：没有神经元组，用隐藏层激活的绝对值。
        """
        eng = self.engine
        if eng is None:
            return None
        net = getattr(eng, "net", None)
        if net is not None and hasattr(net, "v"):
            v = np.asarray(net.v, dtype=np.float32)
            return v - float(v.mean())
        act = getattr(eng, "activations", None)
        if act is not None:
            return np.abs(np.asarray(act, dtype=np.float32))
        return None

    def _reset_for_candidate(self) -> None:
        """把引擎恢复到逐位相同的初始状态，供下一个候选使用。"""
        eng = self.engine
        if eng is None:
            return
        if hasattr(eng, "reset_for_evaluation"):
            eng.reset_for_evaluation()
            return
        # 对照组模型：网络状态就是 activations，不清零的话每个候选继承上一个的状态
        if hasattr(eng, "activations"):
            eng.activations = np.zeros_like(eng.activations)
        if hasattr(eng, "reset_run_stats"):
            eng.reset_run_stats()

    def plasticity(self) -> dict[str, Any]:
        """读出器的可塑性状态，页面上展示"它学了多少"。"""
        return {"updates": self._updates,
                "delta_norm": float(np.abs(self._w_delta).sum()),
                "delta_max": float(np.abs(self._w_delta).max()) if self._w_delta.size else 0.0,
                "rewards": len(self.reward_history)}

    def _trace_candidate(self, obs: list[float]) -> dict[str, Any] | None:
        """对一份观测做一次带逐步追踪的评估，返回 20 帧的放电明细。"""
        eng = self.engine
        if eng is None or not hasattr(eng, "trace"):
            return None
        from flycore.schemas import ObservationPacket

        eng.trace = []
        try:
            self._reset_for_candidate()
            eng.step(BrainStepRequest(
                node_id="game-fly", tick=1,
                observation=ObservationPacket(agent_id=0, tick=1, vector=list(obs),
                                              perception_valid=True),
                run_context={}))
            frames = list(eng.trace)
        finally:
            eng.trace = None

        keys = sorted({k for f in frames for k in f["groups"]})
        return {
            "steps": len(frames),
            "spikes_total": int(sum(f["spikes"] for f in frames)),
            "spikes_per_step": [f["spikes"] for f in frames],
            "group_totals": {k: int(sum(f["groups"].get(k, 0) for f in frames)) for k in keys},
            "group_per_step": {k: [f["groups"].get(k, 0) for f in frames] for k in keys},
            "frames": [{"step": i + 1, "spikes": f["spikes"],
                        "firing": f["firing"], "groups": f["groups"]}
                       for i, f in enumerate(frames)],
        }

    def give_reward(self, reward: float) -> bool:
        """把上一步的收益交给读出器，**立即生效**。

        早先是存成 `pending_reward` 等"下一次评估的第一步"再消费。
        这在棋局**终局**是致命的：赢了之后没有下一步了，
        ±1 的胜负奖励永远不会被用上——果蝇因此从来不知道"什么叫赢"。
        用户的原话很准：光知道棋子怎么走不够，还得知道什么叫赢。
        """
        if not self.learning:
            return False
        eng = self.engine
        if eng is None:
            return False
        self.reward_history.append(float(reward))
        # (a) 更新候选读出器 —— **这个才是真正决定落子的东西**
        applied = False
        if self._last_state is not None and float(reward):
            self._w_delta += self.lr * float(reward) * self._last_state
            cap = self.max_delta_ratio * float(np.abs(self._w_base).max() + 1e-9)
            np.clip(self._w_delta, -cap, cap, out=self._w_delta)
            self._updates += 1
            applied = True
        # (b) 同时驱动脑内读出器（与其他实验口径一致）
        if hasattr(eng, "apply_reward"):
            applied = bool(eng.apply_reward(reward)) or applied
        return applied

    # ---------------------------------------------------------------- 决策
    def decide(self, game, limit: int = 48) -> Decision:
        t0 = time.perf_counter()
        cands = game.candidates(limit=limit)
        if not cands:
            raise RuntimeError("没有候选着法")

        kind = self.spec_meta["kind"]
        if kind == "baseline":
            return self._decide_baseline(game, cands, t0)

        scored: list[dict[str, Any]] = []
        probes: list[dict[str, Any] | None] = []
        for m in cands:
            obs = self._obs(game, m)
            scores, probe, counts = self._evaluate(obs)
            value = float(np.dot(counts, self._w_base + self._w_delta))
            scored.append({
                "move": [list(x) if isinstance(x, (list, tuple)) else x for x in m]
                if isinstance(m, (list, tuple)) else m,
                "label": game.move_label(m),
                "action_scores": [round(float(s), 5) for s in scores],
                "readout_activity": round(float(np.abs(counts).sum()), 2),
                "value": round(value, 6),
            })
            probes.append(probe)

        vals = np.array([c["value"] for c in scored], dtype=np.float64)
        pick = int(np.argmax(vals))
        # 以 epsilon 概率改选，避免完全确定性带来的"永远同一步"
        if self.epsilon > 0 and self.rng.random() < self.epsilon and len(scored) > 1:
            pick = int(self.rng.integers(0, len(scored)))

        chosen = cands[pick]
        # 记住被选中候选的状态向量：奖励回来后要沿它更新权重。
        # 不记住的话，奖励更新的是"另一个候选"的表征，学习就没有意义了。
        self._last_state = np.asarray(
            self._state_of(self._obs(game, chosen)), dtype=np.float32)
        # 只对**选中**的候选做一次带逐步追踪的复算，用来画"思考过程"。
        # 不为所有候选开追踪：那会把每步的放电明细都记下来，慢且没人看。
        trace = self._trace_candidate(self._obs(game, chosen))
        return Decision(
            move=chosen, move_label=game.move_label(chosen), structure=self.structure,
            candidates=scored, chosen_index=pick,
            chosen_probe=trace,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            note=("每个候选都从逐位相同的初始状态出发（含噪声序列），"
                  "唯一差别是观测——否则热力图就是噪声图"),
        )

    def _decide_baseline(self, game, cands, t0: float) -> Decision:
        if self.structure == "random":
            pick = int(self.rng.integers(0, len(cands)))
            vals = [None] * len(cands)
            note = "纯随机基线：在合法着法里均匀随机，不经过任何网络"
        else:
            # 外围启发式：直接用棋类自身的评估（对棋种是"上帝视角"，只作参考上界）
            best, bv = 0, -1e18
            for i, m in enumerate(cands):
                game.apply(m)
                v = game.heuristic()
                game.undo()
                if v > bv:
                    best, bv = i, v
            pick = best
            vals = None
            note = "外围启发式基线：直接用棋类评估函数选点，不经过脑——性能上界参考"
        chosen = cands[pick]
        return Decision(
            move=chosen, move_label=game.move_label(chosen), structure=self.structure,
            candidates=[{"move": m, "label": game.move_label(m), "value": None,
                         "action_scores": None} for m in cands],
            chosen_index=pick, chosen_probe=None,
            elapsed_ms=(time.perf_counter() - t0) * 1000, note=note)

    def _evaluate(self, obs: list[float]) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
        """把一份观测送进脑，返回（5 个动作分, 逐步探测记录, 读出向量）。"""
        from flycore.schemas import ObservationPacket

        eng = self.engine
        self._reset_for_candidate()

        # **一次 engine.step() 就是一个决策步**——它内部已经跑了 steps_per_tick 次网络步进。
        # 早先在外面又套了一层 20 次循环，等于每个候选跑 400 步仿真（预期的 20 倍），
        # 单步决策从毫秒级劣化到 3.8~7.5 秒。
        ctx: dict[str, Any] = {}
        if self.pending_reward:
            ctx["last_reward"] = float(self.pending_reward)
            self.reward_history.append(float(self.pending_reward))
            self.pending_reward = 0.0
        req = BrainStepRequest(
            node_id="game-fly", tick=1,
            observation=ObservationPacket(agent_id=0, tick=1, vector=list(obs),
                                          perception_valid=True),
            run_context=ctx,   # 不能传 None：schema 要求 dict
        )
        resp: BrainStepResponse = eng.step(req)
        scores = np.asarray(resp.action_scores, dtype=np.float32)
        src = self._readout_source()
        counts = src if src is not None else np.zeros(1, dtype=np.float32)
        return scores, [self._step_probe(1, resp)], counts

    def _step_probe(self, step: int, resp: BrainStepResponse) -> dict[str, Any]:
        rec = {"step": step, "spikes": int(resp.spikes_emitted),
               "groups": {}, "firing": []}
        eng = self.engine
        groups = getattr(getattr(eng, "io", None), "groups", None)
        net = getattr(eng, "net", None)
        if net is not None and groups is not None:
            fired = np.flatnonzero(net.last_spikes)
            rec["firing"] = [int(i) for i in fired[:400]]
            for gname in ("sensor", "central", "memory", "peer",
                          "message_out", "action_out"):
                idx = getattr(groups, gname, None)
                if idx is None:
                    continue
                rec["groups"][gname] = int(np.isin(fired, idx).sum())
        return rec


# --------------------------------------------------------------------- 对局管理

@dataclass
class GameSession:
    session_id: str
    game_key: str
    difficulty: str
    structure: str
    seed: int
    game: Any
    player: FlyGamePlayer
    moves: list[dict[str, Any]] = field(default_factory=list)
    last_decision: dict[str, Any] | None = None
    last_pc_move: dict[str, Any] | None = None
    created_at: str = ""
    heuristic_curve: list[float] = field(default_factory=list)
    last_reward: dict[str, Any] | None = None

    def state(self) -> dict[str, Any]:
        r = self.game.render()
        r["meta"].update({
            "session_id": self.session_id, "game_key": self.game_key,
            "difficulty": self.difficulty, "structure": self.structure,
            "structure_label": FLY_STRUCTURES[self.structure]["label"],
            "seed": self.seed,
        })
        r["moves"] = self.moves
        r["heuristic_curve"] = self.heuristic_curve
        return r


class SessionStore:
    def __init__(self, limit: int = 40):
        self.sessions: dict[str, GameSession] = {}
        self.limit = limit

    def put(self, s: GameSession) -> None:
        self.sessions[s.session_id] = s
        if len(self.sessions) > self.limit:
            oldest = sorted(self.sessions.values(),
                            key=lambda x: x.created_at)[: len(self.sessions) - self.limit]
            for o in oldest:
                self.sessions.pop(o.session_id, None)

    def get(self, sid: str) -> GameSession:
        if sid not in self.sessions:
            raise KeyError(sid)
        return self.sessions[sid]
