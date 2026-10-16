#!/usr/bin/env python3
"""端到端冒烟测试：不依赖网络与 Docker，逐项验证关键不变量。

跑法：`make smoke` 或 `PYTHONPATH=. .venv/bin/python tools/smoke_test.py`

覆盖的检查项：
  连接组   度序列/权重多重集/抑制集保持、输入→输出可达、重连确实改变了结构
  神经内核 数值有界、脉冲上限触发时会计入 dropped_spikes、实时比可测
  编解码   消息 int16 往返、带宽口径、观测维度
  环境     目标可拾取、观测维度、碰撞与重复探索计数
  集群     单次运行能产出评分、遥测、通信与记忆记账、能力归因
  记忆中枢 防回声、派生不计证据、冲突保留、过期排除、快照回溯、来源链
  开放 API 鉴权、任务注册、跑一次运行并取回报告
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 整轮冒烟使用独立的 data 目录：否则上一次运行吊销过的密钥会让下一次直接 401，
# 排查起来会误以为是代码退化。必须在导入 services 之前设置。
os.environ.setdefault("FLYCLUSTER_DATA", tempfile.mkdtemp(prefix="flycluster-smoke-data-"))
os.environ.setdefault("FLYCLUSTER_API_KEYS", "smoke:dev-local-key:6000")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn() or ""
        RESULTS.append((name, True, str(detail)))
    except Exception as exc:
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        if "--verbose" in sys.argv:
            traceback.print_exc()


def _demo_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="flycluster-smoke-"))


# ------------------------------------------------------------------ 连接组


def t_connectome_invariants() -> str:
    from flycore.connectome import generate_synthetic_connectome, rewire_matched
    g = generate_synthetic_connectome(n_neurons=2000, seed=0)
    r = rewire_matched(g, seed=0)
    assert g.meta.io_reachable, "合成图的输入→输出必须可达"
    assert r.meta.io_reachable, "重连图的输入→输出必须可达"
    o1, i1 = g.degree_sequences()
    o2, i2 = r.degree_sequences()
    assert np.array_equal(np.sort(o1), np.sort(o2)), "重连改变了出度序列"
    assert np.array_equal(np.sort(i1), np.sort(i2)), "重连改变了入度序列"
    assert np.allclose(np.sort(g.weight), np.sort(r.weight)), "重连改变了权重多重集"
    assert np.array_equal(g.inhibitory, r.inhibitory), "重连改变了抑制神经元集合"
    assert abs(g.meta.modularity - r.meta.modularity) > 0.2, "重连应显著破坏模块化"
    assert r.meta.rewire_swaps > 0, "重连未执行任何交换"
    return (f"n={g.n} edges={g.edge_count()} 模块化 {g.meta.modularity:.3f}→"
            f"{r.meta.modularity:.3f} 交换 {r.meta.rewire_swaps} 次")


def t_connectome_license_recorded() -> str:
    from flycore.connectome import generate_synthetic_connectome
    m = generate_synthetic_connectome(n_neurons=300, seed=1).meta
    assert m.license and m.license != "UNKNOWN", "许可必须登记"
    assert m.dynamics_modeled, "必须声明动力学为人为建模"
    assert "非任何真实" in m.notes or "不是任何真实" in m.notes, "合成图必须标注非真实数据"
    return f"license={m.license} source={m.source}"


# ------------------------------------------------------------------ 神经内核


def t_lif_bounded_and_drop_accounting() -> str:
    from flycore.connectome import generate_synthetic_connectome
    from flycore.network import SpikingNetwork
    from flycore.schemas import DynamicsConfig
    g = generate_synthetic_connectome(n_neurons=800, seed=0)
    net = SpikingNetwork(g, DynamicsConfig(max_spikes_per_step=5), seed=0)
    for _ in range(60):
        # 注入确定性电流把网络推到明显高于上限的发放水平，
        # 否则"上限没触发"可能只是网络本来就安静，测不出记账是否正确。
        if _ < 10:
            net.add_current(np.full(g.n, 60.0, dtype=np.float32))
        net.step()
    h = net.health()
    assert h["bounded"], "膜电位出现 NaN/Inf"
    assert h["nan_events"] == 0
    total = h["spikes"] + h["dropped_spikes"]
    assert total > 0, "网络完全没有活动，工作点失效"
    assert h["dropped_spikes"] > 0, "单步上限设为 5 却从未触发，丢弃计数未生效"
    assert 0.0 <= h["drop_rate"] <= 1.0
    assert h["realtime_ratio"] > 0, "实时比未记录"
    return (f"放电 {h['spikes']} 丢弃 {h['dropped_spikes']} 丢弃率 {h['drop_rate']:.4f} "
            f"实时比 {h['realtime_ratio']:.1f} 有界 {h['bounded']}")


def t_connectome_direction_matters() -> str:
    """方向反了会让输入对读出毫无影响——这是曾经真实发生过的缺陷，必须守住。"""
    from flycore.coding import BrainIO
    from flycore.connectome import generate_synthetic_connectome
    from flycore.network import SpikingNetwork
    from flycore.schemas import DynamicsConfig, OBS_DIM
    g = generate_synthetic_connectome(n_neurons=1000, seed=0)
    out = g.output_ids

    def measure(gain: float) -> float:
        net = SpikingNetwork(g, DynamicsConfig(), seed=0)
        io = BrainIO.build(g, seed=0)
        io.sensory.proj.gain = gain
        rng = np.random.default_rng(3)
        rates = []
        for t in range(120):
            obs = (rng.random(OBS_DIM) < 0.15).astype(np.float32)
            cur, _ = io.sensory.encode(obs)
            c = np.zeros(g.n, dtype=np.float32)
            c[io.groups.sensor] = cur
            net.add_current(c)
            o = net.step()
            if t > 40:
                rates.append(float(o.spikes[out].mean()))
        return float(np.mean(rates))

    base, hi = measure(0.0), measure(200.0)
    assert hi > base * 1.3, (f"提高感觉增益后读出组放电率未显著变化（{base:.5f}→{hi:.5f}），"
                             "怀疑突触传播方向又反了")
    return f"读出组放电率 {base:.5f} → {hi:.5f}（×{hi / max(1e-9, base):.2f}）"


# ------------------------------------------------------------------ 编解码


def t_message_codec() -> str:
    from flycore.coding import MessageCodec
    c = MessageCodec()
    v = np.linspace(-1, 1, 8, dtype=np.float32)
    enc = c.encode(v)
    assert len(enc) == 8 and all(-32767 <= x <= 32767 for x in enc)
    back = c.decode(enc)
    assert np.max(np.abs(back - v)) < 1e-3, "int16 量化往返误差过大"
    assert MessageCodec.raw_bytes_per_second(4, 5.0) == 320, "带宽口径应为 n×8×2×频率"
    return f"往返最大误差 {float(np.max(np.abs(back - v))):.2e}；4 节点 5Hz = 320 B/s"


def t_observation_dim() -> str:
    from flycore.schemas import OBS_DIM
    from services.env.world import GridWorld, TaskSpec
    w = GridWorld(TaskSpec(n_agents=1, n_targets=2, max_ticks=5, seed=0))
    obs, valid = w.observe(0)
    assert obs.size == OBS_DIM, f"观测维度 {obs.size} != {OBS_DIM}"
    assert valid is True
    return f"OBS_DIM={OBS_DIM}"


# ------------------------------------------------------------------ 环境


def t_env_mechanics() -> str:
    from services.env.world import GridWorld, TaskSpec
    w = GridWorld(TaskSpec(n_agents=1, n_targets=3, max_ticks=50, seed=7))
    tx, ty = sorted(w.targets)[0]
    w.agents[0].x, w.agents[0].y = tx, ty
    w.step([4])
    assert w.agents[0].targets_found == 1, "站到目标格上却没有判定命中"
    w2 = GridWorld(TaskSpec(n_agents=2, n_targets=1, max_ticks=10, seed=3))
    a, b = w2.agents
    b.x, b.y = a.x, a.y + 1
    w2.step([1, 4])   # a 向下移入 b 所在格，b 原地不动 → 同格碰撞
    assert w2.agents[0].collisions == 1 and w2.agents[1].collisions == 1, "同格碰撞未被计入"
    return "目标拾取与碰撞判定正确"


def t_env_perception_dropout() -> str:
    """感知失效必须真的让观测变成"无信息帧"。

    `RunConfig` 里曾经存在一个声明了却从未被 runner 读取的 `perception_dropout` 字段——
    **配置项存在但没接通，比没有这个配置更危险**：实验会静默跑成满感知，
    而报告里写的却是"感知失效 50% 条件下"。现在感知失效只有 TaskSpec.perception_fraction
    一个来源，这条测试守住它真的生效。
    """
    from services.env.world import GridWorld, TaskSpec
    # 注意语义：`perception_fraction` 是"感知**有效**的 tick 占比"，
    # 不是失效率。设 0.35 意味着约 65% 的 tick 拿不到信息。
    # 这个字段名极易读反——第一版测试就把它当成了失效率。
    keep = 0.35
    w = GridWorld(TaskSpec(n_agents=1, n_targets=2, max_ticks=2000, seed=3,
                           perception_fraction=keep))
    valids = []
    for _ in range(2000):
        w.step([4])
        valids.append(w.observe(0)[1])
    valid_rate = sum(valids) / len(valids)
    assert 0.25 < valid_rate < 0.46, \
        f"实测感知有效率 {valid_rate:.2f} 与 perception_fraction={keep} 相差过大"
    rate = 1.0 - valid_rate

    w0 = GridWorld(TaskSpec(n_agents=1, n_targets=2, max_ticks=10, seed=3,
                            perception_fraction=0.0))
    obs0, valid0 = w0.observe(0)
    assert valid0 is False, "perception_fraction=0 时不应有有效观测"
    from flycore.schemas import OBS_DIM, POSE_DIM, VIEW_CHANNELS, VIEW_SIZE
    view_dim = VIEW_SIZE * VIEW_SIZE * VIEW_CHANNELS
    assert not obs0[:view_dim].any(), "感知失效时外感受视野必须清零"
    assert obs0[view_dim:].any(), "感知失效不应抹掉本体感觉与时间进度"
    assert np.isfinite(obs0).all() and obs0.size == OBS_DIM
    from flycore.schemas import RunConfig
    assert not hasattr(RunConfig(), "perception_dropout"), \
        "RunConfig 不应再有 perception_dropout：两个来源必然静默漂移"
    return (f"感知有效率实测 {valid_rate:.2f}（设定 {keep}，即失效 {rate:.2f}）；"
            "失效时视野清零但保留本体感觉；RunConfig 无重复字段")


def t_env_visibility_split() -> str:
    from services.env.world import GridWorld, TaskSpec
    spec = TaskSpec(n_agents=2, n_targets=4, max_ticks=10, seed=5,
                    visibility_regions=[[0, 0, 15, 31], [16, 0, 31, 31]])
    w = GridWorld(spec)
    assert not w.visible_to(0, 20, 5), "左半区节点不应看到右半区目标"
    assert w.visible_to(0, 3, 5) and w.visible_to(1, 20, 5)
    return "信息拆分生效：0 号只见左半区，1 号只见右半区"


# ------------------------------------------------------------------ 集群


def t_cluster_run() -> str:
    from flycore.schemas import RunConfig
    from services.env.world import TaskSpec
    from services.orchestrator.runner import ClusterRunner, RunnerOptions
    run = RunConfig(run_id="smoke-run", seed=2, backend="brain", connectome="synthetic-1000",
                    agents=4, max_ticks=30, learning_mode="plastic", enable_memory_channel=True)
    task = TaskSpec(n_agents=4, n_targets=4, max_ticks=30, seed=2)
    rep = ClusterRunner(run, task, RunnerOptions(data_dir=str(_demo_dir()))).run_cluster()
    assert rep.telemetry.get("steps", 0) == 4 * 30 * 20, f"仿真步数异常：{rep.telemetry}"
    assert rep.comm_accounting["messages_issued"] > 0, "开通信后没有任何消息发出"
    assert rep.comm_accounting["formula_bytes_per_second"] == 320, "通信口径公式不符"
    assert rep.memory_accounting["hub"]["records_total"] > 0, "中心记忆未收到任何经验"
    assert rep.attribution, "缺少能力归因"
    assert len(rep.gates) == 3, "应输出三道门"
    return (f"score={rep.score.score:.2f} found={rep.score.targets_found}/{rep.score.targets_total} "
            f"steps={rep.telemetry['steps']} 消息={rep.comm_accounting['messages_issued']} "
            f"记忆={rep.memory_accounting['hub']['records_total']} 实时比={rep.telemetry['realtime_ratio_min']}")


# ------------------------------------------------------------------ 记忆中枢


def t_comm_ablation_effective() -> str:
    """通信消融必须真的改变行为。

    早期版本漏了"关闭通信时不投递消息"这一步，导致 comm/nocomm 两组的分数逐位相同——
    消融看似做了，实际什么都没变。这条检查专门守住它。
    """
    from flycore.schemas import RunConfig
    from services.env.world import TaskSpec
    from services.orchestrator.runner import ClusterRunner, RunnerOptions
    d = _demo_dir()
    task = TaskSpec(n_agents=4, n_targets=4, max_ticks=25, seed=6)
    out = {}
    for on in (True, False):
        run = RunConfig(run_id=f"ablate-{on}", seed=6, backend="brain", connectome="synthetic-800",
                        agents=4, max_ticks=25, learning_mode="plastic",
                        enable_memory_channel=False, enable_peer_messages=on)
        rep = ClusterRunner(run, task, RunnerOptions(data_dir=str(d / str(on)))).run_cluster()
        out[on] = rep
    on, off = out[True], out[False]
    assert on.comm_accounting["messages_issued"] > 0, "开通信时应发出消息"
    assert on.comm_accounting["messages_delivered"] > 0, "开通信时应有消息送达"
    assert off.comm_accounting["messages_issued"] == 0, "关通信时不应有消息进入通道"
    assert off.comm_accounting["messages_delivered"] == 0, "关通信时不应有任何消息送达"
    assert off.comm_accounting["raw_send_bytes"] == 0, "关通信时发送字节应为 0"
    # 运行级设置必须真的下发到节点上，并在报告里回报"实际生效值"。
    # 远端容器只认环境变量，不推的话 enable_memory_channel 会被静默忽略：
    # 实测过一次 16 节点 Docker 运行——请求里写着关闭中心记忆，逐节点却仍报 central_hits=234。
    pushed = off.comm_accounting.get("node_settings_pushed") or []
    assert pushed, "报告里没有回报下发的节点设置，无法核对开关是否生效"
    assert pushed[0]["central_memory_enabled"] is False, \
        f"中心记忆开关未落到节点上：{pushed[0]}"
    assert off.comm_accounting["messages_produced_by_nodes"] > 0, \
        "节点仍应产出消息，只是不进通道——这才是消融生效的证据"
    # 逐节点计数对比：关通信时 central/local hits 应为 0（未接中心记忆）
    assert sum(n["central_hits"] or 0 for n in off.memory_accounting["per_node"]) == 0
    return (f"开：发出 {on.comm_accounting['messages_issued']} 送达 "
            f"{on.comm_accounting['messages_delivered']}；关：发出 0、节点仍产出 "
            f"{off.comm_accounting['messages_produced_by_nodes']} 条未投递")


def t_local_memory_persists() -> str:
    """本地经历必须真的落盘。

    实验 A 的保留测试 1（重启后仍能检索旧记录）就是为这条准备的：早期版本只把记录放在
    内存里、从不写文件，这道测试会永远失败，而失败原因很难从性能数字上看出来。
    """
    from flycore.schemas import BrainNodeConfig, BrainStepRequest, ObservationPacket
    from services.brain.engine import BrainEngine
    from flycore.connectome import generate_synthetic_connectome
    from services.env.world import GridWorld, TaskSpec
    root = _demo_dir() / "local"
    graph = generate_synthetic_connectome(n_neurons=300, seed=0)
    cfg = BrainNodeConfig(node_id="persist-node", connectome="x", seed=0, memory_scope="project")
    w = GridWorld(TaskSpec(n_agents=1, n_targets=2, max_ticks=20, seed=1))

    def make():
        e = BrainEngine(graph, cfg, hub=None, steps_per_tick=5, local_root=root)
        e.central_memory_enabled = False  # 只测本地落盘，不牵扯中心
        return e

    first = make()
    for t in range(1, 6):
        obs, _ = w.observe(0)
        a = w.agents[0]
        first.step(BrainStepRequest(
            node_id="persist-node", tick=t,
            observation=ObservationPacket(agent_id=0, tick=t, vector=obs.tolist()),
            run_context={"run_id": "r", "task_id": "t", "self_cell": [a.x, a.y]}))
    size_before = first.local.stats()["size"]
    assert size_before >= 5, f"本地记忆条数异常：{size_before}"

    second = make()                        # 新建引擎 == 进程重启
    size_after = second.local.stats()["size"]
    assert size_after == size_before, \
        f"重启后记录的条数不一致（{size_before} → {size_after}），说明回填逻辑有重复写入或未落盘"
    cue = np.asarray(first.local.cues[0], dtype=np.float32)
    hits = second.local.retrieve(cue, k=2)
    assert hits, "重启后检索不到任何旧记录，说明经验没有落盘"
    lines = (root / "persist-node.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == size_after, f"文件行数 {len(lines)} 与内存条数 {size_after} 不一致"
    return f"落盘 {size_after} 条，重启后回填 {size_after} 条，检索命中 {len(hits)} 条"


def t_local_memory_knob_live() -> str:
    """`k_local` 必须是活的旋钮：本地记忆真的被检索到。

    背景：`runner` 曾把 `enable_memory_channel`（本意是**中心**记忆通道）直接接到一个
    记忆总开关上，导致主对照里**本地记忆被全程关闭**。症状是在分条件扫描中，
    `k_local = 0 / 2 / 8 / 16` 四个取值给出了**完全相同的 Δ**——
    和"消融后两组结果逐位相同"是同一类失效：旋钮接错了地方。

    这条测试直接断言检索真的发生过，而不是只看配置字段有没有传下去。
    """
    from flycore.schemas import RunConfig
    from services.env.world import TaskSpec
    from services.orchestrator.runner import ClusterRunner, RunnerOptions
    d = _demo_dir()
    task = TaskSpec(n_agents=2, n_targets=4, max_ticks=15, seed=11)

    def run(k_local: int):
        cfg = RunConfig(run_id=f"knob-{k_local}", seed=11, backend="brain",
                        connectome="synthetic-800", agents=2, max_ticks=15,
                        learning_mode="accumulate",
                        enable_memory_channel=False,   # 中心通道关闭
                        enable_peer_messages=False,
                        k_local=k_local, k_central=0)
        return ClusterRunner(cfg, task, RunnerOptions(data_dir=str(d / str(k_local)))).run_cluster()

    off, on = run(0), run(16)
    hits_off = sum(int(n.get("local_hits") or 0) for n in off.memory_accounting["per_node"])
    hits_on = sum(int(n.get("local_hits") or 0) for n in on.memory_accounting["per_node"])
    assert hits_off == 0, f"k_local=0 时不应发生本地检索，实际 {hits_off} 次"
    assert hits_on > 0, ("k_local=16 时本地检索一次都没发生，说明开关接错了地方"
                        "（本地记忆被中心通道开关连带关掉）")
    return (f"k_local=0 → 检索 0 次；k_local=16 → 检索 {hits_on} 次；"
            "本地记忆开关独立于中心通道")


def t_goal_knowledge() -> str:
    """棋局：果蝇必须能"感觉到"规则与目标。

    没有这条，它只是在玩"挪棋子、争子力"而不是下棋——一步将死的棋
    和一步普通将军在它眼里没有区别（**将军 ≠ 将死**）。

    检查三件事：
      1. 观测最后 4 维是目标槽位，且能识别出"这步能赢"
      2. 关掉目标信息（清零这 4 维）后，观测确实变了——消融真的生效
      3. 三个棋种都遵守这个槽位契约
    """
    import numpy as np

    from flycore.schemas import OBS_DIM
    from services.games.base import GOAL_SLOTS
    from services.games.checkers import Checkers
    from services.games.gomoku import Gomoku
    from services.games.xiangqi import BLACK, CELLS, EMPTY, KING, RED, ROOK, Xiangqi, _Engine, _idx

    def xq(pieces, side=RED):
        g = Xiangqi()
        b = [EMPTY] * CELLS
        for (r, c), p_ in pieces.items():
            b[_idx(r, c)] = p_
        g._board = b
        g._kings = {RED: -1, BLACK: -1}
        for sq in range(CELLS):
            if b[sq] == KING:
                g._kings[RED] = sq
            elif b[sq] == -KING:
                g._kings[BLACK] = sq
        g._history = [] if side == RED else [None]
        from services.games.base import Outcome as _O
        g._result = _O()
        g._last = None
        g._ver += 1
        g._cache_ver = -1
        g._legal_cache = None
        g._ann_cache = None
        g._scratch = _Engine(b, g._kings, share=True)
        return g

    g = xq({(9, 3): KING, (0, 4): -KING, (1, 8): ROOK, (5, 0): ROOK})
    mates = g._goal_facts()["mates"]
    assert mates, "该局面应存在一步将死，测试本身失效"
    v = g.observe(sorted(mates)[0])
    assert len(v) == OBS_DIM
    assert v[OBS_DIM - GOAL_SLOTS] == 1.0, f"将死着法的 win_now 槽位应为 +1，实为 {v[OBS_DIM-GOAL_SLOTS]}"
    other = next(m for m in g.legal_moves() if tuple(m) not in {tuple(x) for x in mates})
    v2 = g.observe(tuple(other))
    assert v2[OBS_DIM - GOAL_SLOTS] == -1.0, "普通着法的 win_now 槽位应为 -1"

    # 消融必须真的改变观测
    zeroed = list(v)
    zeroed[OBS_DIM - GOAL_SLOTS:] = [0.0] * GOAL_SLOTS
    assert zeroed != list(v), "清零目标槽位后观测没变，消融不生效"

    # 三个棋种都遵守契约
    for gm in (Gomoku(), Xiangqi(), Checkers()):
        o = gm.observe(None)
        assert len(o) == OBS_DIM, f"{gm.name} 观测 {len(o)} 维"
        for x in o[OBS_DIM - GOAL_SLOTS:]:
            assert x in (1.0, -1.0), f"{gm.name} 的目标槽位取值应为 ±1，实为 {x}"

    return (f"象棋一步将死被识别为 win_now=+1（共 {len(mates)} 手取胜着法）；"
            f"清零目标槽位后观测改变（消融生效）；三个棋种均遵守 {GOAL_SLOTS} 维目标槽位契约")


def t_backend_protocol_complete() -> str:
    """两种节点后端必须实现协议里声明的全部方法。

    真实故障：`HttpNodeBackend` 漏了 `reset`，而 runner 当时用
    `getattr(n, "reset", None)` + `if callable(...)` 兜底——于是"方法没实现"
    被静默变成"跳过"，远端容器从未被重置，报告里的"本次运行步数"一直是累计值。
    这条测试直接比对协议成员，缺一个就报错。
    """
    from services.orchestrator import backends as B

    proto = sorted(m for m in dir(B.NodeBackend) if not m.startswith("_"))
    assert proto, "协议里没有可检查的方法，测试本身失效"
    missing: dict[str, list[str]] = {}
    for cls in (B.HttpNodeBackend, B.InProcessNodeBackend):
        miss = [m for m in proto if not callable(getattr(cls, m, None))]
        if miss:
            missing[cls.__name__] = miss
    assert not missing, f"后端未实现协议声明的全部方法：{missing}"
    return f"HttpNodeBackend 与 InProcessNodeBackend 均实现 {len(proto)} 个协议方法"


def t_container_run_isolation() -> str:
    """脑容器连续跑两次，第二次的遥测必须是干净的，本地记忆必须换到新目录。

    实测过的真实故障：`Telemetry` 是累积累加器，而容器里的引擎是常驻单例。
    同一容器连跑三次，报告分别显示 64000 / 128000 / 192000 步——而每次真实步数都是 64000。
    "本次运行的仿真步数"变成了"从容器启动至今的累计值"。
    这类错误**不报错**，但让"性能数字必须与可信度遥测同时出现"这条纪律直接失效——
    而且是在最容易被引用的那一列上失效。
    """
    import os as _os
    d = _demo_dir()
    _os.environ["DATA_DIR"] = str(d)
    _os.environ["BRAIN_CONNECTOME"] = "synthetic-300"
    _os.environ["BRAIN_CONNECTOME_DIR"] = "."
    from fastapi.testclient import TestClient
    from services.brain import app as brain_mod

    c = TestClient(brain_mod.app)
    req = {"node_id": "node-0", "tick": 1,
           "observation": {"agent_id": 0, "tick": 1, "vector": [0.0] * 105},
           "peer_messages": [], "run_context": {"run_id": "r1", "task_id": "t"}}

    def steps_after_a_run(run_id: str, n_steps: int = 5) -> int:
        r = c.post("/reset", json={"run_id": run_id})
        assert r.status_code == 200, r.text
        for i in range(n_steps):
            req["tick"] = i + 1
            c.post("/step", json=req)
        h = c.get("/health").json()
        return int(h["simulation"]["steps"]), r.json()

    s1, r1 = steps_after_a_run("run-A")
    s2, r2 = steps_after_a_run("run-B")
    assert s1 > 0, "第一次运行没有产生任何仿真步数，测试本身失效"
    assert s2 == s1, (
        f"第二次运行的步数 {s2} 与第一次 {s1} 不一致——遥测在容器里累加了，"
        "报告里的本次运行步数其实是容器启动至今的累计值")
    assert r1["run_id"] == "run-A" and r2["run_id"] == "run-B"

    # 本地记忆必须换到按 run 隔离的目录，否则第二次运行会带着第一次的经验起跑
    assert (d / "local" / "run-A").exists() and (d / "local" / "run-B").exists(), \
        "本地记忆没有按 run 隔离：容器内仍是所有运行共用一个目录"
    return (f"连跑两次步数均为 {s1}（未累加）；本地记忆已按 run 隔离到 "
            f"{d.name}/local/run-A 与 run-B")


def t_memory_echo_guard() -> str:
    from flycore.schemas import ExperienceRecord
    from flycore.util import now_iso
    from services.memory_hub.service import MemoryHubService
    hub = MemoryHubService(str(_demo_dir() / "hub.sqlite3"))

    def rec(rid: str, producer: str, origin: str, kind: str = "observation", parents=None,
            derived=None):
        return ExperienceRecord(
            record_id=rid, origin_event_id=origin, producer_node=producer, kind=kind,
            task_id="t1", scope="project", payload={"key": "cell:1,1", "conclusion": 3},
            parents=parents or [], derived_from_origin=derived, confidence=0.6,
            observed_at=now_iso(), valid_from=now_iso())

    a = hub.submit(rec("r-a", "node-A", "evt-1"))
    assert a.accepted and a.evidence_count_delta == 1, "首条原始经验应计入证据"
    dup = hub.submit(rec("r-a", "node-A", "evt-1"))
    assert not dup.accepted, "同一节点重复提交应被拒绝"
    echo = hub.submit(rec("r-b", "node-B", "evt-1"))
    assert echo.is_echo and echo.evidence_count_delta == 0, "转发来的同一事件必须判为回声且不计证据"
    derived = hub.submit(rec("r-c", "node-C", "evt-2", kind="inference",
                             parents=["r-a"], derived="evt-1"))
    assert derived.accepted and derived.evidence_count_delta == 0, "派生经验不应计入独立证据"
    prov = hub.provenance("evt-1")
    assert prov["independent_sources"] == 1, f"独立来源应为 1，实际 {prov['independent_sources']}"
    hub.close()
    return (f"原始 1 条、重复被拒、回声标记 {echo.is_echo}、派生不计证据、"
            f"独立来源 {prov['independent_sources']}")


def t_memory_conflict_and_expire() -> str:
    from flycore.schemas import ExperienceRecord, MemoryQuery
    from flycore.util import now_iso
    from services.memory_hub.service import MemoryHubService
    hub = MemoryHubService(str(_demo_dir() / "hub2.sqlite3"))
    # 时间戳必须落在"快照之前"：用过去的固定时间，否则 time-travel 语义测不出来
    for i, (concl, tick) in enumerate([(3, 1), (2, 5)]):
        ts = f"2020-01-0{i + 1}T10:00:00+0800"
        hub.submit(ExperienceRecord(
            record_id=f"c{i}", origin_event_id=f"e{i}", producer_node=f"n{i}", kind="observation",
            task_id="t2", scope="project", payload={"key": "cell:9,9", "conclusion": concl},
            conditions={"tick": tick}, observed_at=ts, valid_from=ts))
    conflicts = hub.conflicts("t2")
    assert conflicts and conflicts[0]["n_conclusions"] == 2, "同一 key 的冲突未被保留"
    assert len(conflicts[0]["records"]) == 2, "冲突记录应两条并存而非覆盖"

    snap = hub.snapshot("before expire")
    exp = hub.expire(["c0", "c1"])          # valid_to 设为当前时间
    assert exp["expired"] == 2, "过期标记未生效"
    cq = MemoryQuery(task_id="t2", scope="project", k=10)
    now = hub.query(cq)
    assert len(now["retrieved"]) == 0, f"过期经验仍被检索到：{len(now['retrieved'])}"
    past = hub.query(MemoryQuery(task_id="t2", scope="project", k=10,
                                 as_of_version=snap["version"]))
    assert len(past["retrieved"]) == 2, "按快照回溯应能看到过期前的经验"
    assert past["query_cost_bytes"] > 0, "检索返回字节数必须计入通信预算"
    hub.close()
    return (f"冲突 {conflicts[0]['n_conclusions']} 种结论并存；过期后当前视图 0 条、"
            f"快照视图 {len(past['retrieved'])} 条；检索计费 {past['query_cost_bytes']} B")


def t_memory_scope_isolation() -> str:
    from flycore.schemas import ExperienceRecord, MemoryQuery
    from flycore.util import now_iso
    from services.memory_hub.service import MemoryHubService
    hub = MemoryHubService(str(_demo_dir() / "hub3.sqlite3"))
    c0 = np.zeros(32, dtype=np.float32)
    c0[0] = 1.0
    hub.submit(ExperienceRecord(
        record_id="s1", origin_event_id="se1", producer_node="n", kind="observation",
        run_id="run-X", task_id="t3", scope="run", obs_vector=c0.tolist(),
        payload={"key": "k", "conclusion": 1}, observed_at=now_iso(), valid_from=now_iso()))
    same = hub.query(MemoryQuery(cue_vector=c0.tolist(), task_id="t3", scope="run",
                                 run_id="run-X", k=5))
    other = hub.query(MemoryQuery(cue_vector=c0.tolist(), task_id="t3", scope="run",
                                  run_id="run-Y", k=5))
    none = hub.query(MemoryQuery(cue_vector=c0.tolist(), task_id="t3", scope="run", k=5))
    assert len(same["retrieved"]) == 1, "同运行应检索到"
    assert len(other["retrieved"]) == 0, "不同运行不应看到彼此的经验"
    assert len(none["retrieved"]) == 0, "未指定 run_id 时不得默认跨运行共享"
    hub.close()
    return "同运行可见 / 跨运行不可见 / 缺 run_id 默认不可见"


# ------------------------------------------------------------------ 开放 API


def t_open_api() -> str:
    from fastapi.testclient import TestClient
    from flycore.schemas import RunConfig
    from services.env.world import TaskSpec
    from services.orchestrator import app as app_mod

    app_mod.RUNS.clear()
    client = TestClient(app_mod.app)
    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/runs").status_code == 401, "缺少 API Key 应被拒绝"
    # 密钥库在模块导入时由环境变量播种；这里直接断言播种结果可用。
    assert app_mod.KEY_STORE.resolve("dev-local-key") is not None, "环境变量密钥未注入密钥库"
    h = {"X-API-Key": "dev-local-key"}

    spec = TaskSpec(task_id="api-task", grid=16, n_agents=2, n_targets=2, max_ticks=6, seed=4)
    spec_json = json.loads(json.dumps(dataclasses.asdict(spec)))
    assert client.post("/v1/tasks", json=spec_json, headers=h).status_code == 200

    run = RunConfig(run_id="api-run", seed=4, backend="classical", agents=2, max_ticks=6,
                    memory_scope="run", learning_mode="readonly", enable_memory_channel=False)
    r = client.post("/v1/runs", json={"run": run.model_dump(), "task": spec_json}, headers=h)
    assert r.status_code == 200, r.text
    # 范围高于 run 却未声明租户时必须被拒
    bad = RunConfig(run_id="api-run-bad", seed=4, backend="classical", agents=2, max_ticks=4,
                    memory_scope="tenant", learning_mode="readonly")
    rb = client.post("/v1/runs", json={"run": bad.model_dump(), "task": spec_json}, headers=h)
    assert rb.status_code == 400, "高范围记忆但未声明租户，必须拒绝而不是默认跨客户共享"

    import time
    for _ in range(200):
        st = client.get("/v1/runs/api-run", headers=h).json()
        if st["status"] in ("finished", "failed"):
            break
        time.sleep(0.1)
    assert st["status"] == "finished", f"运行未完成：{st}"
    rep = client.get("/v1/runs/api-run/report", headers=h).json()
    assert rep["score"]["steps_used"] == 6
    assert len(rep["gates"]) == 3
    return f"鉴权/任务注册/运行/报告 全通；run 状态 {st['status']}"


def t_apikey_management() -> str:
    """接入管理：只存哈希、作用域隔离、环境变量密钥只能吊销不能删。

    这三条都是"长期运维会出事"的地方：
      * 存明文 → 库一泄漏，所有接入方的凭据一起泄漏
      * 不分作用域 → 只读接入方也能提交/中止运行
      * 允许删除环境变量注入的密钥 → 页面显示已删除，重启后又复活
    """
    import sqlite3
    from fastapi.testclient import TestClient
    from services.orchestrator import app as app_mod

    app_mod.RUNS.clear()
    client = TestClient(app_mod.app)
    admin = "dev-local-key"
    H = {"X-API-Key": admin}

    assert client.get("/v1/admin/keys").status_code == 401, "管理端未鉴权"
    assert client.get("/v1/admin/keys", headers=H).status_code == 200

    created = client.post("/v1/admin/keys", headers=H,
                          json={"name": "smoke-third-party", "rpm": 30,
                                "scopes": ["read"], "note": "冒烟"}).json()
    raw, kid = created["key"], created["record"]["key_id"]
    assert raw and len(raw) > 16
    listed = client.get("/v1/admin/keys", headers=H).json()["keys"]
    assert all("key" not in k and "key_hash" not in k for k in listed), "列表泄漏了明文或哈希"

    spec = {"task_id": "smoke-adm", "grid": 12, "n_agents": 2, "n_targets": 2,
            "max_ticks": 3, "obstacle_fraction": 0.12, "perception_fraction": 1.0,
            "seed": 1, "env_version": "grid-v1", "visibility_regions": None}
    body = {"run": {"run_id": "smoke-adm-run", "seed": 1, "backend": "classical", "agents": 2,
                    "max_ticks": 3, "memory_scope": "run", "learning_mode": "readonly",
                    "enable_peer_messages": False, "enable_memory_channel": False,
                    "tags": {"tenant": "default", "group": "classical", "protocol": "smoke"}},
            "task": spec}
    assert client.post("/v1/runs", headers={"X-API-Key": raw}, json=body).status_code == 403, \
        "只读作用域的密钥竟然能提交运行"
    assert client.get("/v1/runs", headers={"X-API-Key": raw}).status_code == 200, \
        "只读作用域的密钥应当能查询"
    assert client.get("/v1/admin/keys", headers={"X-API-Key": raw}).status_code == 403, \
        "只读作用域的密钥竟然能进管理端"

    rot = client.post(f"/v1/admin/keys/{kid}/rotate", headers=H).json()
    assert client.get("/v1/runs", headers={"X-API-Key": raw}).status_code == 401, "轮换后旧密钥未失效"
    assert client.get("/v1/runs", headers={"X-API-Key": rot["key"]}).status_code == 200

    env_key = next((k for k in listed if k["source"] == "env"), None)
    assert env_key is not None, "应有环境变量注入的密钥"
    H2 = {"X-API-Key": rot["key"]}
    client.post("/v1/admin/keys", headers=H, json={"name": "smoke-ops", "scopes": ["admin"]})
    r = client.delete(f"/v1/admin/keys/{env_key['key_id']}", params={"purge": "true"}, headers=H2)
    assert r.status_code in (400, 403), f"环境变量密钥竟然被删除：{r.status_code}"

    # 数据库里不能出现明文
    db = Path(app_mod.KEY_STORE.path)
    assert db.exists()
    blob = db.read_bytes()
    assert admin.encode() not in blob, "密钥明文出现在数据库文件里"
    con = sqlite3.connect(str(db))
    n = con.execute("SELECT COUNT(*) FROM api_key").fetchone()[0]
    con.close()

    assert client.delete(f"/v1/admin/keys/{kid}", headers=H).status_code == 200
    assert client.get("/v1/runs", headers={"X-API-Key": rot["key"]}).status_code == 401, "吊销后未失效"
    return (f"创建/轮换/吊销闭环；只读作用域被正确拒绝；环境变量密钥不可删除；"
            f"库内 {n} 条记录且无明文")


def main() -> int:
    checks = [
        ("连接组：保度重连不变量", t_connectome_invariants),
        ("连接组：许可与口径登记", t_connectome_license_recorded),
        ("神经内核：有界性与丢弃计数", t_lif_bounded_and_drop_accounting),
        ("神经内核：突触传播方向", t_connectome_direction_matters),
        ("编解码：int16 往返与带宽口径", t_message_codec),
        ("编解码：观测维度", t_observation_dim),
        ("环境：目标拾取与碰撞", t_env_mechanics),
        ("环境：感知失效真实生效", t_env_perception_dropout),
        ("环境：信息拆分可见性", t_env_visibility_split),
        ("集群：单次运行产出完整报告", t_cluster_run),
        ("集群：通信消融真实生效", t_comm_ablation_effective),
        ("棋局：规则与目标知识进入观测", t_goal_knowledge),
        ("后端：协议方法实现完整", t_backend_protocol_complete),
        ("容器：连跑两次遥测不累加且记忆按 run 隔离", t_container_run_isolation),
        ("记忆：本地经历跨重启持久化", t_local_memory_persists),
        ("记忆：k_local 旋钮真的接通", t_local_memory_knob_live),
        ("记忆：防回声与来源链", t_memory_echo_guard),
        ("记忆：冲突保留与过期/快照", t_memory_conflict_and_expire),
        ("记忆：范围隔离", t_memory_scope_isolation),
        ("开放 API：鉴权与运行闭环", t_open_api),
        ("接入管理：哈希存储与作用域隔离", t_apikey_management),
    ]
    for name, fn in checks:
        check(name, fn)

    width = max(len(n) for n, _, _ in RESULTS)
    print()
    print("=" * (width + 60))
    for name, ok, detail in RESULTS:
        print(f"{'✅' if ok else '❌'}  {name:<{width}}  {detail}")
    print("=" * (width + 60))
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"通过 {len(RESULTS) - len(failed)}/{len(RESULTS)}")
    if failed:
        print("未通过：" + "、".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
