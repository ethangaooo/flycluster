"""构思文档 §七 的三个记忆验证实验。

实验 A：一个脑是否真正从经历中改变行为？
  两个保留测试必须分开报告：
    保留测试 1「重启后仍能检索旧记录」→ 证明经验被持久化；
    保留测试 2「清除即时状态、关闭检索后仍保留能力改善」→ 才支持改善进入了学习参数。
  两者不能混为一个指标。

实验 B：一个脑的经验能否有效帮助另一个脑？
  同时设置 无共享 / 有共享 / 无关经验 / 过期经验 / 来源重复 五个条件。

实验 C：多个脑能否组合各自不完整的经验？
  把目标可见性拆给不同节点，分别关闭直接通信与中心记忆，再叠加中心规划器版本。
  若只有中心规划后才能成功，能力必须归因于组合系统，而非脑间自主涌现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from flycore.connectome import get_graph
from flycore.schemas import BrainNodeConfig, ExperienceRecord, RunConfig
from flycore.util import content_hash, now_iso
from services.brain.engine import BrainEngine
from services.brain.hub_client import InProcessMemoryBackend
from services.env.world import TaskSpec
from services.memory_hub.service import MemoryHubService
from services.orchestrator.runner import ClusterRunner, RunnerOptions

from .protocol import ProtocolConfig

ACTION_NAMES = ("up", "down", "left", "right", "stay")


# --------------------------------------------------------------------- 工具


def make_engine(proto: ProtocolConfig, node_id: str, seed: int, hub: MemoryHubService | None,
                data_dir: Path, *, k_local: int = 4, k_central: int = 6,
                learning_mode: str = "plastic", memory_scope: str = "project") -> BrainEngine:
    cfg = BrainNodeConfig(
        node_id=node_id, connectome=proto.connectome, seed=seed,
        learning_mode=learning_mode,  # type: ignore[arg-type]
        memory_scope=memory_scope,    # type: ignore[arg-type]
    )
    graph = get_graph(proto.connectome, data_dir=data_dir)
    return BrainEngine(graph, cfg, hub=(InProcessMemoryBackend(hub) if hub else None),
                       steps_per_tick=proto.steps_per_tick, local_root=data_dir / "local",
                       k_local=k_local, k_central=k_central)


def drive(proto: ProtocolConfig, engine: BrainEngine, task: TaskSpec, data_dir: Path,
          *, run_id: str, memory: bool, peer: bool, planner: bool = False,
          learning_mode: str = "plastic") -> dict[str, Any]:
    """用同一套编排循环驱动一个（或一组）已有引擎实例，保证与主实验口径一致。"""
    n_agents = task.n_agents
    run = RunConfig(
        run_id=run_id, task_id=task.resolved_id(), seed=task.seed, backend="brain",
        connectome=proto.connectome, agents=n_agents, max_ticks=task.max_ticks,
        memory_scope=("project" if memory else "run"),  # type: ignore[arg-type]
        learning_mode=learning_mode,  # type: ignore[arg-type]
        enable_peer_messages=peer, enable_memory_channel=memory,
        enable_central_planner=planner,
        comm_latency_ticks=proto.comm_latency_ticks, comm_loss=proto.comm_loss,
        message_kind=proto.message_kind,  # type: ignore[arg-type]
        tags={"tenant": proto.tenant, "protocol": "abc"},
    )
    rep = ClusterRunner(run, task, RunnerOptions(data_dir=str(data_dir),
                                                 steps_per_tick=proto.steps_per_tick,
                                                 reuse_nodes=[engine] * n_agents)).run_cluster()
    return rep.model_dump()


def _actions_of(rep: dict) -> dict[str, int]:
    h = rep["per_agent"][0]["action_histogram"]
    return {k: int(v) for k, v in h.items()}


def _tv(p: dict[str, int], q: dict[str, int]) -> float:
    """动作分布的总变差距离。0 = 分布相同，1 = 完全不相交。"""
    ps = sum(p.values()) or 1
    qs = sum(q.values()) or 1
    return float(0.5 * sum(abs(p.get(k, 0) / ps - q.get(k, 0) / qs) for k in ACTION_NAMES))


# --------------------------------------------------------------------- 实验 A


def experiment_a(proto: ProtocolConfig, seeds: list[int], *, episodes: int = 6,
                 ticks: int = 60, data_dir: Path) -> dict[str, Any]:
    conditions = ("A0_no_memory", "A1_local_recall", "A2_recall_plus_learn")
    out: dict[str, Any] = {"experiment": "A", "episodes": episodes, "ticks": ticks,
                           "started_at": now_iso(), "conditions": {}}

    for cond in conditions:
        per_seed = []
        for seed in seeds:
            hub_path = data_dir / f"hubA-{cond}-{seed}.sqlite3"
            hub = MemoryHubService(str(hub_path))
            memory_on = cond != "A0_no_memory"
            engine = make_engine(proto, f"A-{seed}", seed,
                                 hub if memory_on else None, data_dir,
                                 k_local=(0 if not memory_on else 4),
                                 learning_mode=("plastic" if cond == "A2_recall_plus_learn"
                                                else "readonly"))
            if not memory_on:
                engine.local_memory_enabled = False
                engine.central_memory_enabled = False
                engine.io.memory.enabled = False
            task = TaskSpec(task_id="assoc-task", n_agents=1, n_targets=3, max_ticks=ticks,
                            seed=seed)

            found, hists = [], []
            for ep in range(episodes):
                rep = drive(proto, engine, task, data_dir,
                            run_id=f"A-{cond}-{seed}-ep{ep}", memory=memory_on, peer=False,
                            learning_mode=("plastic" if cond == "A2_recall_plus_learn"
                                           else "readonly"))
                found.append(rep["score"]["targets_found"])
                hists.append(_actions_of(rep))

            # 保留测试 1：同 node_id 重建引擎（等价于进程重启），看旧记录是否还在
            restarted = make_engine(proto, f"A-{seed}", seed,
                                    hub if memory_on else None, data_dir,
                                    k_local=(0 if not memory_on else 4),
                                    learning_mode=("plastic" if cond == "A2_recall_plus_learn"
                                                   else "readonly"))
            cue = restarted.cue(np.zeros(105, dtype=np.float32))
            hits = restarted.local.retrieve(cue, k=4) if restarted.k_local else []
            persisted = {
                "local_records_after_restart": restarted.local.stats()["size"],
                "local_retrieval_hit": bool(hits),
                "hub_records": hub.stats()["records_total"],
            }

            # 保留测试 2：清除即时状态 + 关闭检索，只留学习参数
            engine.k_local = 0
            engine.hub = None
            engine.local_memory_enabled = False
            engine.central_memory_enabled = False
            engine.io.memory.enabled = False
            engine.working.buf.clear()
            engine.counter.reset()
            after = drive(proto, engine, task, data_dir, run_id=f"A-{cond}-{seed}-norecall",
                          memory=False, peer=False,
                          learning_mode=("plastic" if cond == "A2_recall_plus_learn"
                                         else "readonly"))
            after_found = after["score"]["targets_found"]
            after_hist = _actions_of(after)

            per_seed.append({
                "seed": seed,
                "targets_found_per_episode": found,
                "first3_mean": float(np.mean(found[:3])),
                "last3_mean": float(np.mean(found[-3:])),
                "learning_gain": float(np.mean(found[-3:]) - np.mean(found[:3])),
                "retention_persistence": persisted,
                "retention_after_retrieval_off": {
                    "targets_found": after_found,
                    "action_histogram": after_hist,
                    "tv_vs_episode1": round(_tv(after_hist, hists[0]), 4),
                    "plasticity_norm": round(engine.counters().get("plasticity_norm", 0.0), 6),
                },
                "hub_stats": hub.stats(),
            })
            hub.close()
        out["conditions"][cond] = {
            "per_seed": per_seed,
            "mean_learning_gain": round(float(np.mean([p["learning_gain"] for p in per_seed])), 4),
            "mean_persistence_hit": float(np.mean(
                [1.0 if p["retention_persistence"]["local_retrieval_hit"] else 0.0 for p in per_seed])),
            "mean_plasticity_norm": round(float(np.mean(
                [p["retention_after_retrieval_off"]["plasticity_norm"] for p in per_seed])), 6),
            "mean_tv_after_retrieval_off": round(float(np.mean(
                [p["retention_after_retrieval_off"]["tv_vs_episode1"] for p in per_seed])), 4),
        }

    a0 = out["conditions"]["A0_no_memory"]
    a2 = out["conditions"]["A2_recall_plus_learn"]
    out["conclusion"] = {
        "learning_gain_A2_minus_A0": round(a2["mean_learning_gain"] - a0["mean_learning_gain"], 4),
        "persistence_test_passed": bool(a2["mean_persistence_hit"] >= 0.5),
        "learning_retained_without_retrieval": bool(a2["mean_tv_after_retrieval_off"] > 0.0
                                                    and a2["mean_plasticity_norm"] > 0),
        "note": ("两个保留测试分开报告：persistence 只看重启后能否检索；"
                 "retained_without_retrieval 只支持'改善进入了学习参数'，"
                 "不区分是脑内还是外围读出器——本条只更新了读出器 ΔW，"
                 "因此结论只能表述为『外围读出器学到了东西』。"),
    }
    return out


# --------------------------------------------------------------------- 实验 B


def experiment_b(proto: ProtocolConfig, seeds: list[int], *, train_episodes: int = 4,
                 ticks: int = 60, data_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"experiment": "B", "started_at": now_iso(), "conditions": {}}
    for seed in seeds:
        hub = MemoryHubService(str(data_dir / f"hubB-{seed}.sqlite3"))
        train_task = TaskSpec(task_id="transfer-task", n_agents=1, n_targets=3,
                              max_ticks=ticks, seed=seed)
        eval_task_same = TaskSpec(task_id="transfer-task", n_agents=1, n_targets=3,
                                  max_ticks=ticks, seed=seed)
        eval_task_other = TaskSpec(task_id="transfer-task", n_agents=1, n_targets=3,
                                   max_ticks=ticks, seed=seed + 500, obstacle_fraction=0.2)

        teacher = make_engine(proto, f"teacher-{seed}", seed, hub, data_dir, k_local=0)
        teacher.io.memory.enabled = False        # 教师只贡献经验，不读取
        for ep in range(train_episodes):
            drive(proto, teacher, train_task, data_dir, run_id=f"B-train-{seed}-{ep}",
                  memory=True, peer=False)
        snapshot = hub.snapshot("实验 B：教师阶段结束")
        teacher_records = hub.store.record_ids(task_id="transfer-task", limit=1000)
        evidence_before = hub.store.stats()

        conditions = {
            "B0_no_share": dict(memory=False, task=eval_task_same, k=0),
            "B1_share_same_env": dict(memory=True, task=eval_task_same, k=6),
            "B2_share_other_env": dict(memory=True, task=eval_task_other, k=6),
            "B3_expired": dict(memory=True, task=eval_task_same, k=6, expire=True),
            "B4_echo_guard": dict(memory=False, task=eval_task_same, k=0),
        }
        cond_out: dict[str, Any] = {}
        for name, cfg in conditions.items():
            if cfg.get("expire"):
                hub.expire(teacher_records)
            student = make_engine(proto, f"student-{seed}-{name}", seed, hub, data_dir,
                                  k_local=0, k_central=cfg["k"])
            student.local_memory_enabled = cfg["memory"]
            student.central_memory_enabled = cfg["memory"]
            student.io.memory.enabled = cfg["memory"]
            rep = drive(proto, student, cfg["task"], data_dir,
                        run_id=f"B-{name}-{seed}", memory=cfg["memory"], peer=False)
            entry = {"targets_found": rep["score"]["targets_found"],
                     "score": rep["score"]["score"],
                     "central_hits": rep["memory_accounting"]["per_node"][0]["central_hits"],
                     "memory_bytes_in": rep["memory_accounting"]["per_node"][0]["memory_bytes_in"]}
            if name == "B4_echo_guard":
                # 把教师的记录以另一个节点名义重投一次（模拟 A→B→中心 的转发）
                echoes = []
                for rid in teacher_records[:10]:
                    src = hub.store.get_record(rid)
                    if src is None:
                        continue
                    rec = ExperienceRecord(
                        record_id=f"echo-{rid}", origin_event_id=src["origin_event_id"],
                        producer_node="echo-node", kind="observation",
                        run_id=src["run_id"], task_id=src["task_id"], tick=src["tick"],
                        scope="project", payload={"conclusion": 0}, confidence=0.9,
                        observed_at=now_iso(), valid_from=now_iso(),
                    )
                    echoes.append(hub.submit(rec).model_dump())
                entry["echo_guard"] = {
                    "submitted": len(echoes),
                    "flagged_as_echo": int(sum(1 for e in echoes if e["is_echo"])),
                    "evidence_delta_total": int(sum(e["evidence_count_delta"] for e in echoes)),
                }
                entry["hub_stats"] = hub.stats()
                if teacher_records:
                    origin = hub.store.get_record(teacher_records[0])["origin_event_id"]
                    entry["provenance_sample"] = hub.provenance(origin)
            cond_out[name] = entry
        cond_out["_meta"] = {"snapshot": snapshot, "evidence_before": evidence_before,
                            "teacher_records": len(teacher_records),
                            "note": ("B1 与 B0 的差值才说明共享是否有效；"
                                     "B2 检验无关经验是否被误当有用；"
                                     "B3 检验过期经验是否被排除；"
                                     "B4 检验转发是否被误计为独立证据。")}
        out["conditions"][f"seed{seed}"] = cond_out
        hub.close()
    return out


# --------------------------------------------------------------------- 实验 C


def experiment_c(proto: ProtocolConfig, seeds: list[int], *, ticks: int = 90,
                 data_dir: Path) -> dict[str, Any]:
    """信息拆分：左半区目标只有 0/1 号节点看得见，右半区只有 2/3 号节点看得见。"""
    g = proto.task.grid
    regions = [[0, 0, g // 2 - 1, g - 1], [0, 0, g // 2 - 1, g - 1],
               [g // 2, 0, g - 1, g - 1], [g // 2, 0, g - 1, g - 1]]
    conditions = {
        "C0_none": dict(peer=False, memory=False, planner=False),
        "C1_peer_only": dict(peer=True, memory=False, planner=False),
        "C2_memory_only": dict(peer=False, memory=True, planner=False),
        "C3_peer_and_memory": dict(peer=True, memory=True, planner=False),
        "C4_plus_central_planner": dict(peer=True, memory=True, planner=True),
    }
    out: dict[str, Any] = {"experiment": "C", "ticks": ticks,
                           "visibility_regions": regions, "started_at": now_iso(),
                           "conditions": {}}
    for seed in seeds:
        hub = MemoryHubService(str(data_dir / f"hubC-{seed}.sqlite3"))
        task = TaskSpec(task_id="complementary-task", n_agents=4, n_targets=8,
                        max_ticks=ticks, seed=seed, visibility_regions=regions)
        for name, cfg in conditions.items():
            nodes = []
            for i in range(4):
                cfg_node = BrainNodeConfig(node_id=f"C-{name}-{seed}-{i}",
                                           connectome=proto.connectome, seed=seed + i,
                                           learning_mode="plastic", memory_scope="project")  # type: ignore[arg-type]
                graph = get_graph(proto.connectome, data_dir=data_dir)
                eng = BrainEngine(graph, cfg_node,
                                  hub=(InProcessMemoryBackend(hub) if cfg["memory"] else None),
                                  steps_per_tick=proto.steps_per_tick,
                                  local_root=data_dir / "local",
                                  k_local=(4 if cfg["memory"] else 0), k_central=6)
                if not cfg["memory"]:
                    eng.local_memory_enabled = False
                    eng.central_memory_enabled = False
                    eng.io.memory.enabled = False
                nodes.append(eng)
            run = RunConfig(run_id=f"C-{name}-{seed}", task_id=task.resolved_id(), seed=seed,
                            backend="brain", connectome=proto.connectome, agents=4,
                            max_ticks=ticks, memory_scope="project",  # type: ignore[arg-type]
                            learning_mode="plastic",  # type: ignore[arg-type]
                            enable_peer_messages=cfg["peer"], enable_memory_channel=cfg["memory"],
                            enable_central_planner=cfg["planner"],
                            comm_latency_ticks=proto.comm_latency_ticks, comm_loss=proto.comm_loss,
                            message_kind=proto.message_kind,  # type: ignore[arg-type]
                            tags={"tenant": proto.tenant, "protocol": "abc"})
            rep = ClusterRunner(run, task, RunnerOptions(data_dir=str(data_dir),
                                                         steps_per_tick=proto.steps_per_tick,
                                                         reuse_nodes=nodes)).run_cluster()
            d = rep.model_dump()
            out["conditions"].setdefault(name, {"per_seed": []})["per_seed"].append({
                "seed": seed,
                "targets_found": d["score"]["targets_found"],
                "score": d["score"]["score"],
                "coverage": d["score"]["coverage"],
                "collisions": d["score"]["collisions"],
                "messages_issued": d["comm_accounting"]["messages_issued"],
                "memory_bytes_in": sum(n["memory_bytes_in"] or 0
                                       for n in d["memory_accounting"]["per_node"]),
                "hub": (d["memory_accounting"].get("hub") or {}),
                "attribution": d["attribution"],
            })
        c0 = out["conditions"]["C0_none"]["per_seed"][-1]["targets_found"]
        c3 = out["conditions"]["C3_peer_and_memory"]["per_seed"][-1]["targets_found"]
        c4 = out["conditions"]["C4_plus_central_planner"]["per_seed"][-1]["targets_found"]
        out.setdefault("per_seed_summary", []).append(
            {"seed": seed, "C0": c0, "C3": c3, "C4": c4,
             "complementary_gain_peer_plus_memory": c3 - c0,
             "planner_marginal_gain": c4 - c3})
        hub.close()
    for name, v in out["conditions"].items():
        v["mean_targets_found"] = round(float(np.mean(
            [p["targets_found"] for p in v["per_seed"]])), 4)
        v["mean_score"] = round(float(np.mean([p["score"] for p in v["per_seed"]])), 4)
    out["note"] = ("若只有 C4 显著高于 C3，说明解题依赖中心规划器，能力应归因于"
                   "「脑集群 + 中心规划器」组合系统，而不是脑间自主涌现。")
    out["hash"] = content_hash(out["conditions"])
    return out
