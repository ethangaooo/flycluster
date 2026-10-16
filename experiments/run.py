#!/usr/bin/env python3
"""实验运行器。

    python experiments/run.py main --seeds 3 --quick
    python experiments/run.py A --seeds 2
    python experiments/run.py B --seeds 2
    python experiments/run.py C --seeds 2
    python experiments/run.py all --seeds 3

产出写入 artifacts/：完整 JSON + 可读的 Markdown 报告。
报告里性能数字一律与可信度遥测（实时比、脉冲丢弃、NaN）同时出现，
并显式列出"哪些能力来自外围代码"。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flycore.scoring import (Aggregate, evaluate_gates, paired_ci_delta)  # noqa: E402
from flycore.util import jdump, now_iso  # noqa: E402
from services.env.world import TaskSpec  # noqa: E402
from services.orchestrator.runner import ClusterRunner, RunnerOptions  # noqa: E402

from experiments.protocol import GROUP_SPECS, ProtocolConfig, run_config_for, task_for  # noqa: E402
from experiments.scan import AXES, markdown_scan, run_scan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- 主对照


def run_main(proto: ProtocolConfig, seeds: list[int], out_dir: Path) -> dict:
    aggs: dict[str, Aggregate] = {g: Aggregate(GROUP_SPECS[g]["label"]) for g in GROUP_SPECS}
    per_seed: dict[str, list[dict]] = {g: [] for g in GROUP_SPECS}
    representative: dict | None = None

    for g in GROUP_SPECS:
        for seed in seeds:
            run = run_config_for(g, proto, seed)
            task = task_for(proto, seed)
            t0 = time.time()
            rep = ClusterRunner(
                run, task,
                RunnerOptions(data_dir=str(out_dir / "runs" / g),
                              steps_per_tick=proto.steps_per_tick),
            ).run_cluster()
            aggs[g].add(rep.score.score)
            per_seed[g].append({
                "seed": seed, "score": rep.score.score,
                "targets_found": rep.score.targets_found,
                "coverage": rep.score.coverage,
                "collisions": rep.score.collisions,
                "redundant_steps": rep.score.redundant_steps,
                "elapsed_seconds": round(time.time() - t0, 3),
                "telemetry": rep.telemetry,
                "trainable_params": rep.trainable_params,
                "output_dim": rep.output_dim,
                "comm": rep.comm_accounting,
                "memory": rep.memory_accounting,
                "gates": [g_.model_dump() for g_ in rep.gates],
            })
            if g == "real_comm" and representative is None:
                representative = rep.model_dump()
            print(f"  [{g:>17}] seed={seed} score={rep.score.score:8.3f} "
                  f"found={rep.score.targets_found}/{rep.score.targets_total} "
                  f"cov={rep.score.coverage:.3f} {time.time() - t0:.1f}s")

    real_comm = aggs["real_comm"].scores
    real_nocomm = aggs["real_nocomm"].scores
    rewire_comm = aggs["rewired_comm"].scores
    rewire_nocomm = aggs["rewired_nocomm"].scores
    controls = [g for g in GROUP_SPECS
                if g not in ("real_comm", "real_nocomm", "rewired_comm", "rewired_nocomm")]
    best_control = max((aggs[g].mean for g in controls), default=0.0)
    best_control_name = max(controls, key=lambda g: aggs[g].mean) if controls else "-"

    delta = paired_ci_delta(real_comm, real_nocomm, rewire_comm, rewire_nocomm)
    screening = {"n": len(seeds), "mean_gain": round(aggs["real_comm"].mean - best_control, 4),
                 "real_comm_mean": round(aggs["real_comm"].mean, 4),
                 "best_control": best_control_name, "best_control_mean": round(best_control, 4)}
    gates = evaluate_gates(representative or {}, screening, delta)

    return {
        # TaskSpec 是 dataclass，直接放进 `proto.__dict__` 会被 json 转成 repr 字符串，
        # 读端一 .get() 就崩。必须显式 asdict——产物是长期资产，形状要稳定。
        "experiment": "main",
        "protocol": {**{k: v for k, v in proto.__dict__.items() if k != "task"},
                     "task": dataclasses.asdict(proto.task)},
        "started_at": now_iso(), "finished_at": now_iso(),
        "aggregates": {g: aggs[g].as_dict() for g in GROUP_SPECS},
        "per_seed": per_seed,
        "interaction_effect_delta": delta,
        "screening": screening,
        "gates": gates,
        "representative_run": representative,
    }


# --------------------------------------------------------------------- 报告


def markdown_main(res: dict) -> str:
    L: list[str] = []
    L.append("# 果蝇连接组约束的多智能体计算验证 · 主对照报告")
    L.append("")
    L.append(f"生成时间：{res['finished_at']}　种子数：{len(res['aggregates']['real_comm']['scores'])}")
    L.append("")
    L.append("## 一、分组结果（同一任务实例、同一观测与消息权限、同一读出器预算）")
    L.append("")
    L.append("| 组 | 均值 S | 标准差 | 95% CI | 可训练参数 | 读出维度 | 各种子 S |")
    L.append("|---|---|---|---|---|---|---|")
    for g, a in res["aggregates"].items():
        ps = res["per_seed"][g][0] if res["per_seed"].get(g) else {}
        L.append(f"| {a['label']} | {a['mean']:.3f} | {a['std']:.3f} | "
                 f"[{a['ci95_low']:.3f}, {a['ci95_high']:.3f}] | "
                 f"{ps.get('trainable_params', '—')} | {ps.get('output_dim', '—')} | {a['scores']} |")
    L.append("")
    L.append("> 可训练参数与读出维度**不一致**必须如实披露：各模型的内部宽度本来就不同，"
             "`同预算`指的是学习规则、探索率、观测与消息权限一致，不是网络规模相同。"
             "若要严格控制预算，应把这些列作为协变量纳入分析。")
    L.append("")
    L.append("## 二、交互效应 Δ（区分「通信对谁都有效」与「真实结构特别会用通信」）")
    L.append("")
    d = res["interaction_effect_delta"]
    L.append("Δ = (S_真实,有通信 − S_真实,无通信) − (S_重连,有通信 − S_重连,无通信)")
    L.append("")
    L.append(f"- Δ 均值 **{d['mean']:.4f}**，95% CI [{d['ci95_low']:.4f}, {d['ci95_high']:.4f}]，"
             f"配对种子数 {d['n']}")
    L.append(f"- 逐种子 Δ：{d['values']}")
    verdict = ("区间不跨 0，支持真实结构对通信的利用优于重连结构。"
               if d["n"] >= 3 and d["ci95_low"] > 0 else
               "区间跨 0 或样本不足，**不支持**「真实结构有独特贡献」的结论；"
               "只能表述为「该协作方法有效」。")
    L.append(f"- 判定：{verdict}")
    L.append("")
    L.append("## 三、任务收益筛查（门 2）")
    L.append("")
    s = res["screening"]
    L.append(f"- 主实验组均值 {s['real_comm_mean']}；最好的普通方法为 {s['best_control']}"
             f"（{s['best_control_mean']}）")
    L.append(f"- 平均增益 {s['mean_gain']}；种子数 {s['n']}（门 2 要求 ≥3 用于筛查、≥10 用于结论）")
    L.append("")
    L.append("## 四、三道门")
    L.append("")
    L.append("| 门 | 通过 | 说明 |")
    L.append("|---|---|---|")
    for g in res["gates"]:
        L.append(f"| {g['gate']} | {'✅' if g['passed'] else '❌'} | {g['detail']} |")
    L.append("")
    rep = res.get("representative_run") or {}
    if rep:
        tel = rep.get("telemetry", {})
        L.append("## 五、可信度遥测（性能数字必须与这些数字同时出现）")
        L.append("")
        L.append(f"- 实时比：均值 {tel.get('realtime_ratio_mean')}，最慢节点 {tel.get('realtime_ratio_min')}")
        L.append(f"- 仿真步数 {tel.get('steps')}，脉冲 {tel.get('spikes')}，"
                 f"丢弃 {tel.get('dropped_spikes')}（丢弃率 {tel.get('drop_rate')}）")
        L.append(f"- NaN 事件 {tel.get('nan_events')}，饱和步数 {tel.get('saturated_steps')}，"
                 f"数值有界 {tel.get('bounded')}")
        L.append("")
        L.append("### 通信与记忆预算")
        L.append("")
        c = rep.get("comm_accounting", {})
        L.append(f"- 通道开关 {c.get('channel_enabled')}；节点产出消息 "
                 f"{c.get('messages_produced_by_nodes')}，进入通道 {c.get('messages_issued')}，"
                 f"送达 {c.get('messages_delivered')}，丢弃 {c.get('messages_dropped')}")
        L.append(f"- 实测 {c.get('raw_bytes_per_second')} B/s；文档口径理论值 "
                 f"{c.get('formula_bytes_per_second')} B/s（n×8×2×频率）")
        L.append(f"- 中心记忆检索返回字节合计 "
                 f"{sum((n.get('memory_bytes_in') or 0) for n in rep.get('memory_accounting', {}).get('per_node', []))}"
                 "（中心记忆是一条通信通道，必须一并计入预算）")
        L.append("")
        L.append("### 能力归因（不是所有成绩都该记在脑结构账上）")
        L.append("")
        for k, v in (rep.get("attribution") or {}).items():
            L.append(f"- **{k}**：{v}")
        L.append("")
    return "\n".join(L)


def markdown_abc(res: dict) -> str:
    L: list[str] = [f"# 实验 {res.get('experiment','?')} 结果", "",
                    f"生成时间：{res.get('started_at')}", "", "```json",
                    jdump(res, indent=2)[:20000], "```", ""]
    return "\n".join(L)


# --------------------------------------------------------------------- CLI


def main() -> None:
    p = argparse.ArgumentParser(description="FlyCluster 实验运行器")
    p.add_argument("experiment", choices=["main", "A", "B", "C", "scan", "all"])
    p.add_argument("--axis", default="perception", choices=sorted(AXES),
                   help="分条件扫描的轴；见 experiments/scan.py 的 AXES")
    p.add_argument("--values", default="",
                   help="逗号分隔的条件值；留空则用该轴的默认值表")
    p.add_argument("--replicate-seeds", type=int, default=0,
                   help="在不相交种子区间上复现候选条件的种子个数（>0 才允许下结论）")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--targets", type=int, default=6)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--connectome", default="synthetic-2000")
    p.add_argument("--learning-mode", default="plastic",
                   choices=["readonly", "accumulate", "plastic"])
    p.add_argument("--out", default="artifacts")
    p.add_argument("--quick", action="store_true", help="小规模快速跑通，不用于得出结论")
    args = p.parse_args()

    if args.quick:
        args.ticks = min(args.ticks, 40)
        args.agents = min(args.agents, 4)
        args.targets = min(args.targets, 4)
        args.connectome = "synthetic-1000"
        print("[quick] 已降为小规模：结论不可用于决策，只验证链路。")

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    proto = ProtocolConfig(
        task=TaskSpec(grid=args.grid, n_agents=args.agents, n_targets=args.targets,
                      max_ticks=args.ticks, seed=seeds[0]),
        connectome=args.connectome, ticks=args.ticks, agents=args.agents,
        learning_mode=args.learning_mode,
    )

    from experiments.memory_experiments import experiment_a, experiment_b, experiment_c

    jobs = []
    if args.experiment in ("main", "all"):
        print("[主对照] 结构 × 通信 六组 + 三类普通方法 + 无脑下限")
        res = run_main(proto, seeds, out_dir)
        jobs.append(("main", res, markdown_main(res)))
    if args.experiment in ("A", "all"):
        print("[实验 A] 个体经历是否改变行为 + 两个保留测试")
        res = experiment_a(proto, seeds, data_dir=out_dir / "expA")
        jobs.append(("A", res, markdown_abc(res)))
    if args.experiment in ("B", "all"):
        print("[实验 B] 经验能否跨节点迁移 + 回声防护")
        res = experiment_b(proto, seeds, data_dir=out_dir / "expB")
        jobs.append(("B", res, markdown_abc(res)))
    if args.experiment in ("scan", "all"):
        vals = [float(x) for x in args.values.split(",") if x.strip()] if args.values else []
        rep_seeds = (list(range(args.seed_start + 1000, args.seed_start + 1000 + args.replicate_seeds))
                     if args.replicate_seeds else None)
        print(f"[分条件扫描] 轴 {args.axis}；探索种子 {seeds}；复现种子 {rep_seeds or '未启用'}")
        res = run_scan(proto, seeds, args.axis, vals, out_dir, replicate_seeds=rep_seeds)
        jobs.append((f"scan-{args.axis}", res, markdown_scan(res)))
    if args.experiment in ("C", "all"):
        print("[实验 C] 互补信息的整合：通信 / 记忆 / 中心规划器")
        res = experiment_c(proto, seeds, data_dir=out_dir / "expC")
        jobs.append(("C", res, markdown_abc(res)))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    for name, res, md in jobs:
        jp = out_dir / f"{name}-{stamp}.json"
        mp = out_dir / f"{name}-{stamp}.md"
        jp.write_text(jdump(res, indent=2), encoding="utf-8")
        mp.write_text(md, encoding="utf-8")
        print(f"[完成] {jp}\n[完成] {mp}")


if __name__ == "__main__":
    main()
