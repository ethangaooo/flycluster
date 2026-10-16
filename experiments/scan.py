"""分条件下的优势搜索。

背景：主对照在 10 个种子上给出 Δ = +5.05、区间跨 0，没有可重复优势。
构思文档对这种情况给了明确出路——

> 初期还应当接受一种有价值的结果：优势只存在于某种噪声条件、记忆长度或通信预算下，
> 那就把研究缩小到一个专用问题。

于是这里把"找优势"本身变成一个受控流程：沿一条轴扫描若干条件，逐条件计算 Δ，
再对表现最好的那些条件**在不相交的种子区间上复现**。

## 为什么必须自动复现，而不是挑最好的那个

扫描 K 个条件，即使真实效应为零，也会有约 0.05K 个条件的 95% 区间偶然不跨 0。
直接挑出最好的一个并报告，就是 p-hacking——只不过发生在"条件"这个维度上。

因此判定规则是：**只有在一个不相交的种子区间上复现，条件才算成立。**
这比 Bonferroni 校正更强，也不需要新的统计代码。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from flycore.scoring import paired_ci_delta
from flycore.util import now_iso
from services.orchestrator.runner import ClusterRunner, RunnerOptions

from experiments.protocol import ProtocolConfig, run_config_for, task_for

# 参与 Δ 计算的四个核心组
CORE = ("real_comm", "real_nocomm", "rewired_comm", "rewired_nocomm")

Axis = dict[str, Any]

AXES: dict[str, Axis] = {
    "perception": {
        "label": "感知失效比例（观测被置零的 tick 占比）",
        "values": [0.0, 0.25, 0.5, 0.75],
        "task": lambda v: {"perception_fraction": 1.0 - float(v)},
        "run": lambda v: {},
        "why": "局部观察是这个任务的前提。感知越差，个体掌握的独占信息越少，"
               "通信与记忆的边际价值理应越高。",
    },
    "comm_loss": {
        "label": "消息丢弃率",
        "values": [0.0, 0.2, 0.5, 0.8],
        "task": lambda v: {},
        "run": lambda v: {"comm_loss": float(v)},
        "why": "如果真实结构的优势需要精确消息才能体现，丢弃率升高时应更快失效；"
               "反之若结构本身提供冗余，则可能更鲁棒。",
    },
    "comm_latency": {
        "label": "消息延迟（决策步）",
        "values": [0, 2, 5, 10],
        "task": lambda v: {},
        "run": lambda v: {"comm_latency_ticks": int(v)},
        "why": "延迟让消息描述的是过时状态，考察结构对陈旧信息的容忍度。",
    },
    "local_memory": {
        "label": "本地记忆检索条数 k_local",
        "values": [0, 2, 8, 16],
        "task": lambda v: {},
        "run": lambda v: {"k_local": int(v)},
        "why": "个体经历只有通过记忆适配器进入神经计算才算被用上；"
               "检索条数是这条通路最直接的强度旋钮。",
    },
    "central_memory": {
        "label": "中心记忆检索条数 k_central（记忆通道开启）",
        "values": [0, 2, 6, 12],
        "task": lambda v: {},
        "run": lambda v: {"k_central": int(v), "enable_memory_channel": True},
        "why": "中心记忆本身是一条通信通道，检索字节计入预算。"
               "这里检验共享经验可否替代直接通信。",
    },
}


def _run_one(group: str, value: float, seed: int, proto: ProtocolConfig, out_dir: Path,
             axis: str, tag: str) -> dict[str, Any]:
    """axis 是 AXES 的键，tag 是产物目录后缀；两者混用会让 AXES 查询直接 KeyError。"""
    ax = AXES[axis]
    run = run_config_for(group, proto, seed,
                         run_overrides=ax["run"](value),
                         tag_overrides={"protocol": "scan", "axis": axis,
                                        "value": str(value)})
    task = task_for(proto, seed, task_overrides=ax["task"](value))
    rep = ClusterRunner(run, task,
                        RunnerOptions(data_dir=str(out_dir / f"{axis}-{tag}" / f"v{value}" / group),
                                      steps_per_tick=proto.steps_per_tick)).run_cluster()
    return {"seed": seed, "score": rep.score.score,
            "targets_found": rep.score.targets_found,
            "coverage": rep.score.coverage,
            "telemetry": rep.telemetry,
            "gates": [g.model_dump() for g in rep.gates]}


def _delta_for_value(value: float, scores: dict[str, list[float]]) -> dict[str, Any]:
    d = paired_ci_delta(scores["real_comm"], scores["real_nocomm"],
                        scores["rewired_comm"], scores["rewired_nocomm"])
    comm_real = float(np.mean(scores["real_comm"]) - np.mean(scores["real_nocomm"]))
    comm_rew = float(np.mean(scores["rewired_comm"]) - np.mean(scores["rewired_nocomm"]))
    return {"value": value, "delta": d,
            "comm_effect_real": round(comm_real, 4),
            "comm_effect_rewired": round(comm_rew, 4),
            "means": {g: round(float(np.mean(s)), 4) for g, s in scores.items()}}


def run_scan(proto: ProtocolConfig, seeds: list[int], axis: str, values: list[float],
             out_dir: Path, *, replicate_seeds: list[int] | None = None) -> dict[str, Any]:
    ax = AXES[axis]
    if not values:
        values = list(ax["values"])

    def sweep(seed_list: list[int], tag: str, vals: list[float]) -> list[dict[str, Any]]:
        """把 values 作为参数显式传入，避免闭包变量在复现阶段被改写造成串味。"""
        results = []
        for v in vals:
            scores: dict[str, list[float]] = {g: [] for g in CORE}
            detail: dict[str, list[dict]] = {g: [] for g in CORE}
            for s in seed_list:
                for g in CORE:
                    r = _run_one(g, v, s, proto, out_dir, axis, tag)
                    scores[g].append(r["score"])
                    detail[g].append(r)
            entry = _delta_for_value(v, scores)
            entry["per_seed_detail"] = detail
            entry["promising"] = bool(entry["delta"]["n"] >= 3 and entry["delta"]["ci95_low"] > 0)
            results.append(entry)
            print(f"  [{axis}={v}] Δ={entry['delta']['mean']:+.3f} "
                  f"CI=[{entry['delta']['ci95_low']:+.3f}, {entry['delta']['ci95_high']:+.3f}] "
                  f"{'← 区间下界 > 0' if entry['promising'] else ''}")
        return results

    print(f"[扫描] 轴 {axis}（{ax['label']}），条件数 K={len(values)}，"
          f"每条件 {len(seeds)} 种子 × {len(CORE)} 组 = {len(values) * len(seeds) * len(CORE)} 次运行")
    first = sweep(seeds, "explore", values)
    promising = [e for e in first if e["promising"]]

    out: dict[str, Any] = {
        "experiment": "scan",
        # 必须记录跑在哪个连接组上：否则这张产物无法追溯，
        # 页面上也就无法按"真实图 / 合成图"切片——那正是最要紧的一个维度。
        "protocol": {**{k: v for k, v in proto.__dict__.items() if k != "task"},
                     "task": dataclasses.asdict(proto.task)},
        "axis": axis, "axis_label": ax["label"], "why": ax["why"],
        "values": values, "seeds": seeds, "n_conditions_tested": len(values),
        "explore": first, "started_at": now_iso(),
        "multiplicity_note": (
            f"本次共检验 {len(values)} 个条件。即使真实效应为零，95% 区间也会有约 "
            f"{0.05 * len(values):.2f} 个条件偶然不跨 0。因此"
            "「某个条件区间下界 > 0」本身不构成发现，必须在不相交的种子区间上复现。"
        ),
    }

    if not promising:
        out["verdict"] = ("在本轴所有扫描条件下，Δ 的 95% 区间下界都不大于 0："
                          "没有出现值得进入复现阶段的信号。按协议不应据此继续投入。")
        out["finished_at"] = now_iso()
        return out

    if not replicate_seeds:
        out["verdict"] = (f"有 {len(promising)} 个条件区间下界 > 0，但未提供复现种子，"
                          "按纪律**不得**据此下结论。请用 --replicate-seeds 在不相交的种子区间复现。")
        out["finished_at"] = now_iso()
        return out

    overlap = set(seeds) & set(replicate_seeds)
    out["replication"] = {"seeds": replicate_seeds, "seed_overlap": sorted(overlap)}
    if overlap:
        out["verdict"] = (f"复现种子与探索种子重叠 {sorted(overlap)}，样本不独立，复现无效。"
                          "请改用完全不相交的种子区间。")
        out["finished_at"] = now_iso()
        return out

    r_values = [e["value"] for e in promising]
    print(f"[复现] 对 {len(r_values)} 个候选条件在不相交种子 {replicate_seeds} 上复现")
    repl = sweep(replicate_seeds, "replicate", r_values)

    confirmed = []
    for e in repl:
        e["replicated"] = bool(e["delta"]["n"] >= 3 and e["delta"]["ci95_low"] > 0)
        if e["replicated"]:
            confirmed.append(e["value"])
    out["replication"]["results"] = repl
    out["replication"]["confirmed_values"] = confirmed

    if confirmed:
        out["verdict"] = (
            f"条件 {confirmed} 在不相交种子区间上复现（两次 Δ 区间下界均 > 0）。"
            "这是**局部条件优势**，不是普遍优势：必须把结论限定在这些条件下，"
            "并说明它不适用于未扫描到的条件。下一步应围绕该条件做机制解释与小规模验证。")
    else:
        out["verdict"] = (
            f"探索阶段有 {len(promising)} 个条件区间下界 > 0，但在不相交种子区间上"
            "**全部未能复现**。按协议判定为多重比较造成的假阳性，不支持任何条件优势。")
    out["finished_at"] = now_iso()
    return out


def markdown_scan(res: dict[str, Any]) -> str:
    L: list[str] = [f"# 分条件优势扫描 · 轴 `{res['axis']}`", "",
                    f"**{res['axis_label']}**", "", f"生成时间：{res.get('finished_at')}", "",
                    f"扫描理由：{res['why']}", ""]
    L.append("## 探索阶段")
    L.append("")
    L.append("| 条件值 | Δ 均值 | 95% CI | 真实结构通信效应 | 重连结构通信效应 | 真实+通信均值 | 重连+通信均值 | 区间下界>0 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for e in res["explore"]:
        d = e["delta"]
        L.append(f"| {e['value']} | {d['mean']:+.3f} | [{d['ci95_low']:+.3f}, {d['ci95_high']:+.3f}] | "
                 f"{e['comm_effect_real']:+.3f} | {e['comm_effect_rewired']:+.3f} | "
                 f"{e['means']['real_comm']:.3f} | {e['means']['rewired_comm']:.3f} | "
                 f"{'✅' if e['promising'] else '—'} |")
    L.append("")
    L.append(f"> {res['multiplicity_note']}")
    L.append("")
    if "replication" in res:
        r = res["replication"]
        L.append("## 复现阶段（不相交种子）")
        L.append("")
        L.append(f"复现种子：{r.get('seeds')}；与探索种子重叠：{r.get('seed_overlap') or '无'}")
        L.append("")
        if r.get("results"):
            L.append("| 条件值 | 复现 Δ 均值 | 95% CI | 是否复现 |")
            L.append("|---|---|---|---|")
            for e in r["results"]:
                d = e["delta"]
                L.append(f"| {e['value']} | {d['mean']:+.3f} | "
                         f"[{d['ci95_low']:+.3f}, {d['ci95_high']:+.3f}] | "
                         f"{'✅ 复现' if e.get('replicated') else '❌ 未复现'} |")
            L.append("")
    L.append("## 判定")
    L.append("")
    L.append(res["verdict"])
    L.append("")
    return "\n".join(L)
