"""实验结果归一化：把两种来源的数据压成同一张扁平表。

来源一：命令行产物 `artifacts/*.json`
  - `main-*.json`  主对照：per_seed[组][种子] 展开成行
  - `scan-*.json`  分条件扫描：explore / replication 的 per_seed_detail 展开成行
来源二：控制台提交的运行（内存中的 RunRecord）

**为什么要归一化**：命令行跑出来的才是正式结论（≥10 种子、含置信区间），
控制台跑的是交互式验证。两者字段名、嵌套层次都不同，直接画图会各画各的。
压成一张表之后，页面上就能按任意维度切片——连接组、结构、通信开关、种子、
扫描条件、来源——而不需要为每种组合单独写一个图。

指标定义集中在 `METRICS`，每个指标带单位与"越大越好/越小越好"的方向，
前端据此决定坐标轴方向与配色，避免各处各写一份。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

# 组语义的唯一来源：与实验协议共用，避免页面上再维护一份映射
try:
    from experiments.protocol import GROUP_SPECS
except Exception:  # pragma: no cover - 极端情况下退化为按名字猜
    GROUP_SPECS = {}

# --------------------------------------------------------------------- 维度

# 可用来筛选或分组的维度。`values` 在构建时动态填。
DIMENSIONS: dict[str, dict[str, str]] = {
    "source": {"label": "数据来源", "desc": "命令行产物 / 控制台运行"},
    "experiment": {"label": "实验类型", "desc": "main 主对照 / scan 分条件扫描 / live 控制台"},
    "artifact": {"label": "产物文件", "desc": "一次跑批对应的产物"},
    "connectome": {"label": "连接组", "desc": "数据来源，决定结论算不算数"},
    "group": {"label": "实验组", "desc": "真实/重连 × 有通信/无通信 等"},
    "backend": {"label": "中间结构", "desc": "脑 / 重连 / 储备池 / GRU / 经典 / 无脑"},
    "comm": {"label": "同伴通信", "desc": "开 / 关"},
    "memory": {"label": "中心记忆通道", "desc": "开 / 关"},
    "seed": {"label": "随机种子", "desc": "独立样本编号"},
    "axis": {"label": "扫描轴", "desc": "分条件扫描扫的是哪个参数"},
    "condition": {"label": "条件值", "desc": "扫描轴上的取值"},
    "phase": {"label": "扫描阶段", "desc": "探索 / 不相交种子复现"},
    "agents": {"label": "节点数", "desc": "参与协作的脑实例数"},
    "ticks": {"label": "步数上限", "desc": "每个节点的决策步上限"},
}

# --------------------------------------------------------------------- 指标

# direction: 1 = 越大越好，-1 = 越小越好
# 页面上用它决定"排序方向"和"好/差"的配色，避免每张图各写一套判断。
METRICS: dict[str, dict[str, Any]] = {
    "score": {"label": "任务得分 S", "unit": "", "direction": 1,
              "desc": "10×命中 − 0.5×碰撞 − 0.2×重复探索 + 5×覆盖率。绝对值无意义，只用于同任务跨组比较"},
    "targets_found": {"label": "命中目标数", "unit": "个", "direction": 1,
                      "desc": "走到格子上才算拾取"},
    "coverage": {"label": "覆盖率", "unit": "", "direction": 1,
                 "desc": "所有节点访问过的格子占可通行格的比例"},
    "collisions": {"label": "碰撞次数", "unit": "次", "direction": -1,
                   "desc": "多个节点同时进入同一格"},
    "redundant_steps": {"label": "重复探索步数", "unit": "步", "direction": -1,
                        "desc": "进入已访问格。实测中它通常主导总分"},
    "elapsed_seconds": {"label": "运行耗时", "unit": "s", "direction": -1,
                        "desc": "墙钟时间，受机器负载影响"},
    "realtime_ratio_min": {"label": "实时比（最慢节点）", "unit": "", "direction": 1,
                           "desc": "模拟时间 ÷ 现实时间。门 1 要求 ≥0.05"},
    "steps_per_second_mean": {"label": "吞吐", "unit": "步/s", "direction": 1,
                              "desc": "每秒仿真的神经元步数"},
    "steps": {"label": "仿真步数", "unit": "步", "direction": 1,
              "desc": "节点数 × 步数 × 每步仿真毫秒"},
    "spikes": {"label": "脉冲总数", "unit": "个", "direction": 1, "desc": "网络活动量"},
    "drop_rate": {"label": "脉冲丢弃率", "unit": "", "direction": -1,
                  "desc": "门 1 要求 ≤0.001。丢弃率高说明实际模型已被改变"},
    "nan_events": {"label": "NaN 事件", "unit": "次", "direction": -1,
                   "desc": "门 1 要求 = 0"},
    "comm_bytes_per_second": {"label": "脑间带宽", "unit": "B/s", "direction": -1,
                              "desc": "实测发出量。4 节点 320 B/s、16 节点 1280 B/s"},
    "memory_bytes_in": {"label": "记忆检索字节", "unit": "B", "direction": -1,
                        "desc": "中心记忆是一条独立通信通道，实测比脑间消息高约三个数量级"},
    "messages_issued": {"label": "发出消息数", "unit": "条", "direction": -1, "desc": ""},
    "messages_dropped": {"label": "丢弃消息数", "unit": "条", "direction": -1,
                         "desc": "由 comm_loss 决定"},
}


def _backend_of(group: str) -> str:
    spec = GROUP_SPECS.get(group)
    if spec:
        return str(spec.get("backend", "?"))
    return group.replace("_comm", "").replace("_nocomm", "")


def _comm_of(group: str) -> bool:
    spec = GROUP_SPECS.get(group)
    if spec is not None:
        return bool(spec.get("comm"))
    return group.endswith("_comm")


def _label_of(group: str) -> str:
    spec = GROUP_SPECS.get(group)
    if spec:
        return str(spec.get("label", group))
    return group


def _flat_row(**kw: Any) -> dict[str, Any]:
    row = {k: None for k in list(DIMENSIONS) + list(METRICS)}
    row.update(kw)
    return row


def _telemetry_fields(t: dict[str, Any] | None) -> dict[str, Any]:
    t = t or {}
    return {
        "realtime_ratio_min": t.get("realtime_ratio_min"),
        "steps_per_second_mean": t.get("steps_per_second_mean"),
        "steps": t.get("steps"),
        "spikes": t.get("spikes"),
        "drop_rate": t.get("drop_rate"),
        "nan_events": t.get("nan_events"),
    }


def _comm_fields(c: dict[str, Any] | None) -> dict[str, Any]:
    c = c or {}
    return {"comm_bytes_per_second": c.get("raw_bytes_per_second"),
            "messages_issued": c.get("messages_issued"),
            "messages_dropped": c.get("messages_dropped")}


def _memory_fields(m: dict[str, Any] | None) -> dict[str, Any]:
    m = m or {}
    per_node = m.get("per_node") or []
    total = sum((n.get("memory_bytes_in") or 0) for n in per_node)
    return {"memory_bytes_in": total or None}


# --------------------------------------------------------------------- 主对照

def _proto_meta(proto: dict[str, Any]) -> dict[str, Any]:
    """协议的规模字段。老产物的 `task` 是 TaskSpec 的 repr 字符串（序列化缺陷），
    这里不强行解析 repr，而是退回用逐种子的 telemetry 里的 nodes / env_ticks——
    那两个值是运行时真实产生的事实，比解析字符串可靠。"""
    task = proto.get("task")
    task_dict = task if isinstance(task, dict) else {}
    return {"agents": proto.get("agents") or task_dict.get("n_agents"),
            "ticks": task_dict.get("max_ticks"),
            "task_is_structured": isinstance(task, dict) if task is not None else None}


def _rows_from_main(path: Path, doc: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proto = doc.get("protocol") or {}
    connectome = proto.get("connectome")
    pm = _proto_meta(proto)
    rows: list[dict[str, Any]] = []
    for group, per_seed in (doc.get("per_seed") or {}).items():
        for e in per_seed:
            tel = e.get("telemetry") or {}
            rows.append(_flat_row(
                source="artifact", experiment="main", artifact=path.name,
                connectome=connectome, group=group, backend=_backend_of(group),
                comm=_comm_of(group), memory=None,
                agents=pm["agents"] or tel.get("nodes"),
                ticks=pm["ticks"] or tel.get("env_ticks"),
                seed=e.get("seed"), axis=None, condition=None, phase=None,
                score=e.get("score"), targets_found=e.get("targets_found"),
                coverage=e.get("coverage"), collisions=e.get("collisions"),
                redundant_steps=e.get("redundant_steps"),
                elapsed_seconds=e.get("elapsed_seconds"),
                **_telemetry_fields(e.get("telemetry")),
                **_comm_fields(e.get("comm")),
                **_memory_fields(e.get("memory")),
            ))
    meta = {
        "artifact": path.name, "experiment": "main",
        "connectome": connectome,
        "traceable": bool(connectome),
        "note": None if pm["task_is_structured"] is not False else
                "旧产物：protocol.task 是 repr 字符串，节点数/步数已从逐种子 telemetry 恢复",
        "n_rows": len(rows),
        "delta": doc.get("interaction_effect_delta"),
        "screening": doc.get("screening"),
        "gates": doc.get("gates"),
        "started_at": doc.get("started_at"), "finished_at": doc.get("finished_at"),
    }
    return rows, meta


# --------------------------------------------------------------------- 扫描

def _rows_from_scan(path: Path, doc: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proto = doc.get("protocol") or {}
    connectome = proto.get("connectome")
    pm = _proto_meta(proto)
    rows: list[dict[str, Any]] = []

    def expand(entries: Iterable[dict[str, Any]], phase: str) -> None:
        for e in entries or []:
            for group, per_seed in (e.get("per_seed_detail") or {}).items():
                for r in per_seed:
                    rows.append(_flat_row(
                        source="artifact", experiment="scan", artifact=path.name,
                        connectome=connectome, group=group, backend=_backend_of(group),
                        comm=_comm_of(group),
                        agents=(r.get("telemetry") or {}).get("nodes") or pm["agents"],
                        ticks=(r.get("telemetry") or {}).get("env_ticks") or pm["ticks"],
                        seed=r.get("seed"), axis=doc.get("axis"),
                        condition=e.get("value"), phase=phase,
                        score=r.get("score"), targets_found=r.get("targets_found"),
                        coverage=r.get("coverage"), collisions=r.get("collisions"),
                        redundant_steps=r.get("redundant_steps"),
                        elapsed_seconds=r.get("elapsed_seconds"),
                        **_telemetry_fields(r.get("telemetry")),
                        **_comm_fields(r.get("comm")),
                        **_memory_fields(r.get("memory")),
                    ))

    expand(doc.get("explore"), "explore")
    expand((doc.get("replication") or {}).get("results"), "replicate")

    meta = {
        "artifact": path.name, "experiment": "scan", "connectome": connectome,
        "traceable": bool(connectome),
        "note": None if connectome else
                "此产物未记录连接组（生产者已修，旧文件无法追溯），按「未知来源」处理",
        "axis": doc.get("axis"), "axis_label": doc.get("axis_label"),
        "n_rows": len(rows), "verdict": doc.get("verdict"),
        "multiplicity_note": doc.get("multiplicity_note"),
        "n_conditions_tested": doc.get("n_conditions_tested"),
        "started_at": doc.get("started_at"), "finished_at": doc.get("finished_at"),
    }
    return rows, meta


# --------------------------------------------------------------------- 控制台

def _rows_from_live(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in runs:
        if r.get("status") != "finished":
            continue
        cfg = r.get("config") or {}
        tags = cfg.get("tags") or {}
        sc = r.get("score") or {}
        group = tags.get("group") or cfg.get("backend") or "?"
        rows.append(_flat_row(
            source="live", experiment=tags.get("protocol") or "live", artifact=None,
            connectome=cfg.get("connectome"), group=group,
            backend=cfg.get("backend") or _backend_of(group),
            comm=cfg.get("enable_peer_messages"), memory=cfg.get("enable_memory_channel"),
            agents=cfg.get("agents"), ticks=cfg.get("max_ticks"), seed=cfg.get("seed"),
            axis=None, condition=None, phase=None,
            score=sc.get("score"), targets_found=sc.get("targets_found"),
            coverage=sc.get("coverage"), collisions=sc.get("collisions"),
            redundant_steps=sc.get("redundant_steps"),
            elapsed_seconds=None,
            run_id=r.get("run_id"),
        ))
    return rows


# --------------------------------------------------------------------- 汇总

def collect(artifact_dir: str | Path, live_runs: list[dict[str, Any]] | None = None,
            limit_artifacts: int = 200) -> dict[str, Any]:
    """把两种来源合成一张表，并回报可用维度取值与产物清单。"""
    adir = Path(artifact_dir)
    rows: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    files = sorted(adir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[:limit_artifacts]:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append({"artifact": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        kind = doc.get("experiment")
        try:
            if kind == "main":
                rs, meta = _rows_from_main(path, doc)
            elif kind == "scan":
                rs, meta = _rows_from_scan(path, doc)
            else:
                continue          # 实验 A/B/C 的记忆验证结果结构不同，暂不并入
        except Exception as exc:
            errors.append({"artifact": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.extend(rs)
        metas.append(meta)

    rows.extend(_rows_from_live(live_runs or []))

    # 动态填出每个维度实际出现过的取值，供前端做下拉
    dim_values: dict[str, list[Any]] = {}
    for d in DIMENSIONS:
        vals = {r.get(d) for r in rows if r.get(d) is not None}
        try:
            dim_values[d] = sorted(vals)
        except TypeError:
            dim_values[d] = sorted(vals, key=str)

    present_metrics = [m for m in METRICS if any(r.get(m) is not None for r in rows)]

    untraceable = [m["artifact"] for m in metas if not m.get("traceable")]

    return {
        "rows": rows,
        "n_rows": len(rows),
        "untraceable_artifacts": untraceable,
        "artifact_dir": str(adir),
        "artifacts": metas,
        "errors": errors,
        "dimensions": {k: {**v, "values": dim_values.get(k, [])} for k, v in DIMENSIONS.items()},
        "metrics": {k: METRICS[k] for k in present_metrics},
        "groups": sorted({r.get("group") for r in rows if r.get("group")}),
        "connectomes": sorted({r.get("connectome") for r in rows if r.get("connectome")}),
    }
