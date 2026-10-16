"""评分与对照指标。

核心不是"4 个比 1 个强"，而是把收益拆开（构思文档 §三.4）：
  1. 观察范围变大本身就能提高成绩 —— 不说明结构特殊。
  2. 因此必须做交互效应 Δ，区分"通信对任何网络都有效"与"真实果蝇结构特别会用通信"：

     Δ = (S_真实结构,有通信 − S_真实结构,无通信) − (S_重连结构,有通信 − S_重连结构,无通信)

  3. 即使 Δ > 0，还要检查主实验组是否优于最好的普通模型，以及多花了多少成本。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schemas import ScoreBreakdown


@dataclass
class ScoreWeights:
    """评分权重。全部为工程选择，必须在报告里显式列出。"""

    target: float = 10.0
    collision: float = 0.5
    redundant: float = 0.2
    coverage: float = 5.0


def compute_score(*, targets_found: int, targets_total: int, collisions: int,
                  redundant_steps: int, coverage: float, steps_used: int,
                  w: ScoreWeights | None = None) -> ScoreBreakdown:
    """任务得分 S。同一批对照必须使用同一套权重与同一套步数上限。"""
    w = w or ScoreWeights()
    s = (w.target * targets_found
         - w.collision * collisions
         - w.redundant * redundant_steps
         + w.coverage * coverage)
    return ScoreBreakdown(
        targets_found=targets_found, targets_total=targets_total, collisions=collisions,
        redundant_steps=redundant_steps, coverage=round(float(coverage), 4),
        steps_used=steps_used, score=round(float(s), 4),
    )


def interaction_effect(real_comm: float, real_nocomm: float,
                       rewire_comm: float, rewire_nocomm: float) -> float:
    """交互效应 Δ。正数表示真实结构对通信的利用优于重连结构。"""
    return float((real_comm - real_nocomm) - (rewire_comm - rewire_nocomm))


@dataclass
class Aggregate:
    """多随机种子聚合结果。不凭一条最佳曲线下结论。"""

    label: str
    scores: list[float] = field(default_factory=list)

    def add(self, s: float) -> None:
        self.scores.append(float(s))

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.scores, ddof=1)) if self.n > 1 else 0.0

    def ci95(self) -> tuple[float, float]:
        """小样本用 t 近似（n<30 时用 2.0 保守系数，避免假装正态）。"""
        if self.n < 2:
            return (self.mean, self.mean)
        se = self.std / np.sqrt(self.n)
        t = 2.0 if self.n < 30 else 1.96
        return (self.mean - t * se, self.mean + t * se)

    def as_dict(self) -> dict[str, object]:
        lo, hi = self.ci95()
        return {"label": self.label, "n": self.n, "mean": round(self.mean, 4),
                "std": round(self.std, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                "scores": [round(s, 4) for s in self.scores]}


def paired_ci_delta(real_comm: list[float], real_nocomm: list[float],
                    rewire_comm: list[float], rewire_nocomm: list[float]) -> dict[str, object]:
    """按种子配对计算 Δ 的均值与 95% 区间。四组必须同种子、同任务、同步数。"""
    n = min(len(real_comm), len(real_nocomm), len(rewire_comm), len(rewire_nocomm))
    if n == 0:
        return {"n": 0, "mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "values": []}
    vals = [interaction_effect(real_comm[i], real_nocomm[i], rewire_comm[i], rewire_nocomm[i])
            for i in range(n)]
    agg = Aggregate("delta", vals)
    lo, hi = agg.ci95()
    return {"n": n, "mean": round(agg.mean, 4), "std": round(agg.std, 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "values": [round(v, 4) for v in vals]}


# ------------------------------------------------------------------ 三道门

GATE1_MIN_REALTIME_RATIO = 0.05     # 至少 5% 实时，否则谈不上"能跑"
GATE1_MAX_DROP_RATE = 0.001         # 丢弃超过千分之一即视为模型被改变
GATE2_MIN_SEEDS = 10
GATE2_MIN_DELTA = 0.0
GATE3_MIN_DELTA = 0.0
GATE3_MIN_TARGET_GAIN_PP = 10.0     # 同预算成功率提高 10 个百分点


def evaluate_gates(report: dict, screening: dict | None = None,
                   delta: dict | None = None) -> list[dict[str, object]]:
    """三道门。任一门不通过，都必须如实报告阻塞原因，不得"为了凑计划继续加功能"。"""
    gates: list[dict[str, object]] = []
    tel = report.get("telemetry", {}) or {}
    has_sim = "steps" in tel          # 对照模型没有脉冲仿真，门 1 的判据要分开
    rr = float(tel.get("realtime_ratio_min", tel.get("realtime_ratio", 0.0)))
    drop = float(tel.get("drop_rate", 0.0))
    nan = int(tel.get("nan_events", 0))
    if has_sim:
        passed = bool(rr >= GATE1_MIN_REALTIME_RATIO and drop <= GATE1_MAX_DROP_RATE
                      and nan == 0 and tel.get("bounded", True))
        detail = (f"实时比（最慢节点）{rr:.4f}（阈值 ≥{GATE1_MIN_REALTIME_RATIO}）；"
                  f"脉冲丢弃率 {drop:.6f}（阈值 ≤{GATE1_MAX_DROP_RATE}）；NaN 事件 {nan}；"
                  f"数值有界 {tel.get('bounded')}。")
    else:
        passed = bool(tel.get("reproducible", True))
        detail = "对照模型无脉冲仿真，门 1 改以确定性复现与数值有界为准。"
    gates.append({
        "gate": "门1 实现可信", "passed": passed, "detail": detail,
        "evidence": {"realtime_ratio_min": rr, "drop_rate": drop, "nan_events": nan,
                     "dropped_spikes": tel.get("dropped_spikes", 0),
                     "has_spiking_simulation": has_sim},
    })

    if screening is None:
        gates.append({"gate": "门2 任务收益", "passed": False, "detail": "未提供 3 种子筛查结果。", "evidence": {}})
    else:
        n = int(screening.get("n", 0))
        gain = float(screening.get("mean_gain", 0.0))
        gates.append({
            "gate": "门2 任务收益",
            "passed": bool(n >= 3 and gain > 0),
            "detail": f"筛查种子数 {n}（要求 ≥3）；平均增益 {gain:.4f}。"
                      "通过筛查后须扩到 ≥10 个独立种子并保留置信区间。",
            "evidence": screening,
        })

    if delta is None:
        gates.append({"gate": "门3 值得继续花钱", "passed": False,
                      "detail": "未提供交互效应 Δ，无法判断结构是否对通信有独特贡献。", "evidence": {}})
    else:
        n = int(delta.get("n", 0))
        d = float(delta.get("mean", 0.0))
        lo = float(delta.get("ci95_low", 0.0))
        gates.append({
            "gate": "门3 值得继续花钱",
            "passed": bool(n >= GATE2_MIN_SEEDS and d > GATE3_MIN_DELTA and lo > 0),
            "detail": f"配对种子数 {n}（要求 ≥{GATE2_MIN_SEEDS}）；Δ={d:.4f}，95%CI=[{lo:.4f},"
                      f"{float(delta.get('ci95_high', 0.0)):.4f}]。区间跨 0 时不支持「结构有独特贡献」。",
            "evidence": delta,
        })
    return gates
