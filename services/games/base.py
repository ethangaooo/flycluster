"""棋类引擎的统一接口。

**为什么要先定接口**：三个棋种（五子棋 / 中国象棋 / 跳棋）规则差异极大，
但要让果蝇集群去下，它们必须对"上层"呈现出完全相同的样子——
一套 `legal_moves / apply / observe / ai_move`，桥接层与页面都不需要知道是哪个棋种。

**能力归因（重要）**：本模块及三个实现全部属于**外围代码**。
棋盘编码、候选着法剪枝、胜负判定、电脑对手的搜索——都不是脑结构做的。
脑只提供一个脉冲动力系统，其输出组活动被外围固定投影解读成"对某个落点的倾向分"。
页面上必须如实标注，否则就成了把外围能力记在脑结构账上。

**接口契约**：

- `legal_moves()` 返回的每个着法都必须是 JSON 可序列化的（列表/元组/整数）。
- `observe(move)` 必须返回**恰好 OBS_DIM（105）个 float**，值域大致归一化到 [-1, 1]。
  `move=None` 表示"只看当前局面、不指定候选"，用于展示。
- `apply` / `undo` 必须严格互逆，允许 AI 在搜索里反复调用。
- `outcome()` 在未结束时返回 `(False, None)`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# 与脑的观测维度一致。桥接层据此校验，不匹配直接报错——
# 静默截断/补零会让"棋盘编码错了"看起来像"果蝇下得很差"。
from flycore.schemas import OBS_DIM

Move = Any  # JSON 可序列化

FLY = "fly"
PC = "pc"


@dataclass
class Outcome:
    over: bool = False
    winner: str | None = None      # FLY / PC / None(未结束或和棋)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"over": self.over, "winner": self.winner, "reason": self.reason}


@dataclass
class Difficulty:
    """电脑对手难度。飞蝇一方的参数不在里面——那属于实验变量，单独配。"""

    key: str
    label: str
    depth: int = 1
    randomness: float = 0.0        # 以多大概率随机挑一个合法着法
    note: str = ""


DIFFICULTIES: list[Difficulty] = [
    Difficulty("easy", "简单", depth=1, randomness=0.55,
               note="多半随机走，基本不防守——用来确认页面链路是否通"),
    Difficulty("normal", "普通", depth=2, randomness=0.12,
               note="会挡你的活三、吃无保护的子"),
    Difficulty("hard", "困难", depth=4, randomness=0.0,
               note="常规搜索深度，会算几步杀棋"),
    Difficulty("brutal", "残酷", depth=6, randomness=0.0,
               note="更深的搜索。思考时间明显变长，但对手更强"),
]

DIFF_BY_KEY = {d.key: d for d in DIFFICULTIES}


class Game(Protocol):
    """棋种实现必须满足的契约。"""

    name: str          # gomoku / xiangqi / checkers
    label: str         # 中文名

    def reset(self) -> None: ...

    def legal_moves(self) -> list[Move]: ...

    def apply(self, move: Move) -> None: ...

    def undo(self) -> None: ...

    def outcome(self) -> Outcome: ...

    def turn(self) -> str:
        """轮到谁：FLY 或 PC。"""
        ...

    def observe(self, move: Move | None = None) -> list[float]: ...

    def candidates(self, limit: int = 48) -> list[Move]:
        """供脑逐个评估的候选着法。

        通常比 `legal_moves()` 少得多——真实棋类都要剪枝，
        否则 15×15 棋盘每步 200 多个候选会让"逐个评估"变得很慢。
        **剪枝是外围启发式，必须如实标注。**
        """
        ...

    def ai_move(self, difficulty: str) -> Move: ...

    def render(self) -> dict[str, Any]:
        """给前端的棋盘状态（JSON）。"""
        ...

    def move_label(self, move: Move) -> str: ...

    def heuristic(self) -> float:
        """从飞蝇视角看当前局面的好坏（用于奖励与曲线展示）。"""
        ...


def check_observe(game: Game) -> None:
    """校验观测维度。宁可在这里报错，也不要让错误编码悄悄变成"下得差"。"""
    v = game.observe(None)
    if len(v) != OBS_DIM:
        raise ValueError(
            f"{game.name}.observe() 返回 {len(v)} 维，必须恰好 {OBS_DIM} 维。"
            "静默截断/补零会把'编码错了'伪装成'果蝇水平差'。")
    if not all(isinstance(x, (int, float)) for x in v):
        raise ValueError(f"{game.name}.observe() 含非数值元素")


# --------------------------------------------------------------------- 目标知识

# 观测的**最后 4 维**留给"规则与目标"，三个棋种统一占用，不得挪作他用。
#
# 为什么必须单独留出来：早先的编码让果蝇**感觉不到"这步能赢"**。
# 以象棋为例，候选着法的特征只有"是否吃子 / 被吃子价值 / 是否将军"——
# 而**将军不等于将死**：一步把对方将死的棋和一步普通的将军，在它眼里没有区别。
# "是否结束 / 胜负方"那两维在候选评估时又恒为 0（那时局面当然还没结束）。
# 结果就是：它在下"挪棋子、争子力"的游戏，而不是下棋。
#
# 用户的说法很准确：**光知道棋子怎么动不够，还得知道什么叫赢。**
GOAL_SLOTS = 4
# 引擎自己可以使用的维数。**目标槽位是框架保留的，引擎不得占用**——
# 第一版没写清这条，象棋把候选位置项放在 s[13]/s[14]，正好被目标槽位覆盖，
# 结果两个对称候选（兵前进 / 兵前进）变成逐位相同的向量，自测立刻抓到。
ENGINE_DIMS = OBS_DIM - GOAL_SLOTS
GOAL_SLOT_NAMES = ("win_now", "lose_after", "opp_can_win_now", "i_can_win_now")
GOAL_SLOT_DESC = {
    "win_now": "该候选着法是否**立刻取胜**（将死 / 成五 / 填满目标区）",
    "lose_after": "走完该着法后，对方是否**立刻取胜**（等于送死）",
    "opp_can_win_now": "当前局面下对方是否有一步取胜（威胁告警）",
    "i_can_win_now": "当前局面下我是否有一步取胜（机会告警）",
}


def goal_slots(*, win_now: bool, lose_after: bool,
               opp_can_win_now: bool, i_can_win_now: bool) -> list[float]:
    """填充最后 4 维。取 ±1 而不是 0/1：让"对我有利/不利"在符号上就分得开，
    随机投影里正负号才有意义。"""
    return [
        1.0 if win_now else -1.0,
        -1.0 if lose_after else 1.0,          # 注意方向：能送死是坏事，给负
        1.0 if opp_can_win_now else -1.0,
        1.0 if i_can_win_now else -1.0,
    ]


def apply_goal_slots(vec: list[float], **kw: bool) -> list[float]:
    """把目标槽位写到向量的最后 4 维，并校验总长。"""
    assert len(vec) == ENGINE_DIMS, (
        f"引擎块长度 {len(vec)} != {ENGINE_DIMS}（= {OBS_DIM} - {GOAL_SLOTS}）。"
        "目标槽位是框架保留的，引擎不能占用最后 4 维，否则会被覆盖。")
    vec = list(vec)
    vec[OBS_DIM - GOAL_SLOTS:] = goal_slots(**kw)
    return vec
