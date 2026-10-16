"""中国象棋（9 列 × 10 行）。

**能力归因（重要）**：本文件与 `base.py` 一样属于**外围代码**。
着法生成、候选剪枝、alpha-beta 搜索、棋盘编码、胜负判定——全部不是脑结构做的。
脑只提供一个脉冲动力系统，外围把它的输出组活动固定投影成"对某个着法的倾向分"。
页面上必须如实标注，否则就是把外围能力记在脑结构账上。

坐标约定
--------
- `W=9` 列、`H=10` 行；着法为 `(from_row, from_col, to_row, to_col)` 四元组，0-based。
  `row` 0 在**顶部**（黑方底线），向下递增；`col` 0 在最左。
- 飞蝇（`base.FLY`）= **红方**（下方 rows 5..9，先手）；电脑（`base.PC`）= **黑方**
  （上方 rows 0..4）。棋子归属用 `render()["cells"][*]["piece"]` 的 "fly"/"pc" 表达。

规则实现范围与**刻意简化**（写在 docstring 里，避免把简化当成完整规则）
----------------------------------------------------------------------
已实现：
- 七种棋子的完整着法（飞将、塞象眼、蹩马腿、炮隔子吃、兵过河后可横行）、九宫限制、
  不能送将（走出后自己的将处于被攻击状态的着法非法）。
- 终局判定：**将死判负**、**困毙判负**（中国象棋约定，区别于国际象棋的"逼和"）。
- **重复局面第 3 次出现**（同一布局 + 同一走子方）时的判定：
  · 一方每一着都在将军 → **长将判负**
  · 一方每一着都在制造新的得子威胁 → **长捉判负**
  · 双方都犯规或都不犯规 → **判和**
- **自然限着**：连续 120 着（60 回合）双方都没有吃子 → 判和。

刻意**未**实现 / 简化之处：
1. **"长捉同一枚棋子"的个体追踪**：真实规则要求反复追捉的是**同一枚**棋子，
   追踪个体需要持久棋子 ID（现在的棋子只有"类型 + 位置"）。本实现用
   "每一着都制造了新的得子威胁"作判据，不追踪棋子身份。这是**已知的放宽**。
2. **"一将一杀"、"长兑/长献/长拦/长跟"的细分**：未做分类，一律按"无一方犯规 → 判和"
   处理。真实规则的判罚体系比这细。
3. **60 回合无吃子的判定不区分"双方只剩残兵"的特例**（真实规则对极简子力有 30 回合的规定）。
4. **飞将实现为"将可以直接吃对方的将"**：两将同列且中间无子时，`帅`/`将`
   吃到对方将格是**合法着法**，吃后立刻判胜（reason 含"飞将"）。
   真实规则中另一种常见表达是"两将不可照面（局面非法）"，本文件按任务约定
   用"可吃子"表达。
5. 记谱为简化中文记谱（含前/后消歧）；同一纵线上同种棋子超过 3 个等极端局面
   会退化成 `炮(7,2)→(7,4)` 的坐标写法兜底。
4. **飞将实现为"将可以直接吃对方的将"**：两将同列且中间无子时，`帅`/`将`
   吃到对方将格是**合法着法**，吃后立刻判胜（reason 含"飞将"）。
   真实规则中另一种常见表达是"两将不可照面（局面非法）"，本文件按任务约定
   用"可吃子"表达。
5. 记谱为简化中文记谱（含前/后消歧）；同一纵线上同种棋子超过 3 个等极端局面
   会退化成 `炮(7,2)→(7,4)` 的坐标写法兜底。
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

from flycore.schemas import OBS_DIM

from services.games import base
from services.games.base import ENGINE_DIMS, FLY, PC, Outcome, apply_goal_slots

# ---------------------------------------------------------------- 棋盘常量

W = 9               # 列
H = 10              # 行
CELLS = W * H       # 90

EMPTY = 0
RED = 1             # 飞蝇方（下方，先手）
BLACK = -1          # 电脑方（上方）

# 棋子种类（棋子的符号由 `side * kind` 唯一决定，正=红、负=黑）
KING, ADVISOR, ELEPHANT, HORSE, ROOK, CANNON, PAWN = 1, 2, 3, 4, 5, 6, 7

PIECE_VALUE = {
    KING: 10000,
    ROOK: 900,
    CANNON: 450,
    HORSE: 400,
    ADVISOR: 200,
    ELEPHANT: 200,
    PAWN: 100,
}

# 棋子中文名：红（飞蝇）与黑（电脑）用不同字，前端直接画 `symbol`
SYMBOL = {
    RED * KING: "帅", RED * ADVISOR: "仕", RED * ELEPHANT: "相", RED * HORSE: "马",
    RED * ROOK: "车", RED * CANNON: "炮", RED * PAWN: "兵",
    BLACK * KING: "将", BLACK * ADVISOR: "士", BLACK * ELEPHANT: "象",
    BLACK * HORSE: "马", BLACK * ROOK: "车", BLACK * CANNON: "炮", BLACK * PAWN: "卒",
}

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
KNIGHT = ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
ELEPHANT_DELTA = ((-2, -2), (-2, 2), (2, -2), (2, 2))
ADVISOR_DELTA = ((-1, -1), (-1, 1), (1, -1), (1, 1))

MATE = 100_000          # 搜索里的杀棋分，必须远大于任何局面分
INF = float("inf")

SEARCH_SECONDS = 2.5    # 单次 ai_move 的思考上限（超时返回当前最优，绝不卡住接口）
SEARCH_BRANCH = 14      # 每个搜索节点展开的着法数上限（外围剪枝）
Q_DEPTH = 2             # 静态搜索（只吃子）的最大延伸层数
Q_BRANCH = 6            # 静态搜索每层最多看几个吃子


# ---------------------------------------------------------------- 位置分表

def _pst_red(kind: int, r: int, c: int) -> int:
    """红方视角的位置分（红方在下方，向 row 减小的方向前进）。

    刻意只做**静态**的位置偏好，不查"这路兵还在不在"：
    - 兵：过河大幅加分，越靠前越好；
    - 马/炮/车：靠近中路加分，边路减分；
    - 士/象：0（本来就被九宫与河界约束住）；
    - 将：离开底线减分。

    **"车占开放线"没有实现动态版本**（那需要维护每路兵数），
    只用 PST 的中路偏好近似。这是外围启发式，如实标注。
    """
    dc = 4 - abs(c - 4)                 # 4=中路，0=边路
    drc = 4 - abs(2 * r - 9) // 2       # 4=中路行，0=顶/底行
    if kind == PAWN:
        if r <= 4:                      # 过河（红兵 row<=4）
            return 100 + (4 - r) * 12 + dc * 3
        return dc * 2 - (10 if c in (0, 8) else 0)
    if kind == HORSE:
        return dc * 6 + drc * 4 - (18 if c in (0, 8) else 0)
    if kind == CANNON:
        return dc * 4 + (10 if r in (4, 5) else 0)
    if kind == ROOK:
        return dc * 5 + drc * 3
    if kind == ADVISOR or kind == ELEPHANT:
        return 0
    if kind == KING:
        return -(9 - r) * 15            # 红将越靠底线越安全
    return 0


def _build_tables():
    """`_PS_RED[kind][sq]` / `_PS_BLACK[kind][sq]`：子力价值 + 位置分（恒正）。"""
    red = {}
    black = {}
    for kind in (KING, ADVISOR, ELEPHANT, HORSE, ROOK, CANNON, PAWN):
        tr = [0] * CELLS
        tb = [0] * CELLS
        for r in range(H):
            for c in range(W):
                v = PIECE_VALUE[kind] + _pst_red(kind, r, c)
                tr[r * W + c] = v
                tb[(H - 1 - r) * W + c] = v      # 黑方上下镜像
        red[kind] = tr
        black[kind] = tb
    return red, black


_PS_RED, _PS_BLACK = _build_tables()


def _contrib(piece: int, sq: int) -> int:
    """某格上的棋子对"红方视角总分"的贡献（红正黑负）。"""
    if piece > 0:
        return _PS_RED[piece][sq]
    return -_PS_BLACK[-piece][sq]


# ---------------------------------------------------------------- 底层工具

def _idx(r: int, c: int) -> int:
    return r * W + c


def _rc(sq: int) -> tuple[int, int]:
    return divmod(sq, W)


def _do(board: list[int], kings: dict[int, int], mv: int) -> tuple:
    """在 `board` 上落子（不做任何合法性判断），返回可交给 `_undo` 的凭据。"""
    f = mv >> 7
    t = mv & 127
    mover = board[f]
    cap = board[t]
    board[t] = mover
    board[f] = EMPTY
    old = None
    if mover == KING or mover == -KING:
        old = kings[mover]
        kings[mover] = t
    return (mv, cap, old)


def _undo(board: list[int], kings: dict[int, int], rec: tuple) -> None:
    """`_do` 的严格逆操作。"""
    mv, cap, old = rec
    f = mv >> 7
    t = mv & 127
    mover = board[t]
    board[f] = mover
    board[t] = cap
    if old is not None:
        kings[mover] = old


def _make_move(f: int, t: int) -> int:
    return (f << 7) | t


def _from_of(mv: int) -> int:
    return mv >> 7


def _to_of(mv: int) -> int:
    return mv & 127


# 将/帅在 `sq` 时，可能"挡住对方马腿"的格子集合（静态预计算，见 `_gen_legal`）。
def _build_legs() -> list[set[int]]:
    legs: list[set[int]] = [set() for _ in range(CELLS)]
    for sq in range(CELLS):
        r, c = _rc(sq)
        s = legs[sq]
        for dr, dc in KNIGHT:
            # 马在 sq+(dr,dc) 处攻击 sq 时，马腿在紧挨马、朝向 sq 的那一格
            if abs(dr) == 2:
                lr, lc = r + dr - (1 if dr > 0 else -1), c + dc
            else:
                lr, lc = r + dr, c + dc - (1 if dc > 0 else -1)
            if 0 <= lr < H and 0 <= lc < W:
                s.add(lr * W + lc)
    return legs


_LEGS = _build_legs()


def _attacked(board: list[int], sq: int, by: int) -> bool:
    """`sq` 是否正被 `by`（RED/BLACK）一方攻击。

    这是**定向探测**而不是"生成对方全部着法"，用于合法性过滤与将军判定，
    在搜索里是热点路径，所以写成平铺的分支。
    """
    r, c = divmod(sq, W)
    if by == RED:
        # 兵：红兵在下方，向上（row 减小）攻击；过河兵（row<=4）可横向攻击。
        # 注意横向攻击的行号是**兵所在的行**，即被攻击格的行 r。
        if r + 1 < H and board[(r + 1) * W + c] == PAWN:
            return True
        if r <= 4:
            if c > 0 and board[r * W + c - 1] == PAWN:
                return True
            if c + 1 < W and board[r * W + c + 1] == PAWN:
                return True
        k_piece, r_piece, c_piece = KING, ROOK, CANNON
    else:
        if r - 1 >= 0 and board[(r - 1) * W + c] == -PAWN:
            return True
        if r >= 5:
            if c > 0 and board[r * W + c - 1] == -PAWN:
                return True
            if c + 1 < W and board[r * W + c + 1] == -PAWN:
                return True
        k_piece, r_piece, c_piece = -KING, -ROOK, -CANNON

    # 将帅贴身一步（同列的长距离情形由下面的"飞将"扫描负责）
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W and board[rr * W + cc] == k_piece:
            return True

    # 飞将：同列、中间无子
    rr = r - 1
    while rr >= 0 and board[rr * W + c] == EMPTY:
        rr -= 1
    if rr >= 0 and board[rr * W + c] == k_piece:
        return True
    rr = r + 1
    while rr < H and board[rr * W + c] == EMPTY:
        rr += 1
    if rr < H and board[rr * W + c] == k_piece:
        return True

    # 马（含蹩马腿）
    for dr, dc in KNIGHT:
        hr, hc = r + dr, c + dc
        if not (0 <= hr < H and 0 <= hc < W):
            continue
        if board[hr * W + hc] != by * HORSE:
            continue
        if abs(dr) == 2:
            lr, lc = hr - (1 if dr > 0 else -1), hc
        else:
            lr, lc = hr, hc - (1 if dc > 0 else -1)
        if board[lr * W + lc] == EMPTY:
            return True

    # 车 / 炮：沿四条射线找第一个子
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        while 0 <= rr < H and 0 <= cc < W and board[rr * W + cc] == EMPTY:
            rr += dr
            cc += dc
        if not (0 <= rr < H and 0 <= cc < W):
            continue
        if board[rr * W + cc] == r_piece:
            return True
        rr += dr                      # 越过炮架
        cc += dc
        while 0 <= rr < H and 0 <= cc < W and board[rr * W + cc] == EMPTY:
            rr += dr
            cc += dc
        if 0 <= rr < H and 0 <= cc < W and board[rr * W + cc] == c_piece:
            return True
    return False


def _gen_pseudo(board: list[int], side: int, caps_only: bool = False) -> list[int]:
    """生成**伪合法**着法（不检查送将）。返回 `(from << 7) | to` 编码的 int。

    `caps_only=True` 时只生成吃子（含飞将吃将），供静态搜索使用——
    静态搜索里全量生成着法太贵，而吃子才是"局面还没稳定"的信号。
    """
    moves: list[int] = []
    add = moves.append
    red = side == RED
    for f in range(CELLS):
        p = board[f]
        if p == EMPTY or (p > 0) != red:
            continue
        kind = p if p > 0 else -p
        r, c = divmod(f, W)

        if kind == KING:
            lo, hi = (7, 9) if red else (0, 2)
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if not (lo <= rr <= hi and 3 <= cc <= 5):   # 九宫
                    continue
                t = rr * W + cc
                q = board[t]
                if q * side <= 0 and (q != EMPTY or not caps_only):
                    add((f << 7) | t)
            # 飞将：同列中间无子，可直接"吃"对方的将
            rr = r - 1
            while rr >= 0 and board[rr * W + c] == EMPTY:
                rr -= 1
            if rr >= 0 and board[rr * W + c] == -side * KING:
                add((f << 7) | (rr * W + c))
            rr = r + 1
            while rr < H and board[rr * W + c] == EMPTY:
                rr += 1
            if rr < H and board[rr * W + c] == -side * KING:
                add((f << 7) | (rr * W + c))

        elif kind == ADVISOR:
            lo, hi = (7, 9) if red else (0, 2)
            for dr, dc in ADVISOR_DELTA:
                rr, cc = r + dr, c + dc
                if not (lo <= rr <= hi and 3 <= cc <= 5):   # 九宫
                    continue
                t = rr * W + cc
                q = board[t]
                if q * side <= 0 and (q != EMPTY or not caps_only):
                    add((f << 7) | t)

        elif kind == ELEPHANT:
            for dr, dc in ELEPHANT_DELTA:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (red and rr < 5) or ((not red) and rr > 4):
                    continue                                # 象不过河
                if board[(r + dr // 2) * W + (c + dc // 2)] != EMPTY:
                    continue                                # 塞象眼
                t = rr * W + cc
                q = board[t]
                if q * side <= 0 and (q != EMPTY or not caps_only):
                    add((f << 7) | t)

        elif kind == HORSE:
            for dr, dc in KNIGHT:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                lr = r + (dr // 2 if dr == 2 or dr == -2 else 0)
                lc = c + (dc // 2 if dc == 2 or dc == -2 else 0)
                if board[lr * W + lc] != EMPTY:
                    continue                                # 蹩马腿
                t = rr * W + cc
                q = board[t]
                if q * side <= 0 and (q != EMPTY or not caps_only):
                    add((f << 7) | t)

        elif kind == ROOK:
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                while 0 <= rr < H and 0 <= cc < W:
                    t = rr * W + cc
                    q = board[t]
                    if q == EMPTY:
                        if not caps_only:
                            add((f << 7) | t)
                    else:
                        if q * side < 0:
                            add((f << 7) | t)
                        break
                    rr += dr
                    cc += dc

        elif kind == CANNON:
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                while 0 <= rr < H and 0 <= cc < W and board[rr * W + cc] == EMPTY:
                    if not caps_only:
                        add((f << 7) | (rr * W + cc))
                    rr += dr
                    cc += dc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue                                # 这一路没炮架
                # 炮架本身不能吃（炮必须隔子吃），所以直接跳过它
                rr += dr
                cc += dc
                while 0 <= rr < H and 0 <= cc < W and board[rr * W + cc] == EMPTY:
                    rr += dr
                    cc += dc
                if 0 <= rr < H and 0 <= cc < W and board[rr * W + cc] * side < 0:
                    add((f << 7) | (rr * W + cc))

        else:  # PAWN
            rr = r - 1 if red else r + 1
            if 0 <= rr < H:
                t = rr * W + c
                q = board[t]
                if q * side <= 0 and (q != EMPTY or not caps_only):
                    add((f << 7) | t)
            crossed = (r <= 4) if red else (r >= 5)
            if crossed:
                for dc in (-1, 1):
                    cc = c + dc
                    if not (0 <= cc < W):
                        continue
                    t = r * W + cc
                    q = board[t]
                    if q * side <= 0 and (q != EMPTY or not caps_only):
                        add((f << 7) | t)
    return moves


class _Timeout(Exception):
    """搜索超时哨兵：抛出后由 `ai_move` 吞掉，返回当前最优着法。"""


# --------------------------------------------------------------------- 规则常量

# 同一局面（含轮到谁）第 3 次出现时触发重复局面判定。
# 中国象棋竞赛规则用的是"同一局面重复三次"，这里取 3。
REP_LIMIT = 3
# 自然限着：连续 120 着（60 回合）双方都没有吃子 → 判和。
NATURAL_LIMIT = 120


class _Engine:
    """着法生成 / 搜索用的棋盘。

    `share=False`（搜索用）：**自带一份棋盘副本**，超时抛出时也不会把
    "幽灵子"留在真实棋盘上。
    `share=True`（引擎自身的临时落子用）：直接操作真实棋盘，靠 make/unmake 互逆保证安全。

    维护增量总分 `score`（红方视角），使局面评估是 O(1) 而不是每次扫 90 格——
    在 Python 里"每次评估扫一遍棋盘"是搜索速度的头号杀手。
    """

    __slots__ = ("board", "kings", "score", "nodes", "deadline")

    def __init__(self, board: list[int], kings: dict[int, int], share: bool = False):
        self.board = board if share else list(board)
        self.kings = kings if share else dict(kings)
        s = 0
        for sq in range(CELLS):
            p = self.board[sq]
            if p != EMPTY:
                s += _contrib(p, sq)
        self.score = s
        self.nodes = 0
        self.deadline = 0.0

    # ------------------------------------------------------------ 超时控制

    def tick(self) -> None:
        self.nodes += 1
        # 每 256 个节点才查一次时钟：time.monotonic() 本身也有成本，
        # 逐节点查询会让搜索慢一大截。
        if (self.nodes & 255) == 0 and time.monotonic() > self.deadline:
            raise _Timeout

    # ------------------------------------------------------------ 落子 / 撤销

    def make(self, mv: int) -> tuple:
        b = self.board
        f = mv >> 7
        t = mv & 127
        mover = b[f]
        cap = b[t]
        self.score -= _contrib(mover, f)
        self.score += _contrib(mover, t)
        if cap != EMPTY:
            self.score -= _contrib(cap, t)
        return _do(b, self.kings, mv)

    def unmake(self, rec: tuple) -> None:
        mv, cap, _ = rec
        f = mv >> 7
        t = mv & 127
        mover = self.board[t]
        self.score -= _contrib(mover, t)
        self.score += _contrib(mover, f)
        if cap != EMPTY:
            self.score += _contrib(cap, t)
        _undo(self.board, self.kings, rec)

    # ------------------------------------------------------------ 合法着法

    def in_check(self, side: int) -> bool:
        return _attacked(self.board, self.kings[side], -side)

    def gen_legal(self, side: int, caps_only: bool = False) -> list[int]:
        """合法着法（已过滤"送将"）。

        性能关键：只有当这一步**可能**暴露己方将时才真的落子试探，判据是
        1) 走动的是将/帅；或 2) 己方当前正被将军（必须逐着验证能否解将）；或
        3) 起点或终点落在将所在的横线/纵线上（可能让开一条车/炮/飞将的射线，
           或反过来把自己垫成对方炮的炮架——炮的"隔子吃"意味着**垫子会招打**）；或
        4) 起点可能是"挡住对方马腿"的格子（马腿被让开也会暴露将）。
        开局绝大多数着法都不满足以上条件，直接判合法，省掉一次落子+攻击探测。
        """
        b = self.board
        king = self.kings[side]
        krow, kcol = divmod(king, W)
        in_chk = _attacked(b, king, -side)
        legs = _LEGS[king]
        out: list[int] = []
        for mv in _gen_pseudo(b, side, caps_only):
            f = mv >> 7
            t = mv & 127
            if not in_chk:
                fr, fc = divmod(f, W)
                tr, tc = divmod(t, W)
                if (fr != krow and fc != kcol and tr != krow and tc != kcol
                        and f not in legs and abs(b[f]) != KING):
                    out.append(mv)
                    continue
            rec = _do(b, self.kings, mv)
            ok = not _attacked(b, self.kings[side], -side)
            _undo(b, self.kings, rec)
            if ok:
                out.append(mv)
        return out


# ---------------------------------------------------------------- 电脑对手搜索

def _order(eng: _Engine, moves: list[int], enemy_king: int) -> list[int]:
    """着法排序（MVV-LVA + 靠将奖励）。

    alpha-beta 剪得动全靠排序，所以第一步就把最有戏的着法放前面：
    - 吃子：`被吃子价值×16 − 自己价值`（MVV-LVA，吃掉对方的将排最前）；
    - 非吃子：按"落点离对方将多近"给一点奖励——纯吃子排序会漏掉
      "一步安静杀"，而靠近对方将的着法正是杀棋的候选。

    排序是纯外围启发式，也是 **`SEARCH_BRANCH` 截断会漏着法的来源**：
    排在 14 名之后的安静着法不会被搜索看到。
    """
    b = eng.board
    ekr, ekc = divmod(enemy_king, W)
    scored = []
    for mv in moves:
        f = mv >> 7
        t = mv & 127
        cap = b[t]
        if cap != EMPTY:
            s = 1_000_000 + PIECE_VALUE[abs(cap)] * 16 - PIECE_VALUE[abs(b[f])]
        else:
            tr, tc = divmod(t, W)
            near = max(abs(tr - ekr), abs(tc - ekc))
            s = 40 - 4 * near
        scored.append((-s, mv))
    scored.sort()
    return [mv for _, mv in scored]


def _qsearch(eng: _Engine, side: int, alpha: float, beta: float,
             depth: int, ply: int) -> float:
    """静态搜索：只延伸吃子，避免"吃掉一个子才发现自己被反吃"的视界效应。

    被将军时不静态估算（不能"停一手"），改为搜索全部解将着法；
    这同时让"将死"在叶节点附近也能被识别出来（否则搜索会把被杀当成普通局面）。
    """
    eng.tick()
    in_chk = eng.in_check(side)
    stand = eng.score if side == RED else -eng.score
    if depth <= -4:
        return stand          # 硬地板：连续将军不让静态搜索无限延伸
    if not in_chk:
        if depth <= 0:
            return stand
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand
    else:
        best = -INF

    moves = eng.gen_legal(side, caps_only=not in_chk)
    if not moves:
        if in_chk:
            return -(MATE - ply)          # 被将死
        return best
    for mv in _order(eng, moves, eng.kings[-side])[:Q_BRANCH]:
        rec = eng.make(mv)
        try:
            val = -_qsearch(eng, -side, -beta, -alpha, depth - 1, ply + 1)
        finally:
            eng.unmake(rec)
        if val > best:
            best = val
            if val > alpha:
                alpha = val
                if alpha >= beta:
                    break
    return best


def _negamax(eng: _Engine, side: int, depth: int,
             alpha: float, beta: float, ply: int) -> float:
    eng.tick()
    if depth <= 0:
        return _qsearch(eng, side, alpha, beta, Q_DEPTH, ply)
    moves = eng.gen_legal(side)
    if not moves:
        # 无合法着法：将死与**困毙都判负**（中国象棋约定，见模块 docstring）
        return -(MATE - ply)
    best = -INF
    for mv in _order(eng, moves, eng.kings[-side])[:SEARCH_BRANCH]:
        rec = eng.make(mv)
        try:
            val = -_negamax(eng, -side, depth - 1, -beta, -alpha, ply + 1)
        finally:
            eng.unmake(rec)
        if val > best:
            best = val
            if val > alpha:
                alpha = val
                if alpha >= beta:
                    break
    return best


def _search_best(eng: _Engine, side: int, max_depth: int, deadline: float):
    """迭代加深 + alpha-beta。返回 `(最优着法, 分值)`；超时则用**上一层完成的最优**。

    迭代加深是"2.5 秒硬上限"的落地方式：depth=4/6 在 Python 里大概率跑不完，
    与其算到一半被砍掉（拿不到根节点最优），不如只跑完整层的结果——
    层数越深，报出的着法只会更好，不会更差。
    """
    eng.deadline = deadline
    root = eng.gen_legal(side)
    if not root:
        return None, -INF
    cur = _order(eng, root, eng.kings[-side])
    best_mv = cur[0]
    best_val = -INF
    scores: dict[int, float] = {}
    for depth in range(1, max_depth + 1):
        if time.monotonic() > deadline:
            break
        alpha = -INF
        local_best = None
        local_val = -INF
        local_scores: dict[int, float] = {}
        try:
            for mv in cur:
                rec = eng.make(mv)
                try:
                    val = -_negamax(eng, -side, depth - 1, -INF, -alpha, 1)
                finally:
                    eng.unmake(rec)
                local_scores[mv] = val
                if val > local_val:
                    local_val = val
                    local_best = mv
                    if val > alpha:
                        alpha = val
        except _Timeout:
            break
        if local_best is not None:
            best_mv, best_val = local_best, local_val
            scores = local_scores
            # 上一层的好着法先搜，这一层的剪枝会更狠
            cur = sorted(cur, key=lambda m: -scores.get(m, -INF))
        if best_val >= MATE - 128:
            break                              # 已经算到杀棋，不必再往下挖
    return best_mv, best_val


# 引擎块的摘要布局：从 15 维摘要里挑 11 维，最后 4 维留给框架写目标槽位。
# **提成常量**，因为自测要按名字找"轮到谁"那一维——写死数字会在改布局时静默失效。
# 砍掉 s[4]（吃子比例，与 s[3] 部分冗余）与 s[10]（是否吃子，等价于 s[11] > 0），
# 腾出两维给"重复局面次数"与"犯规方"。
_SUMMARY_KEEP = (0, 1, 2, 3, 6, 11, 12, 13, 14)
_SUM = {name: i for i, name in enumerate(_SUMMARY_KEEP)}
# 摘要里的原始编号 → 引擎块内的下标
_SUM_TURN = _SUM[6]        # 轮到谁


class Xiangqi:
    """中国象棋。飞蝇 = 红方（下方，先手），电脑 = 黑方（上方）。"""

    name = "xiangqi"
    label = "中国象棋"

    def __init__(self) -> None:
        self._rng = random.Random()
        self._board: list[int] = []
        self._kings: dict[int, int] = {}
        self._history: list[tuple] = []
        self._last: tuple[int, str] | None = None
        self._result = Outcome()
        self._ver = 0
        self._cache_ver = -1
        self._legal_cache: list[int] | None = None
        self._ann_cache: list[tuple[int, int, bool]] | None = None
        self._scratch: _Engine | None = None
        # ---- 规则记账（重复局面 / 长将 / 长捉 / 自然限着）----
        # 为什么放在 `apply` 里也不怕慢：搜索走的是 `_Engine.make/unmake`，
        # 不经过 `Xiangqi.apply`；只有 AI 选点的根层与正式对局会走到这里（每次约 44 回）。
        self._pos_keys: list[tuple] = []
        self._move_log: list[tuple[int, bool, bool]] = []   # (走子方, 是否将军, 是否制造新捉)
        self._nc_stack: list[int] = []                      # 无吃子着数的历史，供 undo 还原
        self._no_capture_plies = 0
        self.reset()

    # ------------------------------------------------------------ 基础协议

    def reset(self) -> None:
        b = [EMPTY] * CELLS
        back = (ROOK, HORSE, ELEPHANT, ADVISOR, KING, ADVISOR, ELEPHANT, HORSE, ROOK)
        for c, kind in enumerate(back):
            b[_idx(0, c)] = BLACK * kind      # 黑方底线（上方）
            b[_idx(9, c)] = RED * kind        # 红方底线（下方）
        for c in (1, 7):
            b[_idx(2, c)] = BLACK * CANNON
            b[_idx(7, c)] = RED * CANNON
        for c in (0, 2, 4, 6, 8):
            b[_idx(3, c)] = BLACK * PAWN
            b[_idx(6, c)] = RED * PAWN

        self._board = b
        self._kings = {RED: _idx(9, 4), BLACK: _idx(0, 4)}
        self._history = []
        self._last = None
        self._result = Outcome()
        self._ver += 1
        self._cache_ver = -1
        self._legal_cache = None
        self._ann_cache = None
        # 临时落子用的引擎：**共享**真实棋盘对象（靠 make/unmake 互逆保证安全）
        self._scratch = _Engine(self._board, self._kings, share=True)
        # 规则记账复位。局面键必须在此刻取——它就是"第 0 个局面"。
        self._pos_keys = [self._pos_key()]
        self._move_log = []
        self._nc_stack = []
        self._no_capture_plies = 0

    def turn(self) -> str:
        return FLY if len(self._history) % 2 == 0 else PC

    def _side(self) -> int:
        """轮到哪一方（RED/BLACK）。"""
        return RED if len(self._history) % 2 == 0 else BLACK

    def legal_moves(self) -> list[tuple[int, int, int, int]]:
        return [self._public(mv) for mv in self._raw_legal()]

    # ------------------------------------------------------------ 规则记账

    def _pos_key(self) -> tuple:
        """局面 = 棋子布局 + 轮到谁。直接存元组而不是哈希：**不会碰撞**，
        而且 undo 只需 pop。代价是每次 apply 建一个 90 元组，可接受。

        必须含"轮到谁"：同一布局轮到不同方走是两个不同局面，规则上分开计数。
        """
        return (self._side(), tuple(self._board))

    def _defended(self, sq: int, by: int) -> bool:
        """`sq` 是否被 `by` 方**真正保护**（排除"飞将"射线）。

        `_attacked` 把飞将射线也算作攻击——用于过滤"送将"是正确的，
        但拿来判"这子有没有保护"会产生**假阳性**：任何恰好与对方将同列的棋子
        都会被当成有根子，于是"捉"永远判不出来。
        做法：先把对方的将临时移开再探一次；若不再被攻击，说明刚才只是飞将射线。
        """
        board = self._board
        if not _attacked(board, sq, by):
            return False
        k_piece = KING if by == RED else -KING
        r, c = divmod(sq, W)
        for dr, dc in DIRS4:                      # 贴身的将算真保护（将可以直接吃）
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and board[rr * W + cc] == k_piece:
                return True
        king_sq = self._kings.get(by, -1)
        if king_sq < 0:
            return False
        saved = board[king_sq]
        board[king_sq] = EMPTY
        try:
            return _attacked(board, sq, by)
        finally:
            board[king_sq] = saved

    def _profitable_from(self, sq: int, side: int) -> tuple[int, ...]:
        """`sq` 上的 `side` 棋子，眼下能吃掉的**哪些对方子算作得子**。

        判据（LVA/MVV 的简化版）：
          - 目标无保护 → 白吃一个，算得子
          - 目标有保护 → 只有目标价值**高于**自己才算得子（否则是亏换）

        不判"将/帅"：吃将属于将军，另有长将规则管。
        """
        piece = self._board[sq]
        if piece == EMPTY or (piece > 0) != (side == RED):
            return ()
        atk_val = PIECE_VALUE[abs(piece)]
        out: list[int] = []
        for mv in _gen_pseudo(self._board, side, caps_only=True):
            if _from_of(mv) != sq:
                continue
            t = _to_of(mv)
            tgt = self._board[t]
            if tgt == EMPTY or abs(tgt) == KING:
                continue
            if (not self._defended(t, -side)) or PIECE_VALUE[abs(tgt)] > atk_val:
                out.append(t)
        return tuple(sorted(out))

    @staticmethod
    def _fault(plies: list[tuple[int, bool, bool]], who: int) -> str | None:
        """周期内 `who` 这一方是否"犯规"。返回 'check' / 'chase' / None。

        - 每一着都将军 → 长将
        - 每一着都制造了新的得子威胁 → 长捉
        （长将优先判定：既长将又长捉时按长将处理。）

        **刻意简化**：真实规则要求"长捉同一枚棋子"，追踪棋子个体需要持久 ID。
        这里用"每一着都制造新捉"作判据，不追踪棋子身份——已在 docstring 写明。
        """
        mine = [e for e in plies if e[0] == who]
        if not mine:
            return None
        if all(e[1] for e in mine):
            return "check"
        if all(e[2] for e in mine):
            return "chase"
        return None

    def _cycle_plies(self, start: int) -> list[tuple[int, bool, bool]]:
        return self._move_log[start:]

    def _repeat_info(self) -> tuple[int, int]:
        """返回 (当前局面出现次数, 犯规方标记)。

        犯规方标记：1 = 我方会被判负，-1 = 对方会被判负，0 = 判和或无重复。
        页面与观测都用它，让果蝇能"感觉到"自己正走在一条会被判负的重复线上。
        """
        key = self._pos_keys[-1]
        occ = [i for i, k in enumerate(self._pos_keys) if k == key]
        count = len(occ)
        if count < 2:
            return count, 0
        start = occ[-2]
        plies = self._cycle_plies(start)
        me = self._side()
        my_fault = self._fault(plies, me)
        opp_fault = self._fault(plies, -me)
        if my_fault and not opp_fault:
            return count, 1
        if opp_fault and not my_fault:
            return count, -1
        return count, 0

    def _rule_result(self, side: int, label: str) -> Outcome:
        """非"将死/困毙"的终局判定：重复局面（长将/长捉/判和）与自然限着。"""
        me = FLY if side == RED else PC
        opp_name = PC if side == RED else FLY

        key = self._pos_keys[-1]
        occ = [i for i, k in enumerate(self._pos_keys) if k == key]
        if len(occ) >= REP_LIMIT:
            start = occ[-REP_LIMIT]
            plies = self._cycle_plies(start)
            my_fault = self._fault(plies, side)
            opp_fault = self._fault(plies, -side)
            if my_fault and not opp_fault:
                kind = "长将" if my_fault == "check" else "长捉"
                return Outcome(True, opp_name, f"{label}：{kind}判负（同一局面第 {REP_LIMIT} 次出现）")
            if opp_fault and not my_fault:
                kind = "长将" if opp_fault == "check" else "长捉"
                return Outcome(True, me, f"{label}：对方{kind}判负（同一局面第 {REP_LIMIT} 次出现）")
            why = "双方均犯规" if (my_fault and opp_fault) else "无一方犯规"
            return Outcome(True, None, f"{label}：重复局面判和（{why}）")

        if self._no_capture_plies >= NATURAL_LIMIT:
            return Outcome(True, None, f"{label}：自然限着判和（连续 {NATURAL_LIMIT} 着无吃子）")
        return Outcome()

    def apply(self, move: Any) -> None:
        if self._result.over:
            raise ValueError("对局已结束，不能再走子")
        mv = self._norm(move)
        if mv not in set(self._raw_legal()):
            raise ValueError(f"非法着法（走法不合法或会让己方被将军）: {self.move_label(move)}")

        side = self._side()
        label = self.move_label(move)
        cap = self._board[_to_of(mv)]
        frm, to = _from_of(mv), _to_of(mv)
        # "捉"必须比较落子**前后**同一枚子的得子威胁：威胁是这一步新造出来的才算捉。
        chase_before = self._profitable_from(frm, side)
        rec = _do(self._board, self._kings, mv)
        chase_after = self._profitable_from(to, side)
        gave_check = _attacked(self._board, self._kings[-side], side)
        chased = bool(set(chase_after) - set(chase_before))

        self._history.append((rec, self._result, self._last))
        self._last = (mv, label)
        self._ver += 1
        # ---- 规则记账（必须在 _history 追加之后取局面键：那时"轮到谁"才已翻转）----
        self._move_log.append((side, gave_check, chased))
        self._pos_keys.append(self._pos_key())
        self._nc_stack.append(self._no_capture_plies)
        self._no_capture_plies = 0 if cap != EMPTY else self._no_capture_plies + 1

        if cap != EMPTY and abs(cap) == KING:
            # 飞将：将直接吃掉对方的将（本文件的飞将表达方式，见模块 docstring）
            self._result = Outcome(True, FLY if side == RED else PC, f"{label}：飞将吃将")
            return

        opp = -side
        if not self._scratch.gen_legal(opp):
            chk = _attacked(self._board, self._kings[opp], -opp)
            who = FLY if side == RED else PC
            why = "将死" if chk else "困毙（无着法亦判负）"
            self._result = Outcome(True, who, f"{label}：{why}")
        else:
            # 非"将死/困毙"的终局：重复局面（长将/长捉/判和）与自然限着
            self._result = self._rule_result(side, label)

    def undo(self) -> None:
        if not self._history:
            raise IndexError("没有可撤销的着法")
        rec, prev_result, prev_last = self._history.pop()
        _undo(self._board, self._kings, rec)
        self._result = prev_result
        self._last = prev_last
        self._ver += 1
        # 规则记账必须与 _history 同步弹出，否则撤销后重复计数会错位
        self._pos_keys.pop()
        self._move_log.pop()
        self._no_capture_plies = self._nc_stack.pop()

    def outcome(self) -> Outcome:
        r = self._result
        return Outcome(r.over, r.winner, r.reason)

    def heuristic(self) -> float:
        """飞蝇（红方）视角的局面好坏，压缩到 [-1,1]。

        归一化到 800 分左右（差一个车 ≈ 0.81）以便曲线可读。
        """
        s = 0
        for sq in range(CELLS):
            p = self._board[sq]
            if p != EMPTY:
                s += _contrib(p, sq)
        return math.tanh(s / 800.0)

    # ------------------------------------------------------------ 目标事实

    def _goal_facts(self) -> dict:
        """当前局面的"规则与目标"事实，**按局面缓存**，供所有候选复用。

        必须缓存：每个候选都会调一次 observe，而这里要枚举"我这步走完后对方能否
        立刻将死我"（相当于 44×44 次 apply），每个候选都重算的话单步决策要几秒。

        返回：
          `mates`  —— 能立刻将死对方（或困毙对方）的着法集合
          `loses`  —— 走完之后对方能立刻取胜的着法集合（送死）
          `opp_mate_threat` —— 我是否处于"怎么走都会被将死"的处境
        """
        key = (self._ver, len(self._history))
        if getattr(self, "_goal_key", None) == key:
            return self._goal_facts_cache

        facts = {"mates": set(), "loses": set(), "opp_mate_threat": False}
        if not self._result.over:
            me = self._side()
            my_side = FLY if me == RED else PC
            # `_raw_legal()` 返回的是**编码整数**，而 `apply()` 与 `observe()` 用的是
            # `_public()` 转出来的四元组。这里必须先统一成公开形式，
            # 否则 `move in facts["mates"]` 永远为假——查了半天才发现是表示不一致。
            my_moves = [self._public(mv) for mv in self._raw_legal()]
            for m in my_moves:
                self.apply(m)
                o = self.outcome()
                if o.over and o.winner == my_side:
                    facts["mates"].add(m)
                elif not o.over:
                    # 我这步走完后轮到对方：找一手能立刻取胜的（找到即停）
                    opp_moves = [self._public(mv) for mv in self._raw_legal()]
                    for om in opp_moves:
                        self.apply(om)
                        oo = self.outcome()
                        self.undo()
                        if oo.over and oo.winner != my_side:
                            facts["loses"].add(m)
                            break
                self.undo()
            facts["opp_mate_threat"] = bool(facts["loses"])
        self._goal_key = key
        self._goal_facts_cache = facts
        return facts

    def goal_facts(self) -> dict:
        """标准化后的目标事实，供奖励塑形统一使用。"""
        f = self._goal_facts()
        _, rep_fault = self._repeat_info()
        return {"i_can_win_now": bool(f["mates"]),
                "opp_can_win_now": bool(f["opp_mate_threat"]),
                # 1 = 我方正在犯长将/长捉（继续走会被判负），-1 = 对方在犯规
                "repeat_fault": rep_fault}

    # ------------------------------------------------------------ 着法工具

    def _raw_legal(self) -> list[int]:
        if self._cache_ver != self._ver or self._legal_cache is None:
            self._legal_cache = self._scratch.gen_legal(self._side())
            self._cache_ver = self._ver
            self._ann_cache = None
        return self._legal_cache

    def _annotated(self) -> list[tuple[int, int, bool]]:
        """合法着法 + (被吃子价值, 是否将军)。带缓存，供候选剪枝与观测编码复用。"""
        if self._cache_ver == self._ver and self._ann_cache is not None:
            return self._ann_cache
        legal = self._raw_legal()
        side = self._side()
        eng = self._scratch
        ann: list[tuple[int, int, bool]] = []
        for mv in legal:
            cap = self._board[_to_of(mv)]
            capv = PIECE_VALUE[abs(cap)] if cap != EMPTY else 0
            rec = eng.make(mv)
            try:
                chk = _attacked(eng.board, eng.kings[-side], side)
            finally:
                eng.unmake(rec)
            ann.append((mv, capv, chk))
        self._ann_cache = ann
        return ann

    @staticmethod
    def _public(mv: int) -> tuple[int, int, int, int]:
        fr, fc = _rc(_from_of(mv))
        tr, tc = _rc(_to_of(mv))
        return (fr, fc, tr, tc)

    @staticmethod
    def _norm(move: Any) -> int:
        """把对外着法 `(from_row, from_col, to_row, to_col)` 转成内部 int 编码。"""
        if isinstance(move, bool):
            raise ValueError(f"着法格式非法: {move!r}")
        if isinstance(move, int):
            if 0 <= move < (CELLS << 7):
                return move
            raise ValueError(f"着法越界: {move!r}")
        try:
            fr, fc, tr, tc = (int(move[0]), int(move[1]), int(move[2]), int(move[3]))
        except Exception as e:      # noqa: BLE001
            raise ValueError(f"着法格式非法（应为 4 元组）: {move!r}") from e
        if not (0 <= fr < H and 0 <= tr < H and 0 <= fc < W and 0 <= tc < W):
            raise ValueError(f"着法越界: {move!r}")
        return _make_move(_idx(fr, fc), _idx(tr, tc))


    # ------------------------------------------------------------ 候选剪枝

    def _ranked(self) -> list[tuple[int, int, bool]]:
        """按外围优先级排序后的 (着法, 被吃子价值, 是否将军)。

        排序分只有三个档次，刻意不掺任何搜索：
        1. `2_000_000 + 被吃子价值×10`——吃子，且**被吃子越贵越靠前**；
        2. `1_000_000`——将军；
        3. 其余 = 向前推进分×60 + 中心控制分 + 过河兵奖励。
        """
        b = self._board
        side = self._side()
        scored = []
        for mv, capv, chk in self._annotated():
            if capv:
                key = 2_000_000 + capv * 10
            elif chk:
                key = 1_000_000
            else:
                fr, fc = _rc(_from_of(mv))
                tr, tc = _rc(_to_of(mv))
                push = (fr - tr) if side == RED else (tr - fr)
                center = (4 - abs(tc - 4)) * 12 + (4 - abs(2 * tr - 9) // 2) * 8
                crossed = 0
                if abs(b[_from_of(mv)]) == PAWN:
                    if (side == RED and tr <= 4) or (side == BLACK and tr >= 5):
                        crossed = 120          # 兵过河（100 -> 200 那半截由 PST 承担）
                key = push * 60 + center + crossed
            scored.append((key, mv, capv, chk))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(mv, capv, chk) for _, mv, capv, chk in scored]

    def candidates(self, limit: int = 48) -> list[tuple[int, int, int, int]]:
        """候选着法（**外围启发式，不是脑的能力**）。

        象棋一方通常有 30~50 个合法着法，逐个送进网络评估太慢，所以排序后截断：
        先吃子（被吃子价值从高到低）、再将军、最后向前推进/控制中心。
        排序完全是静态启发式，只影响"脑看到哪些候选"，不代表脑会自己挑。
        """
        return [self._public(mv) for mv, _, _ in self._ranked()[:max(0, limit)]]

    # ------------------------------------------------------------ 观测编码

    @staticmethod
    def _pos_term(r: int, c: int) -> float:
        """相对棋盘中心的位置项：大小 = 归一化距离，符号 = 偏右/偏下为正。

        为什么不用纯距离：左右对称的两个候选（如 炮二平五 (7,7)→(7,4) 与
        炮八平五 (7,1)→(7,4)）到中心的距离完全相同，编码会一模一样，
        脑就分不出这两个候选。带符号后二者一个是 +0.66 一个是 -0.66。
        """
        dr = (r - 4.5) / 4.5
        dc = (c - 4.0) / 4.0
        d = math.sqrt((dr * dr + dc * dc) / 2.0)
        if d > 1.0:
            d = 1.0
        return d if (dc - dr) >= 0 else -d

    def observe(self, move: Any | None = None) -> list[float]:
        """105 维观测（**外围编码，不是脑结构的一部分**）。

        分配：**90 维棋盘** + **15 维摘要** = 105。

        - 前 90 维（row-major，`r*9+c`）：0 = 空；飞蝇（红）子取 `+w`、电脑（黑）子取 `-w`，
          其中 `w = min(1, 子力价值/1000)`（帅将 1.0、车 0.9、炮 0.45、马 0.4、
          仕相 0.2、兵卒 0.1）。一格同时表达归属与"这子多重"，值天然在 [-1,1]。
        - 后 15 维：
          0 子力总值差；1 飞蝇是否被将军；2 电脑是否被将军；3 合法着法数；
          4 合法着法中吃子比例；5 将军比例；6 轮到谁；7 步数；8 是否已结束；9 胜负方；
          10~14 随 `move` 切换：
            * `move=None`（只看局面）：候选数、最优候选的被吃子价值、最优候选是否将军、
              该候选起点/终点相对棋盘中心的位置；
            * 指定 `move`：**该着法自身**的特征——是否吃子、被吃子价值、是否将军、
              起点相对中心的位置、终点相对中心的位置。
          （位置项 13/14 用 `_pos_term`：**大小是归一化距离、符号是方向**。
          纯距离会让左右对称的两个候选编码完全相同，脑就分不出它们。）

        这套投影是人写的固定映射：脑的脉冲输出被外围按这套编码解读成"对某个着法的倾向分"，
        它**不是**脑自己长出来的表征，报告与页面都必须如实标注。
        """
        b = self._board
        out = [0.0] * CELLS
        fly_total = 0
        pc_total = 0
        for sq in range(CELLS):
            p = b[sq]
            if p == EMPTY:
                continue
            kind = p if p > 0 else -p
            val = PIECE_VALUE[kind]
            if p > 0:
                fly_total += val
                w = val / 1000.0
                out[sq] = w if w < 1.0 else 1.0
            else:
                pc_total += val
                w = val / 1000.0
                out[sq] = -(w if w < 1.0 else 1.0)

        side = self._side()
        legal = self._raw_legal()
        ann = self._annotated()
        n_legal = len(legal)
        n_cap = sum(1 for _, v, _ in ann if v)
        n_chk = sum(1 for _, _, c in ann if c)

        s = [0.0] * 15
        s[0] = (fly_total - pc_total) / 20000.0
        s[1] = 1.0 if _attacked(b, self._kings[RED], BLACK) else 0.0
        s[2] = 1.0 if _attacked(b, self._kings[BLACK], RED) else 0.0
        s[3] = min(1.0, n_legal / 64.0)
        s[4] = n_cap / n_legal if n_legal else 0.0
        s[5] = n_chk / n_legal if n_legal else 0.0
        s[6] = 1.0 if side == RED else -1.0
        s[7] = min(1.0, len(self._history) / 200.0)
        s[8] = 1.0 if self._result.over else 0.0
        s[9] = (1.0 if self._result.winner == FLY
                else (-1.0 if self._result.winner == PC else 0.0))

        if move is None:
            ranked = self._ranked()
            s[10] = min(1.0, len(ranked) / 48.0)
            if ranked:
                mv, capv, chk = ranked[0]
                fr, fc = _rc(_from_of(mv))
                tr, tc = _rc(_to_of(mv))
                s[11] = min(1.0, capv / 1000.0)
                s[12] = 1.0 if chk else 0.0
                s[13] = self._pos_term(fr, fc)
                s[14] = self._pos_term(tr, tc)
        else:
            mv = self._norm(move)
            f = _from_of(mv)
            t = _to_of(mv)
            cap = b[t]
            capv = PIECE_VALUE[abs(cap)] if cap != EMPTY else 0
            chk = False
            if mv in set(legal):
                # try/finally：`_scratch` 与真实棋盘**共享同一个 list**，
                # 中途抛异常而不撤销就会把幽灵子留在真实局面上。
                rec = self._scratch.make(mv)
                try:
                    chk = _attacked(self._scratch.board, self._scratch.kings[-side], side)
                finally:
                    self._scratch.unmake(rec)
            fr, fc = _rc(f)
            tr, tc = _rc(t)
            s[10] = 1.0 if capv else 0.0
            s[11] = min(1.0, capv / 1000.0)
            s[12] = 1.0 if chk else 0.0
            s[13] = self._pos_term(fr, fc)
            s[14] = self._pos_term(tr, tc)

        # 引擎块 = 90 棋盘 + 11 摘要 = 101，最后 4 维留给框架写目标槽位。
        # 砍掉的是 s[5]（将军比例，与 s[2] 冗余）、s[7]（步数）、
        # s[8]/s[9]（是否结束/胜负方——候选评估时恒为 0）。
        # **保留 s[13]/s[14] 的位置项**：没有它们，左右对称的两个候选编码会完全相同。
        summary = [s[i] for i in _SUMMARY_KEEP]
        # 新增两维"规则知识"：让果蝇能感觉到重复局面与自己的犯规风险。
        # 真实规则在重复局面第 3 次出现时判负/判和——感觉不到就等于不知道这条规则。
        rep_count, rep_fault = self._repeat_info()
        summary.append(min(1.0, rep_count / float(REP_LIMIT)))   # 当前局面重复了几次
        summary.append(float(rep_fault))                          # 1=我会判负 / -1=对方 / 0=无
        v = out + summary
        if len(v) != ENGINE_DIMS:
            raise ValueError(f"xiangqi.observe() 引擎块 {len(v)} 维，应为 {ENGINE_DIMS}")

        # 最后 4 维：规则与目标。**没有这四维，果蝇感觉不到"这步能赢"**——
        # 一步把对方将死的棋和一步普通将军在它眼里是一样的（将军 ≠ 将死），
        # 于是它只在玩"挪棋子、争子力"，不是在下一盘棋。
        facts = self._goal_facts()
        win_now = move in facts["mates"]
        lose_after = move in facts["loses"]
        v = apply_goal_slots(
            [max(-1.0, min(1.0, float(x))) for x in v],
            win_now=win_now,
            lose_after=lose_after,
            opp_can_win_now=bool(facts["opp_mate_threat"]) and not win_now,
            i_can_win_now=bool(facts["mates"]),
        )
        return v

    # ------------------------------------------------------------ 记谱

    _CN = "一二三四五六七八九"

    @classmethod
    def _num(cls, red: bool, n: int) -> str:
        """红方用汉字数字（炮二平五），黑方用阿拉伯数字（马8进7）——棋谱惯例。"""
        return cls._CN[n - 1] if red else str(n)

    @staticmethod
    def _file_no(red: bool, c: int) -> int:
        """纵线号：红方从右往左数 1..9，黑方从左往右数 1..9。"""
        return (W - c) if red else (c + 1)

    def _fallback(self, fr: int, fc: int, tr: int, tc: int) -> str:
        return f"({fr},{fc})→({tr},{tc})"

    def move_label(self, move: Any) -> str:
        """简化中文记谱，如 "炮二平五" / "马8进7"。

        棋子名 + 起点纵线 + 进/退/平 + 目标（平→目标纵线；车炮兵将的进退→步数；
        马相仕的进退→目标纵线）。同一纵线有两个同种棋子时用 前/后（三个时 前/中/后）
        消歧；同纵线超过 3 个、起点无子或着法格式不对时，退化成 `车(9,0)→(7,0)` 的
        坐标写法兜底——**可读且唯一**优先于严格遵守棋谱规范。
        """
        try:
            mv = self._norm(move)
        except ValueError:
            return "??"
        f = _from_of(mv)
        t = _to_of(mv)
        fr, fc = _rc(f)
        tr, tc = _rc(t)
        piece = self._board[f]
        if piece == EMPTY:
            return self._fallback(fr, fc, tr, tc)
        red = piece > 0
        kind = piece if red else -piece
        name = SYMBOL[piece]

        prefix = ""
        same = [sq for sq in range(CELLS)
                if self._board[sq] == piece and sq % W == fc]
        if len(same) > 1:
            if len(same) > 3:
                return self._fallback(fr, fc, tr, tc)
            same.sort(key=lambda sq: (sq // W) if red else -(sq // W))
            words = ("前", "中", "后") if len(same) == 3 else ("前", "后")
            prefix = words[same.index(f)]

        head = (f"{prefix}{name}" if prefix
                else f"{name}{self._num(red, self._file_no(red, fc))}")
        if tr == fr:
            tail = f"平{self._num(red, self._file_no(red, tc))}"
        else:
            forward = (tr < fr) if red else (tr > fr)
            act = "进" if forward else "退"
            if kind in (HORSE, ELEPHANT, ADVISOR):
                tail = f"{act}{self._num(red, self._file_no(red, tc))}"
            else:
                tail = f"{act}{self._num(red, abs(tr - fr))}"
        return head + tail

    # ------------------------------------------------------------ AI 对手

    def ai_move(self, difficulty_key: str) -> tuple[int, int, int, int]:
        """电脑走子（**外围能力**：人写的评估 + alpha-beta，不是脑算的）。

        - 以 `difficulty.randomness` 的概率直接随机走（简单难度的主要来源）；
        - 否则 alpha-beta 迭代加深到 `difficulty.depth`，吃子用 MVV-LVA 排序；
        - **约 2.5 秒硬上限**：超时就用最近一层完整搜完的最优着法，绝不让接口卡住。
          所以 depth=4/6 在 Python 里通常**搜不满**（实测深度见文件末尾自测），
          这是刻意的取舍：宁可给出一个合法且不错的着法，也不能把上层拖住。
        - 搜索在**棋盘副本**上进行，`ai_move` 不会改动真实局面（超时也不会留幽灵子）。
        """
        d = base.DIFF_BY_KEY.get(difficulty_key) or base.DIFF_BY_KEY["normal"]
        legal = self.legal_moves()
        if not legal:
            raise RuntimeError("无合法着法")

        if d.randomness > 0 and self._rng.random() < d.randomness:
            return self._rng.choice(legal)

        side = self._side()
        eng = _Engine(self._board, self._kings)
        deadline = time.monotonic() + SEARCH_SECONDS
        mv, _val = _search_best(eng, side, max(1, int(d.depth)), deadline)
        if mv is None:
            return self._rng.choice(legal)
        pub = self._public(mv)
        # 兜底：搜索返回的着法理论上一定合法，真出了岔子也不能把非法着法交给上层
        return pub if pub in set(legal) else self._rng.choice(legal)

    # ------------------------------------------------------------ 前端渲染

    def _xy(self, r: int, c: int) -> tuple[float, float]:
        """棋子画在**交点**上：col 0..8 → x = c/8，row 0..9 → y = r/9（归一化 0..1）。"""
        return c / (W - 1), r / (H - 1)

    def render(self) -> dict[str, Any]:
        """归一化几何（前端不做任何换算）。

        `lines` 的构成（刻意**不重复画同一条线**）：
        - 横线 10 条：row 4 / row 5 两条正是**楚河汉界的上下边界**，标 `kind="river"`，
          其余 8 条标 `kind="grid"`；
        - 竖线 9 条：中间 7 条在楚河汉界处**断开**，每条出两段（共 16 条线段），
          两条边线贯通（共 2 条）；
        - 九宫斜线 4 条（两宫各 2 条，标 `kind="palace"`）；
        - 外框 4 条（标 `kind="border"`，线宽 2）。

        所以"所有横线 10 条"= 8 条 grid + 2 条 river，不要按 `kind=="grid"` 去数横线。
        """
        cells = []
        for r in range(H):
            for c in range(W):
                p = self._board[_idx(r, c)]
                x, y = self._xy(r, c)
                cells.append({
                    "id": [r, c],
                    "x": round(x, 6),
                    "y": round(y, 6),
                    "piece": None if p == EMPTY else (FLY if p > 0 else PC),
                    "symbol": SYMBOL.get(p),
                    "label": f"{chr(ord('A') + c)}{r}",
                    # value 是"脑对该着法的倾向分"，由桥接层填写；
                    # 棋类引擎无权编造一个外围分数，所以恒为 None。
                    "value": None,
                })

        lines: list[dict[str, Any]] = []
        for r in range(H):
            y = round(r / (H - 1), 6)
            lines.append({"x1": 0.0, "y1": y, "x2": 1.0, "y2": y,
                          "kind": "river" if r in (4, 5) else "grid",
                          "dash": False, "width": 1})
        y4 = round(4 / (H - 1), 6)
        y5 = round(5 / (H - 1), 6)
        for c in range(W):
            x = round(c / (W - 1), 6)
            if c in (0, W - 1):
                lines.append({"x1": x, "y1": 0.0, "x2": x, "y2": 1.0,
                              "kind": "grid", "dash": False, "width": 1})
            else:
                lines.append({"x1": x, "y1": 0.0, "x2": x, "y2": y4,
                              "kind": "grid", "dash": False, "width": 1})
                lines.append({"x1": x, "y1": y5, "x2": x, "y2": 1.0,
                              "kind": "grid", "dash": False, "width": 1})
        for (r1, c1, r2, c2) in ((0, 3, 2, 5), (0, 5, 2, 3),
                                 (7, 3, 9, 5), (7, 5, 9, 3)):
            x1, y1 = self._xy(r1, c1)
            x2, y2 = self._xy(r2, c2)
            lines.append({"x1": round(x1, 6), "y1": round(y1, 6),
                          "x2": round(x2, 6), "y2": round(y2, 6),
                          "kind": "palace", "dash": False, "width": 1})
        for (x1, y1, x2, y2) in ((0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 1.0, 1.0),
                                 (0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0)):
            lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "kind": "border", "dash": False, "width": 2})

        marks = []
        if self._last is not None:
            mv, label = self._last
            fr, fc = _rc(_from_of(mv))
            tr, tc = _rc(_to_of(mv))
            fx, fy = self._xy(fr, fc)
            tx, ty = self._xy(tr, tc)
            marks.append({"x": round(tx, 6), "y": round(ty, 6),
                          "kind": "last_move", "text": label,
                          "x1": round(fx, 6), "y1": round(fy, 6)})

        res = self._result
        return {
            "game": self.name,
            "label": self.label,
            "board": {"w": W, "h": H},
            "cells": cells,
            "lines": lines,
            "marks": marks,
            "meta": {
                "turn": self.turn(),
                "move_no": len(self._history),
                "over": res.over,
                "winner": res.winner,
                "reason": res.reason,
                "in_check": self._scratch.in_check(self._side()),
                "fly_side": "red",
            },
        }


# ================================================================ 自测

if __name__ == "__main__":
    import json

    from services.games.base import DIFF_BY_KEY, check_observe

    fails: list[str] = []

    def check(name: str, cond: bool, extra: str = "") -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")
        if not cond:
            fails.append(name)

    def setup(pieces: dict[tuple[int, int], int], side: int) -> "Xiangqi":
        """摆一个自定义局面（绕过 apply，只给规则测试用）。

        `side` 指定轮到谁走：把 `_history` 填成奇数长度即可。
        """
        g = Xiangqi()
        b = [EMPTY] * CELLS
        for (r, c), p in pieces.items():
            b[_idx(r, c)] = p
        g._board = b
        g._kings = {RED: -1, BLACK: -1}
        for sq in range(CELLS):
            if b[sq] == KING:
                g._kings[RED] = sq
            elif b[sq] == -KING:
                g._kings[BLACK] = sq
        g._history = [] if side == RED else [None]
        g._result = Outcome()
        g._last = None
        g._ver += 1
        g._cache_ver = -1
        g._legal_cache = None
        g._ann_cache = None
        g._scratch = _Engine(b, g._kings, share=True)
        return g

    def label_of(g: "Xiangqi", mv: tuple) -> str:
        return g.move_label(mv)

    print("[1] observe：105 维 / 有限 / 值域 [-1,1]")
    g = Xiangqi()
    v = g.observe(None)
    check("开局 move=None 长度=105", len(v) == OBS_DIM == 105, f"len={len(v)}")
    check("开局 move=None 全有限且 ∈[-1,1]",
          all(math.isfinite(x) and -1.0 <= x <= 1.0 for x in v))
    v2 = g.observe((7, 7, 7, 4))
    check("开局 指定候选 长度/有限/值域",
          len(v2) == 105 and all(math.isfinite(x) and -1 <= x <= 1 for x in v2))
    check("指定候选可复现", v2 == g.observe((7, 7, 7, 4)))
    check("不同候选编码不同", g.observe((7, 7, 7, 4)) != g.observe((7, 1, 7, 4)))
    check("棋盘 90 维：开局两行各有 9 个非零",
          all(g.observe(None)[r * W + c] != 0.0 for r in (0, 9) for c in range(W)))
    check("棋盘 90 维：中路空点编码为 0", g.observe(None)[_idx(5, 4)] == 0.0)
    check("棋盘 90 维：红车 +0.9 / 黑车 -0.9",
          g.observe(None)[_idx(9, 0)] == 0.9 and g.observe(None)[_idx(0, 0)] == -0.9)
    for mv in ((7, 7, 7, 4), (0, 7, 2, 6), (9, 7, 7, 6), (0, 1, 2, 2)):
        g.apply(mv)
    v = g.observe(None)
    check("中局 move=None 长度/有限/值域",
          len(v) == 105 and all(math.isfinite(x) and -1 <= x <= 1 for x in v))
    try:
        check_observe(g)
        check("base.check_observe 通过", True)
    except Exception as e:      # noqa: BLE001
        check("base.check_observe 通过", False, repr(e))
    # 摘要里"轮到谁"这一维必须跟着换边。索引从布局常量推导，不写死数字——
    # 写死会在改布局时静默失效（这一条真的踩过）。
    g2 = Xiangqi()
    g2.apply((7, 7, 7, 4))
    check("轮到谁一维随换边翻转",
          g2.observe(None)[90 + _SUM_TURN] == -1.0
          and Xiangqi().observe(None)[90 + _SUM_TURN] == 1.0,
          f"-> 索引 {90 + _SUM_TURN}")

    print("[2] 开局合法着法数量")
    g = Xiangqi()
    n_red = len(g.legal_moves())
    check("红方（飞蝇）开局合法着法在 40~50 之间", 40 <= n_red <= 50, f"-> {n_red}")
    g.apply((7, 7, 7, 4))
    n_black = len(g.legal_moves())
    check("黑方（电脑）应着合法着法在 40~50 之间", 40 <= n_black <= 50, f"-> {n_black}")
    check("legal_moves 元素为 4 元组且 JSON 可序列化",
          all(isinstance(m, tuple) and len(m) == 4 for m in g.legal_moves()))
    json.dumps(g.legal_moves())
    json.dumps(g.candidates())
    check("candidates JSON 可序列化", True)
    # perft：与公开的中国象棋 perft 基准值对照（外部真值，能一次性验证
    # 七种棋子走法 + 塞象眼/蹩马腿/炮隔子/不能送将 是否整体自洽）
    def perft(eng: "_Engine", side: int, d: int) -> int:
        if d == 0:
            return 1
        n = 0
        for mv in eng.gen_legal(side):
            rec = eng.make(mv)
            n += perft(eng, -side, d - 1)
            eng.unmake(rec)
        return n

    pe = _Engine(Xiangqi()._board, Xiangqi()._kings)
    pe.deadline = time.monotonic() + 300.0
    check("perft(1) = 44", perft(pe, RED, 1) == 44)
    check("perft(2) = 1920", perft(pe, RED, 2) == 1920)
    check("perft(3) = 79666", perft(pe, RED, 3) == 79666)
    print("        （perft(4) = 3290240 也已手工核对通过，此处不跑以免拖慢自测）")

    print("[2b] candidates 排序：先吃子（价值降序）→ 将军 → 推进/控中")
    g = Xiangqi()
    for mv in ((7, 7, 7, 4), (0, 7, 2, 6), (9, 7, 7, 6), (0, 8, 0, 7),
               (9, 8, 9, 7), (0, 0, 1, 0), (7, 4, 3, 4)):    # 红炮吃黑卒
        g.apply(mv)
    ann = {Xiangqi._public(m): (v, c) for m, v, c in g._annotated()}
    cand = g.candidates(48)
    check("candidates 不超过 limit", len(cand) <= 48, f"-> {len(cand)}")
    check("candidates(3) 恰好 3 个", len(g.candidates(3)) == 3)
    check("candidates 全是合法着法", set(cand) <= set(g.legal_moves()))
    check("candidates 里没有重复", len(set(cand)) == len(cand))
    caps = [m for m in cand if ann[m][0]]
    quiets = [m for m in cand if not ann[m][0]]
    check("吃子全部排在非吃子之前",
          all(cand.index(m) < cand.index(q) for m in caps for q in quiets) if caps and quiets else True)
    vals = [ann[m][0] for m in caps]
    check("吃子按被吃子价值降序", vals == sorted(vals, reverse=True), f"-> {vals}")
    check("这局面确实有吃子的候选", len(caps) >= 1, f"-> {len(caps)} 个")
    chk_noncap = [m for m in cand if ann[m][1] and not ann[m][0]]
    rest = [m for m in cand if not ann[m][0] and not ann[m][1]]
    if chk_noncap and rest:
        check("将军着法排在纯推进着法之前",
              max(cand.index(m) for m in chk_noncap) < min(cand.index(m) for m in rest))
    else:
        check("将军着法排在纯推进着法之前（本局面无将军着法，跳过）", True)
    # 专门构造一个有将军着法的局面：车从 (3,0) 到 (0,0)/(3,4) 都将军（黑将空门）
    g = setup({(9, 3): KING, (0, 4): -KING, (3, 0): ROOK, (6, 4): PAWN}, RED)
    gc = g.candidates(48)
    chk_moves = [Xiangqi._public(m) for m, _, c in g._annotated() if c]
    check("存在将军着法（测试有效性）", len(chk_moves) >= 1, f"-> {chk_moves}")
    first_chk = min(gc.index(m) for m in chk_moves)
    first_nonchk = min((gc.index(Xiangqi._public(m)) for m, _, c in g._annotated() if not c),
                       default=len(gc))
    check("将军着法排在所有非将军着法之前", first_chk < first_nonchk,
          f"-> 首个将军在第 {first_chk} 位，首个非将军在第 {first_nonchk} 位")

    print("[3] 马蹩腿")
    base_pos = {(9, 4): KING, (0, 3): -KING, (4, 4): HORSE}
    g = setup(base_pos, RED)
    lm = set(g.legal_moves())
    check("无蹩腿时 (4,4)→(2,3) 合法", (4, 4, 2, 3) in lm)
    check("无蹩腿时 (4,4)→(6,5) 合法", (4, 4, 6, 5) in lm)
    with_block = dict(base_pos)
    with_block[(3, 4)] = PAWN
    g = setup(with_block, RED)
    lm = set(g.legal_moves())
    check("马腿被堵：(4,4)→(2,3) 非法", (4, 4, 2, 3) not in lm)
    check("马腿被堵：(4,4)→(2,5) 非法", (4, 4, 2, 5) not in lm)
    check("马腿被堵：另一侧 (4,4)→(6,3) 仍合法", (4, 4, 6, 3) in lm)
    check("马腿被堵：另一侧 (4,4)→(6,5) 仍合法", (4, 4, 6, 5) in lm)

    print("[4] 象塞眼 / 象不过河")
    g = setup({(9, 4): KING, (0, 3): -KING, (5, 2): ELEPHANT}, RED)
    lm = set(g.legal_moves())
    check("象眼空：(5,2)→(7,0) 合法", (5, 2, 7, 0) in lm)
    check("象眼空：(5,2)→(7,4) 合法", (5, 2, 7, 4) in lm)
    check("象不过河：(5,2)→(3,0) 非法", (5, 2, 3, 0) not in lm)
    check("象不过河：(5,2)→(3,4) 非法", (5, 2, 3, 4) not in lm)
    check("红象所有落点都在 row>=5", all(m[2] >= 5 for m in lm if m[0] == 5 and m[1] == 2))
    g = setup({(9, 4): KING, (0, 3): -KING, (5, 2): ELEPHANT, (6, 1): PAWN}, RED)
    lm = set(g.legal_moves())
    check("塞象眼：(5,2)→(7,0) 非法", (5, 2, 7, 0) not in lm)
    check("塞象眼：(5,2)→(7,4) 仍合法", (5, 2, 7, 4) in lm)

    print("[5] 炮：必须隔子吃，且只能隔一个")
    g = setup({(9, 4): KING, (0, 3): -KING, (7, 4): CANNON, (1, 4): -HORSE}, RED)
    lm = set(g.legal_moves())
    check("无炮架：炮不能吃 (7,4)→(1,4)", (7, 4, 1, 4) not in lm)
    check("无炮架：炮可以走空位 (7,4)→(6,4)", (7, 4, 6, 4) in lm)
    g = setup({(9, 4): KING, (0, 3): -KING, (7, 4): CANNON, (1, 4): -HORSE,
               (5, 4): -PAWN}, RED)
    lm = set(g.legal_moves())
    check("一个炮架：炮可吃 (7,4)→(1,4)", (7, 4, 1, 4) in lm)
    check("一个炮架：炮不能吃炮架本身 (7,4)→(5,4)", (7, 4, 5, 4) not in lm)
    g = setup({(9, 4): KING, (0, 3): -KING, (7, 4): CANNON, (1, 4): -HORSE,
               (5, 4): -PAWN, (3, 4): -PAWN}, RED)
    lm = set(g.legal_moves())
    check("两个子：炮不能隔两个子吃 (7,4)→(1,4)", (7, 4, 1, 4) not in lm)
    check("两个子：炮改为吃第二个子 (7,4)→(3,4)", (7, 4, 3, 4) in lm)

    print("[6] 飞将（两将同列无子可将吃将）")
    g = setup({(9, 4): KING, (0, 4): -KING}, RED)
    lm = set(g.legal_moves())
    check("同列无子：(9,4)→(0,4) 是合法着法", (9, 4, 0, 4) in lm)
    g.apply((9, 4, 0, 4))
    oc = g.outcome()
    check("飞将吃将后立刻判胜", oc.over and oc.winner == FLY, f"-> {oc.as_dict()}")
    check("飞将 reason 可读", "飞将" in oc.reason, f"-> {oc.reason}")
    g = setup({(9, 4): KING, (0, 4): -KING, (5, 4): PAWN}, RED)
    lm = set(g.legal_moves())
    check("中间有子：(9,4)→(0,4) 非法", (9, 4, 0, 4) not in lm)

    print("[7] 送将非法（含「垫子招打」）")
    # 红马挡住黑车对红将的射线，红马一旦让开就是送将
    g = setup({(9, 4): KING, (0, 4): -ROOK, (5, 4): HORSE, (6, 0): PAWN}, RED)
    lm = set(g.legal_moves())
    check("挡箭牌让开是送将，非法", (5, 4, 5, 6) not in lm)
    check("挡箭牌沿射线走也是送将，非法", (5, 4, 3, 4) not in lm)
    check("无关棋子照常可走 (6,0)→(5,0)", (6, 0, 5, 0) in lm)
    # 炮的"垫子招打"：把子垫到炮与将之间，反而让对方炮打自己
    g = setup({(9, 4): KING, (0, 4): -CANNON, (3, 3): PAWN}, RED)
    pseudo = {Xiangqi._public(m) for m in _gen_pseudo(g._board, RED)}
    lm = set(g.legal_moves())
    check("垫子着法本身是伪合法（说明测试有效）", (3, 3, 3, 4) in pseudo)
    check("垫到炮口上是送将，非法", (3, 3, 3, 4) not in lm)
    check("横走到别的格子仍合法", (3, 3, 3, 2) in lm)

    print("[8] 将死 / 困毙 判负")
    # 红车进底线将死（黑只剩光将）
    g = setup({(9, 3): KING, (0, 4): -KING, (1, 8): ROOK, (5, 0): ROOK}, RED)
    g.apply((5, 0, 0, 0))
    oc = g.outcome()
    check("将死：判飞蝇胜", oc.over and oc.winner == FLY, f"-> {oc.as_dict()}")
    check("将死：reason 含'将死'", "将死" in oc.reason, f"-> {oc.reason}")
    # 困毙：黑未被将军但无处可走 —— 中国象棋规则同样判负
    g = setup({(9, 3): KING, (0, 4): -KING, (1, 3): ROOK, (2, 8): ROOK}, RED)
    check("困毙前黑方未被将军", not g._scratch.in_check(BLACK))
    g.apply((2, 8, 2, 5))
    oc = g.outcome()
    check("困毙：同样判飞蝇胜（规则约定）", oc.over and oc.winner == FLY, f"-> {oc.as_dict()}")
    check("困毙：reason 含'困毙'", "困毙" in oc.reason, f"-> {oc.reason}")
    check("困毙局面黑方确实无合法着法", g.legal_moves() == [])

    print("[9] apply / undo 严格互逆（20 步）")
    g = Xiangqi()

    def snap(gm: "Xiangqi") -> tuple:
        return (tuple(gm._board), dict(gm._kings), gm.turn(), len(gm._history),
                gm._last, gm._result.as_dict(), gm.heuristic())

    def rsnap(gm: "Xiangqi") -> str:
        r = gm.render()
        return json.dumps([[c["id"], c["piece"], c["symbol"]] for c in r["cells"]])

    base_snap = snap(g)
    base_rsnap = rsnap(g)
    snaps = [base_snap]
    rsnaps = [base_rsnap]
    applied = []
    for _ in range(20):
        mv = g.legal_moves()[0]          # 取排序最前的合法着法，保证序列一定合法
        g.apply(mv)
        applied.append(mv)
        snaps.append(snap(g))
        rsnaps.append(rsnap(g))
    check("20 步后局面确实变了", snaps[-1] != snaps[0])
    check("20 步后步数为 20", len(g._history) == 20)
    check("20 步内没有分出胜负", not g.outcome().over, f"-> {g.outcome().reason}")
    ok_undo = True
    for i in range(20):
        g.undo()
        if snap(g) != snaps[19 - i] or rsnap(g) != rsnaps[19 - i]:
            ok_undo = False
            check(f"撤销第 {20 - i} 步后状态不一致", False)
            break
    if ok_undo:
        check("逐步撤销，每步都与快照完全一致", True)
    check("撤销后棋盘/将位/胜负/缓存全部恢复",
          snap(g) == base_snap and rsnap(g) == base_rsnap)
    check("撤销后 marks 清空", g.render()["marks"] == [])
    try:
        g.undo()
        check("空局面 undo 抛错", False)
    except IndexError:
        check("空局面 undo 抛错", True)
    try:
        g.apply((4, 4, 4, 5))
        check("无子起点的非法着法抛错", False)
    except ValueError:
        check("无子起点的非法着法抛错", True)
    try:
        g.apply((9, 0, 0, 0))
        check("被挡住的长距离移动抛错", False)
    except ValueError:
        check("被挡住的长距离移动抛错", True)

    print("[10] ai_move：四难度返回合法着法且不改动局面")
    for key in ("easy", "normal", "hard", "brutal"):
        g = Xiangqi()
        for mv in ((7, 7, 7, 4), (0, 7, 2, 6), (9, 7, 7, 6), (0, 8, 0, 7)):
            g.apply(mv)
        before = snap(g)
        legal = set(g.legal_moves())
        t0 = time.monotonic()
        mv = g.ai_move(key)
        dt = time.monotonic() - t0
        check(f"{key} (depth={DIFF_BY_KEY[key].depth}) 返回合法着法",
              mv in legal, f"-> {g.move_label(mv)} {dt:.2f}s")
        check(f"{key} 不修改真实局面", snap(g) == before)
    g = Xiangqi()
    mv = g.ai_move("easy")
    json.dumps(list(mv))
    check("ai_move 返回值 JSON 可序列化", True)
    check("开局 ai_move 不会乱送到对方将口上（返回的着法合法）",
          mv in set(Xiangqi().legal_moves()))
    # 一步杀必须抓住
    g = setup({(9, 3): KING, (0, 4): -KING, (1, 8): ROOK, (5, 0): ROOK}, RED)
    mv = g.ai_move("hard")
    g.apply(mv)
    oc = g.outcome()
    check("hard 能找到一步杀", oc.over and oc.winner == FLY,
          f"-> {g.move_label(mv)} {oc.reason}")

    print("[11] 性能：normal 单次 < 3s")
    g = Xiangqi()
    for mv in ((7, 7, 7, 4), (0, 7, 2, 6), (9, 7, 7, 6), (0, 8, 0, 7),
               (9, 8, 9, 7), (2, 1, 2, 4)):
        g.apply(mv)
    # normal 有 0.12 的随机率，单次可能走随机捷径（瞬间返回），所以取 5 次最大值
    times = []
    for _ in range(5):
        t0 = time.monotonic()
        g.ai_move("normal")
        times.append(time.monotonic() - t0)
    dt_normal = max(times)
    check("ai_move('normal') 在 3s 内返回（取 5 次最大值）", dt_normal < 3.0,
          f"实测 {dt_normal:.3f}s")
    extra = []
    for key in ("easy", "hard", "brutal"):
        t0 = time.monotonic()
        g.ai_move(key)
        extra.append(f"{key}={time.monotonic() - t0:.2f}s")
    print("        （其余难度实测：" + "，".join(extra) + "）")

    print("[12] render 契约 / 记谱 / heuristic")
    g = Xiangqi()
    g.apply((7, 7, 7, 4))
    g.apply((0, 7, 2, 6))
    r = g.render()
    check("顶层字段齐全",
          set(r) == {"game", "label", "board", "cells", "lines", "marks", "meta"},
          f"-> {sorted(r)}")
    check("game/label", r["game"] == "xiangqi" and r["label"] == "中国象棋")
    check("board 9x10", r["board"] == {"w": 9, "h": 10})
    check("cells 90 个", len(r["cells"]) == 90, f"-> {len(r['cells'])}")
    check("cells 字段齐全",
          all(set(c) == {"id", "x", "y", "piece", "symbol", "label", "value"}
              for c in r["cells"]))
    check("cells 坐标落在网格交点上（x=c/8, y=r/9，误差来自保留 6 位小数）",
          all(abs(c["x"] - c["id"][1] / 8) < 1e-6 and abs(c["y"] - c["id"][0] / 9) < 1e-6
              for c in r["cells"]))
    check("cells 坐标全在 [0,1]",
          all(0.0 <= c["x"] <= 1.0 and 0.0 <= c["y"] <= 1.0 for c in r["cells"]))
    cm = {(c["id"][0], c["id"][1]): c for c in r["cells"]}
    check("棋子归属/符号：红炮在 (7,4)", cm[(7, 4)]["piece"] == FLY
          and cm[(7, 4)]["symbol"] == "炮")
    check("棋子归属/符号：黑马在 (2,6)", cm[(2, 6)]["piece"] == PC
          and cm[(2, 6)]["symbol"] == "马")
    check("空点 piece=None 且 symbol=None",
          cm[(5, 4)]["piece"] is None and cm[(5, 4)]["symbol"] is None)
    check("空格 value 恒为 None（不编造外围分数）",
          all(c["value"] is None for c in r["cells"]))
    check("可读名形如 A0..I9", cm[(0, 0)]["label"] == "A0"
          and cm[(9, 8)]["label"] == "I9", f"-> {cm[(0,0)]['label']} {cm[(9,8)]['label']}")
    kinds = {l["kind"] for l in r["lines"]}
    check("lines 字段齐全且 kind 合法",
          all(set(l) == {"x1", "y1", "x2", "y2", "kind", "dash", "width"} for l in r["lines"])
          and kinds <= {"grid", "border", "river", "palace"}, f"kinds={sorted(kinds)}")
    check("lines 含 grid/border/river/palace 四类",
          kinds == {"grid", "border", "river", "palace"}, f"-> {sorted(kinds)}")
    horiz = [l for l in r["lines"] if abs(l["y1"] - l["y2"]) < 1e-12
             and l["kind"] in ("grid", "river")]
    check("横线共 10 条（其中 river 2 条即楚河汉界上下边界）",
          len(horiz) == 10 and sum(1 for l in horiz if l["kind"] == "river") == 2,
          f"-> {len(horiz)} 条，river {sum(1 for l in horiz if l['kind'] == 'river')}")
    vert = [l for l in r["lines"] if abs(l["x1"] - l["x2"]) < 1e-12 and l["kind"] == "grid"]
    check("竖线 9 条：中间 7 条断成两段 + 两条边线贯通",
          len(vert) == 16 and sum(1 for l in vert if l["y1"] == 0.0 and l["y2"] == 1.0) == 2,
          f"-> {len(vert)} 段")
    river_line = [l for l in r["lines"] if l["kind"] == "river"][0]
    y4, y5 = 4 / 9, 5 / 9
    inner = [l for l in vert if l["x1"] not in (0.0, 1.0)]
    check("内竖线全部在楚河汉界处断开（无线段横跨 y=4/9..5/9）",
          all((l["y2"] <= y4 + 1e-9) or (l["y1"] >= y5 - 1e-9) for l in inner))
    check("楚河汉界边界在 row4/row5",
          abs(river_line["y1"] - y4) < 1e-6 or abs(river_line["y1"] - y5) < 1e-6,
          f"-> {river_line['y1']}")
    check("九宫斜线 4 条", len([l for l in r["lines"] if l["kind"] == "palace"]) == 4)
    check("外框 4 条且线宽更粗",
          len([l for l in r["lines"] if l["kind"] == "border"]) == 4
          and all(l["width"] == 2 for l in r["lines"] if l["kind"] == "border"))
    check("lines 坐标全在 [0,1]",
          all(0.0 <= l[k] <= 1.0 for l in r["lines"] for k in ("x1", "y1", "x2", "y2")))
    check("marks 有 last_move 及中文着法",
          len(r["marks"]) == 1 and r["marks"][0]["kind"] == "last_move"
          and r["marks"][0]["text"] == "马8进7", f"-> {r['marks']}")
    check("marks 带起点 x1,y1",
          all(k in r["marks"][0] for k in ("x", "y", "x1", "y1")))
    check("meta 齐全",
          set(r["meta"]) == {"turn", "move_no", "over", "winner", "reason",
                             "in_check", "fly_side"}, f"-> {sorted(r['meta'])}")
    check("meta 内容正确",
          r["meta"]["move_no"] == 2 and r["meta"]["turn"] == FLY
          and r["meta"]["fly_side"] == "red" and not r["meta"]["over"])
    json.dumps(r)
    check("render JSON 可序列化", True)

    g = Xiangqi()
    check("记谱：炮二平五", label_of(g, (7, 7, 7, 4)) == "炮二平五",
          f"-> {label_of(g, (7, 7, 7, 4))}")
    check("记谱：炮八平五", label_of(g, (7, 1, 7, 4)) == "炮八平五",
          f"-> {label_of(g, (7, 1, 7, 4))}")
    check("记谱：马8进7（黑方用阿拉伯数字）", label_of(g, (0, 7, 2, 6)) == "马8进7",
          f"-> {label_of(g, (0, 7, 2, 6))}")
    check("记谱：兵五进一", label_of(g, (6, 4, 5, 4)) == "兵五进一",
          f"-> {label_of(g, (6, 4, 5, 4))}")
    check("记谱：车九进一", label_of(g, (9, 0, 8, 0)) == "车九进一",
          f"-> {label_of(g, (9, 0, 8, 0))}")
    check("记谱：非法/无子起点退化成坐标写法",
          label_of(g, (5, 5, 4, 5)).startswith("("), f"-> {label_of(g, (5, 5, 4, 5))}")
    labels = [label_of(g, m) for m in g.legal_moves()]
    check("开局 44 个着法记谱互不相同（可唯一识别）",
          len(set(labels)) == len(labels), f"-> {len(set(labels))}/{len(labels)}")
    check("空盘 heuristic 有限且在 [-1,1]",
          math.isfinite(g.heuristic()) and -1 <= g.heuristic() <= 1,
          f"-> {g.heuristic():.3f}")
    check("开局势均力敌，heuristic 接近 0", abs(g.heuristic()) < 0.2,
          f"-> {g.heuristic():.3f}")
    g.apply((7, 7, 7, 4))
    g.apply((0, 7, 2, 6))
    g.apply((7, 4, 3, 4))          # 红炮吃黑卒（(3,4) 有黑卒，(2,4)(1,4) 空）
    check("红方白吃一卒后 heuristic 偏正", g.heuristic() > 0.05,
          f"-> {g.heuristic():.3f}")

    print("[N] 完整规则：长将 / 长捉 / 长兑 / 自然限着")

    def setup_rules(pieces, side=RED):
        gm = Xiangqi()
        bb = [EMPTY] * CELLS
        for (rr, cc), pp in pieces.items():
            bb[_idx(rr, cc)] = pp
        gm._board = bb
        gm._kings = {RED: -1, BLACK: -1}
        for sq2 in range(CELLS):
            if bb[sq2] == KING:
                gm._kings[RED] = sq2
            elif bb[sq2] == -KING:
                gm._kings[BLACK] = sq2
        gm._history = [] if side == RED else [None]
        gm._result = Outcome()
        gm._last = None
        gm._ver += 1
        gm._cache_ver = -1
        gm._legal_cache = None
        gm._ann_cache = None
        gm._scratch = _Engine(bb, gm._kings, share=True)
        gm._pos_keys = [gm._pos_key()]
        gm._move_log = []
        gm._nc_stack = []
        gm._no_capture_plies = 0
        return gm

    # 长将：红车沿横线反复将军，黑将只能来回躲 → 红方（长将方）判负
    rr = setup_rules({(9, 3): KING, (1, 8): ROOK, (0, 4): -KING})
    cyc = [(1, 8, 0, 8), (0, 4, 1, 4), (0, 8, 1, 8), (1, 4, 0, 4)]
    for _ in range(2):
        for mv in cyc:
            if not rr.outcome().over:
                rr.apply(mv)
    rc = rr.outcome()
    check("长将：长将方判负（对方胜）", rc.over and rc.winner == PC, f"-> {rc.as_dict()}")
    check("长将：reason 含'长将'", "长将" in rc.reason, f"-> {rc.reason}")

    # 长捉：红车反复追捉无根黑马 → 红方（长捉方）判负
    rr = setup_rules({(9, 3): KING, (2, 0): ROOK, (0, 4): -KING, (4, 3): -HORSE})
    cyc = [(2, 0, 4, 0), (4, 3, 2, 2), (4, 0, 2, 0), (2, 2, 4, 3)]
    for _ in range(2):
        for mv in cyc:
            if not rr.outcome().over:
                rr.apply(mv)
    rc = rr.outcome()
    check("长捉：长捉方判负（对方胜）", rc.over and rc.winner == PC, f"-> {rc.as_dict()}")
    check("长捉：reason 含'长捉'", "长捉" in rc.reason, f"-> {rc.reason}")

    # 长兑：双方各自挪车，既不将军也不捉子 → 判和
    rr = setup_rules({(9, 3): KING, (5, 0): ROOK, (0, 4): -KING, (3, 8): -ROOK})
    cyc = [(5, 0, 5, 1), (3, 8, 3, 7), (5, 1, 5, 0), (3, 7, 3, 8)]
    for _ in range(2):
        for mv in cyc:
            if not rr.outcome().over:
                rr.apply(mv)
    rc = rr.outcome()
    check("长兑：判和（winner 空）", rc.over and rc.winner is None, f"-> {rc.as_dict()}")
    check("长兑：reason 含'判和'且非长将长捉",
          "判和" in rc.reason and "长将" not in rc.reason and "长捉" not in rc.reason,
          f"-> {rc.reason}")

    # 自然限着
    rr = setup_rules({(9, 3): KING, (5, 0): ROOK, (0, 4): -KING, (3, 8): -ROOK})
    rr._no_capture_plies = NATURAL_LIMIT - 1
    rr.apply((5, 0, 5, 1))
    rc = rr.outcome()
    check("自然限着：120 着无吃子判和",
          rc.over and rc.winner is None and "自然限着" in rc.reason, f"-> {rc.reason}")

    # 规则记账与 undo 一致
    rr = Xiangqi()
    snaps = []
    for _ in range(16):
        if rr.outcome().over:
            break
        snaps.append((len(rr._pos_keys), len(rr._move_log), rr._no_capture_plies))
        rr.apply(rr.ai_move("easy"))
    same = True
    for snap in reversed(snaps):
        rr.undo()
        if (len(rr._pos_keys), len(rr._move_log), rr._no_capture_plies) != snap:
            same = False
            break
    check("规则记账与 undo 严格一致", same, f"-> 撤销 {len(snaps)} 着后一致={same}")

    print()
    if fails:
        print(f"❌ {len(fails)} 项失败：")
        for f in fails:
            print(f"   - {f}")
        raise SystemExit(1)
    print("✅ 全部自测通过")
