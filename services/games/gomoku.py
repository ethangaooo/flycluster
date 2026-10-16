"""自由五子棋（15×15，无禁手，黑先，五子及以上连成一线即胜）。

**能力归因（重要）**：本文件与 `base.py` 一样属于**外围代码**。
候选剪枝、形状扫描、alpha-beta 搜索、棋盘编码、胜负判定——全部不是脑结构做的。
脑只提供脉冲动力系统，外围把它的输出组活动固定投影成"对某个落点的倾向分"。
页面上必须如实标注，否则就是把外围能力记在脑结构账上。

坐标约定：着法为 `(row, col)` 元组，0-based，`row` 向下、`col` 向右。
黑 = 飞蝇（`base.FLY`，先手），白 = 电脑（`base.PC`）。
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

N = 15
CELLS = N * N
EMPTY = 0
FLY_C = 1   # 飞蝇 = 黑
PC_C = 2    # 电脑 = 白

DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))   # 横 / 竖 / 主对角 / 副对角
STAR_POINTS = ((3, 3), (3, 11), (11, 3), (11, 11), (7, 7))

# 单点形状分值（己方视角）。数量级拉开是为了让搜索天然偏好更快的杀棋。
S_WIN = 10_000_000      # 成五
S_FOUR_OPEN = 500_000   # 活四
S_FOUR = 60_000         # 冲四 / 眠四
S_THREE_OPEN = 20_000   # 活三
S_THREE = 1_600         # 眠三
S_TWO_OPEN = 900        # 活二
S_TWO = 200             # 眠二
S_ONE_OPEN = 80
S_ONE = 10

SEARCH_SECONDS = 2.0    # 单次 ai_move 的思考上限，超时返回当前最优
SEARCH_BRANCH = 12      # 每个搜索节点展开的着法数上限


def _squash(x: float, scale: float = 10.0) -> float:
    """把跨越数量级的分数压进 [-1,1]。

    直接用 tanh(x/k) 不行：成子潜力从"活二 900"到"成五 1e7"跨了四个数量级，
    tanh 会在第一个活三处就饱和成 ±1，曲线就退化成二值信号。
    先取 log1p 再 tanh，能把活二/活三/冲四/活四分层显示出来。
    """
    if x == 0:
        return 0.0
    return math.tanh(math.copysign(math.log1p(abs(x)) / scale, x))


def _dir_score(w: list[int]) -> int:
    """单方向形状分。`w` 是 9 格窗口，中心 w[4] 恒为"假设已落子"的己方子。

    编码：0 空 / 1 己方 / 2 对方或墙（墙与对方子对本方等价）。
    只做"连续子 + 两端是否开"的扫描——因为调用方是**逐点试探**（假设在该点落子），
    断点形状（如 `11011`）在试探断点本身时会自然变成连续五连，不需要额外模式表。
    """
    i = 3
    while i >= 0 and w[i] == 1:
        i -= 1
    j = 5
    while j < 9 and w[j] == 1:
        j += 1
    run = j - i - 1

    if run >= 5:
        return S_WIN

    opens = 0
    if i >= 0 and w[i] == 0:
        opens += 1
    if j < 9 and w[j] == 0:
        opens += 1

    if run == 4:
        return S_FOUR_OPEN if opens == 2 else (S_FOUR if opens == 1 else 0)
    if run == 3:
        if opens == 2:
            # 活三还要求有再延伸一格的空间，否则只是"看着像活三"的眠三
            l2 = i - 1 >= 0 and w[i - 1] == 0
            r2 = j + 1 < 9 and w[j + 1] == 0
            return S_THREE_OPEN if (l2 or r2) else S_THREE
        return S_THREE if opens == 1 else 0
    if run == 2:
        return S_TWO_OPEN if opens == 2 else S_TWO
    if run == 1:
        return S_ONE_OPEN if opens == 2 else S_ONE
    return 0


class _Timeout(Exception):
    """搜索超时哨兵。"""


class Gomoku:
    """自由五子棋。"""

    name = "gomoku"
    label = "五子棋"

    def __init__(self) -> None:
        self._rng = random.Random()
        self.reset()

    # ------------------------------------------------------------ 基础协议

    def reset(self) -> None:
        self._board = [EMPTY] * CELLS
        self._history: list[tuple[int, Outcome, int | None]] = []
        self._last: int | None = None
        self._result = Outcome()

    def turn(self) -> str:
        return FLY if len(self._history) % 2 == 0 else PC

    def legal_moves(self) -> list[tuple[int, int]]:
        b = self._board
        return [(i // N, i % N) for i in range(CELLS) if b[i] == EMPTY]

    def apply(self, move: Any) -> None:
        r, c = int(move[0]), int(move[1])
        if not (0 <= r < N and 0 <= c < N):
            raise ValueError(f"着法越界: {(r, c)}")
        idx = r * N + c
        if self._board[idx] != EMPTY:
            raise ValueError(f"该点已有子: {(r, c)}")

        player = FLY_C if self.turn() == FLY else PC_C
        self._board[idx] = player
        self._history.append((idx, self._result, self._last))
        self._last = idx

        if self._is_five(self._board, idx, player):
            who = FLY if player == FLY_C else PC
            self._result = Outcome(True, who, f"{'飞蝇' if player == FLY_C else '电脑'}五连成线")
        elif len(self._history) == CELLS:
            self._result = Outcome(True, None, "棋盘已满，和棋")
        else:
            self._result = Outcome()

    def undo(self) -> None:
        if not self._history:
            raise IndexError("没有可撤销的着法")
        idx, prev_result, prev_last = self._history.pop()
        self._board[idx] = EMPTY
        self._result = prev_result
        self._last = prev_last

    def outcome(self) -> Outcome:
        r = self._result
        return Outcome(r.over, r.winner, r.reason)

    # ------------------------------------------------------------ 目标事实

    def _winning_cells(self, player: int) -> set[tuple[int, int]]:
        """当前局面上，`player` 一步就能成五的点（返回 (行,列)，与 legal_moves 同形）。"""
        out: set[tuple[int, int]] = set()
        for idx in range(CELLS):
            if self._board[idx] != EMPTY:
                continue
            self._board[idx] = player
            if self._is_five(self._board, idx, player):
                out.add((idx // N, idx % N))
            self._board[idx] = EMPTY
        return out

    def _goal_facts(self) -> dict:
        """目标事实，按局面缓存——每个候选都会调一次 observe，不能重复算。"""
        key = (len(self._history), self._last)
        if getattr(self, "_goal_key", None) == key:
            return self._goal_facts_cache
        if self._result.over:
            facts = {"mine": set(), "opp": set()}
        else:
            mine_c = FLY_C if self.turn() == FLY else PC_C
            facts = {"mine": self._winning_cells(mine_c),
                     "opp": self._winning_cells(PC_C if mine_c == FLY_C else FLY_C)}
        self._goal_key = key
        self._goal_facts_cache = facts
        return facts

    @staticmethod
    def _to_idx(move) -> int:
        return int(move[0]) * N + int(move[1])

    def goal_facts(self) -> dict:
        f = self._goal_facts()
        return {"i_can_win_now": bool(f["mine"]), "opp_can_win_now": bool(f["opp"])}

    # ------------------------------------------------------------ 规则工具

    def _is_five(self, board: list[int], idx: int, player: int) -> bool:
        """`idx` 处刚落 `player` 子后，四个方向是否连成五连及以上。"""
        r, c = divmod(idx, N)
        for dr, dc in DIRS:
            cnt = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < N and 0 <= cc < N and board[rr * N + cc] == player:
                cnt += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < N and 0 <= cc < N and board[rr * N + cc] == player:
                cnt += 1
                rr -= dr
                cc -= dc
            if cnt >= 5:
                return True
        return False

    def _longest_run(self, board: list[int], player: int, dr: int, dc: int) -> int:
        best = 0
        for i in range(CELLS):
            if board[i] != player:
                continue
            r, c = divmod(i, N)
            pr, pc = r - dr, c - dc
            if 0 <= pr < N and 0 <= pc < N and board[pr * N + pc] == player:
                continue    # 不是起点
            n = 0
            rr, cc = r, c
            while 0 <= rr < N and 0 <= cc < N and board[rr * N + cc] == player:
                n += 1
                rr += dr
                cc += dc
            if n > best:
                best = n
        return best

    def _point_scores(self, board: list[int], idx: int, me: int) -> tuple[int, int]:
        """假设在空点 `idx` 落 `me` 子，返回 (己方收益, 对方收益)。

        "对方收益"= 同一空点若被对手占据的价值，用于攻防兼顾的排序与评估。
        """
        if board[idx] != EMPTY:
            return 0, 0
        r, c = divmod(idx, N)
        opp = 3 - me
        wa = [0] * 9
        wb = [0] * 9
        my = 0
        op = 0
        for dr, dc in DIRS:
            for k in range(9):
                rr = r + dr * (k - 4)
                cc = c + dc * (k - 4)
                if 0 <= rr < N and 0 <= cc < N:
                    v = board[rr * N + cc]
                    if v == me:
                        wa[k], wb[k] = 1, 2
                    elif v == opp:
                        wa[k], wb[k] = 2, 1
                    else:
                        wa[k], wb[k] = 0, 0
                else:
                    wa[k], wb[k] = 2, 2   # 出界视作被堵
            wa[4] = 1
            wb[4] = 1
            my += _dir_score(wa)
            op += _dir_score(wb)
        return my, op

    def _frontier(self, board: list[int]) -> list[int]:
        """已有棋子周围 2 格（切比雪夫）内的空点；空盘返回中心点。"""
        pts: set[int] = set()
        for i in range(CELLS):
            if board[i] == EMPTY:
                continue
            r, c = divmod(i, N)
            for dr in range(-2, 3):
                rr = r + dr
                if not (0 <= rr < N):
                    continue
                for dc in range(-2, 3):
                    cc = c + dc
                    if 0 <= cc < N:
                        j = rr * N + cc
                        if board[j] == EMPTY:
                            pts.add(j)
        if not pts:
            return [(N // 2) * N + N // 2]
        return list(pts)

    def _stone_neighbors(self, board: list[int], idx: int) -> int:
        r, c = divmod(idx, N)
        n = 0
        for dr in range(-2, 3):
            rr = r + dr
            if not (0 <= rr < N):
                continue
            for dc in range(-2, 3):
                cc = c + dc
                if 0 <= cc < N and board[rr * N + cc] != EMPTY:
                    n += 1
        return n

    # ------------------------------------------------------------ 候选剪枝

    def candidates(self, limit: int = 48) -> list[tuple[int, int]]:
        """候选着法（**外围启发式**）。

        真实五子棋每步有 200+ 个空点，逐个送进神经网络太慢，所以先剪枝：

        1. 只取**已有棋子周围 2 格（切比雪夫距离 ≤2）内的空点**——这是行业标准做法，
           任何有效着法都必须与现有棋子相邻到一定程度。空盘时直接返回中心点。
        2. 排序分 = 己方在该点的成子潜力 + 0.9×对手在该点的成子潜力 + 邻域棋子密度，
           高的在前。这样 `limit` 截断时留下的是最值得看的点（攻点与挡点都在里面）。
        3. 候选**恒不超出半径 2**，池子不足 `limit` 时就直接少给几个，不用远处的空点补齐。

        第 3 条是刻意的：把远处的点塞进候选，既违反"候选必须紧邻已有棋子"的前提，
        也白白浪费神经网络的评估次数（池子不足只发生在开局前三手，那时远处点毫无价值）。
        密度项已经承担了"按周围棋子数排序"的作用。

        **这不是脑的能力**：剪枝规则是人写的外围启发式，只影响"脑看到哪些候选"。
        """
        if not self._history:
            return [(N // 2, N // 2)]

        side = FLY_C if self.turn() == FLY else PC_C
        board = self._board
        scored: list[tuple[float, int]] = []
        for idx in self._frontier(board):
            my, op = self._point_scores(board, idx, side)
            dens = self._stone_neighbors(board, idx)
            scored.append((my + 0.9 * op + dens * 40.0, idx))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [(i // N, i % N) for _, i in scored[:limit]]

    # ------------------------------------------------------------ AI 对手

    def _order(self, board: list[int], pts: list[int], player: int, limit: int) -> list[int]:
        """着法排序：先试攻防价值高的，alpha-beta 才剪得动。"""
        scored = []
        for idx in pts:
            a, b = self._point_scores(board, idx, player)
            scored.append((a + 0.9 * b, idx))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [idx for _, idx in scored[:limit]]

    def _eval_points(self, board: list[int], pts: list[int], me: int) -> int:
        """局面评估（己方视角）：取双方各自最优两点的成子潜力之差。

        只统计"空点若被占"的潜力，所以每个点的价值天然包含攻与防。
        对方威胁乘 1.05 的轻微防守偏置，避免等价时选择对攻。
        """
        my1 = my2 = op1 = op2 = 0
        for idx in pts:
            a, b = self._point_scores(board, idx, me)
            if a > my1:
                my2, my1 = my1, a
            elif a > my2:
                my2 = a
            if b > op1:
                op2, op1 = op1, b
            elif b > op2:
                op2 = b
        return (my1 + 0.5 * my2) - 1.05 * (op1 + 0.5 * op2)

    def _negamax(self, board: list[int], player: int, depth: int,
                 alpha: float, beta: float, deadline: float, me: int) -> float:
        if time.monotonic() > deadline:
            raise _Timeout

        pts = self._frontier(board)
        if depth == 0:
            return self._eval_points(board, pts, me)

        cand = self._order(board, pts, player, SEARCH_BRANCH)
        if not cand:
            return 0.0

        best = -float("inf")
        opp = 3 - player
        for idx in cand:
            board[idx] = player
            # 必须用 finally 复位：_Timeout 会在递归中途抛出，
            # 若不复位就会在搜索超时后把"幽灵子"留在真实棋盘上。
            try:
                if self._is_five(board, idx, player):
                    return S_WIN - (10 - depth)   # 越快取胜越好
                val = -self._negamax(board, opp, depth - 1, -beta, -alpha, deadline, me)
            finally:
                board[idx] = EMPTY
            if val > best:
                best = val
                if val > alpha:
                    alpha = val
                    if alpha >= beta:
                        break
        return best

    def ai_move(self, difficulty_key: str) -> tuple[int, int]:
        d = base.DIFF_BY_KEY.get(difficulty_key) or base.DIFF_BY_KEY["normal"]
        moves = self.legal_moves()
        if not moves:
            raise RuntimeError("无合法着法")

        me = FLY_C if self.turn() == FLY else PC_C
        opp = 3 - me
        board = self._board

        # 简单难度的主要来源：以一定概率完全随机走
        if d.randomness > 0 and self._rng.random() < d.randomness:
            return self._rng.choice(moves)

        # 战术捷径（所有难度都做，只是为了让对手不至于"看不见一步杀"）
        for (r, c) in moves:
            idx = r * N + c
            board[idx] = me
            win = self._is_five(board, idx, me)
            board[idx] = EMPTY
            if win:
                return (r, c)

        blocks = []
        for (r, c) in moves:
            idx = r * N + c
            board[idx] = opp
            win = self._is_five(board, idx, opp)
            board[idx] = EMPTY
            if win:
                blocks.append(idx)
        if blocks:
            best = max(blocks, key=lambda i: self._point_scores(board, i, me)[0])
            return (best // N, best % N)

        # 主搜索
        pts = self._frontier(board)
        cand = self._order(board, pts, me, SEARCH_BRANCH)
        best_move = (cand[0] // N, cand[0] % N)
        if d.depth <= 1:
            return best_move

        deadline = time.monotonic() + SEARCH_SECONDS
        best_val = -float("inf")
        alpha = -float("inf")
        try:
            for idx in cand:
                board[idx] = me
                try:
                    if self._is_five(board, idx, me):
                        return (idx // N, idx % N)
                    val = -self._negamax(board, opp, d.depth - 1,
                                         -float("inf"), -alpha, deadline, me)
                finally:
                    board[idx] = EMPTY
                if val > best_val:
                    best_val = val
                    best_move = (idx // N, idx % N)
                    if val > alpha:
                        alpha = val
        except _Timeout:
            pass    # 超时就用当前最优，绝不让接口卡住
        return best_move

    # ------------------------------------------------------------ 观测编码

    def observe(self, move: Any | None = None) -> list[float]:
        """105 维观测（**外围编码，不是脑结构的一部分**）。

        分配：`5×5×3 = 75` 维局部窗口 + `30` 维全局摘要 = 105。

        - 局部 75 维：以候选点为中心取 5×5 邻域，每格 3 通道
          （空 / 飞蝇子 / 电脑子，one-hot，值 0/1）。出界的格子三通道全 0，
          以此与"棋盘内的空点"（[1,0,0]）区分开。
          注意通道是**绝对归属**（飞蝇/电脑）而非"我方/对方"，所以同一局面在
          双方视角下编码不同——这是有意的，飞蝇只有一方。
        - 全局 30 维：4 方向最长连子（飞蝇）+ 4（电脑）、双方子数、子数差、
          步数、阶段、该点对双方的潜力（tanh 压缩）、该点占据 one-hot、
          该点行列与离中心距离、该点能否立刻成五（双方）、双方活三以上威胁数、
          双方冲四以上点数、轮到谁、是否已结束。

        `move=None` 时用**当前最优点**作为参考中心——**这只是给页面展示用的默认视角**，
        不代表脑真的"在看"那个点。
        """
        if move is None:
            cand = self.candidates(1)
            idx = cand[0][0] * N + cand[0][1]
        else:
            r, c = int(move[0]), int(move[1])
            if not (0 <= r < N and 0 <= c < N):
                raise ValueError(f"观测点越界: {(r, c)}")
            idx = r * N + c

        v = self._encode(idx)
        if len(v) != ENGINE_DIMS:
            raise ValueError(f"gomoku._encode() 返回 {len(v)} 维，应为 {ENGINE_DIMS}")

        # 最后 4 维：规则与目标。**没有这几维，果蝇感觉不到"这步能赢"**——
        # 原编码虽然含"该点能否立刻成五"，但缺少"走完之后对方能否立刻成五"（会不会输），
        # 也不知道"我现在有没有一步取胜"。
        facts = self._goal_facts()
        win_now = (idx // N, idx % N) in facts["mine"] if move is not None else False
        v = apply_goal_slots(
            v,
            win_now=win_now,
            lose_after=bool(facts["opp"]),
            opp_can_win_now=bool(facts["opp"]),
            i_can_win_now=bool(facts["mine"]),
        )
        return v

    def _encode(self, idx: int) -> list[float]:
        board = self._board
        side = FLY_C if self.turn() == FLY else PC_C
        r, c = divmod(idx, N)
        out: list[float] = []

        # ---- 75 维：5×5×3 局部窗口 ----
        for dr in range(-2, 3):
            rr = r + dr
            for dc in range(-2, 3):
                cc = c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    v = board[rr * N + cc]
                    out.append(1.0 if v == EMPTY else 0.0)
                    out.append(1.0 if v == FLY_C else 0.0)
                    out.append(1.0 if v == PC_C else 0.0)
                else:
                    out.extend((0.0, 0.0, 0.0))

        # ---- 30 维：全局摘要 ----
        n_fly = board.count(FLY_C)
        n_pc = board.count(PC_C)
        for p in (FLY_C, PC_C):
            for dr, dc in DIRS:
                out.append(min(1.0, self._longest_run(board, p, dr, dc) / 5.0))
        out.append(n_fly / CELLS)
        out.append(n_pc / CELLS)
        tot = n_fly + n_pc
        out.append((n_fly - n_pc) / tot if tot else 0.0)
        out.append(len(self._history) / CELLS)
        out.append(min(1.0, len(self._history) / 40.0))

        if board[idx] == EMPTY:
            pot_me, pot_op = self._point_scores(board, idx, side)
        else:
            pot_me = pot_op = 0
        t_me = _squash(pot_me)
        t_op = _squash(pot_op)
        out.append(t_me)
        out.append(t_op)
        out.append((t_me - t_op) / 2.0)

        out.append(1.0 if board[idx] == EMPTY else 0.0)
        out.append(1.0 if board[idx] == FLY_C else 0.0)
        out.append(1.0 if board[idx] == PC_C else 0.0)

        out.append(r / (N - 1))
        out.append(c / (N - 1))
        out.append(max(abs(r - 7), abs(c - 7)) / 7.0)

        out.append(1.0 if pot_me >= S_WIN else 0.0)
        out.append(1.0 if pot_op >= S_WIN else 0.0)

        pts = self._frontier(board)
        my_th = op_th = my_fo = op_fo = 0
        for i in pts:
            a, b = self._point_scores(board, i, side)
            if a >= S_THREE_OPEN:
                my_th += 1
            if a >= S_FOUR:
                my_fo += 1
            if b >= S_THREE_OPEN:
                op_th += 1
            if b >= S_FOUR:
                op_fo += 1
        out.append(min(1.0, my_th / 20.0))
        out.append(min(1.0, op_th / 20.0))
        out.append(min(1.0, my_fo / 10.0))
        out.append(min(1.0, op_fo / 10.0))

        out.append(1.0 if self.turn() == FLY else -1.0)
        out.append(1.0 if self._result.over else 0.0)

        # 引擎块只能占前 101 维（原摘要 30 维 → 26），最后 4 维留给框架写目标槽位。
        # 目标槽位由 observe() 填充——这里只有 idx，拿不到候选着法。
        return [max(-1.0, min(1.0, float(x))) for x in out[:ENGINE_DIMS]]

    # ------------------------------------------------------------ 展示

    def move_label(self, move: Any) -> str:
        r, c = int(move[0]), int(move[1])
        if not (0 <= r < N and 0 <= c < N):
            return "??"
        return f"{chr(ord('A') + c)}{N - r}"

    def heuristic(self) -> float:
        """飞蝇视角的局面好坏，压缩到 [-1, 1]（用于奖励与曲线展示）。"""
        pts = self._frontier(self._board)
        return _squash(self._eval_points(self._board, pts, FLY_C))

    # 归一化坐标：棋盘内容内缩一点，给边框留空隙；前端不做任何换算
    _INSET = 0.04
    _STAR = 0.012

    def _xy(self, r: int, c: int) -> tuple[float, float]:
        span = 1.0 - 2 * self._INSET
        return (self._INSET + c / (N - 1) * span,
                self._INSET + r / (N - 1) * span)

    def render(self) -> dict[str, Any]:
        lo, hi = self._INSET, 1.0 - self._INSET
        cells = []
        for r in range(N):
            for c in range(N):
                x, y = self._xy(r, c)
                v = self._board[r * N + c]
                cells.append({
                    "id": [r, c],
                    "x": round(x, 6),
                    "y": round(y, 6),
                    "piece": FLY if v == FLY_C else (PC if v == PC_C else None),
                    "label": self.move_label((r, c)),
                    # value 是"脑对该点的倾向分"，属于桥接层的输出，棋类引擎无权填写，
                    # 所以这里恒为 None，由上层覆盖。留空比编造一个外围分数更诚实。
                    "value": None,
                })

        lines: list[dict[str, Any]] = []
        for r in range(N):
            _, y = self._xy(r, 0)
            lines.append({"x1": lo, "y1": y, "x2": hi, "y2": y,
                          "kind": "grid", "dash": False, "width": 1})
        for c in range(N):
            x, _ = self._xy(0, c)
            lines.append({"x1": x, "y1": lo, "x2": x, "y2": hi,
                          "kind": "grid", "dash": False, "width": 1})

        b = round(self._INSET - 0.02, 6)
        e = round(1.0 - self._INSET + 0.02, 6)
        for (x1, y1, x2, y2) in ((b, b, e, b), (b, e, e, e), (b, b, b, e), (e, b, e, e)):
            lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "kind": "border", "dash": False, "width": 2})

        for (sr, sc) in STAR_POINTS:
            x, y = self._xy(sr, sc)
            lines.append({"x1": round(x - self._STAR, 6), "y1": y,
                          "x2": round(x + self._STAR, 6), "y2": y,
                          "kind": "star", "dash": False, "width": 2})
            lines.append({"x1": x, "y1": round(y - self._STAR, 6),
                          "x2": x, "y2": round(y + self._STAR, 6),
                          "kind": "star", "dash": False, "width": 2})

        marks = []
        if self._last is not None:
            lr, lc = divmod(self._last, N)
            x, y = self._xy(lr, lc)
            marks.append({"x": round(x, 6), "y": round(y, 6),
                          "kind": "last_move", "text": self.move_label((lr, lc))})

        res = self._result
        return {
            "game": self.name,
            "label": self.label,
            "board": {"w": N, "h": N},
            "cells": cells,
            "lines": lines,
            "marks": marks,
            "meta": {
                "turn": self.turn(),
                "move_no": len(self._history),
                "over": res.over,
                "winner": res.winner,
                "reason": res.reason,
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

    def fresh() -> Gomoku:
        return Gomoku()

    print("[1] observe 维度 / 有限性 / 值域")
    g = fresh()
    for label, mv in (("空盘 move=None", None), ("空盘 move=(7,7)", (7, 7))):
        v = g.observe(mv)
        ok_len = len(v) == OBS_DIM == 105
        ok_fin = all(math.isfinite(x) for x in v)
        ok_rng = all(-1.0 <= x <= 1.0 for x in v)
        check(f"{label}: 长度=105", ok_len, f"len={len(v)}")
        check(f"{label}: 全有限且 ∈[-1,1]", ok_fin and ok_rng)
    g.apply((7, 7)); g.apply((7, 8)); g.apply((8, 8))
    v = g.observe(None)
    check("有子局面 move=None: 长度/有限/值域",
          len(v) == 105 and all(math.isfinite(x) and -1 <= x <= 1 for x in v))
    v2 = g.observe((6, 6))
    check("指定候选点编码可复现", v2 == g.observe((6, 6)))
    check("不同候选点编码不同", g.observe((0, 0)) != g.observe((14, 14)))
    try:
        check_observe(g)
        check("base.check_observe 通过", True)
    except Exception as e:
        check("base.check_observe 通过", False, repr(e))

    # 75 维局部窗口的结构：界内每格三通道恰有一个 1，界外三通道全 0
    gg = fresh()
    gg.apply((7, 7)); gg.apply((7, 8)); gg.apply((8, 8))
    for center in ((0, 0), (7, 7), (14, 14)):
        v = gg.observe(center)
        r0, c0 = center
        expect_oob = sum(1 for dr in range(-2, 3) for dc in range(-2, 3)
                         if not (0 <= r0 + dr < 15 and 0 <= c0 + dc < 15))
        zeros = sum(1 for k in range(25) if v[k * 3:k * 3 + 3] == [0.0, 0.0, 0.0])
        ok = (zeros == expect_oob
              and all(sum(v[k * 3:k * 3 + 3]) in (0.0, 1.0) for k in range(25)))
        check(f"局部窗口 one-hot 合法、界外格全 0 (中心 {center})", ok,
              f"界外={expect_oob} 零格={zeros}")
    v = gg.observe((0, 0))
    check("左上角窗口首格（界外）三通道全 0", v[0:3] == [0.0, 0.0, 0.0], f"-> {v[0:3]}")
    v = gg.observe((14, 14))
    check("右下角窗口末格（界外）三通道全 0", v[72:75] == [0.0, 0.0, 0.0], f"-> {v[72:75]}")
    v = gg.observe((7, 8))
    check("中心格反映真实棋子（电脑子）", v[36:39] == [0.0, 0.0, 1.0], f"-> {v[36:39]}")
    v = gg.observe((7, 7))
    check("中心格反映真实棋子（飞蝇子）", v[36:39] == [0.0, 1.0, 0.0], f"-> {v[36:39]}")
    v = gg.observe((6, 6))
    check("中心格为空时编码 [1,0,0]", v[36:39] == [1.0, 0.0, 0.0], f"-> {v[36:39]}")

    print("[2] 胜负判定：四个方向的五连")
    # (row0, col0) 起点 + (dr, dc)，摆 5 连；穿插对方的子避免提前结束
    cases = {
        "横向": ((7, 3), (0, 1), [(0, 0), (0, 2), (0, 4), (0, 6)]),
        "纵向": ((3, 7), (1, 0), [(0, 0), (2, 0), (4, 0), (6, 0)]),
        "主对角": ((3, 3), (1, 1), [(0, 14), (0, 13), (0, 12), (0, 11)]),
        "副对角": ((3, 11), (1, -1), [(0, 0), (0, 1), (0, 2), (0, 3)]),
    }
    for cname, (start, (dr, dc), filler) in cases.items():
        gg = fresh()
        for i in range(5):
            gg.apply((start[0] + dr * i, start[1] + dc * i))   # 飞蝇(黑)连下
            if i < 4:
                gg.apply(filler[i])                            # 电脑(白)随便垫
        oc = gg.outcome()
        check(f"{cname}五连判胜", oc.over and oc.winner == FLY, f"-> {oc.as_dict()}")

    gg = fresh()
    for i in range(6):                       # 六连也应判胜（"五子或以上"）
        gg.apply((7, i))
        if i < 5:
            gg.apply((0, i))
    check("六连也判胜", gg.outcome().over and gg.outcome().winner == FLY)

    gg = fresh()
    for i in range(4):
        gg.apply((7, 3 + i))
        gg.apply((0, 3 + i))
    check("四连不算胜", not gg.outcome().over)

    print("[3] apply / undo 严格互逆（10 步全撤销）")
    gg = fresh()
    seq = [(7, 7), (7, 8), (8, 6), (6, 6), (6, 7), (5, 5), (9, 9), (8, 8), (4, 4), (3, 3)]
    before_tuple = tuple(gg._board)

    def digest(gm: Gomoku) -> str:
        r = gm.render()
        return json.dumps([[c["id"], c["piece"]] for c in r["cells"]], sort_keys=True)

    before_cells = digest(gg)
    snaps = [digest(gg)]
    for mv in seq:
        gg.apply(mv)
        snaps.append(digest(gg))
    check("10 步后棋盘确实变了", snaps[-1] != snaps[0])
    check("步数=10", len(gg._history) == 10)
    for i in range(10):
        gg.undo()
        if digest(gg) != snaps[9 - i]:
            check(f"撤销第 {10 - i} 步后与快照一致", False)
            break
    else:
        check("逐步撤销每一步都与快照一致", True)
    check("撤销后棋盘元组完全恢复", tuple(gg._board) == before_tuple)
    check("撤销后 render 完全恢复", digest(gg) == before_cells)
    check("撤销后 last_move 清空", gg.render()["marks"] == [])
    check("撤销后步数归零", len(gg._history) == 0)
    check("撤销后轮到飞蝇", gg.turn() == FLY)
    try:
        gg.undo()
        check("空局面 undo 抛错", False)
    except IndexError:
        check("空局面 undo 抛错", True)
    try:
        gg.apply((7, 7)); gg.apply((7, 7))
        check("重复落子抛错", False)
    except ValueError:
        check("重复落子抛错", True)

    print("[4] legal_moves 数量")
    gg = fresh()
    check("空盘 225", len(gg.legal_moves()) == 225, f"-> {len(gg.legal_moves())}")
    prev = 225
    ok = True
    for i in range(225):
        n = 225 - i
        if len(gg.legal_moves()) != n:
            ok = False
            break
        gg.apply((i // 15, i % 15))
        prev = n
    check("随落子严格递减", ok, f"最后 {prev} -> {len(gg.legal_moves())}")
    check("下满后为 0", len(gg.legal_moves()) == 0)
    check("非法着法不在列表里", (7, 7) not in gg.legal_moves())
    gg = fresh()
    mv = gg.legal_moves()[0]
    json.dumps(mv)
    json.dumps(gg.legal_moves())
    check("legal_moves JSON 可序列化", True)

    print("[5] candidates 剪枝")
    gg = fresh()
    check("空盘返回 1 个中心点", gg.candidates() == [(7, 7)], f"-> {gg.candidates()}")
    gg.apply((7, 7))
    cand = gg.candidates()
    expect = {(r, c) for r in range(5, 10) for c in range(5, 10)} - {(7, 7)}
    check("单子时恰好是周围 2 格内的空点", set(cand) == expect,
          f"n={len(cand)} 期望={len(expect)}")
    check("单子时全部满足切比雪夫 ≤2",
          all(max(abs(r - 7), abs(c - 7)) <= 2 for r, c in cand))
    check("候选不含已占点", (7, 7) not in cand)
    check("limit 截断生效", len(gg.candidates(limit=5)) == 5)
    gg.apply((7, 8)); gg.apply((6, 6)); gg.apply((8, 8))
    cand = gg.candidates(limit=48)
    check("多子时不超过 limit", len(cand) <= 48, f"-> {len(cand)}")
    check("多子时候选均为空点", all(gg._board[r * 15 + c] == EMPTY for r, c in cand))
    # 已有一个活三时，其两端必须排在候选最前面（否则剪枝会把关键点丢掉）
    gg = fresh()
    for i in range(3):
        gg.apply((7, 5 + i))
        gg.apply((0, i))
    top2 = set(gg.candidates(limit=2))
    check("活三两端被排进候选前二", top2 == {(7, 4), (7, 8)}, f"-> {sorted(top2)}")
    json.dumps(gg.candidates())
    check("candidates JSON 可序列化", True)

    print("[6] ai_move：四难度合法 + 困难挡活三")
    for key in ("easy", "normal", "hard", "brutal"):
        gg = fresh()
        gg.apply((7, 7))
        rounds = 6 if key in ("easy", "normal") else 2
        ok = True
        for _ in range(rounds):
            legal_now = set(gg.legal_moves())
            mv = gg.ai_move(key)
            if mv not in legal_now:
                ok = False
                break
            gg.apply(mv)
        check(f"{key} (depth={DIFF_BY_KEY[key].depth}) 返回合法着法", ok)

    # ai_move 只思考不落子：绝不能改动真实局面（搜索超时很容易留下幽灵子）
    gg = fresh()
    for i in range(6):
        gg.apply((i, i))
        gg.apply((i, 14 - i))
    snapshot = (tuple(gg._board), gg.turn(), len(gg._history), gg._last)
    for key in ("easy", "normal", "hard", "brutal"):
        gg.ai_move(key)
    check("ai_move 不修改真实局面",
          (tuple(gg._board), gg.turn(), len(gg._history), gg._last) == snapshot)

    # 构造：飞蝇在 (7,5)(7,6)(7,7) 形成活三，(7,4)/(7,8) 皆空，轮到电脑
    gg = fresh()
    for i in range(3):
        gg.apply((7, 5 + i))
        if i < 2:
            gg.apply((0, i))
    assert gg.turn() == PC
    blockers = {(7, 4), (7, 8)}
    mv = gg.ai_move("hard")
    check("困难挡住活三", mv in blockers, f"-> {gg.move_label(mv)} {mv}")
    mm = gg.ai_move("brutal")
    check("残酷挡住活三", mm in blockers, f"-> {gg.move_label(mm)} {mm}")
    easy_blocks = sum(1 for _ in range(20) if gg.ai_move("easy") in blockers)
    check("简单难度确实经常不挡（随机性来源）", easy_blocks < 20,
          f"20 次里挡了 {easy_blocks} 次")

    gg2 = fresh()
    for i in range(3):
        gg2.apply((7, 5 + i))
        if i < 2:
            gg2.apply((0, i))
    # normal 有 0.12 的随机率，单次调用可能故意乱走，所以按多数判定
    normal_blocks = sum(1 for _ in range(20) if gg2.ai_move("normal") in blockers)
    check("普通难度多数情况下挡活三（允许 randomness=0.12）", normal_blocks >= 12,
          f"20 次里挡了 {normal_blocks} 次")

    # 一步成五必须抓住（轮到飞蝇，飞蝇已有四连）
    gg = fresh()
    for i in range(4):
        gg.apply((7, 3 + i))
        gg.apply((0, 2 * i))       # 电脑散开垫子，不构成威胁
    assert gg.turn() == FLY and len(gg._history) == 8
    mv = gg.ai_move("hard")
    check("轮到飞蝇时会补上第五子", mv in {(7, 2), (7, 7)}, f"-> {mv}")

    # 对方（电脑）已有四连时，飞蝇必须去堵
    gg = fresh()
    gg.apply((0, 0))                       # 飞
    gg.apply((10, 3))                      # 电
    gg.apply((0, 1))                       # 飞
    gg.apply((10, 4))                      # 电
    gg.apply((0, 2))                       # 飞
    gg.apply((10, 5))                      # 电
    gg.apply((5, 5))                       # 飞
    gg.apply((10, 6))                      # 电 —— 电脑四连 (10,3..6)，轮到飞蝇
    assert gg.turn() == FLY
    mv = gg.ai_move("hard")
    check("飞蝇会堵电脑的四连", mv in {(10, 2), (10, 7)}, f"-> {gg.move_label(mv)} {mv}")

    print("[7] ai_move 不落已占点（长对局）")
    gg = fresh()
    occupied_ok = True
    for step in range(40):
        key = ("easy", "normal", "hard", "brutal")[step % 4]
        mv = gg.ai_move(key)
        if gg._board[mv[0] * 15 + mv[1]] != EMPTY:
            occupied_ok = False
            break
        gg.apply(mv)
        if gg.outcome().over:
            break
    check("40 步内从不落到已占点", occupied_ok)

    print("[8] render 契约")
    gg = fresh()
    gg.apply((7, 7)); gg.apply((0, 0))
    r = gg.render()
    check("顶层字段齐全",
          set(r) == {"game", "label", "board", "cells", "lines", "marks", "meta"},
          f"-> {sorted(r)}")
    check("game/label", r["game"] == "gomoku" and r["label"] == "五子棋")
    check("board 15x15", r["board"] == {"w": 15, "h": 15})
    check("cells 225 个", len(r["cells"]) == 225)
    check("cells 字段齐全",
          all(set(c) == {"id", "x", "y", "piece", "label", "value"} for c in r["cells"]))
    check("cells 坐标全在 [0,1]",
          all(0.0 <= c["x"] <= 1.0 and 0.0 <= c["y"] <= 1.0 for c in r["cells"]))
    check("lines 字段齐全且 kind 合法",
          all(set(l) >= {"x1", "y1", "x2", "y2", "kind", "dash", "width"}
              and l["kind"] in ("grid", "border", "star") for l in r["lines"]),
          f"kinds={sorted({l['kind'] for l in r['lines']})}")
    check("marks 有 last_move", len(r["marks"]) == 1
          and r["marks"][0]["kind"] == "last_move")
    check("meta 齐全", set(r["meta"]) == {"turn", "move_no", "over", "winner", "reason"})
    check("meta.move_no 与步数一致", r["meta"]["move_no"] == 2)
    check("meta.turn 为飞蝇", r["meta"]["turn"] == FLY)
    cell_map = {(c["id"][0], c["id"][1]): c for c in r["cells"]}
    check("棋子归属正确",
          cell_map[(7, 7)]["piece"] == FLY and cell_map[(0, 0)]["piece"] == PC)
    check("可读坐标 A15 / O1",
          cell_map[(0, 0)]["label"] == "A15" and cell_map[(14, 14)]["label"] == "O1",
          f"-> {cell_map[(0, 0)]['label']} / {cell_map[(14, 14)]['label']}")
    json.dumps(r)
    check("render JSON 可序列化", True)

    print("[9] move_label / heuristic")
    gg = fresh()
    check("中心点 H8", gg.move_label((7, 7)) == "H8", f"-> {gg.move_label((7, 7))}")
    check("越界返回 ??", gg.move_label((99, 99)) == "??")
    check("空盘 heuristic 有限且在 [-1,1]", -1 <= gg.heuristic() <= 1)
    for i in range(3):
        gg.apply((7, 5 + i))
        gg.apply((0, i))
    h = gg.heuristic()
    check("飞蝇活三时 heuristic 偏正", h > 0.5, f"-> {h:.3f}")
    gg.reset()
    for i in range(3):
        gg.apply((0, i))
        gg.apply((7, 5 + i))
    check("电脑活三时 heuristic 偏负", gg.heuristic() < -0.5, f"-> {gg.heuristic():.3f}")

    # 分层压缩：活二/活三/冲四应该有区分，不能一有成子威胁就饱和到 1.0
    gg = fresh()
    vals = []
    for i in range(4):
        gg.apply((7, 3 + i))
        gg.apply((0, 2 * i))
        vals.append(gg.heuristic())
    check("heuristic 随飞蝇连子变长严格上升",
          all(b > a for a, b in zip(vals, vals[1:])),
          f"-> {[round(v, 3) for v in vals]}")
    check("活四也不饱和到 1.0", vals[-1] < 0.99, f"-> {vals[-1]:.3f}")
    check("heuristic 全程有限", all(math.isfinite(v) for v in vals))

    print("[10] ai_move 耗时保护")
    gg = fresh()
    for i in range(6):
        gg.apply((2, i * 2))            # 飞：隔列散开，不构成五连
        gg.apply((10, i * 2 + 1))       # 电：错列散开
    check("耗时测试局面无胜负", not gg.outcome().over, f"-> {len(gg._history)} 手")
    t0 = time.monotonic()
    mv = gg.ai_move("brutal")
    dt = time.monotonic() - t0
    check("brutal 单次耗时 < 4s（受 2s 上限保护）", dt < 4.0, f"-> {dt:.2f}s")
    check("brutal 超时后仍返回合法着法", mv in set(gg.legal_moves()))

    print()
    if fails:
        print(f"❌ {len(fails)} 项失败：")
        for f in fails:
            print(f"   - {f}")
        raise SystemExit(1)
    print("✅ 全部自测通过")
