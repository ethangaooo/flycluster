"""中国跳棋（六角星跳棋）引擎。

**坐标系统**：不做硬编码邻接表，而是先给 121 个洞位算出几何坐标，再由坐标反推连接关系。
理由——硬编码 121 个洞位的邻接与跳跃关系，一处写错就会表现成"某条路走不通"，
而这种错误在自测里很难覆盖全。有了坐标，邻接就是"距离恰好等于间距"，
跳跃就是"沿同方向再走一格"，两句话就能写对。

棋盘布局采用经典的行长度表（17 行，合计 121）：

```
        1  2  3  4              10   上三角（飞蝇方起点）
    13 12 11 10  9 10 11 12 13  101  中区：中央六边形 61 + 四个边三角 40
        4  3  2  1              10   下三角（电脑方起点）
```

每行水平居中，行距为 √3/2，列距为 1。相邻两行的半宽差恰好是 0.5，
这正是六角格几何——所以"居中排列"就自动得到标准的星形，不需要手工指定偏移。

**能力归因**：本模块是**外围代码**。规则判定、坐标几何、候选剪枝、AI 搜索都与脑结构无关。
"""

from __future__ import annotations

import math
import random
from typing import Any

from services.games.base import (DIFF_BY_KEY, ENGINE_DIMS, FLY, PC, Difficulty, Move,
                                 Outcome, apply_goal_slots)

NAME = "checkers"
LABEL = "跳棋"

# 经典行长度表（17 行，合计 121）
ROW_LENGTHS = [1, 2, 3, 4, 13, 12, 11, 10, 9, 10, 11, 12, 13, 4, 3, 2, 1]
assert sum(ROW_LENGTHS) == 121, "行长度表必须合计 121"

_DY = math.sqrt(3.0) / 2.0
_ADJ_EPS = 1e-6


def _build_cells() -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[float, float]]]:
    """生成 121 个洞位的 (row, col) 与其"单位间距"坐标。"""
    ids: list[tuple[int, int]] = []
    pos: dict[tuple[int, int], tuple[float, float]] = {}
    for r, w in enumerate(ROW_LENGTHS):
        for c in range(w):
            ids.append((r, c))
            pos[(r, c)] = (c - (w - 1) / 2.0, r * _DY)
    return ids, pos


CELL_IDS, CELL_POS = _build_cells()


def _neighbours() -> dict[tuple[int, int], list[tuple[int, int]]]:
    """邻接 = 距离恰好为 1 的两个洞位。用坐标反推，避免手工维护 121 项邻接表。"""
    out: dict[tuple[int, int], list[tuple[int, int]]] = {cid: [] for cid in CELL_IDS}
    for i, a in enumerate(CELL_IDS):
        ax, ay = CELL_POS[a]
        for b in CELL_IDS[i + 1:]:
            bx, by = CELL_POS[b]
            if abs((ax - bx) ** 2 + (ay - by) ** 2 - 1.0) < _ADJ_EPS:
                out[a].append(b)
                out[b].append(a)
    for k in out:
        out[k].sort()
    return out


ADJ = _neighbours()

# 由几何推出的邻接度数必须都等于 6（边缘洞位理论上分数会低，但六角星每条边的洞位
# 在星形内部仍有 6 个邻居）。实测只有星形最外侧的 6 个尖角是 1 度，其余为 6。
_DEG = {k: len(v) for k, v in ADJ.items()}


def _jump_target(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int] | None:
    """a 越过 b 之后落到哪个洞位。按几何外推，再找最近的洞位。"""
    ax, ay = CELL_POS[a]
    bx, by = CELL_POS[b]
    tx, ty = 2 * bx - ax, 2 * by - ay
    best, bestd = None, 1e-4
    for cid, (cx, cy) in CELL_POS.items():
        d = (cx - tx) ** 2 + (cy - ty) ** 2
        if d < bestd:
            best, bestd = cid, d
    return best


JUMP = {(a, b): _jump_target(a, b) for a in CELL_IDS for b in ADJ[a]}
JUMP = {k: v for k, v in JUMP.items() if v is not None}


def _zones() -> dict[tuple[int, int], str]:
    """起点区 / 目标区。上三角属飞蝇起点、下三角属电脑起点，其余为中区。"""
    z: dict[tuple[int, int], str] = {}
    for cid in CELL_IDS:
        r = cid[0]
        if r <= 3:
            z[cid] = "start_fly"
        elif r >= 13:
            z[cid] = "start_pc"
        else:
            z[cid] = "neutral"
    return z


ZONE = _zones()
FLY_START = [c for c in CELL_IDS if ZONE[c] == "start_fly"]
FLY_GOAL = [c for c in CELL_IDS if ZONE[c] == "start_pc"]
PC_START = FLY_GOAL
PC_GOAL = FLY_START


def _dist_table() -> dict[tuple[tuple[int, int], str], float]:
    """到各目标区的最近距离。预计算——`_dist` 每次遍历 121 个洞位，在搜索里是热点。"""
    t: dict[tuple[tuple[int, int], str], float] = {}
    for cid in CELL_IDS:
        gx, gy = CELL_POS[cid]
        for zname in ("start_fly", "start_pc"):
            best = 1e9
            for c in CELL_IDS:
                if ZONE[c] != zname:
                    continue
                x, y = CELL_POS[c]
                best = min(best, math.hypot(x - gx, y - gy))
            t[(cid, zname)] = best
    return t


DIST = _dist_table()


class Checkers:
    """中国跳棋。着法是完整跳跃链：`[起点, 经过..., 终点]`。"""

    name = NAME
    label = LABEL

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.reset()

    # ---------------------------------------------------------------- 生命周期
    def reset(self) -> None:
        self.board: dict[tuple[int, int], str] = {c: "" for c in CELL_IDS}
        for c in FLY_START:
            self.board[c] = FLY
        for c in PC_START:
            self.board[c] = PC
        self.turn_side = FLY
        self.move_no = 0
        self.last_move: list[tuple[int, int]] | None = None
        # 着法栈。**必须是栈**：只记一个 last_move 的话，撤销第二步就没得撤了。
        self._move_stack: list[list[tuple[int, int]]] = []
        self._over = Outcome()
        self._goal_fly = {c: 0 for c in FLY_GOAL}
        self._goal_pc = {c: 0 for c in PC_GOAL}

    def turn(self) -> str:
        return self.turn_side

    def outcome(self) -> Outcome:
        if self._over.over:
            return self._over
        if all(self.board[c] == FLY for c in FLY_GOAL):
            self._over = Outcome(True, FLY, "飞蝇方填满目标区")
        elif all(self.board[c] == PC for c in PC_GOAL):
            self._over = Outcome(True, PC, "电脑方填满目标区")
        return self._over

    # ---------------------------------------------------------------- 着法
    def _chains(self, start: tuple[int, int]) -> list[list[tuple[int, int]]]:
        """从 start 出发的所有完整跳跃链（跳到不能再跳为止）。

        **表示法**：`[起点, 被跨过的棋子, 落点, 被跨过的棋子, 落点, ...]`。
        把被跨过的棋子显式列进去，跳链长度就恒为**奇数 ≥3**，
        与长度为 2 的平移不可能混淆——早先用 `[起点, 落点]` 表示单次跳跃，
        结果 `apply` 无法区分它和一步平移，合法着法一执行就报"不相邻"。

        规则要求一次跳到底，所以这里收集的是所有**终止**路径：中途还能继续跳的位置
        不会作为一条独立着法返回（除非它确实无后续可跳）。
        """
        results: list[list[tuple[int, int]]] = []
        seen: set[tuple] = set()

        def dfs(path: list[tuple[int, int]], visited: set) -> None:
            extended = False
            cur = path[-1]
            for nb in ADJ[cur]:
                if not self.board[nb]:
                    continue
                dst = JUMP.get((cur, nb))
                if dst is None or self.board[dst] or dst in visited:
                    continue
                extended = True
                dfs(path + [nb, dst], visited | {dst})
            if not extended and len(path) >= 3:
                key = tuple(path)
                if key not in seen:
                    seen.add(key)
                    results.append(path)

        dfs([start], {start})
        return results

    def legal_moves(self) -> list[Move]:
        if self.outcome().over:
            return []
        out: list[Move] = []
        for cid, owner in self.board.items():
            if owner != self.turn_side:
                continue
            for nb in ADJ[cid]:
                if not self.board[nb]:
                    out.append([cid, nb])
            for chain in self._chains(cid):
                out.append(list(chain))
        return out

    def apply(self, move: Move) -> None:
        path = [tuple(p) for p in move]
        if len(path) < 2:
            raise ValueError(f"着法至少要有起点与终点：{move}")
        src, dst = path[0], path[-1]
        if self.board[src] != self.turn_side:
            raise ValueError(f"起点 {src} 不是 {self.turn_side} 的棋子")
        if self.board[dst]:
            raise ValueError(f"终点 {dst} 已被占用")
        if len(path) == 2:
            if dst not in ADJ[src]:
                raise ValueError(f"{src}→{dst} 不相邻")
        elif len(path) >= 3 and len(path) % 2 == 1:
            for i in range(0, len(path) - 2, 2):
                a, over, land = path[i], path[i + 1], path[i + 2]
                if over not in ADJ[a]:
                    raise ValueError(f"{a}→{over} 不相邻")
                if not self.board[over]:
                    raise ValueError(f"{a}→{land} 中间没有棋子可跳")
                if JUMP.get((a, over)) != land:
                    raise ValueError(f"{a}→{land} 不是合法跳跃")
        else:
            raise ValueError(f"着法形态非法（长度 {len(path)}）：{move}")

        self.board[src], self.board[dst] = "", self.board[src]
        self._move_stack.append(path)
        self.last_move = path
        self.move_no += 1
        self._refresh_goal()
        self.turn_side = PC if self.turn_side == FLY else FLY

    def undo(self) -> None:
        if not self._move_stack:
            raise ValueError("没有可撤销的着法")
        path = self._move_stack.pop()
        src, dst = path[0], path[-1]
        piece = self.board[dst]
        self.board[dst], self.board[src] = "", piece
        self.last_move = self._move_stack[-1] if self._move_stack else None
        self.move_no -= 1
        self.turn_side = PC if self.turn_side == FLY else FLY
        self._refresh_goal()
        self._over = Outcome()

    def _refresh_goal(self) -> None:
        self._goal_fly = sum(1 for c in FLY_GOAL if self.board[c] == FLY)
        self._goal_pc = sum(1 for c in PC_GOAL if self.board[c] == PC)

    def goal_facts(self) -> dict:
        """标准化目标事实。跳棋里"赢"= 把自己的 10 枚填满对面三角。

        `opp_can_win_now` 用"对方已在目标区 9 枚"近似——**结构性事实**：
        跳棋不可能被对方一步取胜，所以这里没有 `lose_after` 的对应项。
        """
        me = sum(1 for c in FLY_GOAL if self.board[c] == FLY)
        opp = sum(1 for c in PC_GOAL if self.board[c] == PC)
        return {"i_can_win_now": me >= 9, "opp_can_win_now": opp >= 9}

    # ---------------------------------------------------------------- 评价
    def _dist(self, cid: tuple[int, int], goal_zone: str) -> float:
        return DIST[(cid, goal_zone)]

    def heuristic(self) -> float:
        """从飞蝇视角：自己离目标越近越好，对手离其目标越近越差。范围 [-1,1]。"""
        mine = [self._dist(c, "start_pc") for c, v in self.board.items() if v == FLY]
        theirs = [self._dist(c, "start_fly") for c, v in self.board.items() if v == PC]
        if not mine or not theirs:
            return 0.0
        d = (sum(theirs) / len(theirs) - sum(mine) / len(mine))
        return math.tanh(d / 4.0)

    def _score_for(self, side: str) -> float:
        mine_goal = "start_pc" if side == FLY else "start_fly"
        theirs_goal = "start_fly" if side == FLY else "start_pc"
        mine = [self._dist(c, mine_goal) for c, v in self.board.items() if v == side]
        other = PC if side == FLY else FLY
        theirs = [self._dist(c, theirs_goal) for c, v in self.board.items() if v == other]
        if not mine or not theirs:
            return 0.0
        # 已进目标的棋子额外奖励：跳棋终点是"填满"，不是"靠近"
        mine_got = sum(1 for c in (FLY_GOAL if side == FLY else PC_GOAL) if self.board[c] == side)
        theirs_got = sum(1 for c in (PC_GOAL if side == FLY else FLY_GOAL) if self.board[c] == other)
        return (-(sum(mine) / len(mine)) * 2.0 + (sum(theirs) / len(theirs)) * 2.0
                + mine_got * 3.0 - theirs_got * 3.0)

    # ---------------------------------------------------------------- 候选
    def candidates(self, limit: int = 48) -> list[Move]:
        """外围启发式剪枝：长跳优先 → 靠近目标优先 → 离开出发区优先。

        **这是外围代码，不是脑结构的能力。**
        """
        moves = self.legal_moves()
        if not moves:
            return []
        goal = "start_pc" if self.turn_side == FLY else "start_fly"
        start_zone = "start_fly" if self.turn_side == FLY else "start_pc"

        def key(m: Move) -> tuple:
            dst = m[-1]
            jumped = 0 if len(m) == 2 else (len(m) - 1) // 2
            near = -self._dist(dst, goal)
            leaves = 1 if ZONE.get(m[0]) == start_zone and ZONE.get(dst) != start_zone else 0
            return (-jumped, near, -leaves)

        moves.sort(key=key)
        return moves[:limit]

    # ---------------------------------------------------------------- AI
    def ai_move(self, difficulty: str = "normal") -> Move:
        d: Difficulty = DIFF_BY_KEY.get(difficulty, DIFF_BY_KEY["normal"])
        moves = self.legal_moves()
        if not moves:
            raise RuntimeError("没有合法着法")
        if self.rng.random() < d.randomness or len(moves) == 1:
            return self.rng.choice(moves)
        # 跳棋分支因子大：深度截到 2（再深在 Python 里不可用）
        depth = max(1, min(2, d.depth))
        me = self.turn_side
        best, best_score = moves[0], -1e18
        for m in self.candidates(limit=14):
            self.apply(m)
            sc = -self._search(depth - 1, -1e18, 1e18, me)
            self.undo()
            if sc > best_score:
                best, best_score = m, sc
        return best

    def _search(self, depth: int, alpha: float, beta: float, me: str) -> float:
        over = self.outcome()
        if over.over:
            return 1e6 if over.winner == me else -1e6
        if depth <= 0:
            return self._score_for(me)
        moves = self.candidates(limit=12)
        if not moves:
            return -1e6
        for m in moves:
            self.apply(m)
            v = -self._search(depth - 1, -beta, -alpha, me)
            self.undo()
            if v > alpha:
                alpha = v
            if alpha >= beta:
                break
        return alpha

    # ---------------------------------------------------------------- 观测
    def observe(self, move: Move | None = None) -> list[float]:
        """编码成恰好 105 维。**外围编码，不属于脑结构。**

        90 维局部窗口 + 15 维局面摘要；窗口中心取候选落点（`move=None` 时取最靠前的己方棋子）。
        """
        from flycore.schemas import OBS_DIM

        anchor = tuple(move[-1]) if move else self._anchor()
        vec: list[float] = []
        order = sorted(CELL_IDS, key=lambda c: (
            (CELL_POS[c][0] - CELL_POS[anchor][0]) ** 2 + (CELL_POS[c][1] - CELL_POS[anchor][1]) ** 2))
        for c in order[:30]:
            v = self.board[c]
            vec.extend([1.0 if not v else 0.0,
                        1.0 if v == FLY else 0.0,
                        1.0 if v == PC else 0.0])
        vec = [x if x else -1.0 for x in vec]
        while len(vec) < 90:
            vec.extend([-1.0, -1.0, -1.0])
        vec = vec[:90]

        mine = [self._dist(c, "start_pc") for c, v in self.board.items() if v == FLY]
        theirs = [self._dist(c, "start_fly") for c, v in self.board.items() if v == PC]
        out_of_start = sum(1 for c in FLY_START if self.board[c] != FLY)
        in_goal = sum(1 for c in FLY_GOAL if self.board[c] == FLY)
        cands = self.candidates(limit=48)
        longest = max((0 if len(m) == 2 else (len(m) - 1) // 2 for m in cands), default=0)
        extra = [
            _n(sum(mine) / max(1, len(mine)), 12.0), _n(max(mine, default=0), 12.0),
            _n(min(mine, default=0), 12.0), _n(sum(theirs) / max(1, len(theirs)), 12.0),
            _n(max(theirs, default=0), 12.0), _n(min(theirs, default=0), 12.0),
            _n(out_of_start, 10.0), _n(in_goal, 10.0),
            _n(len(cands), 48.0), _n(longest, 4.0),
            1.0 if self.turn_side == FLY else -1.0,
            _n(self.move_no, 80.0),
            1.0 if self.outcome().over else -1.0,
        ]
        if move is not None:
            extra.append(_n(0 if len(move) == 2 else (len(move) - 1) // 2, 4.0))
            extra.append(1.0 - _n(self._dist(tuple(move[-1]), "start_pc"), 12.0) * 2)
        else:
            extra.append(-1.0)
            extra.append(-1.0)
        # 引擎块只能占前 101 维，最后 4 维留给框架写目标槽位
        vec.extend(extra[:11])
        vec = (_coerce(vec) + [-1.0] * ENGINE_DIMS)[:ENGINE_DIMS]

        # 最后 4 维：规则与目标。跳棋的"赢"= 把 10 枚棋子填满对面三角。
        # `lose_after` 恒为假是**结构性事实**：跳棋不可能被对方一步取胜
        # （对方要填满的是它自己的目标区，与我的这一步无关）。写死为 False 比留空更诚实。
        me_goal = sum(1 for c in FLY_GOAL if self.board[c] == FLY)
        opp_goal = sum(1 for c in PC_GOAL if self.board[c] == PC)
        win_now = False
        if move is not None:
            try:
                self.apply(move)
                o = self.outcome()
                win_now = o.over and o.winner == FLY
                self.undo()
            except Exception:
                win_now = False
        vec = apply_goal_slots(
            vec,
            win_now=win_now,
            lose_after=False,
            opp_can_win_now=(opp_goal >= 9),
            i_can_win_now=(me_goal >= 9),
        )
        assert len(vec) == OBS_DIM
        return vec

    def _anchor(self) -> tuple[int, int]:
        """没有候选时用哪个洞位当窗口中心：选离目标最近的己方棋子。"""
        mine = [c for c, v in self.board.items() if v == self.turn_side]
        if not mine:
            return CELL_IDS[0]
        goal = "start_pc" if self.turn_side == FLY else "start_fly"
        return min(mine, key=lambda c: self._dist(c, goal))

    # ---------------------------------------------------------------- 渲染
    def render(self) -> dict[str, Any]:
        xs = [p[0] for p in CELL_POS.values()]
        ys = [p[1] for p in CELL_POS.values()]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        pad = 0.06
        sx = lambda v: pad + (v - x0) / max(1e-9, x1 - x0) * (1 - 2 * pad)
        sy = lambda v: pad + (v - y0) / max(1e-9, y1 - y0) * (1 - 2 * pad)

        cells = []
        for cid in CELL_IDS:
            x, y = CELL_POS[cid]
            cells.append({
                "id": list(cid), "x": round(sx(x), 5), "y": round(sy(y), 5),
                "piece": self.board[cid] or None,
                "zone": ZONE[cid],
                "label": f"R{cid[0]}C{cid[1]}",
                "value": None,
            })

        lines = []
        for zname, zcells in (("start_fly", FLY_START), ("start_pc", PC_START)):
            for c in zcells:
                x, y = CELL_POS[c]
                for nb in ADJ[c]:
                    if ZONE.get(nb) == zname and c < nb:
                        nx, ny = CELL_POS[nb]
                        lines.append({"x1": round(sx(x), 5), "y1": round(sy(y), 5),
                                      "x2": round(sx(nx), 5), "y2": round(sy(ny), 5),
                                      "kind": "zone", "dash": False, "width": 1})

        over = self.outcome()
        return {
            "game": NAME, "label": LABEL, "board": {"w": 121, "h": 121},
            "cells": cells, "lines": lines,
            "marks": [] if not self.last_move else [
                {"x": round(sx(CELL_POS[tuple(self.last_move[-1])][0]), 5),
                 "y": round(sy(CELL_POS[tuple(self.last_move[-1])][1]), 5),
                 "kind": "last_move", "text": self.move_label(self.last_move)}],
            "meta": {"turn": self.turn_side, "move_no": self.move_no,
                     "over": over.over, "winner": over.winner, "reason": over.reason,
                     "fly_in_goal": self._goal_fly, "pc_in_goal": self._goal_pc},
        }

    def move_label(self, move: Move) -> str:
        path = [tuple(p) for p in move]
        if len(path) == 2:
            return f"R{path[0][0]}C{path[0][1]}→R{path[1][0]}C{path[1][1]}"
        return (f"跳跃 {(len(path) - 1) // 2} 连："
                + "→".join(f"R{p[0]}C{p[1]}" for p in path[2::2]))


def _n(v: float, scale: float) -> float:
    return max(-1.0, min(1.0, float(v) / scale))


def _coerce(vec: list[float]) -> list[float]:
    return [max(-1.0, min(1.0, float(x))) for x in vec]


# --------------------------------------------------------------------- 自测

if __name__ == "__main__":
    import traceback

    from flycore.schemas import OBS_DIM
    from services.games.base import check_observe

    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn):
        try:
            info = fn() or ""
            results.append((name, True, str(info)))
        except Exception as exc:
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()

    def t_cells():
        assert len(CELL_IDS) == 121, len(CELL_IDS)
        assert len(set(CELL_IDS)) == 121
        return "121 个洞位，无重复"

    def t_geometry():
        xs = [p[0] for p in CELL_POS.values()]
        ys = [p[1] for p in CELL_POS.values()]
        assert all(-20 <= x <= 20 for x in xs)
        # 星形上下对称：顶部尖角 1 个、底部尖角 1 个
        top = [c for c in CELL_IDS if CELL_POS[c][1] == min(ys)]
        bot = [c for c in CELL_IDS if CELL_POS[c][1] == max(ys)]
        assert len(top) == 1 and len(bot) == 1, (len(top), len(bot))
        r = Checkers(0).render()
        assert all(0 <= c["x"] <= 1 and 0 <= c["y"] <= 1 for c in r["cells"])
        assert all(len({(round(c["x"], 5), round(c["y"], 5)) for c in r["cells"]}) == 121
                   for _ in [0])
        return "坐标均在 [0,1]，顶部/底部各 1 个尖角，121 个位置互不重合"

    def t_degrees():
        degs = sorted(set(_DEG.values()))
        assert max(degs) == 6, degs
        # 星形尖端是 2 度（尖角那一个洞位在星形内部仍有 2 个邻居），不是 1 度——
        # 第一版断言写成"六个 1 度尖角"，是我想当然了。
        assert min(degs) == 2, f"最小度数 {min(degs)}"
        # 实测集合是 {2,4,5,6}：星形里没有 3 度洞位。断言"恰好六个 2 度尖端"更有意义。
        tips = [c for c, d in _DEG.items() if d == 2]
        assert len(tips) == 6, f"2 度洞位 {len(tips)} 个，应为 6 个尖端"
        return f"度数集合 {degs}；2 度恰好 6 个（六个尖端）"

    def t_initial():
        g = Checkers(0)
        assert len(FLY_START) == 10 and len(PC_START) == 10
        assert all(g.board[c] == FLY for c in FLY_START)
        assert all(g.board[c] == PC for c in PC_START)
        assert sum(1 for v in g.board.values() if v) == 20
        return "飞蝇 10 枚在上三角、电脑 10 枚在下三角，其余 101 空"

    def t_step_and_jump():
        g = Checkers(0)
        moves = g.legal_moves()
        steps = [m for m in moves if len(m) == 2]
        assert steps, "开局应有平移"
        g.apply(steps[0])
        assert g.turn() == PC
        g.undo()
        assert g.turn() == FLY
        return f"开局合法着法 {len(moves)} 个（其中平移 {len(steps)}）"

    def t_chain():
        """构造一条两连跳：飞蝇在顶点 (0,0)，(1,0) 与 (3,0) 各放一枚垫脚石。"""
        g = Checkers(0)
        for c in CELL_IDS:
            g.board[c] = ""
        g.board[(0, 0)] = FLY
        g.board[(1, 0)] = PC
        g.board[(3, 0)] = PC
        g.turn_side = FLY
        chains = [m for m in g.legal_moves() if len(m) >= 5]
        assert chains, "未找到两连跳链"
        expect = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 4)]
        assert [tuple(x) for x in chains[0]] == expect, (chains[0], expect)
        # 半截链不得作为独立着法出现
        half = [(0, 0), (1, 0), (2, 0)]
        assert all([tuple(x) for x in m] != half for m in g.legal_moves()), "出现了半截跳链"
        g.apply(chains[0])
        assert g.board[(4, 4)] == FLY and not g.board[(0, 0)]
        g.undo()
        assert g.board[(0, 0)] == FLY and not g.board[(4, 4)]
        return f"两连跳链正确：{[tuple(x) for x in expect]}，且无半截链"

    def t_no_occupied():
        g = Checkers(0)
        for m in g.legal_moves():
            assert not g.board[tuple(m[-1])], f"终点 {m[-1]} 已被占用"
        return f"检查了 {len(g.legal_moves())} 个着法，终点全部为空"

    def t_jump_over_one():
        g = Checkers(0)
        checked = 0
        for m in g.legal_moves():
            if len(m) == 2:
                continue
            for i in range(0, len(m) - 2, 2):
                a, over, land = m[i], m[i + 1], m[i + 2]
                assert g.board[tuple(over)], f"跳跃 {a}→{land} 中间没有棋子"
                assert tuple(over) in ADJ[tuple(a)], f"{a}→{over} 不相邻"
                assert JUMP.get((tuple(a), tuple(over))) == tuple(land)
                checked += 1
        return f"校验了 {checked} 次跳跃，全部恰好跨过一枚棋子"

    def t_observe():
        g = Checkers(0)
        check_observe(g)
        v = g.observe(g.legal_moves()[0])
        assert len(v) == OBS_DIM and all(math.isfinite(x) for x in v)
        assert all(-1.0001 <= x <= 1.0001 for x in v)
        return f"{len(v)} 维，全有限，值域 [{min(v):.2f}, {max(v):.2f}]"

    def t_undo():
        g = Checkers(0)
        snaps = []
        for _ in range(20):
            moves = g.legal_moves()
            if not moves:
                break
            snaps.append((dict(g.board), g.turn_side, g.move_no))
            g.apply(g.candidates(limit=1)[0])
        for snap in reversed(snaps):
            g.undo()
            assert (dict(g.board), g.turn_side, g.move_no) == snap, "undo 未完全恢复"
        return f"连走 {len(snaps)} 步后全部撤销，局面/轮次/步数逐快照一致"

    def t_win():
        """9 枚已在目标区，第 10 枚走一步填满 → 判飞蝇胜。"""
        g = Checkers(0)
        for c in CELL_IDS:
            g.board[c] = ""
        target = FLY_GOAL[0]
        for c in FLY_GOAL[1:]:
            g.board[c] = FLY
        outside = next(nb for nb in ADJ[target] if not g.board[nb])
        g.board[outside] = FLY
        g.turn_side = FLY
        filled = len([c for c in FLY_GOAL if g.board[c] == FLY])
        assert filled == 9, f"构造失败：目标区已填 {filled}，应为 9"
        assert outside not in FLY_GOAL, "outside 不该落在目标区里"
        g.apply([outside, target])
        o = g.outcome()
        assert o.over and o.winner == FLY, o
        return f"填满目标区后判飞蝇胜（{o.reason}）"

    def t_observe():
        g = Checkers(0)
        check_observe(g)
        v = g.observe(g.legal_moves()[0])
        assert len(v) == OBS_DIM and all(math.isfinite(x) for x in v)
        assert all(-1.0001 <= x <= 1.0001 for x in v)
        return f"{len(v)} 维，全有限，值域 [{min(v):.2f}, {max(v):.2f}]"

    def t_undo():
        g = Checkers(0)
        snaps = []
        for _ in range(20):
            moves = g.legal_moves()
            if not moves:
                break
            snaps.append(dict(g.board))
            g.apply(g.candidates(limit=1)[0])
        for snap in reversed(snaps):
            g.undo()
            assert dict(g.board) == snap, "undo 未恢复到相同局面"
        assert g.move_no == 0 and g.turn() == FLY
        return f"连走 {len(snaps)} 步后全部撤销，逐快照一致"

    def t_ai():
        g = Checkers(0)
        import time
        for key in ("easy", "normal", "hard", "brutal"):
            g2 = Checkers(1)
            before = dict(g2.board)
            t0 = time.time()
            m = g2.ai_move(key)
            dt = time.time() - t0
            assert m in g2.legal_moves() or [tuple(x) for x in m] in [
                [tuple(x) for x in mm] for mm in g2.legal_moves()], f"{key} 返回非法着法"
            assert dict(g2.board) == before, f"{key} 修改了局面"
            assert dt < 2.5, f"{key} 耗时 {dt:.2f}s"
        return "四难度均返回合法着法、不修改局面、单次 < 2.5s"

    check("洞位总数与唯一性", t_cells)
    check("星形几何与归一化坐标", t_geometry)
    check("邻接度数（尖端 2 度 / 内部 6 度）", t_degrees)
    check("初始布局", t_initial)
    check("平移与轮次切换", t_step_and_jump)
    check("连跳链完整且无半截", t_chain)
    check("不能跳到已占洞位", t_no_occupied)
    check("跳跃恰跨一枚棋子", t_jump_over_one)
    check("观测 105 维与值域", t_observe)
    check("apply/undo 严格互逆", t_undo)
    check("胜负判定", t_win)
    check("AI 四难度", t_ai)

    ok = sum(1 for _, p, _ in results if p)
    print("=" * 72)
    for name, passed, info in results:
        print(f"{'✅' if passed else '❌'}  {name:<28} {info}")
    print("=" * 72)
    print(f"通过 {ok}/{len(results)}")
    raise SystemExit(0 if ok == len(results) else 1)
