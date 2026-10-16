"""共同环境：32×32 网格、多个隐藏目标、5×5 局部观测、碰撞与重复探索惩罚。

环境是整个多智能体系统的唯一"事实来源"。所有对照模型（真实结构 / 重连结构 /
储备池 / GRU / 经典策略 / 无脑模型）看到完全相同的观测序列、拥有完全相同的
动作集与消息权限，并在完全相同的任务实例上评估。这是公平对照的前提。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flycore.schemas import GRID_SIZE, N_ACTIONS, OBS_DIM, VIEW_CHANNELS, VIEW_SIZE
from flycore.util import content_hash, rng_for
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]  # 上 下 左 右 停
ACTION_NAMES = ["up", "down", "left", "right", "stay"]


@dataclass
class TaskSpec:
    """任务规格。第三方系统可通过 API 提交自定义任务，但必须可复现。"""

    task_id: str = ""
    grid: int = GRID_SIZE
    n_agents: int = 4
    n_targets: int = 6
    max_ticks: int = 120
    obstacle_fraction: float = 0.12
    # 「感知**有效**的 tick 占比」，不是失效率。设 0.5 表示约一半 tick 拿不到信息。
    # 分条件扫描里轴值 v 是失效率，映射为 perception_fraction = 1 - v。
    perception_fraction: float = 1.0
    seed: int = 0
    env_version: str = "grid-v1"
    # 每个智能体只能看到 [x0,y0,x1,y1]（闭区间）内的目标。
    # 用于实验 C：把任务所需信息拆给不同节点，任何单个节点都不掌握完整信息。
    visibility_regions: list[list[int]] | None = None

    def resolved_id(self) -> str:
        return self.task_id or f"grid{self.grid}-a{self.n_agents}-t{self.n_targets}-s{self.seed}"

    def fingerprint(self) -> str:
        return content_hash(self.model_dump() if hasattr(self, "model_dump") else self.__dict__)


@dataclass
class AgentState:
    x: int
    y: int
    heading: int = 0
    visited: set[tuple[int, int]] = field(default_factory=set)
    targets_found: int = 0
    collisions: int = 0
    redundant_steps: int = 0
    steps: int = 0


class GridWorld:
    """32×32 网格世界。给定种子后完全确定。"""

    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self._build()

    def _build(self) -> None:
        spec = self.spec
        g = spec.grid
        rng = rng_for("world", spec.seed, g, spec.n_targets, spec.obstacle_fraction)
        self.walls = rng.random((g, g)) < spec.obstacle_fraction
        self.targets: set[tuple[int, int]] = set()
        self.n_targets = spec.n_targets
        self.agents: list[AgentState] = []
        self.tick = 0
        self.events: list[dict] = []
        self._place(rng)

    def visible_to(self, agent_id: int, x: int, y: int) -> bool:
        """实验 C 的信息拆分：目标只有落在该智能体的可见区域内才出现在它的观测里。"""
        regions = self.spec.visibility_regions
        if not regions:
            return True
        if agent_id >= len(regions):
            return False
        x0, y0, x1, y1 = regions[agent_id]
        return x0 <= x <= x1 and y0 <= y <= y1

    # ---------------------------------------------------------- 初始化
    def _free_cells(self) -> list[tuple[int, int]]:
        g = self.spec.grid
        out = [(x, y) for y in range(g) for x in range(g) if not self.walls[y, x]]
        return out

    def _place(self, rng: np.random.Generator) -> None:
        free = self._free_cells()
        if len(free) < self.spec.n_agents + self.n_targets:
            raise ValueError("可用空格不足，减小 obstacle_fraction 或 n_targets")
        order = rng.permutation(len(free))
        agent_cells = [free[i] for i in order[: self.spec.n_agents]]
        self.agents = [AgentState(x=a[0], y=a[1], visited={a}) for a in agent_cells]
        for i in order[self.spec.n_agents: self.spec.n_agents + self.n_targets]:
            self.targets.add(free[i])
        self._free = len(free)

    def reset(self, seed: int | None = None) -> list[np.ndarray]:
        if seed is not None:
            self.spec.seed = seed
        self._build()
        return [self.observe(i)[0] for i in range(self.spec.n_agents)]

    # ---------------------------------------------------------- 观测
    def observe(self, agent_id: int) -> tuple[np.ndarray, bool]:
        """返回该智能体的局部观测向量与"本 tick 感知是否有效"。

        感知失效语义（**环境是观测的唯一事实来源**）：
          * 零掉的是**外感受**的 5×5 视野四个通道（墙/目标/同伴/足迹）；
          * **保留**本体感觉（自身 x、y、朝向）与时间进度。
        理由是失效相当于"闭眼"，而不是失去对自己身体的感知。

        脑节点不得再对返回向量做二次解释（例如整体清零）——
        否则同一帧的语义在两处不一致，实验条件就说不清了。
        """
        g = self.spec.grid
        rng = rng_for("perception", self.spec.seed, self.tick, agent_id)
        valid = rng.random() <= self.spec.perception_fraction
        r = VIEW_SIZE // 2
        a = self.agents[agent_id]
        vec = np.zeros((VIEW_SIZE, VIEW_SIZE, VIEW_CHANNELS), dtype=np.float32)
        if valid:
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x, y = a.x + dx, a.y + dy
                    i, j = dy + r, dx + r
                    if not (0 <= x < g and 0 <= y < g):
                        vec[i, j, 0] = 1.0
                        continue
                    if self.walls[y, x]:
                        vec[i, j, 0] = 1.0
                    if (x, y) in self.targets and self.visible_to(agent_id, x, y):
                        vec[i, j, 1] = 1.0
                    if any(o.x == x and o.y == y for k, o in enumerate(self.agents) if k != agent_id):
                        vec[i, j, 2] = 1.0
                    if (x, y) in a.visited:
                        vec[i, j, 3] = 1.0
        head = DIRECTIONS[a.heading]
        pose = np.array([a.x / (g - 1), a.y / (g - 1), float(head[0]), float(head[1])], dtype=np.float32)
        aux = np.array([self.tick / max(1, self.spec.max_ticks)], dtype=np.float32)
        out = np.concatenate([vec.ravel(), pose, aux]).astype(np.float32)
        assert out.size == OBS_DIM, (out.size, OBS_DIM)
        return out, bool(valid)

    # ---------------------------------------------------------- 步进
    def step(self, actions: list[int]) -> dict:
        g = self.spec.grid
        self.tick += 1
        intended: list[tuple[int, int]] = []
        for a, act in zip(self.agents, actions):
            if not (0 <= act < N_ACTIONS):
                raise ValueError(f"非法动作 {act}")
            dx, dy = DIRECTIONS[act]
            nx, ny = a.x + dx, a.y + dy
            if not (0 <= nx < g and 0 <= ny < g) or self.walls[ny, nx]:
                nx, ny = a.x, a.y        # 撞墙不移动，但保留该动作的朝向
            intended.append((nx, ny))
            a.heading = act if act < 4 else a.heading

        counts: dict[tuple[int, int], int] = {}
        for cell in intended:
            counts[cell] = counts.get(cell, 0) + 1

        found_now = 0
        for agent_id, (a, cell) in enumerate(zip(self.agents, intended)):
            a.steps += 1
            a.x, a.y = cell
            if counts[cell] > 1:
                a.collisions += 1
            if cell in self.targets:
                self.targets.discard(cell)
                a.targets_found += 1
                found_now += 1
            elif cell in a.visited:
                a.redundant_steps += 1
            a.visited.add(cell)
            self.events.append({"tick": self.tick, "agent": agent_id, "cell": cell,
                                "action": actions[agent_id]})
        return {
            "tick": self.tick,
            "targets_found_now": found_now,
            "targets_remaining": len(self.targets),
            "collisions_now": sum(1 for c in intended if counts[c] > 1),
        }

    # ---------------------------------------------------------- 统计
    def coverage(self) -> float:
        seen: set[tuple[int, int]] = set()
        for a in self.agents:
            seen |= a.visited
        denom = max(1, self._free - len(self.targets))
        return float(len(seen) / denom)

    def summary(self) -> dict:
        return {
            "task_id": self.spec.resolved_id(),
            "env_version": self.spec.env_version,
            "seed": self.spec.seed,
            "grid": self.spec.grid,
            "tick": self.tick,
            "max_ticks": self.spec.max_ticks,
            "targets_total": self.n_targets,
            "targets_remaining": len(self.targets),
            "targets_found": self.n_targets - len(self.targets),
            "collisions": sum(a.collisions for a in self.agents),
            "redundant_steps": sum(a.redundant_steps for a in self.agents),
            "coverage": round(self.coverage(), 4),
            "per_agent": [
                {"agent_id": i, "x": a.x, "y": a.y, "heading": a.heading,
                 "targets_found": a.targets_found, "visited_cells": len(a.visited),
                 "collisions": a.collisions, "redundant_steps": a.redundant_steps}
                for i, a in enumerate(self.agents)
            ],
        }

    def render_grid(self) -> dict:
        """给管理端渲染用：把环境状态压成可直接画的数组。"""
        g = self.spec.grid
        grid = np.zeros((g, g), dtype=np.int8)
        grid[self.walls] = 1
        for (x, y) in self.targets:
            grid[y, x] = 2
        for i, a in enumerate(self.agents):
            grid[a.y, a.x] = 3 + i
        seen = np.zeros((g, g), dtype=np.int8)
        for a in self.agents:
            for (x, y) in a.visited:
                seen[y, x] = 1
        return {"grid": grid.tolist(), "visited": seen.tolist(),
                "agents": [{"id": i, "x": a.x, "y": a.y, "heading": a.heading}
                           for i, a in enumerate(self.agents)],
                "targets_remaining": len(self.targets)}
