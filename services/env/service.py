"""环境与评估服务：把共同环境独立成一个可被替换/远端运行的模块。

编排器默认在进程内持有环境（单机模式）；设置 ENV_URL 后可切到独立容器，
便于"环境与评估"与"脑集群"分开部署或换成别的任务。
"""

from __future__ import annotations

from typing import Any

from flycore.scoring import ScoreWeights, compute_score
from flycore.util import content_hash

from .world import GridWorld, TaskSpec


class EnvService:
    def __init__(self):
        self.world: GridWorld | None = None
        self.weights = ScoreWeights()
        self._tasks: dict[str, TaskSpec] = {}

    def register_task(self, spec: TaskSpec) -> dict[str, Any]:
        """第三方系统提交自定义任务。同一 task_id 必须可复现，否则拒绝覆盖。"""
        tid = spec.resolved_id()
        existing = self._tasks.get(tid)
        if existing is not None and existing.fingerprint() != spec.fingerprint():
            raise ValueError(f"任务 {tid} 已存在且参数不同，拒绝静默覆盖；请改用新的 task_id")
        self._tasks[tid] = spec
        return {"task_id": tid, "fingerprint": spec.fingerprint(), **spec.__dict__}

    def list_tasks(self) -> list[dict[str, Any]]:
        return [{"task_id": s.resolved_id(), "fingerprint": s.fingerprint(), **s.__dict__}
                for s in self._tasks.values()]

    def get_task(self, task_id: str) -> TaskSpec:
        if task_id not in self._tasks:
            raise KeyError(f"未知任务 {task_id}")
        return self._tasks[task_id]

    def reset(self, spec: TaskSpec) -> dict[str, Any]:
        self.world = GridWorld(spec)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        if self.world is None:
            raise RuntimeError("环境尚未 reset")
        return {**self.world.summary(), "grid_state": self.world.render_grid()}

    def step(self, actions: list[int]) -> dict[str, Any]:
        if self.world is None:
            raise RuntimeError("环境尚未 reset")
        info = self.world.step(actions)
        return {**info, "targets_found_total": self.world.n_targets - len(self.world.targets),
                "coverage": round(self.world.coverage(), 4)}

    def score(self) -> dict[str, Any]:
        w = self.world
        if w is None:
            raise RuntimeError("环境尚未 reset")
        b = compute_score(
            targets_found=w.n_targets - len(w.targets), targets_total=w.n_targets,
            collisions=sum(a.collisions for a in w.agents),
            redundant_steps=sum(a.redundant_steps for a in w.agents),
            coverage=w.coverage(), steps_used=w.tick, w=self.weights,
        )
        return b.model_dump()

    def health(self) -> dict[str, Any]:
        return {"ok": True, "has_world": self.world is not None,
                "tasks": len(self._tasks),
                "weights": content_hash(self.weights.__dict__)}
