"""中心记忆中枢服务：经验汇集、检索、冲突、版本与回放。

服务只做"记忆"，不做规划。研究模式下它绝不返回"这一步该往哪走"的最终指令；
如果某个版本开启了中心规划（enable_central_planner），必须在报告里把能力归因于
"脑集群 + 中心规划器的组合系统"，不能算作脑间自主涌现。
"""

from __future__ import annotations

from typing import Any

from flycore.schemas import ExperienceRecord, MemoryQuery, MemorySubmitResult
from flycore.util import content_hash, now_iso

from .store import MemoryStore


class MemoryHubService:
    def __init__(self, db_path: str = "data/memory_hub.sqlite3"):
        self.store = MemoryStore(db_path)
        self.started_at = now_iso()

    # ------------------------------------------------------------ 写入
    def submit(self, rec: ExperienceRecord) -> MemorySubmitResult:
        return self.store.submit(rec)

    def submit_many(self, records: list[ExperienceRecord]) -> list[MemorySubmitResult]:
        return [self.store.submit(r) for r in records]

    # ------------------------------------------------------------ 读取
    def query(self, q: MemoryQuery, caller: str = "") -> dict[str, Any]:
        return self.store.query(q, caller=caller)

    def conflicts(self, task_id: str | None = None) -> list[dict[str, Any]]:
        return self.store.conflicts(task_id)

    def provenance(self, origin_event_id: str) -> dict[str, Any]:
        return self.store.provenance_chain(origin_event_id)

    def replay(self, *, task_id: str | None = None, limit: int = 64,
               min_evidence: int = 1, kinds: list[str] | None = None) -> list[dict[str, Any]]:
        return self.store.replay(task_id=task_id, limit=limit, min_evidence=min_evidence, kinds=kinds)

    # ------------------------------------------------------------ 版本
    def expire(self, record_ids: list[str], at: str | None = None) -> dict[str, Any]:
        return self.store.expire(record_ids, at)

    def snapshot(self, note: str = "") -> dict[str, Any]:
        return self.store.snapshot(note)

    def stats(self) -> dict[str, Any]:
        s = self.store.stats()
        s.update({"started_at": self.started_at, "service_hash": content_hash(s)})
        return s

    def health(self) -> dict[str, Any]:
        s = self.store.stats()
        return {"ok": True, "head_version": s["head_version"], "records_total": s["records_total"],
                "records_primary": s["records_primary"], "echo_events": s["echo_events"]}

    def close(self) -> None:
        self.store.close()
