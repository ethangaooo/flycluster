"""中心记忆中枢的 HTTP 接口。

数据放在独立持久化卷（MEMORY_DB_PATH），因此容器重建后记忆仍在——
容器可写层会随容器删除消失，只有卷有独立生命周期。
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from flycore.schemas import ExperienceRecord, MemoryQuery

from .service import MemoryHubService

app = FastAPI(title="FlyCluster Memory Hub", version="0.1.0")
_service: MemoryHubService | None = None


def get_service() -> MemoryHubService:
    global _service
    if _service is None:
        _service = MemoryHubService(os.environ.get("MEMORY_DB_PATH", "data/memory_hub.sqlite3"))
    return _service


class SubmitBatch(BaseModel):
    records: list[ExperienceRecord]
    caller: str = ""


class SnapshotRequest(BaseModel):
    note: str = ""


class ExpireRequest(BaseModel):
    record_ids: list[str]
    at: str | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return get_service().health()


@app.get("/stats")
def stats() -> dict[str, Any]:
    return get_service().stats()


@app.get("/access-log")
def access_log(limit: int = 50) -> list[dict[str, Any]]:
    return get_service().store.recent_access(limit)


@app.post("/submit")
def submit(rec: ExperienceRecord, caller: str = "") -> dict[str, Any]:
    return get_service().submit(rec).model_dump()


@app.post("/submit-batch")
def submit_batch(body: SubmitBatch) -> list[dict[str, Any]]:
    return [r.model_dump() for r in get_service().submit_many(body.records)]


@app.post("/query")
def query(q: MemoryQuery, caller: str = "") -> dict[str, Any]:
    try:
        return get_service().query(q, caller=caller)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/conflicts")
def conflicts(task_id: str | None = None) -> list[dict[str, Any]]:
    return get_service().conflicts(task_id)


@app.get("/provenance/{origin_event_id}")
def provenance(origin_event_id: str) -> dict[str, Any]:
    return get_service().provenance(origin_event_id)


@app.get("/replay")
def replay(task_id: str | None = None, limit: int = 64, min_evidence: int = 1) -> list[dict[str, Any]]:
    return get_service().replay(task_id=task_id, limit=limit, min_evidence=min_evidence)


@app.post("/snapshot")
def snapshot(body: SnapshotRequest) -> dict[str, Any]:
    return get_service().snapshot(body.note)


@app.post("/expire")
def expire(body: ExpireRequest) -> dict[str, Any]:
    return get_service().expire(body.record_ids, body.at)
