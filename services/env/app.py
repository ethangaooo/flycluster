"""环境与评估服务的 HTTP 接口。"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .service import EnvService
from .world import TaskSpec

app = FastAPI(title="FlyCluster Environment", version="0.1.0")
_service = EnvService()


class StepRequest(BaseModel):
    actions: list[int]


class ResetRequest(BaseModel):
    spec: TaskSpec


@app.get("/health")
def health() -> dict[str, Any]:
    return _service.health()


@app.get("/tasks")
def list_tasks() -> list[dict[str, Any]]:
    return _service.list_tasks()


@app.post("/tasks")
def register_task(spec: TaskSpec) -> dict[str, Any]:
    try:
        return _service.register_task(spec)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/reset")
def reset(body: ResetRequest) -> dict[str, Any]:
    return _service.reset(body.spec)


@app.get("/state")
def state() -> dict[str, Any]:
    try:
        return _service.snapshot()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/step")
def step(body: StepRequest) -> dict[str, Any]:
    try:
        return _service.step(body.actions)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/score")
def score() -> dict[str, Any]:
    try:
        return _service.score()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
