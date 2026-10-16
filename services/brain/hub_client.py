"""脑节点访问中心记忆中枢的传输层。

两种实现共享同一接口，使同一套实验既能跑 Docker 多容器，也能在单进程内跑：
  HttpMemoryBackend      — 容器间 HTTP（Docker 模式）
  InProcessMemoryBackend — 直接调用 MemoryHubService（单机/冒烟测试）
传输方式不影响模型与评分，只影响部署形态。
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from flycore.schemas import ExperienceRecord, MemoryQuery, MemorySubmitResult
from services.memory_hub.service import MemoryHubService


class MemoryBackend(Protocol):
    def submit(self, rec: ExperienceRecord) -> MemorySubmitResult: ...
    def query(self, q: MemoryQuery, caller: str = "") -> dict[str, Any]: ...


class HttpMemoryBackend:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def submit(self, rec: ExperienceRecord) -> MemorySubmitResult:
        r = self.client.post(f"{self.base_url}/submit", json=rec.model_dump())
        r.raise_for_status()
        return MemorySubmitResult(**r.json())

    def query(self, q: MemoryQuery, caller: str = "") -> dict[str, Any]:
        r = self.client.post(f"{self.base_url}/query?caller={caller}", json=q.model_dump())
        r.raise_for_status()
        return r.json()


class InProcessMemoryBackend:
    def __init__(self, service: MemoryHubService):
        self.service = service

    def submit(self, rec: ExperienceRecord) -> MemorySubmitResult:
        return self.service.submit(rec)

    def query(self, q: MemoryQuery, caller: str = "") -> dict[str, Any]:
        return self.service.query(q, caller=caller)


def build_memory_backend(url: str | None, local_service: MemoryHubService | None = None) -> MemoryBackend:
    if url:
        return HttpMemoryBackend(url)
    if local_service is None:
        raise ValueError("既没有给出记忆中枢 URL，也没有提供本地服务实例")
    return InProcessMemoryBackend(local_service)
