"""脑节点/对照模型的传输层。

同一套实验既能跑 Docker 多容器（HTTP），也能在单进程内跑（InProcess）：
传输方式不影响模型与评分，只影响部署形态。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import httpx

from flycore.connectome import get_graph
from flycore.schemas import BrainNodeConfig, BrainStepRequest, BrainStepResponse
from services.baselines.models import build_control
from services.brain.engine import BrainEngine
from services.brain.hub_client import MemoryBackend
from services.memory_hub.service import MemoryHubService


class NodeBackend(Protocol):
    node_id: str

    def step(self, req: BrainStepRequest) -> BrainStepResponse: ...
    def health(self) -> dict[str, Any]: ...
    def set_message_kind(self, kind: str) -> None: ...
    def configure(self, settings: dict[str, Any]) -> dict[str, Any]: ...
    def reset(self, run_id: str) -> dict[str, Any]: ...
    def counters(self) -> dict[str, Any]: ...


class HttpNodeBackend:
    """通过 HTTP 访问远端脑节点容器。"""

    def __init__(self, node_id: str, base_url: str, timeout: float = 60.0):
        self.node_id = node_id
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def step(self, req: BrainStepRequest) -> BrainStepResponse:
        r = self.client.post(f"{self.base_url}/step", json=req.model_dump())
        r.raise_for_status()
        return BrainStepResponse(**r.json())

    def health(self) -> dict[str, Any]:
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def set_message_kind(self, kind: str) -> None:
        self.client.post(f"{self.base_url}/message-kind", json={"kind": kind}).raise_for_status()

    def configure(self, settings: dict[str, Any]) -> dict[str, Any]:
        """把运行级设置推给远端容器。

        远端容器启动时只知道环境变量，不知道"这一次运行"要不要用记忆、要不要学习。
        不推的话，消融开关在 Docker 模式下会被静默忽略——不报错，但结果已经错了。
        """
        body = {k: v for k, v in settings.items() if v is not None}
        if not body:
            return {}
        r = self.client.post(f"{self.base_url}/configure", json=body)
        r.raise_for_status()
        return r.json().get("applied", {})

    def reset(self, run_id: str) -> dict[str, Any]:
        """清零容器的累计统计，并把本地记忆切到按 run 隔离的目录。

        容器是常驻的，不清零的话报告里的"本次运行步数/脉冲/实时比"会是**累计值**。
        实测：同容器连跑三次分别报 19200 / 38400 / 57600 步，而每次真实都是 19200。
        这类错误不报错，但让"性能数字必须与可信度遥测同时出现"直接失效。
        """
        r = self.client.post(f"{self.base_url}/reset", json={"run_id": run_id})
        r.raise_for_status()
        return r.json()

    def counters(self) -> dict[str, Any]:
        return {}


class InProcessNodeBackend:
    """把 BrainEngine 或对照模型直接包成后端，省掉网络往返。"""

    def __init__(self, node_id: str, impl: Any):
        self.node_id = node_id
        self.impl = impl

    def step(self, req: BrainStepRequest) -> BrainStepResponse:
        return self.impl.step(req)

    def health(self) -> dict[str, Any]:
        return self.impl.health()

    def set_message_kind(self, kind: str) -> None:
        self.impl.set_message_kind(kind)

    def configure(self, settings: dict[str, Any]) -> dict[str, Any]:
        """进程内模式：节点在构造时就按本次运行配好了，这里只回报"实际生效"的值。

        回报而不是空操作，是为了让报告里能看出开关确实落到了节点上——
        如果哪天有人把构造时的传参接错了，这个回报会和请求对不上。
        """
        applied: dict[str, Any] = {}
        for f in ("k_local", "k_central", "local_memory_enabled", "central_memory_enabled"):
            if settings.get(f) is not None and hasattr(self.impl, f):
                applied[f] = getattr(self.impl, f)
        return applied

    def reset(self, run_id: str) -> dict[str, Any]:
        """进程内模式：本地记忆构造时已按 run 隔离，这里只清统计。"""
        if hasattr(self.impl, "reset_run_stats"):
            self.impl.reset_run_stats()
        return {"run_id": run_id, "local_memory_records":
                (self.impl.local.stats()["size"] if hasattr(self.impl, "local") else 0)}

    def counters(self) -> dict[str, Any]:
        return self.impl.counters()


def build_inprocess_nodes(run_backend: str, agents: int, connectome: str, seed: int,
                          hub: MemoryBackend | None, message_kind: str,
                          steps_per_tick: int, data_dir: str | Path,
                          learning_mode: str = "accumulate", memory_scope: str = "run",
                          dynamics_overrides: dict[str, Any] | None = None,
                          k_local: int = 4, k_central: int = 6,
                          exploration_epsilon: float = 0.2,
                          central_memory_enabled: bool = True,
                          connectome_dir: str | Path = "data",
                          run_id: str = "") -> list[NodeBackend]:
    """原地构造 N 个节点。backend=brain/rewired 走脉冲网络，其余走对照模型。"""
    nodes: list[NodeBackend] = []
    spec = connectome
    if run_backend == "rewired" and not spec.endswith("-rewired"):
        spec = f"{spec}-rewired"
    for i in range(agents):
        node_id = f"node-{i}"
        cfg = BrainNodeConfig(
            node_id=node_id, connectome=spec, seed=seed + i, learning_mode=learning_mode,  # type: ignore[arg-type]
            memory_scope=memory_scope,  # type: ignore[arg-type]
            exploration_epsilon=exploration_epsilon,
        )
        if dynamics_overrides:
            cfg.dynamics = cfg.dynamics.model_copy(update=dynamics_overrides)
        # 连接组数据目录与运行产物目录是两件事：前者是只读的 data/，后者是每次运行的输出。
        graph_dir = Path(connectome_dir)
        # 本地记忆必须按 run 隔离：否则同一组的种子 2 会带着种子 1 的本地经验起跑，
        # 整批对照就不再是独立样本。中心记忆由于范围是 scope=run 而不受影响。
        local_root = Path(data_dir) / "local" / (run_id or "default")
        if run_backend in ("brain", "rewired"):
            graph = get_graph(spec, data_dir=graph_dir)
            impl = BrainEngine(graph, cfg, hub=hub, steps_per_tick=steps_per_tick,
                              local_root=local_root, k_local=k_local, k_central=k_central)
        else:
            impl = build_control(run_backend, cfg, hub, steps_per_tick, local_root)
        # 中心记忆通道跟随 run.enable_memory_channel；本地记忆不受它影响（它是每个节点自己的经历，
        # 不占通信预算）。把两者绑在一起会让 k_local 变成死旋钮。
        impl.central_memory_enabled = central_memory_enabled
        impl.set_message_kind(message_kind)
        nodes.append(InProcessNodeBackend(node_id, impl))
    return nodes


def build_http_nodes(backend: str, agents: int, urls: list[str]) -> list[NodeBackend]:
    if len(urls) < agents:
        raise ValueError(f"需要 {agents} 个脑节点 URL，只给了 {len(urls)} 个")
    return [HttpNodeBackend(f"node-{i}", urls[i]) for i in range(agents)]


def local_hub(data_dir: str | Path) -> MemoryHubService:
    return MemoryHubService(str(Path(data_dir) / "memory_hub.sqlite3"))
