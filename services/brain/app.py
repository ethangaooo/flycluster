"""脑节点 HTTP 服务。

Docker 模式下每个容器跑一个实例；单机模式下由编排器在同一进程内直接构造
BrainEngine，走 InProcess 传输，两套路径共用同一份引擎代码。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from flycore.connectome import get_graph
from flycore.schemas import BrainNodeConfig, BrainStepRequest, BrainStepResponse, DynamicsConfig

from .engine import BrainEngine
from .hub_client import build_memory_backend

app = FastAPI(title="FlyCluster Brain Node", version="0.1.0")
_engine: BrainEngine | None = None


def build_engine() -> BrainEngine:
    """从环境变量构造脑节点。所有可调项都显式来自环境，便于实验脚本批量改参。"""
    cfg = BrainNodeConfig(
        node_id=os.environ.get("BRAIN_NODE_ID", "brain-1"),
        connectome=os.environ.get("BRAIN_CONNECTOME", "synthetic-2000"),
        seed=int(os.environ.get("BRAIN_SEED", "0")),
        learning_mode=os.environ.get("BRAIN_LEARNING_MODE", "accumulate"),      # type: ignore[arg-type]
        memory_scope=os.environ.get("BRAIN_MEMORY_SCOPE", "run"),              # type: ignore[arg-type]
        memory_version=os.environ.get("BRAIN_MEMORY_VERSION", "head"),         # type: ignore[arg-type]
        enable_peer_input=os.environ.get("BRAIN_PEER_INPUT", "1") == "1",
        enable_memory_input=os.environ.get("BRAIN_MEMORY_INPUT", "1") == "1",
        dynamics=DynamicsConfig(
            synaptic_scale=float(os.environ.get("BRAIN_SYNAPTIC_SCALE", "0.35")),
            background_current=float(os.environ.get("BRAIN_BACKGROUND", "14")),
            noise_sigma=float(os.environ.get("BRAIN_NOISE", "6")),
            max_spikes_per_step=int(os.environ.get("BRAIN_MAX_SPIKES", "0")),
        ),
    )
    # 连接组目录（只读，可能以只读卷挂入）与运行数据目录（必须可写，放本地记忆）分开。
    # 混用会让"把真实连接组挂进容器"变成不可能：/app/data 是卷，不能只读覆盖。
    data_dir = os.environ.get("DATA_DIR", "data")
    graph_dir = os.environ.get("BRAIN_CONNECTOME_DIR", data_dir)
    graph = get_graph(cfg.connectome, data_dir=graph_dir)
    hub_url = os.environ.get("MEMORY_HUB_URL") or None
    hub = build_memory_backend(hub_url) if hub_url else None
    engine = BrainEngine(graph, cfg, hub=hub,
                         steps_per_tick=int(os.environ.get("BRAIN_STEPS_PER_TICK", "20")),
                         local_root=Path(data_dir) / "local")
    engine.set_message_kind(os.environ.get("BRAIN_MESSAGE_KIND", "learned_compressed"))
    return engine


def get_engine() -> BrainEngine:
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


class MessageKindRequest(BaseModel):
    kind: str


class RestoreRequest(BaseModel):
    checkpoint: dict[str, Any]


@app.get("/health")
def health() -> dict[str, Any]:
    return get_engine().health()


@app.post("/step", response_model=BrainStepResponse)
def step(req: BrainStepRequest) -> BrainStepResponse:
    try:
        return get_engine().step(req)
    except Exception as exc:      # 出错必须显式暴露，不静默降级
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/checkpoint")
def checkpoint() -> dict[str, Any]:
    ck = get_engine().checkpoint()
    return {k: (v if k not in ("net_state", "readout") else "<numpy arrays omitted in http mode>")
            for k, v in ck.items()}


@app.post("/restore")
def restore(body: RestoreRequest) -> dict[str, Any]:
    get_engine().restore(body.checkpoint)
    return {"ok": True}


@app.post("/message-kind")
def message_kind(body: MessageKindRequest) -> dict[str, Any]:
    get_engine().set_message_kind(body.kind)
    return {"ok": True, "kind": body.kind}


class ResetRequest(BaseModel):
    run_id: str | None = None


class RunSettingsRequest(BaseModel):
    """一次运行级的设置，由中枢调度在运行开始时下发。

    **为什么必须有这个端点**：容器启动时从环境变量读配置，而"这条运行要不要用中心记忆"
    是**每次运行**的属性。缺了它，`enable_memory_channel=False` 在 Docker 模式下会被静默忽略——
    实测过一次 16 节点运行：请求里写着关闭中心记忆，逐节点却仍报 `central_hits=234`、
    合计检索 6.1 MB。这类"配置项存在但没接通"的失效最难发现，因为它不报错。
    """
    learning_mode: str | None = None
    memory_scope: str | None = None
    k_local: int | None = None
    k_central: int | None = None
    local_memory_enabled: bool | None = None
    central_memory_enabled: bool | None = None
    enable_peer_input: bool | None = None
    enable_memory_input: bool | None = None
    message_kind: str | None = None
    run_id: str | None = None


@app.post("/configure")
def configure(body: RunSettingsRequest) -> dict[str, Any]:
    eng = get_engine()
    applied: dict[str, Any] = {}
    for f in ("learning_mode", "memory_scope"):
        v = getattr(body, f)
        if v is not None:
            setattr(eng.cfg, f, v)
            applied[f] = v
    for f in ("k_local", "k_central"):
        v = getattr(body, f)
        if v is not None:
            setattr(eng, f, int(v))
            applied[f] = int(v)
    for f, target in (("local_memory_enabled", "local_memory_enabled"),
                      ("central_memory_enabled", "central_memory_enabled"),
                      ("enable_peer_input", None), ("enable_memory_input", None)):
        v = getattr(body, f)
        if v is None:
            continue
        if target:
            setattr(eng, target, bool(v))
        else:
            # 输入通道开关同时管住脑内适配器与检索，两处必须一起改
            if f == "enable_peer_input":
                eng.io.peer.enabled = bool(v)
                eng.cfg.enable_peer_input = bool(v)
            else:
                eng.io.memory.enabled = bool(v)
                eng.cfg.enable_memory_input = bool(v)
        applied[f] = bool(v)
    if body.message_kind:
        eng.set_message_kind(body.message_kind)
        applied["message_kind"] = body.message_kind
    if body.run_id:
        applied["local_memory_records"] = _bind_run(eng, body.run_id)
    return {"ok": True, "applied": applied, "node_id": eng.node_id}


def _bind_run(eng: BrainEngine, run_id: str) -> int:
    """把引擎绑定到一次运行：清统计、本地记忆切到按 run 隔离的目录。

    顺序很重要——先清统计再切目录，否则回填进来的旧记录会污染本次运行的计数。
    """
    base = Path(os.environ.get("BRAIN_LOCAL_ROOT") or (Path(os.environ.get("DATA_DIR", "data")) / "local"))
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in run_id)[:120]
    eng.reset_run_stats()
    return eng.local.retarget(base / safe)


@app.post("/reset")
def reset(body: ResetRequest) -> dict[str, Any]:
    eng = get_engine()
    n = _bind_run(eng, body.run_id or "default")
    return {"ok": True, "run_id": body.run_id, "local_memory_records": n,
            "telemetry": eng.net.telemetry.as_dict(eng.graph.n)}
