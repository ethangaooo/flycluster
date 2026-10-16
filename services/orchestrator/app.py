"""中枢调度的对外开放 API。

第三方系统的入口。三条必须显式配置的语义（构思文档 §八）：

| 配置 | 行为 |
|---|---|
| `memory_scope` | 本次运行 / 当前会话 / 指定项目 / 租户范围；**禁止默认跨客户共享** |
| `learning_mode` | 只读推理 / 积累经历 / 允许更新学习参数 |
| `memory_version` | 固定记忆快照，或按 `head` 接收更新 |

鉴权：`X-API-Key`。密钥从环境变量 `FLYCLUSTER_API_KEYS`（`name:key[:rpm]` 逗号分隔）读取，
不做明文落盘、不写日志。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from flycore.schemas import RunConfig
from flycore.util import now_iso
from services.env.service import EnvService
from services.env.world import TaskSpec
from services.memory_hub.service import MemoryHubService

from .analytics import collect as collect_analytics
from services.games.base import (DIFFICULTIES, FLY, GOAL_SLOT_DESC, GOAL_SLOT_NAMES,
                                  PC)
from services.games.play import FLY_STRUCTURES, FlyGamePlayer, SessionStore, games_catalog
from .apikeys import ENDPOINTS, ApiKeyStore, KeyRecord
from .runner import ClusterRunner, RunnerOptions

DATA_DIR = Path(os.environ.get("FLYCLUSTER_DATA", "data"))
app = FastAPI(title="FlyCluster Orchestrator", version="0.1.0")
env_service = EnvService()


KEY_STORE = ApiKeyStore(Path(DATA_DIR) / "api_keys.sqlite3")
KEY_STORE.seed_from_env(os.environ.get("FLYCLUSTER_API_KEYS", ""))
_KEY_ENV = os.environ.get("FLYCLUSTER_API_KEYS", "").strip()
if not _KEY_ENV:
    # 未配置时注入一个开发密钥，方便本地联调；生产必须显式配置。
    KEY_STORE.seed_from_env(f"dev:{os.environ.get('FLYCLUSTER_DEV_KEY', 'dev-local-key')}:600")


def _authenticate(x_api_key: str | None) -> KeyRecord:
    rec = KEY_STORE.resolve(x_api_key)
    if rec is None:
        raise HTTPException(status_code=401, detail="缺少或无效的 X-API-Key")
    if rec.revoked_at:
        raise HTTPException(status_code=401, detail=f"密钥 {rec.name} 已被吊销")
    if not rec.allow():
        raise HTTPException(status_code=429,
                            detail=f"密钥 {rec.name} 超过每分钟 {rec.rpm} 次调用上限")
    KEY_STORE.touch(rec)
    return rec


def require_key(x_api_key: str | None = Header(default=None)) -> KeyRecord:
    """任一有效密钥即可（查询类接口）。"""
    return _authenticate(x_api_key)


def require_run(x_api_key: str | None = Header(default=None)) -> KeyRecord:
    """需要 run 作用域（提交/中止运行、写记忆快照）。"""
    rec = _authenticate(x_api_key)
    if not rec.has("run"):
        raise HTTPException(status_code=403, detail=f"密钥 {rec.name} 缺少 run 作用域")
    return rec


def require_admin(x_api_key: str | None = Header(default=None)) -> KeyRecord:
    """需要 admin 作用域（管理密钥）。"""
    rec = _authenticate(x_api_key)
    if not rec.has("admin"):
        raise HTTPException(status_code=403, detail=f"密钥 {rec.name} 缺少 admin 作用域")
    return rec


# --------------------------------------------------------------------- 运行管理


class RunRecord:
    def __init__(self, run: RunConfig, task: TaskSpec):
        self.run = run
        self.task = task
        self.events: deque[dict[str, Any]] = deque(maxlen=8000)
        self.report: dict[str, Any] | None = None
        self.error: str | None = None
        self.status = "queued"
        self.started_at = now_iso()
        self.finished_at: str | None = None
        self.abort = threading.Event()
        self.lock = threading.Lock()

    def on_event(self, ev: dict[str, Any]) -> None:
        with self.lock:
            self.events.append(ev)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            last = self.events[-1] if self.events else {}
            return {"run_id": self.run.run_id, "status": self.status,
                    "task_id": self.task.resolved_id(),
                    "backend": self.run.backend, "seed": self.run.seed, "agents": self.run.agents,
                    "tick": last.get("tick", 0), "started_at": self.started_at,
                    "finished_at": self.finished_at, "error": self.error,
                    "score": (self.report or {}).get("score"),
                    "gates": (self.report or {}).get("gates"),
                    # 控制台需要按 tags.group 汇总对照结果，这里把配置一并带出
                    "config": self.run.model_dump(),
                    "progress": {"targets_found": last.get("targets_found"),
                                 "targets_total": last.get("targets_total"),
                                 "coverage": last.get("coverage"),
                                 "messages": last.get("messages")}}


RUNS: dict[str, RunRecord] = {}
RUNS_LOCK = threading.Lock()


def _webhook(url: str, payload: dict[str, Any]) -> None:
    try:
        httpx.post(url, json=payload, timeout=15.0)
    except Exception:
        pass


def execute(rec: RunRecord) -> None:
    rec.status = "running"
    opts = RunnerOptions(
        data_dir=str(DATA_DIR),
        brain_urls=([u.strip() for u in os.environ.get("FLYCLUSTER_BRAIN_URLS", "").split(",") if u.strip()]
                    or None),
        steps_per_tick=int(os.environ.get("FLYCLUSTER_STEPS_PER_TICK", "20")),
        on_event=rec.on_event,
        abort_flag=rec.abort,
    )
    try:
        runner = ClusterRunner(rec.run, rec.task, opts)
        report = runner.run_cluster()
        rec.report = report.model_dump()
        rec.status = "finished"
    except Exception as exc:
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.status = "failed"
        rec.on_event({"type": "run_failed", "error": rec.error})
    finally:
        rec.finished_at = now_iso()
        if rec.run.callback_url and rec.report is not None:
            _webhook(rec.run.callback_url, {"run_id": rec.run.run_id, "status": rec.status,
                                            "report": rec.report})


# --------------------------------------------------------------------- 请求模型


class CreateRunRequest(BaseModel):
    run: RunConfig
    task: TaskSpec


@app.get("/v1/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "orchestrator", "at": now_iso(),
            "runs_total": len(RUNS), "brain_urls": bool(os.environ.get("FLYCLUSTER_BRAIN_URLS")),
            "auth_required": True}


@app.get("/v1/cluster")
def cluster(key: KeyRecord = Depends(require_key)) -> dict[str, Any]:
    urls = [u.strip() for u in os.environ.get("FLYCLUSTER_BRAIN_URLS", "").split(",") if u.strip()]
    nodes = []
    for i, u in enumerate(urls):
        try:
            r = httpx.get(f"{u.rstrip('/')}/health", timeout=5.0)
            nodes.append({"node_id": f"node-{i}", "url": u, "reachable": True,
                          "health": r.json() if r.status_code == 200 else {"status": r.status_code}})
        except Exception as exc:
            nodes.append({"node_id": f"node-{i}", "url": u, "reachable": False,
                          "error": f"{type(exc).__name__}: {exc}"})
    return {"caller": key.name, "mode": "docker" if urls else "in-process",
            "nodes": nodes, "memory": _memory().stats()}


@app.post("/v1/tasks")
def register_task(spec: TaskSpec, key: KeyRecord = Depends(require_run)) -> dict[str, Any]:
    try:
        return env_service.register_task(spec)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/tasks")
def list_tasks(key: KeyRecord = Depends(require_key)) -> list[dict[str, Any]]:
    return env_service.list_tasks()


@app.post("/v1/runs")
def create_run(body: CreateRunRequest, key: KeyRecord = Depends(require_run)) -> dict[str, Any]:
    run = body.run
    if not run.run_id:
        run.run_id = f"run-{run.seed}-{int(time.time() * 1000) % 100000}"
    if run.memory_scope != "run" and not run.tags.get("tenant"):
        # 范围高于单次运行时必须显式声明租户，避免默认跨客户共享。
        raise HTTPException(status_code=400,
                            detail="memory_scope 高于 run 时必须提供 tags.tenant，禁止默认跨客户共享")
    with RUNS_LOCK:
        if run.run_id in RUNS:
            raise HTTPException(status_code=409, detail=f"run_id {run.run_id} 已存在")
        if body.task.task_id:
            env_service.register_task(body.task)
        rec = RunRecord(run, body.task)
        RUNS[run.run_id] = rec
    threading.Thread(target=execute, args=(rec,), daemon=True).start()
    return {"run_id": run.run_id, "status": rec.status, "caller": key.name,
            "memory_scope": run.memory_scope, "learning_mode": run.learning_mode,
            "memory_version": run.memory_version}


@app.get("/v1/runs")
def list_runs(key: KeyRecord = Depends(require_key)) -> list[dict[str, Any]]:
    return [r.snapshot() for r in RUNS.values()]


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str, key: KeyRecord = Depends(require_key)) -> dict[str, Any]:
    rec = RUNS.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="未知 run_id")
    return rec.snapshot()


@app.get("/v1/runs/{run_id}/events")
def stream_events(run_id: str, request: Request, key: KeyRecord = Depends(require_key)):
    rec = RUNS.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="未知 run_id")

    def gen():
        sent = 0
        while True:
            with rec.lock:
                batch = list(rec.events)[sent:]
                sent += len(batch)
            for ev in batch:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if rec.status in ("finished", "failed") and sent >= len(rec.events):
                yield f"data: {json.dumps({'type': 'stream_end', 'status': rec.status}, ensure_ascii=False)}\n\n"
                break
            if request.client is None:
                break
            time.sleep(0.2)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/runs/{run_id}/abort")
def abort_run(run_id: str, key: KeyRecord = Depends(require_run)) -> dict[str, Any]:
    rec = RUNS.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="未知 run_id")
    rec.abort.set()
    return {"run_id": run_id, "abort_requested": True}


@app.get("/v1/runs/{run_id}/report")
def get_report(run_id: str, key: KeyRecord = Depends(require_key)) -> dict[str, Any]:
    rec = RUNS.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="未知 run_id")
    if rec.report is None:
        raise HTTPException(status_code=409, detail=f"运行尚未完成，当前状态 {rec.status}")
    return rec.report


# --------------------------------------------------------------------- 记忆


def _hub_get(path: str, **params: Any) -> Any:
    """Docker 模式下记忆中枢是独立容器，这里透明转发；单机模式直接读本地库。"""
    url = os.environ.get("FLYCLUSTER_MEMORY_HUB_URL")
    if url:
        r = httpx.get(f"{url.rstrip('/')}{path}", params=params or None, timeout=15.0)
        r.raise_for_status()
        return r.json()
    return None


def _memory() -> MemoryHubService:
    return MemoryHubService(str(DATA_DIR / "memory_hub.sqlite3"))


@app.get("/v1/memory/stats")
def memory_stats(key: KeyRecord = Depends(require_key)) -> dict[str, Any]:
    remote = _hub_get("/stats")
    return remote if remote is not None else _memory().stats()


@app.get("/v1/memory/conflicts")
def memory_conflicts(task_id: str | None = None, key: KeyRecord = Depends(require_key)) -> Any:
    remote = _hub_get("/conflicts", task_id=task_id) if task_id else _hub_get("/conflicts")
    return remote if remote is not None else _memory().conflicts(task_id)


@app.get("/v1/memory/provenance/{origin_event_id}")
def memory_provenance(origin_event_id: str, key: KeyRecord = Depends(require_key)) -> Any:
    remote = _hub_get(f"/provenance/{origin_event_id}")
    return remote if remote is not None else _memory().provenance(origin_event_id)


@app.post("/v1/memory/snapshot")
def memory_snapshot(note: str = "", key: KeyRecord = Depends(require_run)) -> dict[str, Any]:
    url = os.environ.get("FLYCLUSTER_MEMORY_HUB_URL")
    if url:
        r = httpx.post(f"{url.rstrip('/')}/snapshot", json={"note": note}, timeout=15.0)
        r.raise_for_status()
        return r.json()
    return _memory().snapshot(note)


# --------------------------------------------------------------------- 棋局演示

# 说明见 services/games/play.py 的模块注释：
# 棋盘编码/候选剪枝/倾向分投影/胜负判定/电脑搜索**全部是外围代码**，
# 脑只提供一个脉冲动力系统。接口返回里带 attribution 字段，页面据此标注。

GAME_SESSIONS = SessionStore(limit=24)


class NewGameRequest(BaseModel):
    game: str = "gomoku"
    difficulty: str = "normal"
    structure: str = "brain"
    seed: int = 0
    connectome: str = "male-cns-gateway"
    learning: bool = False
    use_goal_info: bool = True   # 让它知道"什么是赢"（观测里的规则与目标槽位）
    note: str = ""


class PlyRequest(BaseModel):
    limit: int = 32
    fly_first: bool = True
    steps_per_tick: int | None = None   # 留空用默认（见 FlyGamePlayer.GAME_STEPS_DEFAULT）


@app.get("/v1/games/catalog")
def games_catalog_api(key: KeyRecord = Depends(require_key)) -> dict[str, Any]:
    return {
        "games": games_catalog(),
        "structures": [{"key": k, **v} for k, v in FLY_STRUCTURES.items()],
        "difficulties": [d.__dict__ for d in DIFFICULTIES],
        "rules": {
            "xiangqi": [
                "重复局面第 3 次出现：长将判负 / 长捉判负 / 无威胁则判和",
                "自然限着：连续 120 着无吃子判和",
                "困毙判负（不同于国际象棋的逼和）",
                "已知放宽：长捉不追踪棋子个体，只判'每一着都制造新捉'",
            ],
            "gomoku": ["成五即胜；无重复局面规则（棋盘只增不减）"],
            "checkers": ["填满对面三角即胜"],
        },
        "goal_slots": {
            "names": list(GOAL_SLOT_NAMES), "desc": GOAL_SLOT_DESC,
            "note": "观测最后 4 维。**没有它们，果蝇感觉不到'这步能赢'**——"
                    "将军不等于将死，于是它只在玩'挪棋子、争子力'。"
                    "消融实测：有这 4 维时选中一步取胜着法 33%（12 次），"
                    "清零后 0%，随机水平 4.2%。",
        },
        "attribution": {
            "brain": "脉冲动力系统（连接组派生结构 / 对照结构）",
            "periphery": [
                "棋盘 → 105 维观测的编码",
                "候选着法剪枝（每个棋种各自的启发式）",
                "读出向量 → 标量倾向分的固定随机投影",
                "胜负判定与棋规",
                "电脑对手的 alpha-beta 搜索",
            ],
            "note": "演示性质：这不是在主张果蝇具备棋类认知。",
        },
    }


@app.post("/v1/games/sessions")
def games_new(body: NewGameRequest, key: KeyRecord = Depends(require_run)) -> dict[str, Any]:
    from services.games.play import GameSession, make_game
    from flycore.util import now_iso
    import uuid

    sid = "g-" + uuid.uuid4().hex[:10]
    game = make_game(body.game, seed=body.seed)
    player = FlyGamePlayer(structure=body.structure, connectome=body.connectome,
                           seed=body.seed, learning=body.learning,
                           use_goal_info=body.use_goal_info)
    sess = GameSession(session_id=sid, game_key=body.game, difficulty=body.difficulty,
                       structure=body.structure, seed=body.seed, game=game,
                       player=player, created_at=now_iso())
    GAME_SESSIONS.put(sess)
    return {"session_id": sid, "state": sess.state()}


@app.get("/v1/games/sessions/{sid}")
def games_state(sid: str, key: KeyRecord = Depends(require_key)) -> dict[str, Any]:
    try:
        return GAME_SESSIONS.get(sid).state()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"对局 {sid} 不存在") from None


@app.post("/v1/games/sessions/{sid}/ply")
def games_ply(sid: str, body: PlyRequest,
              key: KeyRecord = Depends(require_run)) -> dict[str, Any]:
    """走一个回合：飞蝇一步 + 电脑一步（若对局未结束）。"""
    try:
        sess = GAME_SESSIONS.get(sid)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"对局 {sid} 不存在") from None
    g = sess.game
    out: dict[str, Any] = {"session_id": sid, "fly": None, "pc": None}

    if g.outcome().over:
        out["state"] = sess.state()
        return out

    # ---- 飞蝇一步
    if g.turn() == FLY:
        h0 = g.heuristic()
        d = sess.player.decide(g, limit=body.limit)
        move = [list(x) if isinstance(x, (list, tuple)) else x for x in d.move] \
            if isinstance(d.move, (list, tuple)) else d.move
        g.apply(d.move)
        h1 = g.heuristic()
        # 奖励 = 局面改善（子力/位置） + **规则与目标塑形** + 终局胜负。
        #
        # 为什么要单独加"目标塑形"：光给子力改善，果蝇只知道"吃子划算"，
        # 不知道什么叫赢。用户的原话很准——光知道棋子怎么走不够，还得知道什么叫赢。
        #   我方眼下有一步取胜   → +
        #   对方眼下有一步取胜   → −（危险）
        #   终局胜负             → ±1（最大权重）
        #
        # 而且必须**立即**施加：终局奖励若等"下一步"再消费，赢了之后没有下一步，
        # 这个信号会永远用不上（这正是修复前的状态）。
        o = g.outcome()
        if o.over:
            r = 1.0 if o.winner == FLY else (-1.0 if o.winner == PC else 0.0)
            r_kind = "终局"
        else:
            r = 0.35 * (h1 - h0)
            facts = g.goal_facts() if hasattr(g, "goal_facts") else {}
            if facts.get("i_can_win_now"):
                r += 0.20
            if facts.get("opp_can_win_now"):
                r -= 0.40
            # 重复局面的判罚：长将/长捉会被判负，重复而无威胁会判和。
            # 不把这条写进奖励，果蝇就学不会"别在重复圈里把自己走死"。
            rf = int(facts.get("repeat_fault", 0) or 0)
            if rf > 0:
                r -= 0.50          # 我正在犯长将/长捉
            elif rf < 0:
                r += 0.15          # 对方在犯规，我占便宜
            r_kind = "过程"
        sess.player.give_reward(r)
        sess.last_reward = {"value": round(float(r), 4), "kind": r_kind,
                            "detail": (g.goal_facts() if hasattr(g, "goal_facts") else {})}
        sess.last_decision = d.as_dict()
        sess.moves.append({"side": FLY, "label": d.move_label, "structure": d.structure,
                           "elapsed_ms": d.as_dict()["elapsed_ms"]})
        out["fly"] = d.as_dict()
    sess.heuristic_curve.append(round(g.heuristic(), 4))

    # ---- 电脑一步
    if not g.outcome().over and g.turn() == PC:
        import time as _t
        t0 = _t.perf_counter()
        m = g.ai_move(sess.difficulty)
        dt = (_t.perf_counter() - t0) * 1000
        label = g.move_label(m)
        g.apply(m)
        sess.last_pc_move = {"move": m, "label": label, "elapsed_ms": round(dt, 2)}
        sess.moves.append({"side": PC, "label": label,
                           "structure": f"pc-{sess.difficulty}", "elapsed_ms": round(dt, 2)})
        out["pc"] = sess.last_pc_move
        sess.heuristic_curve.append(round(g.heuristic(), 4))

    out["reward"] = sess.last_reward
    out["state"] = sess.state()
    return out


# --------------------------------------------------------------------- 结果分析


@app.get("/v1/analytics/runs")
def analytics_runs(artifact_dir: str | None = None,
                   key: KeyRecord = Depends(require_key)) -> dict[str, Any]:
    """把命令行产物与控制台运行归一化成一张扁平表，供页面按任意维度切片。

    `artifact_dir` 默认取 `FLYCLUSTER_ARTIFACTS`（compose 里挂的是 /app/artifacts）。
    读取失败的产物会出现在 `errors` 里而不是让整个接口挂掉——
    一个损坏的 JSON 不应该让所有历史结果都看不到。
    """
    adir = artifact_dir or os.environ.get("FLYCLUSTER_ARTIFACTS", "artifacts")
    live = [rec.snapshot() for rec in RUNS.values()]
    return collect_analytics(adir, live)


# --------------------------------------------------------------------- 接入管理

# 密钥管理的三个原则写在 services/orchestrator/apikeys.py 的模块注释里。
# 这里只负责把它暴露出去，不做额外逻辑。


class CreateKeyRequest(BaseModel):
    name: str
    rpm: int = Field(default=120, ge=1, le=100000)
    scopes: list[str] = Field(default_factory=lambda: ["run", "read"])
    note: str = ""


@app.get("/v1/connectomes")
def list_connectomes(key: KeyRecord = Depends(require_key)) -> dict[str, Any]:
    """可用的连接组清单，供页面下拉选择。

    - `synthetic-*`：程序化合成，**不是真实数据**，仅用于打通链路
    - 磁盘上的 `.npz`：由 tools/fetch_connectome.py 接入的真实子图，带许可与来源

    每个条目都带上「是否真实数据」与规模，避免在界面上把合成图误当真实连接组。
    """
    from flycore.connectome import get_graph

    items: list[dict[str, Any]] = []
    for spec, n in (("synthetic-500", 500), ("synthetic-1000", 1000),
                    ("synthetic-2000", 2000), ("synthetic-5000", 5000)):
        items.append({"spec": spec, "real": False, "neuron_count": n,
                      "edge_count": None, "license": "CC0-1.0", "source": "synthetic",
                      "notes": "程序化合成的对照图，不是任何真实果蝇数据"})

    cdir = Path(os.environ.get("FLYCLUSTER_CONNECTOME_DIR", DATA_DIR))
    for path in sorted(cdir.glob("*.npz")) if cdir.exists() else []:
        spec = path.stem
        entry: dict[str, Any] = {"spec": spec, "real": True, "path": str(path)}
        try:
            g = get_graph(spec, data_dir=cdir)
            m = g.meta
            entry.update({
                "neuron_count": int(g.n), "edge_count": int(g.edge_count()),
                "mean_indegree": round(float(m.mean_indegree), 2),
                "largest_scc_fraction": round(float(m.largest_scc_fraction), 4),
                "modularity": round(float(m.modularity), 4),
                "synapse_count": int(m.synapse_count or 0),
                "inhibitory_fraction": round(float(m.inhibitory_fraction), 4),
                "input_count": int(m.input_count), "output_count": int(m.output_count),
                "license": m.license, "license_url": m.license_url,
                "source": m.source, "notes": m.notes, "created_at": m.created_at,
            })
            entry["rewired_available"] = True
        except Exception as exc:      # 文件损坏不应让整个清单接口挂掉
            entry["error"] = f"{type(exc).__name__}: {exc}"
        items.append(entry)

    # 顺带回报磁盘占用，方便判断"能不能再加一个更大的图"
    total = sum(p.stat().st_size for p in cdir.glob("*.npz")) if cdir.exists() else 0
    return {"connectomes": items, "dir": str(cdir),
            "disk_bytes": int(total), "real_count": sum(1 for i in items if i.get("real"))}


@app.get("/v1/admin/endpoints")
def admin_endpoints(key: KeyRecord = Depends(require_admin)) -> dict[str, Any]:
    """接口目录。放在后端而不是前端，保证"代码里的端点"与"页面上列的端点"只有一个来源。"""
    groups: dict[str, list[dict[str, Any]]] = {}
    for e in ENDPOINTS:
        groups.setdefault(e["group"], []).append(e)
    return {"base_url": "http://127.0.0.1:8800", "groups": groups,
            "auth_header": "X-API-Key", "scopes": ["run", "read", "admin"]}


@app.get("/v1/admin/keys")
def admin_list_keys(key: KeyRecord = Depends(require_admin)) -> dict[str, Any]:
    return {"keys": KEY_STORE.list(), "stats": KEY_STORE.stats(),
            "caller": key.name, "caller_scopes": sorted(key.scopes)}


@app.post("/v1/admin/keys")
def admin_create_key(body: CreateKeyRequest,
                     key: KeyRecord = Depends(require_admin)) -> dict[str, Any]:
    try:
        raw, rec = KEY_STORE.create(body.name, body.rpm, set(body.scopes), body.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # 明文只在这一刻出现一次，之后数据库里只有哈希。
    return {"key": raw, "record": rec,
            "warning": "这是唯一一次显示明文密钥，请立即保存；服务端只保留哈希，无法再取回。"}


@app.post("/v1/admin/keys/{key_id}/rotate")
def admin_rotate_key(key_id: str,
                     key: KeyRecord = Depends(require_admin)) -> dict[str, Any]:
    try:
        raw, rec = KEY_STORE.rotate(key_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"密钥 {key_id} 不存在") from None
    return {"key": raw, "record": rec,
            "warning": "旧密钥已立即失效。明文只显示这一次，请立即保存。"}


@app.delete("/v1/admin/keys/{key_id}")
def admin_revoke_key(key_id: str, purge: bool = False,
                     key: KeyRecord = Depends(require_admin)) -> dict[str, Any]:
    """默认吊销（保留记录便于审计）。`purge=true` 时才真正删除。

    环境变量注入的密钥**不能删除**：删了之后重启会由环境变量重新注入，
    出现"页面显示已删除、实际仍可用"的矛盾。只能吊销，且吊销状态落库、重启后依然有效。
    """
    if key_id == key.key_id:
        raise HTTPException(status_code=400, detail="不能吊销当前正在使用的密钥，请换一个密钥操作")
    try:
        if purge:
            KEY_STORE.delete(key_id)
            return {"purged": key_id}
        return {"revoked": KEY_STORE.revoke(key_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"密钥 {key_id} 不存在") from None
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --------------------------------------------------------------------- 静态控制台

_DIST = Path(os.environ.get("FLYCLUSTER_ADMIN_DIST", "admin-ui/dist"))
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="admin")
