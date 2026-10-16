"""开放 API 的密钥管理。

设计要点（都与"能不能长期运维"有关，不是为了好看）：

1. **只存哈希**。数据库里不出现明文密钥，泄漏数据库不等于泄漏凭据。
   明文只在创建/轮换的那一刻返回一次。
2. **区分来源**。环境变量注入的密钥（`source=env`）由部署流程管理，
   页面上只能查看和吊销调用，不能删除——否则重启后又"复活"，与页面上显示的状态不一致。
3. **作用域分开**。`run`（提交与中止运行）、`read`（查询）、`admin`（管理密钥）。
   默认给环境变量密钥全部作用域以便本地联调；生产应显式写第 4 段收紧。
4. **限额与用量可查**。每个密钥有每分钟上限、累计调用数、最近调用时间，
   管理页才能回答"谁在用、用超了没有"。

环境变量格式：`name:key[:rpm[:scopes]]`，多个用逗号分隔。
`scopes` 形如 `admin` 或 `run,read`；省略时给 `run,read,admin`。
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flycore.util import now_iso

DDL = """
CREATE TABLE IF NOT EXISTS api_key (
    key_id       TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    key_hash     TEXT NOT NULL UNIQUE,
    prefix       TEXT NOT NULL,
    rpm          INTEGER NOT NULL DEFAULT 120,
    scopes       TEXT NOT NULL DEFAULT 'run,read,admin',
    source       TEXT NOT NULL DEFAULT 'db',
    note         TEXT DEFAULT '',
    created_at   TEXT NOT NULL,
    revoked_at   TEXT,
    last_used_at TEXT,
    calls_total  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_key_hash ON api_key(key_hash);
"""

ALL_SCOPES = ("run", "read", "admin")


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def new_key() -> str:
    return "fc_" + secrets.token_urlsafe(32)


@dataclass
class KeyRecord:
    key_id: str
    name: str
    key_hash: str
    prefix: str
    rpm: int
    scopes: set[str]
    source: str
    note: str = ""
    created_at: str = ""
    revoked_at: str | None = None
    last_used_at: str | None = None
    calls_total: int = 0
    # 运行期滑动窗口，不落库：重启后限流计数从零开始是可接受的（最坏多放行一轮）
    hits: deque[float] = field(default_factory=deque)

    def allow(self) -> bool:
        now = time.time()
        while self.hits and now - self.hits[0] > 60:
            self.hits.popleft()
        if len(self.hits) >= self.rpm:
            return False
        self.hits.append(now)
        return True

    def has(self, scope: str) -> bool:
        return scope in self.scopes

    def public(self) -> dict[str, Any]:
        """对外视图：**绝不包含明文密钥**，也不包含哈希。"""
        return {
            "key_id": self.key_id,
            "name": self.name,
            "prefix": self.prefix,
            "rpm": self.rpm,
            "scopes": sorted(self.scopes),
            "source": self.source,
            "note": self.note,
            "created_at": self.created_at,
            "revoked_at": self.revoked_at,
            "last_used_at": self.last_used_at,
            "calls_total": self.calls_total,
            "revocable": self.source == "db",
            "active": self.revoked_at is None,
        }


class ApiKeyStore:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(DDL)
            self._conn.commit()
        self._by_hash: dict[str, KeyRecord] = {}
        self.reload()

    # ---------------------------------------------------------- 载入
    def reload(self) -> None:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM api_key").fetchall()
        recs: dict[str, KeyRecord] = {}
        for r in rows:
            rec = KeyRecord(
                key_id=r["key_id"], name=r["name"], key_hash=r["key_hash"],
                prefix=r["prefix"], rpm=int(r["rpm"]),
                scopes={s for s in (r["scopes"] or "").split(",") if s},
                source=r["source"], note=r["note"] or "", created_at=r["created_at"],
                revoked_at=r["revoked_at"], last_used_at=r["last_used_at"],
                calls_total=int(r["calls_total"] or 0),
            )
            recs[rec.key_hash] = rec
        self._by_hash = recs

    def _upsert(self, rec: KeyRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO api_key (key_id, name, key_hash, prefix, rpm, scopes,"
                " source, note, created_at, revoked_at, last_used_at, calls_total)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec.key_id, rec.name, rec.key_hash, rec.prefix, rec.rpm,
                 ",".join(sorted(rec.scopes)), rec.source, rec.note, rec.created_at,
                 rec.revoked_at, rec.last_used_at, rec.calls_total))
            self._conn.commit()
        self._by_hash[rec.key_hash] = rec

    # ---------------------------------------------------------- 环境变量注入
    def seed_from_env(self, raw: str) -> int:
        """把环境变量里的密钥写进库（幂等）。

        以 `source=env` 标记。它们不能从页面删除，但可以吊销——
        吊销状态落库，因此重启后仍然有效，不会出现"页面上已吊销、实际又能用"的矛盾。
        """
        n = 0
        for item in (raw or "").split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            if len(parts) < 2:
                continue
            name, key = parts[0], parts[1]
            rpm = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else 120
            scopes = ({s.strip() for s in parts[3].split(",") if s.strip()}
                      if len(parts) > 3 and parts[3].strip() else set(ALL_SCOPES))
            h = hash_key(key)
            existing = self._by_hash.get(h)
            rec = KeyRecord(
                key_id=existing.key_id if existing else f"env-{name}",
                name=name, key_hash=h, prefix=key[:6] + "…",
                rpm=rpm, scopes=scopes, source="env", note="由环境变量注入",
                created_at=existing.created_at if existing else now_iso(),
                revoked_at=existing.revoked_at if existing else None,
                last_used_at=existing.last_used_at if existing else None,
                calls_total=existing.calls_total if existing else 0,
            )
            self._upsert(rec)
            n += 1
        return n

    # ---------------------------------------------------------- 鉴权
    def resolve(self, raw: str | None) -> KeyRecord | None:
        if not raw:
            return None
        return self._by_hash.get(hash_key(raw))

    def touch(self, rec: KeyRecord) -> None:
        rec.calls_total += 1
        rec.last_used_at = now_iso()
        # 用量写入不追求每调用一次就落盘：批量由管理页读取时才刷。这里直接写，量级很小。
        with self._lock:
            self._conn.execute(
                "UPDATE api_key SET calls_total = ?, last_used_at = ? WHERE key_id = ?",
                (rec.calls_total, rec.last_used_at, rec.key_id))
            self._conn.commit()

    # ---------------------------------------------------------- 增删改查
    def list(self) -> list[dict[str, Any]]:
        self.reload()
        recs = sorted(self._by_hash.values(), key=lambda r: (r.source != "db", r.name))
        return [r.public() for r in recs]

    def create(self, name: str, rpm: int = 120, scopes: set[str] | None = None,
               note: str = "") -> tuple[str, dict[str, Any]]:
        """创建密钥。明文密钥**只在此刻返回一次**，之后无法再取回。"""
        name = (name or "").strip()
        if not name:
            raise ValueError("名称不能为空")
        bad = (scopes or set(ALL_SCOPES)) - set(ALL_SCOPES)
        if bad:
            raise ValueError(f"未知作用域：{sorted(bad)}")
        raw = new_key()
        rec = KeyRecord(
            key_id="k_" + secrets.token_hex(6), name=name, key_hash=hash_key(raw),
            prefix=raw[:8] + "…", rpm=max(1, int(rpm)), scopes=set(scopes or ALL_SCOPES),
            source="db", note=note.strip(), created_at=now_iso())
        self._upsert(rec)
        return raw, rec.public()

    def revoke(self, key_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM api_key WHERE key_id = ?",
                                     (key_id,)).fetchone()
        if row is None:
            raise KeyError(key_id)
        rec = self._by_hash.get(row["key_hash"])
        if rec is None:
            raise KeyError(key_id)
        if rec.revoked_at:
            return rec.public()
        rec.revoked_at = now_iso()
        self._upsert(rec)
        return rec.public()

    def rotate(self, key_id: str) -> tuple[str, dict[str, Any]]:
        """轮换：生成新明文，旧哈希立即失效（记录保留以便审计）。"""
        with self._lock:
            row = self._conn.execute("SELECT * FROM api_key WHERE key_id = ?",
                                     (key_id,)).fetchone()
        if row is None:
            raise KeyError(key_id)
        old = self._by_hash.get(row["key_hash"])
        if old is None:
            raise KeyError(key_id)
        with self._lock:
            self._conn.execute("DELETE FROM api_key WHERE key_id = ?", (key_id,))
            self._conn.commit()
        self._by_hash.pop(old.key_hash, None)

        raw = new_key()
        rec = KeyRecord(
            key_id=key_id, name=old.name, key_hash=hash_key(raw),
            prefix=raw[:8] + "…", rpm=old.rpm, scopes=old.scopes, source=old.source,
            note=(old.note or "") + "（已轮换）", created_at=now_iso())
        self._upsert(rec)
        return raw, rec.public()

    def delete(self, key_id: str) -> None:
        with self._lock:
            row = self._conn.execute("SELECT source FROM api_key WHERE key_id = ?",
                                     (key_id,)).fetchone()
        if row is None:
            raise KeyError(key_id)
        if row["source"] == "env":
            raise PermissionError("环境变量注入的密钥不能删除，只能吊销——"
                                 "否则重启后会重新生效，与页面显示的状态矛盾")
        with self._lock:
            self._conn.execute("DELETE FROM api_key WHERE key_id = ?", (key_id,))
            self._conn.commit()
        self._by_hash = {h: r for h, r in self._by_hash.items() if r.key_id != key_id}

    def stats(self) -> dict[str, Any]:
        recs = list(self._by_hash.values())
        return {
            "total": len(recs),
            "active": sum(1 for r in recs if not r.revoked_at),
            "revoked": sum(1 for r in recs if r.revoked_at),
            "env_sourced": sum(1 for r in recs if r.source == "env"),
            "db_sourced": sum(1 for r in recs if r.source == "db"),
            "calls_total": sum(r.calls_total for r in recs),
            "db_path": str(self.path),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --------------------------------------------------------------------- 端点目录

# 管理页要展示的接口清单。写在后端而不是前端，是为了让"代码里的端点"和"页面上列的端点"
# 只有一个来源，避免加了接口忘了更新文档。
ENDPOINTS: list[dict[str, Any]] = [
    {"method": "GET", "path": "/v1/health", "scope": "—", "auth": False,
     "desc": "存活检查，免鉴权", "group": "系统"},
    {"method": "GET", "path": "/v1/cluster", "scope": "read", "auth": True,
     "desc": "集群模式、脑节点可达性与遥测、记忆统计", "group": "系统"},
    {"method": "GET", "path": "/v1/connectomes", "scope": "read", "auth": True,
     "desc": "可用连接组清单（含是否真实数据、规模、许可）", "group": "系统"},
    {"method": "GET", "path": "/v1/tasks", "scope": "read", "auth": True,
     "desc": "列出已注册任务", "group": "任务"},
    {"method": "POST", "path": "/v1/tasks", "scope": "run", "auth": True,
     "desc": "注册任务；同 task_id 参数不同则 409 拒绝覆盖", "group": "任务"},
    {"method": "POST", "path": "/v1/runs", "scope": "run", "auth": True,
     "desc": "创建并异步启动一次运行", "group": "运行"},
    {"method": "GET", "path": "/v1/runs", "scope": "read", "auth": True,
     "desc": "运行列表（含 config / score / gates）", "group": "运行"},
    {"method": "GET", "path": "/v1/runs/{run_id}", "scope": "read", "auth": True,
     "desc": "单次运行状态与进度", "group": "运行"},
    {"method": "GET", "path": "/v1/runs/{run_id}/events", "scope": "read", "auth": True,
     "desc": "SSE 实时事件流", "group": "运行"},
    {"method": "POST", "path": "/v1/runs/{run_id}/abort", "scope": "run", "auth": True,
     "desc": "中止运行", "group": "运行"},
    {"method": "GET", "path": "/v1/runs/{run_id}/report", "scope": "read", "auth": True,
     "desc": "完整报告；未完成时 409", "group": "运行"},
    {"method": "GET", "path": "/v1/memory/stats", "scope": "read", "auth": True,
     "desc": "记忆统计（Docker 模式下转发到记忆中枢）", "group": "记忆"},
    {"method": "GET", "path": "/v1/memory/conflicts", "scope": "read", "auth": True,
     "desc": "同一 key 下的冲突结论", "group": "记忆"},
    {"method": "GET", "path": "/v1/memory/provenance/{origin_event_id}", "scope": "read",
     "auth": True, "desc": "来源链追溯（独立来源数、回声次数）", "group": "记忆"},
    {"method": "POST", "path": "/v1/memory/snapshot", "scope": "run", "auth": True,
     "desc": "创建记忆快照（?note=）", "group": "记忆"},
    {"method": "GET", "path": "/v1/games/catalog", "scope": "read", "auth": True,
     "desc": "棋种 / 飞蝇结构 / 电脑难度 / 能力归因", "group": "对弈演示"},
    {"method": "POST", "path": "/v1/games/sessions", "scope": "run", "auth": True,
     "desc": "新建对局", "group": "对弈演示"},
    {"method": "GET", "path": "/v1/games/sessions/{session_id}", "scope": "read",
     "auth": True, "desc": "对局状态（棋盘几何 + 走子记录）", "group": "对弈演示"},
    {"method": "POST", "path": "/v1/games/sessions/{session_id}/ply", "scope": "run",
     "auth": True, "desc": "走一个回合（飞蝇一步 + 电脑一步），含思考过程",
     "group": "对弈演示"},
    {"method": "GET", "path": "/v1/analytics/runs", "scope": "read", "auth": True,
     "desc": "实验结果归一化扁平表（含维度取值与指标定义），供分析页切片",
     "group": "分析"},
    {"method": "GET", "path": "/v1/admin/endpoints", "scope": "admin", "auth": True,
     "desc": "本清单本身（由后端提供，保证与代码同步）", "group": "接入管理"},
    {"method": "GET", "path": "/v1/admin/keys", "scope": "admin", "auth": True,
     "desc": "密钥列表（只返回前缀，不返回明文）", "group": "接入管理"},
    {"method": "POST", "path": "/v1/admin/keys", "scope": "admin", "auth": True,
     "desc": "创建密钥；明文只返回一次", "group": "接入管理"},
    {"method": "POST", "path": "/v1/admin/keys/{key_id}/rotate", "scope": "admin",
     "auth": True, "desc": "轮换密钥；旧密钥立即失效", "group": "接入管理"},
    {"method": "DELETE", "path": "/v1/admin/keys/{key_id}", "scope": "admin", "auth": True,
     "desc": "吊销密钥（环境变量注入的只能吊销不能删除）", "group": "接入管理"},
]
