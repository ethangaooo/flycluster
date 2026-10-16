"""中心记忆中枢的持久化层（SQLite）。

设计要点（对应构思文档 §五、§八）：
  * 来源链：每条经验保留 origin_event_id / producer_node / parents / derived_from_origin，
    使「同一次原始事件被转发多次」不会变成「多个独立样本」。
  * 冲突保留：同一 key 的不同结论按时间与条件并列保留，不做最后写入覆盖。
  * 版本：快照式 memory_version，支持按快照时间点回溯查询。
  * 通信预算：检索返回的字节数必须计入通信预算，否则中心记忆会变成免费通道。
  * 持久化：数据库落在独立卷路径，容器重建不丢记忆。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

from flycore.schemas import ExperienceRecord, MemoryQuery, MemorySubmitResult
from flycore.util import content_hash, now_iso

CUES_DIM = 32

DDL = """
CREATE TABLE IF NOT EXISTS seq (
    id    INTEGER PRIMARY KEY CHECK (id = 1),
    value INTEGER NOT NULL
);
INSERT OR IGNORE INTO seq (id, value) VALUES (1, 0);

CREATE TABLE IF NOT EXISTS experience (
    record_id           TEXT PRIMARY KEY,
    seq                 INTEGER NOT NULL DEFAULT 0,
    origin_event_id     TEXT NOT NULL,
    producer_node       TEXT NOT NULL,
    kind                TEXT NOT NULL,
    run_id              TEXT DEFAULT '',
    task_id             TEXT DEFAULT '',
    episode             INTEGER DEFAULT 0,
    tick                INTEGER DEFAULT 0,
    env_version         TEXT DEFAULT '',
    schema_version      TEXT DEFAULT '1',
    scope               TEXT DEFAULT 'run',
    cue                 BLOB,
    payload             TEXT DEFAULT '{}',
    conditions          TEXT DEFAULT '{}',
    parents             TEXT DEFAULT '[]',
    derived_from_origin TEXT,
    confidence          REAL DEFAULT 0.5,
    observed_at         TEXT DEFAULT '',
    valid_from          TEXT DEFAULT '',
    valid_to            TEXT,
    expire_seq          INTEGER,
    is_primary          INTEGER DEFAULT 1,
    evidence_count      INTEGER DEFAULT 1,
    echo_count          INTEGER DEFAULT 0,
    status              TEXT DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_exp_origin  ON experience(origin_event_id);
CREATE INDEX IF NOT EXISTS idx_exp_task    ON experience(task_id, scope, kind, status);
CREATE INDEX IF NOT EXISTS idx_exp_seq     ON experience(seq);
CREATE INDEX IF NOT EXISTS idx_exp_expseq  ON experience(expire_seq);
CREATE INDEX IF NOT EXISTS idx_exp_derived ON experience(derived_from_origin);

CREATE TABLE IF NOT EXISTS snapshot (
    version      TEXT PRIMARY KEY,
    seq          INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    note         TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS access_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    caller      TEXT DEFAULT '',
    op          TEXT NOT NULL,
    n_records   INTEGER DEFAULT 0,
    cost_bytes  INTEGER DEFAULT 0,
    detail      TEXT DEFAULT ''
);
"""


class MemoryStore:
    """SQLite 支撑的经验存储。单写者模型：所有写入经由此类，脑节点不直接碰数据库文件。"""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._conn.executescript(DDL)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """给早期版本建的库补列。缺列不报错，但必须补上，否则序号语义失效。"""
        cols_exp = {r["name"] for r in self._conn.execute("PRAGMA table_info(experience)")}
        for name, ddl in (("seq", "ALTER TABLE experience ADD COLUMN seq INTEGER NOT NULL DEFAULT 0"),
                          ("expire_seq", "ALTER TABLE experience ADD COLUMN expire_seq INTEGER")):
            if name not in cols_exp:
                self._conn.execute(ddl)
        cols_snap = {r["name"] for r in self._conn.execute("PRAGMA table_info(snapshot)")}
        if "seq" not in cols_snap:
            self._conn.execute("ALTER TABLE snapshot ADD COLUMN seq INTEGER NOT NULL DEFAULT 0")
        if self._conn.execute("SELECT COUNT(*) c FROM seq").fetchone()["c"] == 0:
            self._conn.execute("INSERT INTO seq (id, value) VALUES (1, 0)")
            max_seq = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) m FROM experience").fetchone()["m"]
            self._conn.execute("UPDATE seq SET value = ? WHERE id = 1", (int(max_seq),))
        # 早期用 valid_to 表达过期、没有 expire_seq，这里把历史数据对齐到序号语义。
        self._conn.execute(
            "UPDATE experience SET expire_seq = (SELECT value FROM seq WHERE id = 1) "
            "WHERE valid_to IS NOT NULL AND expire_seq IS NULL")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ 事件序号
    def _next_seq(self, n: int = 1) -> int:
        """单调递增事件序号。

        **不能用墙钟时间给事件排序**：快照与过期经常落在同一毫秒里，
        `valid_to > snapshot_time` 这种比较会给出错误答案。
        用序号后，快照回溯与过期排除的语义与时钟分辨率彻底解耦。
        """
        with self._lock:
            cur = self._conn.execute("SELECT value FROM seq WHERE id = 1").fetchone()
            base = int(cur["value"])
            self._conn.execute("UPDATE seq SET value = ? WHERE id = 1", (base + n,))
            self._conn.commit()
            return base + 1

    def _max_seq(self) -> int:
        with self._lock:
            r = self._conn.execute("SELECT value FROM seq WHERE id = 1").fetchone()
            return int(r["value"])

    # ------------------------------------------------------------ 写入
    def submit(self, rec: ExperienceRecord, *, strict_echo_check: bool = True) -> MemorySubmitResult:
        """写入一条经验，并执行防回声与来源判定。

        判定规则（三条，全部落库留痕）：
          1. 同 (origin_event_id, producer_node) 重复提交 → 拒绝，不增加证据。
          2. 同一 origin_event_id 由**另一个**节点提交且自称原始观察 → 判定为回声，
             记录 echo_of 且不给证据计数。这条用来抓
             「A 产生经验 → B 读到 → B 当成自己的经历回传」。
          3. 采纳了检索结论的派生经验（kind=inference 或带 derived_from_origin）
             → 记为非原始（is_primary=0），不增加证据计数。
             仅仅是"检索返回过上下文"不构成派生，否则会系统性低估计数。
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT record_id, producer_node, origin_event_id FROM experience "
                "WHERE origin_event_id = ? AND producer_node = ?",
                (rec.origin_event_id, rec.producer_node),
            )
            dup = cur.fetchone()
            if dup is not None:
                cur.execute(
                    "UPDATE experience SET echo_count = echo_count + 1 WHERE record_id = ?",
                    (dup["record_id"],),
                )
                self._conn.commit()
                self._log("submit", 0, 0, f"duplicate:{rec.record_id}")
                return MemorySubmitResult(accepted=False, record_id=dup["record_id"],
                                          is_echo=True, echo_of=dup["record_id"],
                                          evidence_count_delta=0,
                                          reason="同一来源节点重复提交同一原始事件，不增加证据")

            cur.execute(
                "SELECT record_id, producer_node FROM experience WHERE origin_event_id = ?",
                (rec.origin_event_id,),
            )
            others = cur.fetchall()

            claims_primary = rec.kind in ("observation", "skill") and not rec.derived_from_origin
            is_primary = claims_primary
            is_echo = False
            echo_of: str | None = None
            delta = 0
            if others and claims_primary and strict_echo_check:
                is_echo = True
                echo_of = others[0]["record_id"]
                is_primary = False
            elif is_primary:
                delta = 1

            seq = self._next_seq()
            cur.execute(
                "INSERT OR REPLACE INTO experience (record_id, seq, origin_event_id, producer_node,"
                " kind, run_id, task_id, episode, tick, env_version, schema_version, scope, cue,"
                " payload, conditions, parents, derived_from_origin, confidence, observed_at,"
                " valid_from, valid_to, expire_seq, is_primary, evidence_count, echo_count, status)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.record_id, seq, rec.origin_event_id, rec.producer_node, rec.kind,
                    rec.run_id, rec.task_id, rec.episode, rec.tick, rec.env_version,
                    rec.schema_version, rec.scope, self._pack_cue(rec.obs_vector),
                    json.dumps(rec.payload, ensure_ascii=False),
                    json.dumps(rec.conditions, ensure_ascii=False),
                    json.dumps(rec.parents), rec.derived_from_origin, float(rec.confidence),
                    rec.observed_at or now_iso(), rec.valid_from or now_iso(), rec.valid_to,
                    None, 1 if is_primary else 0, 1 if is_primary else 0, 0, "active",
                ),
            )
            if is_echo and echo_of:
                cur.execute("UPDATE experience SET echo_count = echo_count + 1 WHERE record_id = ?",
                            (echo_of,))
            self._conn.commit()
        self._log("submit", 1, 0, f"echo={is_echo} primary={is_primary}")
        reason = ("判定为回声：同一原始事件已由其他节点上报，不增加独立证据"
                  if is_echo else ("已接收为原始经验" if is_primary else "已接收为派生经验（不增加独立证据）"))
        return MemorySubmitResult(accepted=True, record_id=rec.record_id, is_echo=is_echo,
                                  echo_of=echo_of, evidence_count_delta=delta, reason=reason)

    # ------------------------------------------------------------ 读取
    def query(self, q: MemoryQuery, *, caller: str = "") -> dict[str, Any]:
        """按线索向量做暴力余弦检索（原型规模足够，规模上来需替换为向量索引）。"""
        with self._lock:
            cur = self._conn.cursor()
            sql = "SELECT * FROM experience WHERE status = 'active'"
            args: list[Any] = []
            if q.task_id:
                sql += " AND task_id = ?"
                args.append(q.task_id)
            if q.scope:
                sql += " AND scope = ?"
                args.append(q.scope)
            if q.tenant:
                sql += " AND json_extract(conditions, '$.tenant') = ?"
                args.append(q.tenant)
            if q.scope == "run" and q.run_id:
                sql += " AND run_id = ?"
                args.append(q.run_id)
            elif q.scope == "run":
                # 运行级隔离：未指定 run_id 时**一条都不返回**，而不是返回全部。
                # "没传参数就默认跨运行共享"是最容易出的数据泄漏，必须从 SQL 层堵死。
                sql += " AND 1 = 0"
            if q.producer_node:
                sql += " AND producer_node = ?"
                args.append(q.producer_node)
            if q.kinds:
                sql += f" AND kind IN ({','.join('?' * len(q.kinds))})"
                args.extend(q.kinds)
            if q.as_of_version:
                row = cur.execute("SELECT seq, created_at FROM snapshot WHERE version = ?",
                                  (q.as_of_version,)).fetchone()
                if row is None:
                    raise KeyError(f"未知记忆快照 {q.as_of_version}")
                snap_seq = int(row["seq"])
                sql += " AND seq <= ?"
                args.append(snap_seq)
                sql += " AND (expire_seq IS NULL OR expire_seq > ?)"
                args.append(snap_seq)
            else:
                # 过期经验默认不参与检索：有效期是经验的一部分，不是可选元数据。
                sql += " AND expire_seq IS NULL"
            rows = cur.execute(sql, args).fetchall()

        items = []
        cue = np.asarray(q.cue_vector, dtype=np.float32) if q.cue_vector else None
        if cue is not None and cue.size:
            norm = float(np.linalg.norm(cue)) or 1.0
            cue = cue / norm
        for r in rows:
            sim = 0.0
            rc = self._unpack_cue(r["cue"])
            if cue is not None and rc is not None and rc.size:
                rn = float(np.linalg.norm(rc)) or 1.0
                sim = float(np.dot(cue, rc / rn))
            items.append((sim, r))
        items.sort(key=lambda t: -t[0])
        chosen = items[: max(0, int(q.k))]

        out: list[dict[str, Any]] = []
        for sim, r in chosen:
            out.append({
                "record_id": r["record_id"],
                "origin_event_id": r["origin_event_id"],
                "producer_node": r["producer_node"],
                "kind": r["kind"],
                "task_id": r["task_id"],
                "episode": r["episode"],
                "tick": r["tick"],
                "env_version": r["env_version"],
                "scope": r["scope"],
                "cue_vector": self._unpack_cue(r["cue"]).tolist() if r["cue"] else [],
                "payload": json.loads(r["payload"] or "{}"),
                "conditions": json.loads(r["conditions"] or "{}"),
                "parents": json.loads(r["parents"] or "[]"),
                "derived_from_origin": r["derived_from_origin"],
                "confidence": r["confidence"],
                "observed_at": r["observed_at"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "is_primary": bool(r["is_primary"]),
                "evidence_count": r["evidence_count"],
                "echo_count": r["echo_count"],
                "similarity": round(sim, 4),
            })
        cost = len(json.dumps(out, ensure_ascii=False).encode("utf-8"))
        conflicts = self.conflicts(task_id=q.task_id)
        version = self.head_version()
        self._log("query", len(out), cost, f"task={q.task_id} k={q.k}")
        return {"version": version, "retrieved": out, "conflicts": conflicts, "query_cost_bytes": cost}

    def conflicts(self, task_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """返回同一 key 下结论不一致的记录组。保留时间与条件，不做覆盖。"""
        with self._lock:
            sql = "SELECT * FROM experience WHERE status='active' AND is_primary = 1"
            args: list[Any] = []
            if task_id:
                sql += " AND task_id = ?"
                args.append(task_id)
            rows = self._conn.execute(sql, args).fetchall()
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            payload = json.loads(r["payload"] or "{}")
            key = payload.get("key")
            if not key:
                continue
            groups.setdefault(key, []).append({
                "record_id": r["record_id"], "producer_node": r["producer_node"],
                "conclusion": payload.get("conclusion"), "confidence": r["confidence"],
                "observed_at": r["observed_at"], "conditions": json.loads(r["conditions"] or "{}"),
                "tick": r["tick"], "evidence_count": r["evidence_count"],
            })
        out = []
        for key, items in groups.items():
            conclusions = {json.dumps(i["conclusion"], sort_keys=True, ensure_ascii=False) for i in items}
            if len(conclusions) > 1:
                out.append({"key": key, "n_records": len(items), "n_conclusions": len(conclusions),
                            "records": sorted(items, key=lambda i: i["observed_at"])})
        out.sort(key=lambda g: -g["n_conclusions"])
        return out[:limit]

    def provenance_chain(self, origin_event_id: str) -> dict[str, Any]:
        """追溯一次原始事件在集群中的传播轨迹。

        `independent_sources` 只数原始记录的产生节点：转发十次的同一条经验，
        独立来源仍然是 1。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_id, producer_node, parents, derived_from_origin, observed_at, kind, "
                "is_primary, echo_count FROM experience "
                "WHERE origin_event_id = ? OR derived_from_origin = ? ORDER BY observed_at",
                (origin_event_id, origin_event_id),
            ).fetchall()
        return {
            "origin_event_id": origin_event_id,
            "n_records": len(rows),
            "independent_sources": len({r["producer_node"] for r in rows if r["is_primary"]}),
            "echo_events": int(sum(r["echo_count"] for r in rows)),
            "chain": [dict(r) for r in rows],
        }

    # ------------------------------------------------------------ 版本与回放
    def expire(self, record_ids: list[str], at: str | None = None) -> dict[str, Any]:
        """标记经验失效（设置 valid_to）。经验不会被删除，历史与来源链保留。

        过期是经验生命周期的一部分：'上午可通行、下午有障碍'应保留两条并各自带时间，
        而不是让最后写入覆盖之前内容。
        """
        when = at or now_iso()
        with self._lock:
            cur = self._conn.cursor()
            seq = self._next_seq()
            n = 0
            for rid in record_ids:
                cur.execute(
                    "UPDATE experience SET valid_to = ?, expire_seq = ? "
                    "WHERE record_id = ? AND expire_seq IS NULL",
                    (when, seq, rid))
                n += cur.rowcount
            self._conn.commit()
        self._log("expire", n, 0, f"at={when} seq={seq}")
        return {"expired": n, "at": when, "expire_seq": seq}

    def head_version(self) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT version FROM snapshot ORDER BY seq DESC, created_at DESC LIMIT 1").fetchone()
        return row["version"] if row else "v0"

    def snapshot(self, note: str = "") -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_id, origin_event_id, producer_node, payload, observed_at "
                "FROM experience WHERE status='active' ORDER BY record_id").fetchall()
            payload = [dict(r) for r in rows]
            created = now_iso()
            seq = self._max_seq()
            version = f"v{len(payload)}-{content_hash(payload)}"
            self._conn.execute(
                "INSERT OR REPLACE INTO snapshot (version, seq, created_at, content_hash,"
                " record_count, note) VALUES (?,?,?,?,?,?)",
                (version, seq, created, content_hash(payload), len(payload), note))
            self._conn.commit()
        return {"version": version, "seq": seq, "created_at": created,
                "record_count": len(payload), "note": note}

    def get_record(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM experience WHERE record_id = ?",
                                   (record_id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["payload"] = json.loads(d.get("payload") or "{}")
        d["conditions"] = json.loads(d.get("conditions") or "{}")
        d["parents"] = json.loads(d.get("parents") or "[]")
        return d

    def record_ids(self, *, task_id: str | None = None, limit: int = 1000) -> list[str]:
        sql = "SELECT record_id FROM experience WHERE status='active' AND is_primary=1"
        args: list[Any] = []
        if task_id:
            sql += " AND task_id = ?"
            args.append(task_id)
        sql += " ORDER BY observed_at LIMIT ?"
        args.append(limit)
        with self._lock:
            return [r["record_id"] for r in self._conn.execute(sql, args).fetchall()]

    def replay(self, *, task_id: str | None = None, limit: int = 64,               min_evidence: int = 1, kinds: list[str] | None = None) -> list[dict[str, Any]]:
        """经验回放：只回放达到最小证据量的原始经验，派生与回声不进入回放集。"""
        sql = ("SELECT * FROM experience WHERE status='active' AND is_primary=1 "
               "AND evidence_count >= ?")
        args: list[Any] = [min_evidence]
        if task_id:
            sql += " AND task_id = ?"
            args.append(task_id)
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        sql += " ORDER BY observed_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [{"record_id": r["record_id"], "origin_event_id": r["origin_event_id"],
                 "producer_node": r["producer_node"], "kind": r["kind"],
                 "payload": json.loads(r["payload"] or "{}"),
                 "cue_vector": self._unpack_cue(r["cue"]).tolist() if r["cue"] else [],
                 "observed_at": r["observed_at"], "evidence_count": r["evidence_count"]}
                for r in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            cur = self._conn.cursor()
            total = cur.execute("SELECT COUNT(*) c FROM experience").fetchone()["c"]
            primary = cur.execute("SELECT COUNT(*) c FROM experience WHERE is_primary=1").fetchone()["c"]
            origins = cur.execute("SELECT COUNT(DISTINCT origin_event_id) c FROM experience").fetchone()["c"]
            echoes = cur.execute("SELECT COALESCE(SUM(echo_count),0) c FROM experience").fetchone()["c"]
            snaps = cur.execute("SELECT COUNT(*) c FROM snapshot").fetchone()["c"]
            acc = cur.execute(
                "SELECT COALESCE(SUM(cost_bytes),0) b, COUNT(*) n FROM access_log WHERE op='query'"
            ).fetchone()
        return {
            "records_total": total, "records_primary": primary, "records_derived": total - primary,
            "distinct_origins": origins, "echo_events": echoes,
            "snapshots": snaps, "head_version": self.head_version(),
            "query_count": acc["n"], "query_cost_bytes_total": acc["b"],
            "db_path": self.db_path, "cue_dim": CUES_DIM,
        }

    # ------------------------------------------------------------ 内部
    @staticmethod
    def _pack_cue(v: list[float]) -> bytes:
        a = np.asarray(v or [], dtype=np.float32)
        return a.tobytes()

    @staticmethod
    def _unpack_cue(b: bytes | None) -> np.ndarray:
        if not b:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(b, dtype=np.float32)

    def _log(self, op: str, n: int, cost: int, detail: str, caller: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO access_log (at, caller, op, n_records, cost_bytes, detail) VALUES (?,?,?,?,?,?)",
                (now_iso(), caller, op, n, cost, detail))
            self._conn.commit()

    def recent_access(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM access_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
