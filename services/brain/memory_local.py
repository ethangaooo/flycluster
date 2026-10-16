"""脑节点本地记忆（L1 工作记忆 / L2 个体经历 / L3 个体技能）。

关键纪律（构思文档 §二、§四）：
  * L1 工作记忆只留在运行内存，**不同步**给其他脑，只通过通信输出间接表达。
  * L2 个体经历保存在本地，选择性提交到中心；提交时携带 parents，
    使中心能识别"这是转发来的经验，不是新的独立样本"。
  * L3 个体技能是学习参数本身（如读出器 ΔW）。保存一次完整脑状态只能恢复运行，
    它**不自动**成为一条其他脑能理解和使用的经验——二者不能混称。
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from flycore.schemas import ExperienceRecord
from flycore.util import blas_quiet


class LocalMemory:
    """按容量的环形经验库 + 暴力余弦检索。原型规模足够，不必上向量数据库。"""

    def __init__(self, node_id: str, root: str | Path = "data/local", capacity: int = 2048,
                 cue_dim: int = 32):
        self.node_id = node_id
        self.capacity = int(capacity)
        self.cue_dim = int(cue_dim)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{node_id}.jsonl"
        self._file_lock = threading.Lock()
        self.records: deque[ExperienceRecord] = deque(maxlen=self.capacity)
        self.cues = np.zeros((self.capacity, self.cue_dim), dtype=np.float32)
        self.valid = np.zeros(self.capacity, dtype=bool)
        self.cursor = 0
        self.n_appended = 0
        self._load()

    # ---------------------------------------------------------- 写入
    def remember(self, rec: ExperienceRecord) -> None:
        """写入一条经验。**立即追加落盘**——"重启后还能检索到旧记录"这道保留测试
        只有在真的写了文件时才可能通过；只放内存的实现会让这道测试永远失败。"""
        self._index(rec)
        line = json.dumps({"record": rec.model_dump()}, ensure_ascii=False) + "\n"
        with self._file_lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def _index(self, rec: ExperienceRecord) -> None:
        """只更新内存索引，不写文件（供启动时从文件回填使用，避免重复写入）。"""
        self.records.append(rec)
        cue = np.asarray(rec.obs_vector or [], dtype=np.float32).ravel()[: self.cue_dim]
        v = np.zeros(self.cue_dim, dtype=np.float32)
        v[: cue.size] = cue
        self.cues[self.cursor] = v
        self.valid[self.cursor] = True
        self.cursor = (self.cursor + 1) % self.capacity
        self.n_appended += 1

    def retarget(self, root: str | Path) -> int:
        """把本地记忆切到另一个目录（按 run 隔离用），并清空内存索引后回填。

        **为什么必须按 run 隔离**：脑容器是常驻的，若所有运行共用同一个
        `<data>/local/<node_id>.jsonl`，第二次运行会带着第一次的本地经验起跑，
        样本之间不再独立——单机模式早就修过这个（`local_root = .../<run_id>`），
        容器里漏了。
        """
        new_root = Path(root)
        new_root.mkdir(parents=True, exist_ok=True)
        self.root = new_root
        self.path = new_root / f"{self.node_id}.jsonl"
        self.records.clear()
        self.cues.fill(0.0)
        self.valid.fill(False)
        self.cursor = 0
        self.n_appended = 0
        self._load()
        return self.stats()["size"]

    # ---------------------------------------------------------- 检索
    def retrieve(self, cue: np.ndarray, k: int = 4) -> list[dict[str, Any]]:
        """返回本地相似经历。空库返回空列表，不返回随机噪声。

        检索数学用 float64：混合 dtype 的 BLAS 调用在部分后端会误报
        overflow / divide-by-zero，污染日志（矩阵很小，精度开销可忽略）。
        """
        if k <= 0 or not self.valid.any() or cue.size == 0:
            return []      # k=0 表示"关闭检索"，必须真的返回空，而不是至少给一条
        q = np.asarray(cue, dtype=np.float64).ravel()[: self.cue_dim]
        q = q / (float(np.linalg.norm(q)) or 1.0)
        mat = self.cues[: self.valid.shape[0]].astype(np.float64)
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        with blas_quiet():
            sim = np.asarray(mat @ q, dtype=np.float64) / norms
        sim[~self.valid] = -np.inf
        idx = np.argsort(-sim)[: int(k)]
        out = []
        for i in idx:
            if not np.isfinite(sim[i]):
                continue
            rec = self._record_at(int(i))
            if rec is None:
                continue
            out.append({
                "record_id": rec.record_id, "origin_event_id": rec.origin_event_id,
                "producer_node": rec.producer_node, "kind": rec.kind,
                "cue_vector": rec.obs_vector, "payload": rec.payload,
                "conditions": rec.conditions, "confidence": rec.confidence,
                "observed_at": rec.observed_at, "parents": rec.parents,
                "is_primary": not rec.parents, "similarity": round(float(sim[i]), 4),
                "source": "local",
            })
        return out

    def _record_at(self, ring_index: int) -> ExperienceRecord | None:
        """环形索引 → 记录：最近写入的一条位于 (cursor-1) % capacity。"""
        back = (self.cursor - 1 - ring_index) % self.capacity   # 距最新一条有多少步
        pos = len(self.records) - 1 - back
        if 0 <= pos < len(self.records):
            return self.records[pos]
        return None

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        out = []
        for rec in list(self.records)[-limit:]:
            out.append({"record_id": rec.record_id, "tick": rec.tick, "kind": rec.kind,
                        "payload": rec.payload, "conditions": rec.conditions})
        return out

    # ---------------------------------------------------------- 持久化
    def _load(self) -> None:
        """启动时从 JSONL 回填内存索引。用 `_index` 而不是 `remember`，
        否则回填过程会把历史记录再追加一遍，文件会随每次启动翻倍。"""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = ExperienceRecord(**json.loads(line)["record"])
                except Exception:
                    continue
                self._index(rec)

    def stats(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "capacity": self.capacity,
                "size": len(self.records), "n_appended": self.n_appended,
                "path": str(self.path)}


class WorkingMemory:
    """L1 即时工作记忆：只在运行内存，不同步。"""

    def __init__(self, size: int = 32):
        self.buf: deque[dict[str, Any]] = deque(maxlen=size)

    def push(self, item: dict[str, Any]) -> None:
        self.buf.append(item)

    def last(self) -> dict[str, Any] | None:
        return self.buf[-1] if self.buf else None

    def summary(self) -> dict[str, Any]:
        if not self.buf:
            return {"size": 0}
        last = self.buf[-1]
        return {"size": len(self.buf), "last_tick": last.get("tick"),
                "last_action": last.get("action"),
                "recent_actions": [b.get("action") for b in list(self.buf)[-8:]],
                "membrane": last.get("membrane", {})}
