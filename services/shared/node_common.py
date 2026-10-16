"""脑模型与全部对照组共享的节点机制：记忆检索/写入、消息收发、计数与检查点。

**为什么必须共享而不是各写一份**：对照组如果用了不同的记忆协议或不同的消息口径，
性能差异就无法归因到"中间那段结构"，实验不再满足单一变量对照。所以线索函数、
检索范围、记录格式、提交规则、消息量化与字节核算全部放在这里，唯一允许各模型
自行实现的是 `comm_values()`（消息输出组的活动）与动作决策。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from flycore.coding import CueEncoder, MessageCodec
from flycore.schemas import (MSG_VALUES, N_ACTIONS, BrainNodeConfig, BrainStepRequest,
                             BrainStepResponse, ExperienceRecord, MemoryQuery)
from flycore.util import content_hash, now_iso

from services.brain.hub_client import MemoryBackend
from services.brain.memory_local import LocalMemory, WorkingMemory

ACTION_NAMES = ("up", "down", "left", "right", "stay")


class NodeCommon:
    """记忆与通信的公共实现。子类只需给出 `comm_values()` 与决策逻辑。"""

    def init_common(self, cfg: BrainNodeConfig, hub: MemoryBackend | None,
                    local_root: str | Path = "data/local", k_local: int = 4, k_central: int = 6,
                    record_cue_dim: int = 32, local_memory_enabled: bool = True,
                    central_memory_enabled: bool = True) -> None:
        self.cfg = cfg
        # 两个**正交**的开关，不能合并成一个：
        #   local_memory_enabled   —— 每个节点自己的经历（本地文件，不占用通信预算）
        #   central_memory_enabled —— 中心记忆中枢（这是一条**通信通道**，检索字节要计入预算）
        # 曾经把 `enable_memory_channel` 直接接到一个总开关上，结果主对照里本地记忆被全程关掉、
        # `k_local` 变成死旋钮：扫描时四个 k_local 取值给出**完全相同的 Δ**。
        self.local_memory_enabled = bool(local_memory_enabled)
        self.central_memory_enabled = bool(central_memory_enabled)
        self.node_id = cfg.node_id
        self.hub = hub
        self.k_local = int(k_local)
        self.k_central = int(k_central)
        self.cue_encoder = CueEncoder(record_cue_dim)
        self.local = LocalMemory(cfg.node_id, root=local_root, cue_dim=record_cue_dim)
        self.working = WorkingMemory()
        self.record_cue_dim = int(record_cue_dim)
        self.message_kind = "learned_compressed"
        self.tick = 0
        self.messages_emitted = 0
        self.payload_bytes = 0
        self.memory_updates = 0
        self.retrieval_count = 0
        self.memory_bytes_in = 0
        self.local_hits = 0
        self.central_hits = 0
        self.derived_submissions = 0
        self._codec = MessageCodec()

    # ------------------------------------------------------------ 线索与检索
    def cue(self, obs: np.ndarray) -> np.ndarray:
        return self.cue_encoder(obs)

    def recall(self, cue: np.ndarray, ctx: dict[str, Any]) -> list[dict[str, Any]]:
        """本地检索 + （按范围配置）中心检索。范围隔离由中心强制。"""
        if not (self.local_memory_enabled or self.central_memory_enabled):
            return []
        hits = self.local.retrieve(cue, k=self.k_local) if self.local_memory_enabled else []
        self.local_hits += len(hits)
        if self.hub is not None and self.central_memory_enabled:
            q = MemoryQuery(
                cue_vector=cue.tolist(),
                task_id=ctx.get("task_id") or None,
                scope=self.cfg.memory_scope,
                run_id=ctx.get("run_id") if self.cfg.memory_scope == "run" else None,
                tenant=ctx.get("tenant"),
                k=self.k_central,
                as_of_version=(None if self.cfg.memory_version == "head"
                               else ctx.get("memory_version")),
            )
            try:
                res = self.hub.query(q, caller=self.node_id)
            except Exception:
                res = None
            if res:
                self.memory_bytes_in += int(res.get("query_cost_bytes", 0))
                self.central_hits += len(res.get("retrieved", []))
                hits = hits + res.get("retrieved", [])
        self.retrieval_count += 1
        return hits

    # ------------------------------------------------------------ 消息
    @staticmethod
    def peer_values(req: BrainStepRequest) -> list[float]:
        """同伴消息压成 MSG_VALUES 维输入：多发送者取逐维均值。

        输入维度固定，因此邻居数量变化不会改变模型输入规模；
        "收到几条"由 source_tags.peer_messages 单独统计，不混进输入。
        """
        if not req.peer_messages:
            return []
        codec = MessageCodec()
        arr = np.stack([codec.decode(m.values) for m in req.peer_messages])
        return arr.mean(axis=0).tolist()

    def make_message(self, obs: np.ndarray) -> list[int]:
        if self.message_kind == "shared_observation":
            return self.observation_message(obs)
        vals = np.asarray(self.comm_values(), dtype=np.float32)
        return self._codec.encode(vals)

    def comm_values(self) -> np.ndarray:
        """消息输出通道的取值（长度 MSG_VALUES，范围 [-1,1]）。子类必须实现。"""
        raise NotImplementedError

    @staticmethod
    def observation_message(obs: np.ndarray) -> list[int]:
        """把局部观测压成 8 个数值。**这是外围代码的产物，不是模型输出。**

        使用该模式时必须把协作收益归因于"共享了实际观测到的有限信息"。
        """
        from flycore.schemas import VIEW_CHANNELS, VIEW_SIZE
        v = np.asarray(obs, dtype=np.float32)
        view = v[: VIEW_SIZE * VIEW_SIZE * VIEW_CHANNELS].reshape(VIEW_SIZE, VIEW_SIZE, VIEW_CHANNELS)
        r = VIEW_SIZE // 2
        quads = [(slice(0, r), slice(0, r)), (slice(0, r), slice(r + 1, VIEW_SIZE)),
                 (slice(r + 1, VIEW_SIZE), slice(0, r)),
                 (slice(r + 1, VIEW_SIZE), slice(r + 1, VIEW_SIZE))]
        vals: list[float] = []
        for qy, qx in quads:
            patch = view[qy, qx]
            vals.append(float(patch[:, :, 0].mean()) * 2 - 1)
            vals.append(float(patch[:, :, 1].mean()) * 2 - 1)
        return MessageCodec().encode(np.asarray(vals, dtype=np.float32))

    def set_message_kind(self, kind: str) -> None:
        if kind not in ("shared_observation", "learned_compressed"):
            raise ValueError(f"未知消息模式 {kind}")
        self.message_kind = kind

    # ------------------------------------------------------------ 记忆写入
    def build_record(self, cue: np.ndarray, action: int, recalled: list[dict[str, Any]],
                     ctx: dict[str, Any], extra: dict[str, Any] | None = None) -> ExperienceRecord:
        cell = ctx.get("self_cell")
        outcome = ctx.get("last_outcome") or {}
        payload = {
            "key": (f"cell:{cell[0]},{cell[1]}" if cell else f"tick:{self.tick}"),
            "conclusion": action,
            "action_name": ACTION_NAMES[action] if 0 <= action < N_ACTIONS else "?",
            "target_found": bool(outcome.get("target_found")),
            "collided": bool(outcome.get("collided")),
            "perception_valid": bool(ctx.get("perception_valid", True)),
        }
        if extra:
            payload.update(extra)

        parents: list[str] = []
        derived_from: str | None = None
        adopted = False
        used = [r for r in recalled if r.get("record_id")]
        if used:
            # parents 记录"本次计算确实用过哪些检索结果"，用于构建来源链。
            top = max(used, key=lambda r: float(r.get("similarity", 0.0)))
            parents = [r["record_id"] for r in used[:3]]
            # 只有**采纳了检索到的结论**才算派生经验，中心因此不会把它计为独立证据。
            # 仅仅是"检索返回了上下文"不构成派生——否则几乎所有经验都会被误判为转发。
            top_conclusion = (top.get("payload") or {}).get("conclusion")
            adopted = top_conclusion is not None and int(top_conclusion) == int(action)
            if adopted:
                derived_from = top.get("origin_event_id")
                payload["adopted_from"] = top.get("record_id")
                payload["adopted_origin"] = derived_from
                payload["conclusion_source"] = "retrieved"
                self.derived_submissions += 1
            else:
                payload["conclusion_source"] = "own_observation"

        return ExperienceRecord(
            record_id=f"{self.node_id}-{ctx.get('run_id','')}-{self.tick}-{action}-{content_hash(payload)[:6]}",
            origin_event_id=f"evt-{ctx.get('run_id','')}-{self.node_id}-{self.tick}",
            producer_node=self.node_id,
            kind="inference" if adopted else "observation",
            run_id=str(ctx.get("run_id", "")),
            task_id=str(ctx.get("task_id", "")),
            episode=int(ctx.get("episode", 0) or 0),
            tick=self.tick,
            env_version=str(ctx.get("env_version", "")),
            scope=self.cfg.memory_scope,
            obs_vector=np.asarray(cue, dtype=np.float32).tolist(),
            payload=payload,
            conditions={"tenant": ctx.get("tenant", "default"), "grid": ctx.get("grid"),
                        "agents": ctx.get("agents"), "tick": self.tick,
                        "decision_hz": ctx.get("decision_hz")},
            parents=parents,
            derived_from_origin=derived_from,
            confidence=0.4 if adopted else 0.6,
            observed_at=now_iso(),
            valid_from=now_iso(),
        )

    def remember(self, rec: ExperienceRecord) -> None:
        if self.local_memory_enabled:
            self.local.remember(rec)
        if self.hub is not None and self.central_memory_enabled:
            try:
                if self.hub.submit(rec).accepted:
                    self.memory_updates += 1
            except Exception:
                pass

    # ------------------------------------------------------------ 单步骨架
    def prepare(self, req: BrainStepRequest) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]],
                                                     list[float], dict[str, Any]]:
        """所有模型共用的单步前置：观测清洗 → 线索 → 检索 → 同伴消息。"""
        ctx = dict(req.run_context or {})
        obs = np.asarray(req.observation.vector, dtype=np.float32)
        # 感知失效已经由环境编码进观测向量本身（视野清零、本体感觉保留）。
        # 这里**不**再做整体清零：那会把环境刻意保留的本体感觉也抹掉，
        # 导致"环境说的失效"和"脑以为的失效"不是同一件事。
        ctx["perception_valid"] = bool(req.observation.perception_valid)
        self.tick = req.tick
        cue = self.cue(obs)
        recalled = self.recall(cue, ctx)
        peer = self.peer_values(req)
        return obs, cue, recalled, peer, ctx

    def external_features(self, obs: np.ndarray, recalled: list[dict[str, Any]],
                          peer_values: list[float], mem_dim: int = 24) -> np.ndarray:
        """给对照组统一的外部信息向量：[观测 | 同伴消息 | 记忆特征]。

        三部分与脑模型的三条电流注入通道一一对应，信息量相同——差别只在中间结构。
        记忆特征使用与 MemoryAdapter 相同的聚合规则（按条均值 × 置信度）。
        """
        from flycore.coding import CUE_DIM
        feats = np.zeros(mem_dim, dtype=np.float32)
        if recalled:
            used = 0
            for r in recalled[:mem_dim]:
                cue = r.get("cue_vector") or []
                if not cue:
                    continue
                c = np.asarray(cue, dtype=np.float32)[:CUE_DIM]
                feats[used % mem_dim] += float(c.mean()) * float(r.get("confidence", 0.5) or 0.5)
                used += 1
            if used:
                feats /= used
        pv = np.asarray(peer_values or [], dtype=np.float32)
        if pv.size != MSG_VALUES:
            pv = np.zeros(MSG_VALUES, dtype=np.float32)
        return np.concatenate([obs, pv, feats]).astype(np.float32)

    def finish(self, *, node_id: str, tick: int, action: int, scores: np.ndarray,
               obs: np.ndarray, cue: np.ndarray, recalled: list[dict[str, Any]],
               ctx: dict[str, Any], spikes: int = 0, dropped: int = 0,
               realtime_ratio: float = 0.0, source_tags: dict[str, int] | None = None,
               telemetry: dict[str, Any] | None = None,
               extra_payload: dict[str, Any] | None = None) -> BrainStepResponse:
        """所有模型共用的单步后置：消息、记录、工作记忆、计量。"""
        message = self.make_message(obs)
        if message:
            self.messages_emitted += 1
            self.payload_bytes += len(message) * 2
        self.working.push({"tick": tick, "action": action,
                           "reward": float(ctx.get("last_reward", 0.0) or 0.0)})
        rec = self.build_record(cue, action, recalled, ctx, extra_payload)
        self.remember(rec)
        return BrainStepResponse(
            node_id=node_id, tick=tick, action=action,
            action_scores=[round(float(s), 5) for s in np.asarray(scores).ravel()],
            outgoing_message=message, spikes_emitted=spikes, dropped_spikes=dropped,
            realtime_ratio=round(float(realtime_ratio), 4),
            source_tags=source_tags or {}, telemetry=telemetry or {},
        )

    # ------------------------------------------------------------ 计量
    def counters(self) -> dict[str, Any]:
        return {
            "messages_emitted": self.messages_emitted,
            "payload_bytes": self.payload_bytes,
            "memory_updates": self.memory_updates,
            "memory_bytes_in": self.memory_bytes_in,
            "retrieval_count": self.retrieval_count,
            "local_hits": self.local_hits,
            "central_hits": self.central_hits,
            "derived_submissions": self.derived_submissions,
        }

    def common_health(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "learning_mode": self.cfg.learning_mode,
            "memory_scope": self.cfg.memory_scope,
            "memory_version": self.cfg.memory_version,
            "memory": {**self.local.stats(), "working": self.working.summary()},
            "counters": self.counters(),
        }
