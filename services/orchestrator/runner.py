"""中枢调度：共同环境 + 多脑实例 + 消息路由 + 经验汇集 + 评分与三道门。

编排器只做四件事：推进环境、分发观测、路由消息、记账与评分。
它**不替脑节点计算动作**——除非显式打开 `enable_central_planner`，
而那种情况下的能力必须归因于"脑集群 + 中心规划器的组合系统"。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from flycore.schemas import (MSG_VALUES, N_ACTIONS, BrainStepRequest, GateResult, MessagePacket,
                             ObservationPacket, RunConfig, RunReport)
from flycore.scoring import compute_score, evaluate_gates
from flycore.util import now_iso, rng_for
from services.brain.hub_client import InProcessMemoryBackend
from services.env.world import GridWorld, TaskSpec

from .backends import (InProcessNodeBackend, build_http_nodes, build_inprocess_nodes, local_hub)

REWARD_TARGET = 1.0
REWARD_COLLISION = -0.35
REWARD_REDUNDANT = -0.10
REWARD_NOVEL_CELL = 0.20
# 新颖格奖励：任务本身就惩罚"重复探索"，因此把"进入新格"作为正向信号是与任务一致的
# 塑形，而不是偷偷替模型解题。没有这一项，奖励几乎只在碰撞与原地打转时出现，
# 未训练的读出器拿不到任何可用的信用分配，所有神经模型都会退化成"极少数动作"。

ACTION_NAMES = ("up", "down", "left", "right", "stay")


@dataclass
class RunnerOptions:
    data_dir: str = "data"
    brain_urls: list[str] | None = None
    steps_per_tick: int = 20
    dynamics_overrides: dict[str, Any] | None = None
    on_event: Callable[[dict[str, Any]], None] | None = None
    abort_flag: threading.Event | None = field(default=None)
    # 复用已有引擎实例：实验 A/B/C 需要跨 run 保留脑内状态与学习参数
    connectome_dir: str = field(
        default="data",
        metadata={"doc": "连接组 .npz 所在目录（只读），与 data_dir（运行产物）分开"},
    )
    reuse_nodes: list[Any] | None = field(default=None)


class ClusterRunner:
    def __init__(self, run: RunConfig, task: TaskSpec, opts: RunnerOptions | None = None):
        self.run = run
        self.task = task
        self.opts = opts or RunnerOptions()
        self.run_id = run.run_id or f"run-{run.seed}-{now_iso()}"
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 内部
    def _emit(self, ev: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(ev)
            if len(self.events) > 4000:
                del self.events[:1000]
        if self.opts.on_event is not None:
            try:
                self.opts.on_event(ev)
            except Exception:
                pass

    def _central_plan(self, env: GridWorld, agent_id: int) -> list[int] | None:
        """中心规划器：直接给出朝最近目标的建议动作。

        研究模式下默认关闭。开启时**必须**在报告里把能力归因于组合系统，
        不能算作脑间自主涌现。
        """
        if not self.run.enable_central_planner or not env.targets:
            return None
        a = env.agents[agent_id]
        tx, ty = min(env.targets, key=lambda t: abs(t[0] - a.x) + abs(t[1] - a.y))
        dx, dy = tx - a.x, ty - a.y
        if abs(dx) >= abs(dy):
            act = 3 if dx > 0 else (2 if dx < 0 else 4)
        else:
            act = 1 if dy > 0 else (0 if dy < 0 else 4)
        vals = [0.0] * MSG_VALUES
        vals[act] = 1.0
        return [int(v * 32767) for v in vals]

    def _route(self, packets: list[MessagePacket], queue: list[tuple[int, MessagePacket]],
               dropped: list[int], tick: int) -> None:
        rng = rng_for("comm-loss", self.run.seed, tick)
        for p in packets:
            if self.run.comm_loss > 0 and rng.random() < self.run.comm_loss:
                dropped[0] += 1
                continue
            queue.append((tick + max(0, self.run.comm_latency_ticks), p))

    # ------------------------------------------------------------------ 主流程
    def run_cluster(self) -> RunReport:
        run, opts = self.run, self.opts
        env = GridWorld(self.task)
        agents = self.task.n_agents

        hub_service = None
        hub = None
        if run.enable_memory_channel:
            hub_service = local_hub(opts.data_dir)
            hub = InProcessMemoryBackend(hub_service)

        if opts.reuse_nodes is not None:
            if len(opts.reuse_nodes) < agents:
                raise ValueError(f"reuse_nodes 需要 {agents} 个实例，只给了 {len(opts.reuse_nodes)} 个")
            nodes = [InProcessNodeBackend(f"node-{i}", opts.reuse_nodes[i]) for i in range(agents)]
            for n in nodes:
                n.set_message_kind(run.message_kind)
        elif opts.brain_urls:
            nodes = build_http_nodes(run.backend, agents, opts.brain_urls)
            for n in nodes:
                n.set_message_kind(run.message_kind)
        else:
            nodes = build_inprocess_nodes(
                run.backend, agents, run.connectome, run.seed, hub, run.message_kind,
                opts.steps_per_tick, opts.data_dir, run.learning_mode, run.memory_scope,
                opts.dynamics_overrides, k_local=run.k_local, k_central=run.k_central,
                exploration_epsilon=run.exploration_epsilon,
                central_memory_enabled=run.enable_memory_channel,
                connectome_dir=opts.connectome_dir,
                run_id=self.run_id,
            )

        # 运行级设置统一下发。这里必须覆盖三种后端：
        #   - 复用实例：构造时已按旧配置建好，需要显式重置
        #   - HTTP 远端容器：容器只看得到环境变量，不知道"这一次运行"要不要用记忆
        #   - 进程内新建：构造时已按 run 配好，configure 只是回报实际生效值
        # 漏掉任何一条，消融开关就会被静默忽略——不报错，但结果已经错了。
        settings = {
            "learning_mode": run.learning_mode,
            "memory_scope": run.memory_scope,
            "k_local": run.k_local,
            "k_central": run.k_central,
            "local_memory_enabled": True,
            "central_memory_enabled": run.enable_memory_channel,
            "enable_peer_input": run.enable_peer_messages,
            "enable_memory_input": run.enable_memory_channel,
            "message_kind": run.message_kind,
        }
        pushed: list[dict[str, Any]] = []
        for n in nodes:
            n.set_message_kind(run.message_kind)
            # 先 reset 再 configure：
            #   reset —— 清掉上一次运行残留在容器里的累计遥测与本地记忆
            #   configure —— 推送本次运行的开关
            # 顺序反了会让"本次运行"的统计里混进上一次的数据。
            # 直接调用，不用 getattr 兜底：协议里声明了这两个方法，
            # 缺实现就该立刻 AttributeError。用 getattr(..., None) + callable 判断
            # 会把"方法没实现"静默变成"跳过"——曾经就因为 HttpNodeBackend 漏了 reset，
            # 导致远端容器从未被重置，而报告看起来一切正常。
            n.reset(self.run_id)
            pushed.append(n.configure(settings))

        queue: list[tuple[int, MessagePacket]] = []
        inbox: list[list[MessagePacket]] = [[] for _ in range(agents)]
        last_reward = [0.0] * agents
        last_outcome: list[dict[str, Any]] = [{} for _ in range(agents)]
        msg_produced = 0
        msg_issued = 0
        msg_delivered = 0
        dropped = [0]
        per_agent: list[dict[str, Any]] = [{"agent_id": i, "actions": [], "targets_found": 0,
                                            "collisions": 0, "redundant_steps": 0} for i in range(agents)]

        self._emit({"type": "run_started", "run_id": self.run_id, "backend": run.backend,
                    "task_id": self.task.resolved_id(), "agents": agents,
                    "max_ticks": self.task.max_ticks})

        aborted = False
        with ThreadPoolExecutor(max_workers=max(1, agents)) as pool:
            for tick in range(1, self.task.max_ticks + 1):
                if opts.abort_flag is not None and opts.abort_flag.is_set():
                    aborted = True
                    break

                due = [p for (t, p) in queue if t <= tick]
                queue[:] = [(t, p) for (t, p) in queue if t > tick]
                for p in due:
                    inbox[p.dst_agent].append(p)
                    msg_delivered += 1

                requests: list[BrainStepRequest] = []
                for i in range(agents):
                    obs, valid = env.observe(i)
                    msgs = list(inbox[i])
                    inbox[i].clear()
                    plan = self._central_plan(env, i)
                    if plan is not None:
                        msgs = msgs + [MessagePacket(src_agent=-1, dst_agent=i, tick=tick,
                                                     issued_tick=tick, values=plan,
                                                     kind="shared_observation")]
                    a = env.agents[i]
                    requests.append(BrainStepRequest(
                        node_id=nodes[i].node_id, tick=tick,
                        observation=ObservationPacket(agent_id=i, tick=tick,
                                                      vector=obs.tolist(),
                                                      perception_valid=valid),
                        peer_messages=msgs,
                        run_context={
                            "run_id": self.run_id, "task_id": self.task.resolved_id(),
                            "env_version": self.task.env_version, "episode": 0,
                            "last_reward": last_reward[i], "last_outcome": last_outcome[i],
                            "self_cell": [a.x, a.y], "grid": self.task.grid,
                            "agents": agents, "decision_hz": self.run.message_rate_hz,
                            "memory_version": "", "tenant": run.tags.get("tenant", "default"),
                        },
                    ))

                futs = [pool.submit(n.step, r) for n, r in zip(nodes, requests)]
                responses = []
                for f in futs:
                    try:
                        responses.append(f.result())
                    except Exception as exc:
                        self._emit({"type": "node_error", "tick": tick,
                                    "error": f"{type(exc).__name__}: {exc}"})
                        responses.append(None)

                actions: list[int] = []
                packets: list[MessagePacket] = []
                for i, resp in enumerate(responses):
                    act = resp.action if resp is not None else 4
                    actions.append(int(np.clip(act, 0, N_ACTIONS - 1)))
                    per_agent[i]["actions"].append(int(act))
                    if resp is not None and resp.outgoing_message:
                        msg_produced += 1
                        # 通信消融开关必须在**投递环节**生效：关掉通信时消息不进入通道，
                        # 节点也就收不到任何东西。早期版本漏了这一步，导致 comm/nocomm
                        # 两组分数逐位相同——消融看似做了，实际什么都没变。
                        if not run.enable_peer_messages:
                            continue
                        msg_issued += 1
                        for j in range(agents):
                            if j == i:
                                continue
                            packets.append(MessagePacket(
                                src_agent=i, dst_agent=j, tick=tick, issued_tick=tick,
                                values=resp.outgoing_message, kind=run.message_kind))
                self._route(packets, queue, dropped, tick)

                before = [(a.targets_found, a.collisions, a.redundant_steps) for a in env.agents]
                info = env.step(actions)
                for i, a in enumerate(env.agents):
                    dt = a.targets_found - before[i][0]
                    dc = a.collisions - before[i][1]
                    dr = a.redundant_steps - before[i][2]
                    novel = (not dr) and (not dt)
                    last_reward[i] = (REWARD_TARGET * dt + REWARD_COLLISION * dc
                                      + REWARD_REDUNDANT * dr
                                      + (REWARD_NOVEL_CELL if novel else 0.0))
                    last_outcome[i] = {"target_found": bool(dt), "collided": bool(dc),
                                       "redundant": bool(dr), "novel_cell": bool(novel)}
                    per_agent[i]["targets_found"] += dt
                    per_agent[i]["collisions"] += dc
                    per_agent[i]["redundant_steps"] += dr

                if tick % 5 == 0 or info["targets_found_now"]:
                    self._emit({
                        "type": "tick", "tick": tick,
                        "targets_found": self.task.n_targets - len(env.targets),
                        "targets_total": self.task.n_targets,
                        "coverage": round(env.coverage(), 4),
                        "messages": msg_issued, "msg_dropped": dropped[0],
                        "node_realtime_ratio": [
                            round(float(r.realtime_ratio), 3) for r in responses if r is not None],
                        "sim_steps_per_second": [
                            (r.telemetry or {}).get("steps_per_second", 0.0) for r in responses
                            if r is not None],
                    })

        # ------------------------------------------------------------ 汇总
        sb = compute_score(
            targets_found=self.task.n_targets - len(env.targets),
            targets_total=self.task.n_targets,
            collisions=sum(a.collisions for a in env.agents),
            redundant_steps=sum(a.redundant_steps for a in env.agents),
            coverage=env.coverage(), steps_used=env.tick,
        )

        healths = []
        for n in nodes:
            try:
                healths.append(n.health())
            except Exception:
                healths.append({})

        telemetry = self._aggregate_telemetry(healths, env)
        duration_s = max(1e-6, env.tick / max(1e-6, run.message_rate_hz))
        comm = {
            "agents": agents,
            "node_settings_pushed": pushed[:1] if pushed else [],
            "channel_enabled": run.enable_peer_messages,
            "messages_produced_by_nodes": msg_produced,
            "messages_issued": msg_issued,
            "messages_delivered": msg_delivered,
            "messages_dropped": dropped[0],
            "values_per_message": MSG_VALUES,
            "bytes_per_message": MSG_VALUES * 2,
            "raw_send_bytes": msg_issued * MSG_VALUES * 2,
            "delivered_bytes": msg_delivered * MSG_VALUES * 2,
            "decision_hz": run.message_rate_hz,
            "duration_seconds": round(duration_s, 4),
            "raw_bytes_per_second": round(msg_issued * MSG_VALUES * 2 / duration_s, 2),
            "formula_bytes_per_second": round(agents * MSG_VALUES * 2 * run.message_rate_hz, 2),
            "note": ("formula_bytes_per_second 为文档口径的理论值（n×8×2×频率），不含报头、"
                     "重传、以及发给多个邻居造成的重复传输；raw_bytes_per_second 为本次实测发出量。"
                     "messages_produced_by_nodes 是节点算出来但未进入通道的消息数"
                     "（关闭通信时该值 > 0 而 messages_issued = 0，正是消融生效的证据）。"),
        }

        memory = {
            "channel_enabled": run.enable_memory_channel,
            "scope": run.memory_scope,
            "version_mode": run.memory_version,
            "per_node": [{k: h.get("counters", {}).get(k) for k in
                          ("memory_updates", "memory_bytes_in", "retrieval_count",
                           "local_hits", "central_hits", "derived_submissions")}
                         for h in healths],
        }
        if hub_service is not None:
            memory["hub"] = hub_service.stats()
            memory["note"] = ("中心记忆本身是一条通信通道：memory_bytes_in 必须与脑间消息一并计入"
                              "通信预算，否则'低带宽协作'会因为共享数据库而失真。")

        attribution = {
            "动作选择": f"外围读出器（可训练，N_ACTIONS×{healths[0].get('trainable_params', 0) if healths else 0} 参数）",
            "感觉编码": "固定随机投影（外围代码，不参与学习）",
            "消息生成": ("脑模型消息输出组脉冲（learned_compressed）"
                         if run.message_kind == "learned_compressed"
                         else "外围代码直接压缩观测（shared_observation）——协作收益不得归因于脑结构"),
            "记忆检索与编码": "本地 + 中心记忆检索，经固定适配器注入独立神经元组",
            "中间计算": ("连接组派生结构（含保度重连对照）" if run.backend in ("brain", "rewired")
                         else f"对照模型：{run.backend}"),
            "中心规划器": ("已开启：动作建议经同伴消息通道注入，能力归属为「脑集群 + 中心规划器」组合系统"
                           if run.enable_central_planner else "未开启（仅提供检索）"),
        }

        report = RunReport(
            run_id=self.run_id, task_id=self.task.resolved_id(),
            backend=run.backend, seed=run.seed,
            config=run.model_dump(),
            score=sb, per_agent=[
                {**p, "action_histogram": self._hist(p["actions"])} for p in per_agent],
            telemetry=telemetry, comm_accounting=comm, memory_accounting=memory,
            trainable_params=int((healths[0].get("trainable_params") or 0) if healths else 0),
            output_dim=int((healths[0].get("output_dim")
                            or (healths[0].get("groups") or {}).get("action_out", 0)) if healths else 0),
            attribution=attribution, finished_at=now_iso(),
            env_snapshot=env.render_grid(),
        )
        gates = evaluate_gates(report.model_dump())
        report.gates = [GateResult(**g) for g in gates]
        if aborted:
            self._emit({"type": "run_aborted", "run_id": self.run_id, "tick": env.tick})
        else:
            self._emit({"type": "run_finished", "run_id": self.run_id, "score": sb.score,
                        "targets_found": sb.targets_found, "coverage": sb.coverage})
        if hub_service is not None:
            hub_service.close()
        return report

    # ------------------------------------------------------------------ 工具
    def _aggregate_telemetry(self, healths: list[dict[str, Any]], env: GridWorld) -> dict[str, Any]:
        sims = [h.get("simulation", {}) for h in healths if h.get("simulation")]
        if not sims:
            return {"note": "本次全部为对照模型，无脉冲仿真遥测"}
        steps = sum(int(s.get("steps", 0)) for s in sims)
        spikes = sum(int(s.get("spikes", 0)) for s in sims)
        dropped = sum(int(s.get("dropped_spikes", 0)) for s in sims)
        nan = sum(int(s.get("nan_events", 0)) for s in sims)
        saturated = sum(int(s.get("saturated_steps", 0)) for s in sims)
        return {
            "nodes": len(sims), "steps": steps, "spikes": spikes,
            "dropped_spikes": dropped,
            "drop_rate": round(dropped / max(1, spikes + dropped), 6),
            "nan_events": nan, "saturated_steps": saturated,
            "realtime_ratio_mean": round(float(np.mean([s.get("realtime_ratio", 0) for s in sims])), 4),
            "realtime_ratio_min": round(float(np.min([s.get("realtime_ratio", 0) for s in sims])), 4),
            "steps_per_second_mean": round(float(np.mean([s.get("steps_per_second", 0) for s in sims])), 2),
            "bounded": all(bool(h.get("bounded", h.get("simulation", {}).get("bounded", True)))
                           for h in healths),
            "env_ticks": env.tick,
        }

    @staticmethod
    def _hist(actions: list[int]) -> dict[str, int]:
        return {ACTION_NAMES[i]: int(sum(1 for a in actions if a == i)) for i in range(N_ACTIONS)}
