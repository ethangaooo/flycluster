# 开放 API 与运行手册

## 一、鉴权与接入管理

所有 `/v1/*`（除 `/v1/health`）都需要请求头 `X-API-Key`。

### 密钥来源有两种

1. **环境变量注入**（`source=env`）：由部署流程管理，页面上**只能吊销、不能删除**。
   删了之后重启会由环境变量重新注入，出现"页面显示已删除、实际仍可用"的矛盾。
   吊销状态落库，因此重启后依然有效。

2. **页面创建**（`source=db`）：在管理端「API 接入」页创建，可随时吊销。

### 环境变量格式

```
FLYCLUSTER_API_KEYS=name:key[:rpm[:scopes]]
```

| 段 | 含义 | 默认 |
|---|---|---|
| `name` | 名称，用于审计与用量归属 | 必填 |
| `key` | 密钥明文 | 必填 |
| `rpm` | 每分钟调用上限 | 120 |
| `scopes` | 作用域，逗号分隔：`run` / `read` / `admin` | `run,read,admin` |

示例：

```bash
# 单个开发密钥（本地联调）
FLYCLUSTER_API_KEYS=dev:dev-local-key:600

# 生产：给外部接入方一个只有读取权限、限 60/分的密钥
FLYCLUSTER_API_KEYS="backend:sk_xxx:600:admin,report:sk_yyy:60:read"
```

未配置时自动启用开发密钥 `dev-local-key`（上限 600/分，全作用域），仅用于本地联调。

### 作用域

| 作用域 | 允许 | 典型用途 |
|---|---|---|
| `run` | 提交运行、中止运行、写记忆快照、注册任务 | 业务系统下发任务 |
| `read` | 集群/运行/记忆的查询接口 | 看板、监控 |
| `admin` | 管理密钥与接入 | 运维 |

缺作用域会被拒绝：**403 缺少 xxx 作用域**。

### 密钥存储

- 数据库里**只存 SHA-256 哈希**，不存明文。库泄漏 ≠ 凭据泄漏。
- 明文**只在创建/轮换的那一刻返回一次**，之后无法再取回，只能重新轮换。
- 列表接口只返回前缀（如 `fc_9lmRbL…`），不返回明文，也不返回哈希。
- 每次调用累加 `calls_total` 并更新 `last_used_at`，管理页可查用量。

> 冒烟测试「接入管理：哈希存储与作用域隔离」会断言数据库文件中不含密钥明文，
> 并验证只读密钥不能提交运行、环境变量密钥不能删除。

## 二、接口一览

| 方法 | 路径 | 作用域 | 用途 |
|---|---|---|---|
| GET | `/v1/health` | — | 免鉴权存活检查 |
| GET | `/v1/connectomes` | read | 可用连接组清单（含**是否真实数据**、规模、许可） |
| GET | `/v1/cluster` | 集群模式、脑节点可达性与遥测、记忆统计 |
| POST | `/v1/tasks` | 注册任务；同 `task_id` 参数不同则 **409 拒绝覆盖** |
| GET | `/v1/tasks` | 列出已注册任务 |
| POST | `/v1/runs` | 创建并异步启动一次运行 |
| GET | `/v1/runs` | 运行列表（含 `config` / `score` / `gates`） |
| GET | `/v1/runs/{id}` | 单次运行状态与进度 |
| GET | `/v1/runs/{id}/events` | SSE 实时事件流 |
| POST | `/v1/runs/{id}/abort` | 中止 |
| GET | `/v1/runs/{id}/report` | 完整报告（未完成时 409） |
| GET | `/v1/memory/stats` | 记忆统计（Docker 模式下透明转发到记忆中枢） |
| GET | `/v1/memory/conflicts` | 冲突记录 |
| GET | `/v1/memory/provenance/{origin_event_id}` | 来源链追溯 |
| POST | `/v1/memory/snapshot` | 创建记忆快照（`?note=`） |
| GET | `/v1/analytics/runs` | read | 实验结果归一化扁平表（见下） |
| GET | `/v1/games/catalog` | read | 棋种 / 飞蝇结构 / 电脑难度 / 能力归因 |
| POST | `/v1/games/sessions` | run | 新建对局 |
| GET | `/v1/games/sessions/{id}` | read | 对局状态（棋盘几何 + 走子记录） |
| POST | `/v1/games/sessions/{id}/ply` | run | 走一个回合，返回含思考过程 |
| GET | `/v1/admin/endpoints` | 接口目录（由后端提供，保证与代码同步） |
| GET | `/v1/admin/keys` | 密钥列表（只返回前缀） |
| POST | `/v1/admin/keys` | 创建密钥；**明文只返回一次** |
| POST | `/v1/admin/keys/{key_id}/rotate` | 轮换；旧密钥立即失效 |
| DELETE | `/v1/admin/keys/{key_id}` | 吊销；环境变量密钥只能吊销不能删除 |

## 三、创建运行

```bash
curl -X POST http://127.0.0.1:8800/v1/runs \
  -H 'X-API-Key: dev-local-key' -H 'Content-Type: application/json' \
  -d '{
    "run": {
      "seed": 7,
      "backend": "brain",
      "connectome": "synthetic-2000",
      "agents": 4,
      "max_ticks": 120,
      "memory_scope": "project",
      "learning_mode": "plastic",
      "memory_version": "head",
      "enable_peer_messages": true,
      "enable_memory_channel": true,
      "enable_central_planner": false,
      "comm_latency_ticks": 1,
      "comm_loss": 0.1,
      "message_kind": "learned_compressed",
      "tags": { "tenant": "pit-an" },
      "callback_url": "https://example.com/flycluster/callback"
    },
    "task": {
      "task_id": "grid32-a4-t6",
      "grid": 32, "n_agents": 4, "n_targets": 6, "max_ticks": 120,
      "obstacle_fraction": 0.12, "perception_fraction": 1.0,
      "seed": 7, "env_version": "grid-v1", "visibility_regions": null
    }
  }'
```

### 三个必须显式配置的语义

| 字段 | 取值 | 行为 |
|---|---|---|
| `memory_scope` | `run` / `session` / `project` / `tenant` | 经验可见范围。**高于 `run` 时必须提供 `tags.tenant`，否则 400**。`scope=run` 且未传 `run_id` 的查询返回 0 条，禁止默认跨运行共享 |
| `learning_mode` | `readonly` / `accumulate` / `plastic` | 只读推理 / 积累经历 / 允许更新学习参数 |
| `memory_version` | `head` / `pinned` | 跟随最新快照，或按预先规定的快照接收更新 |

### 其他重要字段

| 字段 | 说明 |
|---|---|
| `backend` | `brain`（真实连接子图）/ `rewired`（保度重连）/ `reservoir` / `gru` / `classical` / `noop` |
| `enable_peer_messages` | 同伴通信开关（消融用） |
| `enable_memory_channel` | 中心记忆通道开关（消融用） |
| `enable_central_planner` | 中心规划器。**开启后能力必须归因于组合系统** |
| `comm_latency_ticks` / `comm_loss` | 消息延迟与丢失（鲁棒性场景） |
| `k_local` / `k_central` | 本地 / 中心经验检索条数；`0` 表示关闭该路径检索 |
| `exploration_epsilon` | 决策时的随机探索比例，全部神经模型共享同一取值 |
| `message_kind` | `learned_compressed`（脑输出组脉冲）/ `shared_observation`（外围代码压缩观测，**收益不得归因于脑结构**） |
| `callback_url` | 运行完成后 POST 报告 |

### 感知失效只能从任务侧配

**没有** `RunConfig.perception_dropout`。感知失效是**环境属性**，只在 `task` 里配：

```json
"task": { "perception_fraction": 0.5, ... }
```

注意语义：`perception_fraction` 是**感知有效的 tick 占比**，不是失效率。
`0.5` 表示约一半的 tick 拿不到信息。分条件扫描里轴值 `v` 是失效率，映射为 `1 - v`。

失效时**零掉的是外感受的 5×5 视野四通道，保留本体感觉（自身 x/y/朝向）与时间进度**——
相当于"闭眼"，不是失去对自己身体的感知。脑节点不得再对观测向量做二次解释
（早期版本在脑侧又整体清零了一次，把环境刻意保留的本体感觉也抹掉了）。

> 为什么删掉了 `perception_dropout`：它曾被声明在 `RunConfig` 上却**从未被 runner 读取**。
> 配置项存在但没接通，比没有这个配置更危险——实验会静默跑成满感知，
> 而报告里写的却是"感知失效 50% 条件下"。两个来源必然静默漂移，所以只保留一个。


## 四、Python 客户端示例

```python
import httpx, time

H = {"X-API-Key": "dev-local-key"}
r = httpx.post("http://127.0.0.1:8800/v1/runs", headers=H, json={...}, timeout=30)
run_id = r.json()["run_id"]

while True:
    st = httpx.get(f"http://127.0.0.1:8800/v1/runs/{run_id}", headers=H).json()
    if st["status"] in ("finished", "failed"):
        break
    time.sleep(1)

rep = httpx.get(f"http://127.0.0.1:8800/v1/runs/{run_id}/report", headers=H).json()
print(rep["score"], rep["telemetry"], [g["gate"] for g in rep["gates"]])
print(rep["attribution"])   # 哪些能力来自外围代码
```

## 四之补、结果分析接口

`GET /v1/analytics/runs` 把两种来源压成**同一张扁平表**，供管理端「结果分析」页按任意维度切片：

| 来源 | 说明 |
|---|---|
| `artifact` | 命令行产物 `artifacts/*.json`（`main-*` 主对照、`scan-*` 分条件扫描） |
| `live` | 控制台提交的运行 |

返回结构：

```
rows          扁平行（每个"组 × 种子"一行）
n_rows        行数
artifacts     产物清单：文件、类型、连接组、行数、是否可追溯、备注
errors        解析失败的产物（不会让整个接口挂掉）
untraceable_artifacts  未记录连接组的产物文件名
dimensions    14 个可用维度 + 各自实际出现过的取值
metrics       可用指标 + 标签 / 单位 / 方向（越大越好 or 越小越好）
```

**14 个维度**：来源、实验类型、产物、连接组、实验组、中间结构、同伴通信、中心记忆、
随机种子、扫描轴、条件值、扫描阶段、节点数、步数上限。
**15 个指标**：得分 S、命中、覆盖率、碰撞、重复探索、耗时、实时比、吞吐、仿真步数、
脉冲数、丢弃率、NaN、脑间带宽、发出消息数、丢弃消息数。

指标定义集中在 `services/orchestrator/analytics.py` 的 `METRICS`，带 `direction` 字段
（越大越好 / 越小越好），前端据此决定排序方向与配色——**不在页面上另写一份判断**。

### 产物必须可追溯

一个实验结果必须说得出跑在哪个连接组上。历史遗留问题：`scan-*.json` 早期**完全没有记录连接组**，
`main-*.json` 的 `protocol.task` 是 `TaskSpec` 的 repr 字符串（dataclass 没做 `asdict` 就被 `json` 转成字符串），
读端一 `.get()` 就崩。

生产者已修（`experiments/run.py` 与 `scan.py` 都显式 `dataclasses.asdict`）。
**旧产物无法追溯，接口会如实标注 `traceable: false`**，页面也提供"只看可追溯"开关——
而不是静默当作有效数据参与统计。

## 四之补二、对弈演示接口

管理端「果蝇对弈」页（`/games`）让连接组派生的脉冲网络集群与电脑对手下
**五子棋 / 中国象棋 / 跳棋**，并可视化它的神经活动。

### 脑怎么下棋（不改核心引擎）

`BrainEngine` 的观测维度（105）与动作数（5）是硬编码的，改造它会动到已验证的实验结论。
所以用**逐候选评估**：

```
对每个合法着法 → 各编码一份 105 维观测 → 同一个脑实例跑一次决策步（内部 120 次网络步进）
→ 取读出组脉冲计数 → 固定随机投影压成标量"倾向分" → 取最高者落子
```

每个候选都从**逐位相同**的初始状态出发（连随机数发生器序列一起重置）。
否则候选之间除了观测还有噪声差异，评估热力图就成了噪声图。

### 一个必须知道的实测差异

**同一个局面下 32 个候选：**

| 仿真窗口 | 网络总放电 | 候选间有差异的个数 |
|---|---|---|
| 20 步（主任务口径） | 5 次 | **0 / 32** |
| 60 步 | 71 次 | 0 / 32 |
| **120 步（本页默认）** | **457 次** | **30 / 32** |
| 240 步 | 1136 次 | 28 / 32（4.3 s，太慢） |

也就是说：**这个连接组网络在 20 ms 的窗口内太稀疏，分辨不出一盘棋的不同落点。**
页面用 120 步窗口并在界面上照实标注，而不是调参掩盖。这是模型的真实属性。

### 飞蝇侧的四种结构与两个基线

| 结构 | 用途 |
|---|---|
| 真实连接组 | 受检验对象 |
| 保度重连对照 | 区分「结构」与「规模」的贡献 |
| 随机储备池 | 同规模的普通时序网络 |
| 纯随机基线 | 在合法着法里均匀随机——判断结构有没有用的**下界** |
| 外围启发式基线 | 直接用棋类评估函数选点，不经过脑——**上界参考** |

### 能力归因（接口返回 `attribution` 字段，页面直接展示）

**脑负责**：脉冲动力系统。
**外围代码负责**：棋盘→105 维观测的编码、候选剪枝、读出向量→倾向分的固定随机投影、
胜负判定与棋规、电脑对手的 alpha-beta 搜索。

> 这是**演示**，不是在主张果蝇具备棋类认知。按项目分类属于「演示有趣但指标无收益」那一类。

## 五、运行手册

### 5.1 单机模式（无 Docker）

```bash
make setup                # 建 venv 装依赖
make smoke                # 13 项不变量检查，必须先过
make local-brain          # 起中枢 + 控制台 + 开放 API  → http://127.0.0.1:8800
```

脑节点在调度器进程内直接构造，适合算法迭代。实时比与丢弃遥测同样被记录。

### 5.2 Docker 多容器模式

```bash
make admin-build          # 先构建控制台到 admin-ui/dist（compose 会只读挂载）
cp .env.example .env      # 按需修改密钥与连接组
docker compose up --build -d
```

#### 三个必须知道的构建事实

| 事项 | 说明 |
|---|---|
| **必须有 `.dockerignore`** | 否则构建上下文会把 `.venv`、`admin-ui/node_modules`、`artifacts/`、`data/` 全发给守护进程。实测未加时上下文 **1.3 GB**、构建 25 分钟仍未完成；加上后只有 **308 KB**、构建 **16 秒** |
| **国内网络要覆盖 pip 源** | 官方 PyPI 可能 TLS 超时。用环境变量覆盖，**不改文件**：<br>`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple docker compose up --build -d` |
| **基础镜像拉不动时** | Docker Hub 常见 TLS 握手超时。可先从可用镜像源拉取并打官方 tag：<br>`docker pull <mirror>/library/python:3.12-slim`<br>`docker tag <mirror>/library/python:3.12-slim python:3.12-slim` |

仓库里**不硬编码**任何第三方镜像地址，避免把环境假设固化进代码。

#### 在容器里使用真实连接组

脑容器的**连接组目录（只读）**与**运行数据目录（可写，放本地记忆）**是分开的两件事：

| 环境变量 | 默认 | 用途 |
|---|---|---|
| `BRAIN_CONNECTOME_DIR` | `DATA_DIR` | 只读，查找 `.npz`；compose 已把 `./data` 以只读方式挂到 `/connectomes` |
| `DATA_DIR` | `data` | 可写，存放本地记忆 JSONL |

混用会导致"真实连接组挂不进容器"——`/app/data` 是卷，不能只读覆盖。

```bash
# 让整个集群切换到真实连接组
BRAIN_CONNECTOME=male-cns-gateway docker compose up -d
```

验证各节点实际加载了什么（比逐个 `docker compose exec` 快得多，也顺带确认调度器能连上）：

```bash
curl -s -H 'X-API-Key: dev-local-key' http://127.0.0.1:8800/v1/cluster \
  | python3 -c "import sys,json;[print(n['node_id'], (n.get('health') or {}).get('connectome',{}).get('name')) for n in json.load(sys.stdin)['nodes']]"
```

#### 运行级设置如何下发到远端脑容器

脑容器启动时只读环境变量，**不知道"这一次运行"要不要用记忆、要不要学习**。
因此调度器在每次运行开始时会调用每个容器的 `POST /configure`，推送：

```
learning_mode / memory_scope / k_local / k_central /
local_memory_enabled / central_memory_enabled /
enable_peer_input / enable_memory_input / message_kind
```

**缺了这一步会静默出错**：实测过一次 16 节点 Docker 运行，请求里写着
`enable_memory_channel: false`，逐节点却仍报 `central_hits=234`、合计检索 6.1 MB——
消融开关根本没生效，而且不报错。修复后同一请求的 `central_hits` 为 0，
而本地记忆 `local_hits=2560` 照常工作。

报告里的 `comm_accounting.node_settings_pushed` 会回放**实际生效**的设置，
用来核对"请求的配置"与"节点上的配置"是否一致。冒烟测试「集群：通信消融真实生效」
会断言这个回报里 `central_memory_enabled` 与请求一致。

#### 脑节点规模

默认 **16 个独立脑容器**（`brain-1` … `brain-16`，`BRAIN_NODE_ID = node-0..node-15`，
种子 0–15），加上记忆中枢、环境服务、中枢调度共 **19 个容器**。

**为什么不用 `--scale brain=16`**：副本共用一个 DNS 名，调度器侧的多个 agent
可能被轮询到同一个容器，从而共享膜电位与本地记忆——"每容器一个独立实例"这个前提
就不再成立。所以每个脑节点都是一个显式服务。

验证 16 个实例确实互相独立（唯一 `node_id` 数应等于节点数）：

```bash
curl -s -H 'X-API-Key: dev-local-key' http://127.0.0.1:8800/v1/cluster \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
seen = {(n.get('health') or {}).get('node_id') for n in d['nodes']}
print('节点数', len(d['nodes']), '可达', sum(n['reachable'] for n in d['nodes']),
      '唯一 node_id', len(seen))
"
```

#### 分布式模式下的通信预算要重新看

Docker 模式下**中心记忆的检索字节会明显超过脑间消息字节**。实测一次 4 节点 / 120 步的运行：

| 通道 | 量 |
|---|---|
| 脑间同伴消息 | 320 B/s（480 条发出、1292 条送达） |
| 中心记忆检索返回 | **约 7.0 MB**（约 292 kB/s） |

相差约三个数量级。这正是文档反复强调"中心记忆本身是一条通信通道"的原因——
如果只统计脑间消息就宣称"低带宽协作"，结论是不成立的。

启动后：

| 地址 | 内容 |
|---|---|
| http://localhost:8800 | 控制台（由调度器托管 dist） |
| http://localhost:8800/v1/health | 开放 API |
| http://localhost:8100/health | 记忆中枢 |
| http://localhost:8200/health | 环境服务 |

**持久化**：`memory-data` 卷承载记忆数据库，容器重建不丢记忆。
容器可写层会随容器删除消失——"记忆容器重建后还能恢复记忆"靠的是卷，不是容器名不变。

### 5.3 硬件在环（通过预研之后）

```bash
FLYCLUSTER_BRAIN_URLS=http://board-1:8000,http://board-2:8000 make local-brain
```

电脑负责环境与日志，开发板各自运行控制器，**电脑只转发消息，不替开发板计算动作**。
成本报告要分别列出"开发板消耗"与"包含主机在内的整个实验系统消耗"，两种口径不能混用。

### 5.4 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `401 缺少或无效的 X-API-Key` | 未设置密钥或与 `FLYCLUSTER_API_KEYS` 不一致 | 控制台右下角"设置 API Key" |
| `429 超过每分钟调用上限` | 触发限流 | 提高 `FLYCLUSTER_API_KEYS` 里的上限 |
| `409 run_id 已存在` | 批跑脚本复用了 `run_id` | 加时间戳后缀 |
| `400 memory_scope 高于 run 时必须提供 tags.tenant` | 见"三个必须显式配置的语义" | 补 `tags.tenant` |
| `未知记忆快照` | `memory_version=pinned` 但快照不存在 | 先 `POST /v1/memory/snapshot` |
| 门 1 实时比不达标 | 连接组规模过大 | 降低 `neuron_count` 或提高 `synaptic_scale` 前的规模；**不要**靠关闭丢弃计数"过门" |
| 所有组得分都为 0 | 任务对当前步数预算不可解 | 增大 `max_ticks`、减小 `grid`、或降低 `n_targets`；先确认经典策略能拿到分 |
| 重连很慢 | 保度交换是 O(swap_multiplier × m) | 降低 `swap_multiplier`（但会减弱对照强度，需在报告里说明） |

### 5.5 发布前检查清单

- [ ] `make smoke` 全绿（13/13）
- [ ] `make admin-build` 通过，控制台能打开并正常拉取数据
- [ ] 用到的连接组在 `ConnectomeMeta` 里记录了许可与来源
- [ ] 报告里的性能数字都配有实时比 / 丢弃率 / NaN
- [ ] `attribution` 字段如实标注了外围代码的贡献
- [ ] 结论措辞与样本量匹配（3 种子只说"筛查"，≥10 种子才谈"结论"）
- [ ] 外部密钥未出现在代码、日志、报告与版本库中
