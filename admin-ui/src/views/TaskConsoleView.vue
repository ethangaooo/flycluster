<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api, streamRun } from '../api/client'
import RunReportPanel from '../components/RunReportPanel.vue'

const conns = ref([])
const connInfo = ref(null)
const cluster = ref(null)
const loading = ref(false)
const report = ref(null)
const live = ref(null)
const taskName = ref('')
let stopStream = null

const form = ref({
  backend: 'brain',
  connectome: 'male-cns-gateway',
  agents: 4,
  grid: 32,
  n_targets: 6,
  max_ticks: 200,
  seed: 7,
  obstacle_fraction: 0.12,
  perception_keep: 1.0,
  memory_scope: 'run',
  learning_mode: 'plastic',
  memory_version: 'head',
  k_local: 4,
  k_central: 6,
  enable_peer_messages: true,
  enable_memory_channel: false,
  enable_central_planner: false,
  comm_latency_ticks: 1,
  comm_loss: 0.1,
  message_kind: 'learned_compressed',
  tenant: 'default'
})

const BACKENDS = [
  { v: 'brain', l: '真实连接子图（脉冲网络）', hint: '受检验对象' },
  { v: 'rewired', l: '保度重连对照（脉冲网络）', hint: '同度序列的对照结构' },
  { v: 'reservoir', l: '随机储备池', hint: '普通时序模型' },
  { v: 'gru', l: '小型 GRU', hint: '普通时序模型' },
  { v: 'classical', l: '经典局部探索策略', hint: '零可训练参数的基线' },
  { v: 'noop', l: '无脑模型（仅接口与读出）', hint: '验证能力是否来自外围代码' }
]

// 预设模板：把常用的几套配置固化下来，避免每次手填出错。
const PRESETS = [
  {
    key: 'smoke', label: '链路自检', hint: '小规模、几秒出结果，只验证链路通不通',
    patch: { backend: 'classical', connectome: 'synthetic-500', agents: 2, grid: 16,
             n_targets: 3, max_ticks: 40, obstacle_fraction: 0.12, perception_keep: 1.0 }
  },
  {
    key: 'real-comm', label: '真实结构 + 通信', hint: '主实验组：真实连接子图、开同伴通信',
    patch: { backend: 'brain', connectome: 'male-cns-gateway', enable_peer_messages: true,
             enable_memory_channel: false, enable_central_planner: false }
  },
  {
    key: 'real-nocomm', label: '真实结构 − 通信', hint: '通信消融臂；与上一组配对才能算 Δ',
    patch: { backend: 'brain', connectome: 'male-cns-gateway', enable_peer_messages: false,
             enable_memory_channel: false, enable_central_planner: false }
  },
  {
    key: 'rewired-comm', label: '重连结构 + 通信', hint: '对照臂；Δ 的另一半由它提供',
    patch: { backend: 'rewired', connectome: 'male-cns-gateway', enable_peer_messages: true,
             enable_memory_channel: false, enable_central_planner: false }
  },
  {
    key: 'perception', label: '感知失效 50%', hint: '半数的 tick 拿不到外部信息',
    patch: { perception_keep: 0.5 }
  },
  {
    key: 'memory', label: '开启中心记忆', hint: '注意：检索字节会计入通信预算，量级远大于脑间消息',
    patch: { enable_memory_channel: true, k_central: 6 }
  }
]

const currentConn = computed(() => conns.value.find((c) => c.spec === form.value.connectome))
const isReal = computed(() => !!currentConn.value?.real)
const stepCount = computed(() => form.value.agents * form.value.max_ticks)

const canDispatch = computed(() =>
  !!form.value.connectome && form.value.agents >= 1 && form.value.max_ticks >= 1 && !loading.value)

function applyPreset(p) {
  Object.assign(form.value, p.patch)
  ElMessage.success(`已套用「${p.label}」`)
}

async function loadCluster() {
  try {
    cluster.value = await api.cluster()
  } catch (e) { /* 顶部状态条会统一提示 */ }
}

const nodePool = computed(() => (cluster.value?.nodes || []).filter((n) => n.reachable).length)

async function loadConns() {
  try {
    const d = await api.connectomes()
    conns.value = d.connectomes
    connInfo.value = d
    if (!conns.value.some((c) => c.spec === form.value.connectome)) {
      form.value.connectome = conns.value[0]?.spec || ''
    }
  } catch (e) {
    ElMessage.error(e.friendly)
  }
}

async function dispatch() {
  const f = form.value
  const runId = (taskName.value.trim() || `task-${f.backend}-s${f.seed}`)
    .replace(/[^\w\-.]/g, '-') + '-' + Date.now().toString(36).slice(-4)

  const realNote = isReal.value
    ? `真实连接组 ${f.connectome}`
    : `**合成图** ${f.connectome}（不是真实果蝇数据）`
  try {
    await ElMessageBox.confirm(
      `<div style="line-height:1.9;font-size:13px">
        即将下发：<b>${runId}</b><br/>
        结构：<b>${BACKENDS.find(b => b.v === f.backend)?.l}</b><br/>
        数据：${realNote}<br/>
        规模：${f.agents} 节点 × ${f.max_ticks} 步 × 每步 20 ms 仿真 ≈ <b>${stepCount.value}</b> 个决策步
       </div>`,
      '确认下发任务', { dangerouslyUseHTMLString: true, confirmButtonText: '下发', cancelButtonText: '再改改' })
  } catch { return }

  loading.value = true
  report.value = null
  live.value = { status: 'queued', tick: 0 }

  const payload = {
    run: {
      run_id: runId, task_id: runId, seed: f.seed, backend: f.backend,
      connectome: f.connectome, agents: f.agents, max_ticks: f.max_ticks,
      memory_scope: f.memory_scope, learning_mode: f.learning_mode,
      memory_version: f.memory_version, k_local: f.k_local, k_central: f.k_central,
      enable_peer_messages: f.enable_peer_messages,
      enable_memory_channel: f.enable_memory_channel,
      enable_central_planner: f.enable_central_planner,
      comm_latency_ticks: f.comm_latency_ticks, comm_loss: f.comm_loss,
      message_kind: f.message_kind,
      tags: { tenant: f.tenant, group: f.backend, protocol: 'console' }
    },
    task: {
      task_id: runId, grid: f.grid, n_agents: f.agents, n_targets: f.n_targets,
      max_ticks: f.max_ticks, obstacle_fraction: f.obstacle_fraction,
      perception_fraction: f.perception_keep, seed: f.seed,
      env_version: 'grid-v1', visibility_regions: null
    }
  }

  try {
    await api.createRun(payload)
    taskName.value = runId
    if (stopStream) stopStream()
    stopStream = streamRun(runId, (ev) => {
      if (ev.type === 'status') live.value = ev
      if (ev.type === 'report') {
        report.value = ev.report
        loading.value = false
        ElMessage.success('任务完成，结果已返回')
      }
      if (ev.type === 'error') {
        live.value = { ...live.value, status: 'failed', error: ev.message }
        loading.value = false
        ElMessage.error(ev.message)
      }
    })
  } catch (e) {
    loading.value = false
    live.value = { status: 'failed', error: e.friendly }
    ElMessage.error(e.friendly)
  }
}

const progressPct = computed(() => {
  const t = live.value?.tick || 0
  const total = form.value.max_ticks || 1
  return Math.min(100, Math.round((t / total) * 100))
})

onMounted(() => { loadConns(); loadCluster() })
onUnmounted(() => stopStream && stopStream())
</script>

<template>
  <div class="fc-page">
    <h2 class="fc-page-title">任务下发</h2>
    <p class="fc-page-sub">
      给集群布置一个搜索任务，完成后在原位返回结果。每个节点只看到 5×5 的局部视野，
      拿不到全局地图；能交换的只有每条 8×int16 的消息。
    </p>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title">
        <el-icon><MagicStick /></el-icon> 预设模板
        <span class="fc-hint">一键填好常用配置，避免每次手填出错</span>
      </div>
      <div style="display: flex; gap: 8px; flex-wrap: wrap">
        <el-button v-for="p in PRESETS" :key="p.key" size="small" @click="applyPreset(p)">
          {{ p.label }}
        </el-button>
      </div>
      <div style="margin-top: 8px; font-size: 11.5px; color: var(--fc-text-faint)">
        {{ PRESETS.map(p => `${p.label}：${p.hint}`).join('　·　') }}
      </div>
    </div>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title">
        <el-icon><Histogram /></el-icon> 数据来源
        <span class="fc-hint">这张卡决定结论算不算数</span>
      </div>
      <div style="display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap">
        <div style="min-width: 300px">
          <el-select v-model="form.connectome" style="width: 100%">
            <el-option-group label="真实连接组（可用作结论依据）">
              <el-option v-for="c in conns.filter(x => x.real)" :key="c.spec"
                         :label="`${c.spec}　${c.neuron_count} 神经元 / ${c.edge_count} 边`"
                         :value="c.spec" />
            </el-option-group>
            <el-option-group label="合成图（只能打通链路，不能下结论）">
              <el-option v-for="c in conns.filter(x => !x.real)" :key="c.spec"
                         :label="`${c.spec}　${c.neuron_count} 神经元`" :value="c.spec" />
            </el-option-group>
          </el-select>
        </div>
        <div v-if="currentConn" style="flex: 1; min-width: 320px">
          <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 8px">
            <span class="fc-tag" :class="isReal ? 'ok' : 'warn'">
              {{ isReal ? '真实数据' : '合成数据' }}
            </span>
            <span class="fc-tag">许可 {{ currentConn.license }}</span>
            <span v-if="currentConn.rewired_available" class="fc-tag">可派生重连对照</span>
          </div>
          <div v-if="isReal" class="fc-num" style="font-size: 11.5px; color: var(--fc-text-dim); line-height: 1.85">
            神经元 {{ currentConn.neuron_count }}　边 {{ currentConn.edge_count }}　
            平均入度 {{ currentConn.mean_indegree }}<br />
            最大强连通分量 {{ (currentConn.largest_scc_fraction * 100).toFixed(1) }}%　
            抑制性 {{ (currentConn.inhibitory_fraction * 100).toFixed(1) }}%　
            输入 {{ currentConn.input_count }} / 输出 {{ currentConn.output_count }}<br />
            来源 {{ currentConn.source }}　
            <a v-if="currentConn.license_url" :href="currentConn.license_url" target="_blank"
               style="color: var(--fc-accent)">{{ currentConn.license_url }}</a>
          </div>
          <div v-else style="font-size: 12px; color: var(--fc-warn); line-height: 1.8">
            合成图只在拓扑统计量上对齐果蝇连接组，<b>不是任何真实果蝇数据</b>。
            用它跑出的差异不能用来支持"真实结构有独特优势"之类的结论。
            真实结构与合成图上的 Δ 符号曾经相反。
          </div>
          <div v-if="currentConn.notes" style="margin-top: 8px; font-size: 11px; color: var(--fc-text-faint); line-height: 1.7">
            {{ currentConn.notes }}
          </div>
        </div>
      </div>
    </div>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title">
        <el-icon><Setting /></el-icon> 任务与集群配置
        <div style="flex: 1"></div>
        <span v-if="cluster" class="fc-tag" :class="nodePool ? 'ok' : 'bad'">
          集群 {{ cluster.mode === 'docker' ? 'Docker 多容器' : '单进程' }} · 可用脑节点 {{ nodePool }}/{{ (cluster.nodes || []).length }}
        </span>
      </div>
      <el-form :model="form" label-width="118px" label-position="left">
        <el-divider content-position="left">任务定义</el-divider>
        <div class="fc-grid fc-grid-4">
          <el-form-item label="任务名">
            <el-input v-model="taskName" placeholder="留空则自动生成" />
          </el-form-item>
          <el-form-item label="网格边长">
            <el-input-number v-model="form.grid" :min="8" :max="64" controls-position="right" />
          </el-form-item>
          <el-form-item label="目标数">
            <el-input-number v-model="form.n_targets" :min="1" :max="40" controls-position="right" />
          </el-form-item>
          <el-form-item label="步数上限">
            <el-input-number v-model="form.max_ticks" :min="5" :max="5000" :step="20" controls-position="right" />
          </el-form-item>
        </div>
        <div class="fc-grid fc-grid-4">
          <el-form-item label="障碍比例">
            <el-input-number v-model="form.obstacle_fraction" :min="0" :max="0.4" :step="0.02"
                             :precision="2" controls-position="right" />
          </el-form-item>
          <el-form-item label="感知有效率">
            <el-input-number v-model="form.perception_keep" :min="0" :max="1" :step="0.05"
                             :precision="2" controls-position="right" />
          </el-form-item>
          <el-form-item label="随机种子">
            <el-input-number v-model="form.seed" :min="0" :max="9999" controls-position="right" />
          </el-form-item>
          <el-form-item label="租户">
            <el-input v-model="form.tenant" />
          </el-form-item>
        </div>
        <div style="font-size: 11.5px; color: var(--fc-text-faint); margin: -6px 0 10px 118px; line-height: 1.7">
          「感知有效率」是感知<b>有效</b>的 tick 占比，不是失效率——填 0.5 表示约一半的 tick 拿不到外部信息。
          失效时视野清零，但自身位置与朝向仍然保留（相当于闭眼，不是失去本体感觉）。
        </div>

        <el-divider content-position="left">中间结构与规模</el-divider>
        <div class="fc-grid fc-grid-4">
          <el-form-item label="中间结构">
            <el-select v-model="form.backend">
              <el-option v-for="b in BACKENDS" :key="b.v" :label="b.l" :value="b.v" />
            </el-select>
          </el-form-item>
          <el-form-item label="节点数">
            <div style="display: flex; align-items: center; gap: 8px">
              <el-input-number v-model="form.agents" :min="1" :max="Math.max(1, nodePool || 16)"
                               controls-position="right" />
              <el-button v-if="nodePool" link size="small" @click="form.agents = nodePool">
                用满 {{ nodePool }} 个
              </el-button>
            </div>
          </el-form-item>
          <el-form-item label="学习模式">
            <el-select v-model="form.learning_mode">
              <el-option label="只读推理 readonly" value="readonly" />
              <el-option label="积累经历 accumulate" value="accumulate" />
              <el-option label="允许更新参数 plastic" value="plastic" />
            </el-select>
          </el-form-item>
          <el-form-item label="消息模式">
            <el-select v-model="form.message_kind">
              <el-option label="脑输出组脉冲（归因于结构）" value="learned_compressed" />
              <el-option label="外围压缩观测（不得归因于结构）" value="shared_observation" />
            </el-select>
          </el-form-item>
        </div>

        <el-divider content-position="left">协作与记忆</el-divider>
        <div class="fc-grid fc-grid-4">
          <el-form-item label="同伴通信">
            <el-switch v-model="form.enable_peer_messages" />
          </el-form-item>
          <el-form-item label="消息延迟（步）">
            <el-input-number v-model="form.comm_latency_ticks" :min="0" :max="50" controls-position="right" />
          </el-form-item>
          <el-form-item label="消息丢弃率">
            <el-input-number v-model="form.comm_loss" :min="0" :max="1" :step="0.05"
                             :precision="2" controls-position="right" />
          </el-form-item>
          <el-form-item label="中心规划器">
            <el-switch v-model="form.enable_central_planner" />
          </el-form-item>
        </div>
        <div class="fc-grid fc-grid-4">
          <el-form-item label="中心记忆通道">
            <el-switch v-model="form.enable_memory_channel" />
          </el-form-item>
          <el-form-item label="本地检索 k_local">
            <el-input-number v-model="form.k_local" :min="0" :max="64" controls-position="right" />
          </el-form-item>
          <el-form-item label="中心检索 k_central">
            <el-input-number v-model="form.k_central" :min="0" :max="64"
                             :disabled="!form.enable_memory_channel" controls-position="right" />
          </el-form-item>
          <el-form-item label="记忆范围">
            <el-select v-model="form.memory_scope">
              <el-option label="本次运行 run" value="run" />
              <el-option label="当前会话 session" value="session" />
              <el-option label="指定项目 project" value="project" />
              <el-option label="租户 tenant" value="tenant" />
            </el-select>
          </el-form-item>
        </div>
        <div v-if="form.enable_memory_channel"
             style="font-size: 11.5px; color: var(--fc-warn); margin: -6px 0 10px 118px; line-height: 1.7">
          中心记忆是一条<b>独立的通信通道</b>。实测检索返回字节比脑间消息高约三个数量级，
          只统计脑间消息就宣称"低带宽协作"是不成立的。
        </div>
      </el-form>

      <div style="display: flex; align-items: center; gap: 16px; border-top: 1px solid var(--fc-border); padding-top: 14px">
        <el-button type="primary" size="large" :loading="loading" :disabled="!canDispatch" @click="dispatch">
          {{ loading ? '执行中…' : '下发给集群' }}
        </el-button>
        <span class="fc-num" style="font-size: 12px; color: var(--fc-text-dim)">
          预计 {{ stepCount }} 个决策步（{{ form.agents }} 节点 × {{ form.max_ticks }} 步 × 20 ms 神经仿真）
        </span>
      </div>
    </div>

    <div v-if="live" class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title"><el-icon><Loading /></el-icon> 执行状态</div>
      <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 12px">
        <span class="fc-tag" :class="live.status === 'finished' ? 'ok' : live.status === 'failed' ? 'bad' : 'warn'">
          {{ live.status }}
        </span>
        <span class="fc-num" style="font-size: 12px; color: var(--fc-text-dim)">{{ live.run_id }}</span>
      </div>
      <el-progress :percentage="progressPct" :stroke-width="8"
                   :status="live.status === 'failed' ? 'exception' : (live.status === 'finished' ? 'success' : '')" />
      <div style="margin-top: 10px; font-size: 12px; color: var(--fc-text-dim)">
        进度 {{ live.tick || 0 }} / {{ form.max_ticks }} 步　
        命中 {{ live.progress?.targets_found ?? '—' }}/{{ live.progress?.targets_total ?? '—' }}　
        覆盖 {{ live.progress?.coverage ?? '—' }}　
        消息 {{ live.progress?.messages ?? '—' }}
      </div>
      <el-alert v-if="live.error" type="error" :closable="false" :title="live.error" style="margin-top: 10px" />
    </div>

    <div v-if="report">
      <h3 class="fc-page-title" style="font-size: 15px; margin-bottom: 12px">返回结果 · {{ report.run_id }}</h3>
      <RunReportPanel :report="report" />
    </div>
    <el-empty v-else-if="!live" description="填好配置后点「下发给集群」，结果会显示在这里" />
  </div>
</template>
