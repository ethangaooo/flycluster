<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api, streamRun } from '../api/client'
import RunReportPanel from '../components/RunReportPanel.vue'

const runs = ref([])
const selected = ref(null)
const report = ref(null)
const loading = ref(false)
const err = ref('')
let stopStream = null
let timer = null

const BACKENDS = [
  { v: 'brain', l: '真实连接子图（脉冲网络）' },
  { v: 'rewired', l: '保度重连对照（脉冲网络）' },
  { v: 'reservoir', l: '随机储备池' },
  { v: 'gru', l: '小型 GRU' },
  { v: 'classical', l: '经典局部探索策略' },
  { v: 'noop', l: '无脑模型（仅接口与读出）' }
]

const form = ref({
  backend: 'brain',
  connectome: 'synthetic-2000',
  agents: 4,
  max_ticks: 120,
  n_targets: 6,
  grid: 32,
  seed: 1,
  memory_scope: 'run',
  learning_mode: 'plastic',
  memory_version: 'head',
  enable_peer_messages: true,
  enable_memory_channel: false,
  enable_central_planner: false,
  message_kind: 'learned_compressed',
  comm_latency_ticks: 1,
  comm_loss: 0.1,
  tenant: 'default'
})

async function loadRuns() {
  try {
    runs.value = (await api.runs()).slice().reverse()
    err.value = ''
  } catch (e) {
    err.value = e.friendly
  }
}

async function openRun(row) {
  selected.value = row
  report.value = null
  if (stopStream) stopStream()
  try {
    if (row.status === 'finished') {
      report.value = await api.report(row.run_id)
      return
    }
    stopStream = streamRun(row.run_id, (ev) => {
      if (ev.type === 'report') report.value = ev.report
      if (ev.type === 'status') {
        const i = runs.value.findIndex((r) => r.run_id === row.run_id)
        if (i >= 0) runs.value[i] = { ...runs.value[i], ...ev }
      }
    })
  } catch (e) {
    ElMessage.error(e.friendly)
  }
}

async function createRun() {
  loading.value = true
  try {
    const f = form.value
    const payload = {
      run: {
        run_id: '',
        task_id: '',
        seed: f.seed,
        backend: f.backend,
        connectome: f.connectome,
        agents: f.agents,
        max_ticks: f.max_ticks,
        memory_scope: f.memory_scope,
        learning_mode: f.learning_mode,
        memory_version: f.memory_version,
        enable_peer_messages: f.enable_peer_messages,
        enable_memory_channel: f.enable_memory_channel,
        enable_central_planner: f.enable_central_planner,
        comm_latency_ticks: f.comm_latency_ticks,
        comm_loss: f.comm_loss,
        message_kind: f.message_kind,
        tags: { tenant: f.tenant, group: f.backend, protocol: 'manual' }
      },
      task: {
        task_id: '',
        grid: f.grid,
        n_agents: f.agents,
        n_targets: f.n_targets,
        max_ticks: f.max_ticks,
        obstacle_fraction: 0.12,
        perception_fraction: 1.0,
        seed: f.seed,
        env_version: 'grid-v1',
        visibility_regions: null
      }
    }
    const res = await api.createRun(payload)
    ElMessage.success(`已提交运行 ${res.run_id}`)
    await loadRuns()
    const row = runs.value.find((r) => r.run_id === res.run_id)
    if (row) openRun(row)
  } catch (e) {
    ElMessage.error(e.friendly)
  }
  loading.value = false
}

async function abort(row) {
  try {
    await api.abort(row.run_id)
    ElMessage.warning('已请求中止')
    loadRuns()
  } catch (e) {
    ElMessage.error(e.friendly)
  }
}

const statusTone = (s) => (s === 'finished' ? 'ok' : s === 'failed' ? 'bad' : 'warn')

onMounted(() => {
  loadRuns()
  timer = setInterval(loadRuns, 2500)
})
onUnmounted(() => {
  if (timer) clearInterval(timer)
  if (stopStream) stopStream()
})
</script>

<template>
  <div class="fc-page">
    <h2 class="fc-page-title">运行与实验</h2>
    <p class="fc-page-sub">
      三个配置决定一次运行能回答什么问题：
      <code class="fc-num">memory_scope</code>（本次运行 / 会话 / 项目 / 租户，禁止默认跨客户共享）、
      <code class="fc-num">learning_mode</code>（只读推理 / 积累经历 / 允许更新学习参数）、
      <code class="fc-num">memory_version</code>（固定快照 / 跟随 head）。
    </p>

    <el-alert v-if="err" type="error" :closable="false" :title="err" style="margin-bottom: 14px" />

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title"><el-icon><Plus /></el-icon> 新建运行</div>
      <el-form :model="form" label-width="112px" label-position="left">
        <div class="fc-grid fc-grid-4">
          <el-form-item label="中间结构">
            <el-select v-model="form.backend">
              <el-option v-for="b in BACKENDS" :key="b.v" :label="b.l" :value="b.v" />
            </el-select>
          </el-form-item>
          <el-form-item label="连接组">
            <el-input v-model="form.connectome" placeholder="synthetic-2000 或 data/xxx.npz" />
          </el-form-item>
          <el-form-item label="智能体数">
            <el-input-number v-model="form.agents" :min="1" :max="16" controls-position="right" />
          </el-form-item>
          <el-form-item label="步数上限">
            <el-input-number v-model="form.max_ticks" :min="5" :max="5000" :step="20" controls-position="right" />
          </el-form-item>
        </div>
        <div class="fc-grid fc-grid-4">
          <el-form-item label="网格边长">
            <el-input-number v-model="form.grid" :min="8" :max="64" controls-position="right" />
          </el-form-item>
          <el-form-item label="目标数">
            <el-input-number v-model="form.n_targets" :min="1" :max="40" controls-position="right" />
          </el-form-item>
          <el-form-item label="随机种子">
            <el-input-number v-model="form.seed" :min="0" :max="9999" controls-position="right" />
          </el-form-item>
          <el-form-item label="租户">
            <el-input v-model="form.tenant" />
          </el-form-item>
        </div>
        <div class="fc-grid fc-grid-4">
          <el-form-item label="记忆范围">
            <el-select v-model="form.memory_scope">
              <el-option label="本次运行 run" value="run" />
              <el-option label="当前会话 session" value="session" />
              <el-option label="指定项目 project" value="project" />
              <el-option label="租户 tenant" value="tenant" />
            </el-select>
          </el-form-item>
          <el-form-item label="学习模式">
            <el-select v-model="form.learning_mode">
              <el-option label="只读推理 readonly" value="readonly" />
              <el-option label="积累经历 accumulate" value="accumulate" />
              <el-option label="允许更新参数 plastic" value="plastic" />
            </el-select>
          </el-form-item>
          <el-form-item label="记忆版本">
            <el-select v-model="form.memory_version">
              <el-option label="跟随 head" value="head" />
              <el-option label="固定快照 pinned" value="pinned" />
            </el-select>
          </el-form-item>
          <el-form-item label="消息模式">
            <el-select v-model="form.message_kind">
              <el-option label="脑输出组脉冲" value="learned_compressed" />
              <el-option label="外围压缩观测" value="shared_observation" />
            </el-select>
          </el-form-item>
        </div>
        <div class="fc-grid fc-grid-4">
          <el-form-item label="同伴通信">
            <el-switch v-model="form.enable_peer_messages" />
          </el-form-item>
          <el-form-item label="中心记忆">
            <el-switch v-model="form.enable_memory_channel" />
          </el-form-item>
          <el-form-item label="中心规划器">
            <el-switch v-model="form.enable_central_planner" />
          </el-form-item>
          <el-form-item label=" ">
            <el-button type="primary" :loading="loading" @click="createRun">提交运行</el-button>
          </el-form-item>
        </div>
      </el-form>
    </div>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title"><el-icon><VideoPlay /></el-icon> 运行列表
        <span class="fc-hint">每 2.5 秒自动刷新</span></div>
      <el-table :data="runs" size="small" highlight-current-row @current-change="(r) => r && openRun(r)">
        <el-table-column prop="run_id" label="run_id" min-width="180" />
        <el-table-column prop="backend" label="结构" width="100" />
        <el-table-column prop="task_id" label="任务" min-width="190" />
        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <span class="fc-tag" :class="statusTone(row.status)">{{ row.status }}</span>
          </template>
        </el-table-column>
        <el-table-column label="进度" width="150">
          <template #default="{ row }">
            <span class="fc-num" style="font-size: 11.5px">
              命中 {{ row.progress?.targets_found ?? '—' }}/{{ row.progress?.targets_total ?? '—' }}
              · 覆盖 {{ row.progress?.coverage ?? '—' }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="88">
          <template #default="{ row }">
            <el-button v-if="row.status === 'running' || row.status === 'queued'" link size="small" @click.stop="abort(row)">
              中止
            </el-button>
            <span v-else style="color: var(--fc-text-faint); font-size: 11.5px">—</span>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <div v-if="report">
      <h3 class="fc-page-title" style="font-size: 15px; margin-bottom: 12px">
        报告 · {{ report.run_id }}
        <span class="fc-tag" style="margin-left: 8px">{{ report.backend }}</span>
      </h3>
      <RunReportPanel :report="report" />
    </div>
    <el-empty v-else description="选择上方一行运行以查看报告" />
  </div>
</template>
