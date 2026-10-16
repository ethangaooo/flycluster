<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { api } from '../api/client'

use([CanvasRenderer, BarChart, GridComponent, TooltipComponent, LegendComponent])

// 与 experiments/protocol.py 的 GROUP_SPECS 保持一致
const GROUPS = [
  { key: 'real_comm', backend: 'brain', comm: true, label: '真实连接子图 + 通信' },
  { key: 'real_nocomm', backend: 'brain', comm: false, label: '真实连接子图 − 通信' },
  { key: 'rewired_comm', backend: 'rewired', comm: true, label: '保度重连 + 通信' },
  { key: 'rewired_nocomm', backend: 'rewired', comm: false, label: '保度重连 − 通信' },
  { key: 'reservoir_comm', backend: 'reservoir', comm: true, label: '随机储备池 + 通信' },
  { key: 'reservoir_nocomm', backend: 'reservoir', comm: false, label: '随机储备池 − 通信' },
  { key: 'gru_comm', backend: 'gru', comm: true, label: '小型 GRU + 通信' },
  { key: 'gru_nocomm', backend: 'gru', comm: false, label: '小型 GRU − 通信' },
  { key: 'classical', backend: 'classical', comm: false, label: '经典局部探索策略' },
  { key: 'classical_comm', backend: 'classical', comm: true, label: '经典策略 + 通信' },
  { key: 'noop', backend: 'noop', comm: false, label: '无脑模型（仅接口与读出）' }
]

const seeds = ref(2)
const ticks = ref(120)
const targets = ref(6)
const running = ref(false)
const progress = ref({ done: 0, total: 0, text: '' })
const runs = ref([])
let timer = null

async function refresh() {
  try {
    runs.value = await api.runs()
  } catch (e) { /* 静默：由顶部状态条统一提示 */ }
}

const tableRows = computed(() => {
  return GROUPS.map((g) => {
    const scores = []
    for (const r of runs.value) {
      if (r.config?.tags?.group !== g.key || r.status !== 'finished') continue
      const s = r.score?.score
      if (typeof s === 'number') scores.push(s)
    }
    const n = scores.length
    const mean = n ? scores.reduce((a, b) => a + b, 0) / n : null
    const sd = n > 1 ? Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : 0
    return { ...g, n, mean, sd, scores }
  })
})

const chartOption = computed(() => {
  const rows = tableRows.value.filter((r) => r.n > 0)
  return {
    backgroundColor: 'transparent',
    grid: { left: 8, right: 20, top: 24, bottom: 8, containLabel: true },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: '#333c4a' } },
      splitLine: { lineStyle: { color: '#222933' } },
      axisLabel: { color: '#9aa7b4' }
    },
    yAxis: {
      type: 'category',
      data: rows.map((r) => r.label),
      axisLine: { lineStyle: { color: '#333c4a' } },
      axisLabel: { color: '#9aa7b4', fontSize: 11 }
    },
    series: [{
      type: 'bar',
      data: rows.map((r) => ({ value: Number(r.mean.toFixed(3)), itemStyle: { color: r.key.includes('rewired') ? '#d9a441' : (r.key.includes('real') || r.key === 'brain' ? '#35c9c0' : '#4a5666') } })),
      barWidth: '55%',
      label: { show: true, position: 'right', color: '#9aa7b4', fontSize: 11, formatter: (p) => p.value }
    }]
  }
})

const delta = computed(() => {
  const pick = (k) => tableRows.value.find((r) => r.key === k)?.mean
  const a = pick('real_comm'), b = pick('real_nocomm'), c = pick('rewired_comm'), d = pick('rewired_nocomm')
  if ([a, b, c, d].some((x) => x === null || x === undefined)) return null
  return (a - b) - (c - d)
})

async function runBatch() {
  running.value = true
  const total = GROUPS.length * seeds.value
  progress.value = { done: 0, total, text: '提交中…' }
  const batch = `batch-${Date.now()}`
  try {
    for (let s = 1; s <= seeds.value; s++) {
      for (const g of GROUPS) {
        await api.createRun({
          run: {
            run_id: `${batch}-${g.key}-s${s}`,
            task_id: '', seed: s, backend: g.backend,
            connectome: 'synthetic-1000', agents: 4, max_ticks: ticks.value,
            memory_scope: 'run', learning_mode: 'plastic', memory_version: 'head',
            enable_peer_messages: g.comm, enable_memory_channel: false,
            enable_central_planner: false, comm_latency_ticks: 1, comm_loss: 0.1,
            message_kind: 'learned_compressed',
            tags: { tenant: 'default', group: g.key, protocol: 'batch-ui', batch }
          },
          task: {
            task_id: `${batch}-${g.key}-s${s}`, grid: 32, n_agents: 4,
            n_targets: targets.value, max_ticks: ticks.value,
            obstacle_fraction: 0.12, perception_fraction: 1.0, seed: s,
            env_version: 'grid-v1', visibility_regions: null
          }
        })
        progress.value = { done: progress.value.done + 1, total, text: `${g.label} · 种子 ${s}` }
        await refresh()
      }
    }
    ElMessage.success(`已提交 ${total} 次运行，等待完成后本页自动汇总`)
  } catch (e) {
    ElMessage.error(e.friendly)
  }
  running.value = false
}

onMounted(() => {
  refresh()
  timer = setInterval(refresh, 3000)
})
onUnmounted(() => timer && clearInterval(timer))
</script>

<template>
  <div class="fc-page">
    <h2 class="fc-page-title">对照结果</h2>
    <p class="fc-page-sub">
      核心指标不是「4 个比 1 个强」，而是交互效应 Δ：
      Δ =（真实结构有通信 − 真实结构无通信）−（重连结构有通信 − 重连结构无通信）。
      它区分「通信对任何网络都有效」与「真实果蝇结构特别会用通信」。Δ 为正也不自动等于商业优势。
    </p>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title"><el-icon><VideoPlay /></el-icon> 跑一批主对照
        <span class="fc-hint">在浏览器里提交全部 11 个组 × N 个种子；结果自动汇总到下方</span>
      </div>
      <div style="display: flex; gap: 14px; align-items: flex-end; flex-wrap: wrap">
        <div>
          <div class="fc-metric-label">种子数</div>
          <el-input-number v-model="seeds" :min="1" :max="10" controls-position="right" />
        </div>
        <div>
          <div class="fc-metric-label">步数上限</div>
          <el-input-number v-model="ticks" :min="20" :max="2000" :step="20" controls-position="right" />
        </div>
        <div>
          <div class="fc-metric-label">目标数</div>
          <el-input-number v-model="targets" :min="1" :max="20" controls-position="right" />
        </div>
        <el-button type="primary" :loading="running" @click="runBatch">
          提交 {{ GROUPS.length * seeds }} 次运行
        </el-button>
        <div v-if="progress.total" class="fc-num" style="font-size: 12px; color: var(--fc-text-dim)">
          {{ progress.done }}/{{ progress.total }} · {{ progress.text }}
        </div>
      </div>
      <div style="margin-top: 10px; font-size: 11.5px; color: var(--fc-text-faint); line-height: 1.7">
        界面跑批使用 <code class="fc-num">synthetic-1000</code> 以便快速反馈。正式结论请用命令行
        <code class="fc-num">experiments/run.py</code>（≥10 种子、保留置信区间、同时输出 JSON 与 Markdown 报告）。
      </div>
    </div>

    <div class="fc-grid fc-grid-3" style="margin-bottom: 14px">
      <div class="fc-card">
        <div class="fc-card-title"><el-icon><DataAnalysis /></el-icon> 交互效应 Δ</div>
        <div class="fc-metric-value" :style="{ color: delta === null ? 'var(--fc-text-faint)' : (delta > 0 ? 'var(--fc-accent)' : 'var(--fc-warn)') }">
          {{ delta === null ? '数据不足' : delta.toFixed(3) }}
        </div>
        <div class="fc-metric-note" style="margin-top: 8px">
          {{ delta === null
            ? '需要真实结构（有/无通信）与重连结构（有/无通信）四组的已完成运行。'
            : (delta > 0 ? '名义上支持真实结构对通信的利用更好；结论仍需 ≥10 种子的置信区间支撑。'
                        : '不支持「真实结构有独特贡献」；只能表述为「该协作方法有效」。') }}
        </div>
      </div>
      <div class="fc-card" style="grid-column: span 2">
        <div class="fc-card-title"><el-icon><Histogram /></el-icon> 各组平均得分</div>
        <v-chart v-if="tableRows.some(r => r.n > 0)" :option="chartOption" style="height: 320px" autoresize />
        <el-empty v-else description="还没有批次运行结果" />
      </div>
    </div>

    <div class="fc-card">
      <div class="fc-card-title"><el-icon><Tickets /></el-icon> 分组明细</div>
      <el-table :data="tableRows" size="small">
        <el-table-column prop="label" label="组" min-width="200" />
        <el-table-column prop="backend" label="结构" width="110" />
        <el-table-column label="通信" width="80">
          <template #default="{ row }">
            <span class="fc-tag" :class="row.comm ? 'ok' : ''">{{ row.comm ? '开' : '关' }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="n" label="完成数" width="80" />
        <el-table-column label="均值 S" width="110">
          <template #default="{ row }">
            <span class="fc-num">{{ row.mean === null ? '—' : row.mean.toFixed(3) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="标准差" width="90">
          <template #default="{ row }">
            <span class="fc-num">{{ row.n > 1 ? row.sd.toFixed(3) : '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="各种子得分" min-width="240">
          <template #default="{ row }">
            <span class="fc-num" style="font-size: 11px; color: var(--fc-text-faint)">
              {{ row.scores.length ? row.scores.join(', ') : '—' }}
            </span>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>
