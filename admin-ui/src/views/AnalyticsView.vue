<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import {
  BarChart, LineChart, ScatterChart, BoxplotChart, CustomChart
} from 'echarts/charts'
import {
  GridComponent, TooltipComponent, LegendComponent, DataZoomComponent,
  ToolboxComponent, MarkLineComponent
} from 'echarts/components'

import { api } from '../api/client'

use([CanvasRenderer, BarChart, LineChart, ScatterChart, BoxplotChart, CustomChart,
     GridComponent, TooltipComponent, LegendComponent, DataZoomComponent,
     ToolboxComponent, MarkLineComponent])

const data = ref(null)
const loading = ref(false)
const err = ref('')

// ---------------- 筛选
const filters = ref({
  experiment: [],
  connectome: [],
  backend: [],
  seedMin: null,
  seedMax: null,
  traceableOnly: false
})

// ---------------- 图表配置
const view = ref({
  chart: 'bar',          // bar | line | scatter | box
  xDim: 'group',
  yMetric: 'score',
  seriesDim: 'connectome',
  sort: 'value'
})

// ---------------- Δ 计算器
const deltaCfg = ref({ treatment: '', control: '', metric: 'score', pairBy: 'seed',
                       scatterX: 'redundant_steps' })

const METRIC_FALLBACK = { label: '—', unit: '', direction: 1, desc: '' }

const metrics = computed(() => data.value?.metrics || {})
const dims = computed(() => data.value?.dimensions || {})
const metricMeta = (m) => metrics.value[m] || METRIC_FALLBACK

// 小样本的 t 值（95%，双侧）。n>30 直接用 1.96。
const T95 = { 1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
              8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
              14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
              20: 2.086, 25: 2.060, 30: 2.042 }
function t95(n) {
  const df = n - 1
  if (df <= 0) return 0
  if (T95[df]) return T95[df]
  if (df > 30) return 1.96
  let best = 30
  for (const k of Object.keys(T95).map(Number)) if (k <= df && k < best) best = k
  return T95[best] || 2.0
}

function stats(values) {
  const v = values.filter((x) => typeof x === 'number' && isFinite(x))
  const n = v.length
  if (!n) return { n: 0, mean: null, sd: null, ci: null, min: null, max: null }
  const mean = v.reduce((a, b) => a + b, 0) / n
  const variance = n > 1 ? v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1) : 0
  const sd = Math.sqrt(variance)
  const half = n > 1 ? t95(n) * sd / Math.sqrt(n) : null
  return { n, mean, sd, ci: half, min: Math.min(...v), max: Math.max(...v), values: v }
}

// ---------------- 取数
async function load() {
  loading.value = true
  try {
    data.value = await api.analytics()
    err.value = ''
  } catch (e) {
    err.value = e.friendly
  }
  loading.value = false
}

const rows = computed(() => data.value?.rows || [])

const filtered = computed(() => {
  const f = filters.value
  return rows.value.filter((r) => {
    if (f.experiment.length && !f.experiment.includes(r.experiment)) return false
    if (f.connectome.length && !f.connectome.includes(r.connectome)) return false
    if (f.backend.length && !f.backend.includes(r.backend)) return false
    if (f.seedMin !== null && (r.seed ?? -Infinity) < f.seedMin) return false
    if (f.seedMax !== null && (r.seed ?? Infinity) > f.seedMax) return false
    if (f.traceableOnly && !r.connectome) return false
    return true
  })
})

const xValues = computed(() => {
  const s = new Set(filtered.value.map((r) => fmt(r[view.value.xDim])))
  const arr = [...s]
  if (view.value.sort === 'value') {
    const dir = metricMeta(view.value.yMetric).direction || 1
    arr.sort((a, b) => {
      const ma = stats(filtered.value.filter((r) => fmt(r[view.value.xDim]) === a)
        .map((r) => r[view.value.yMetric])).mean ?? 0
      const mb = stats(filtered.value.filter((r) => fmt(r[view.value.xDim]) === b)
        .map((r) => r[view.value.yMetric])).mean ?? 0
      return dir >= 0 ? mb - ma : ma - mb
    })
  } else arr.sort()
  return arr
})

function fmt(v) {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'boolean') return v ? '开' : '关'
  if (typeof v === 'number') return String(v)
  return String(v)
}

const seriesValues = computed(() => {
  if (view.value.seriesDim === 'none') return ['全部']
  return [...new Set(filtered.value.map((r) => fmt(r[view.value.seriesDim])))].sort()
})

// ---------------- 图表
const PALETTE = ['#35c9c0', '#d9a441', '#7c9cff', '#e07bd0', '#8fd14f', '#e5534b',
                 '#9aa7b4', '#4a5666']

function groupRows(xv, sv) {
  return filtered.value.filter((r) =>
    fmt(r[view.value.xDim]) === xv &&
    (view.value.seriesDim === 'none' || fmt(r[view.value.seriesDim]) === sv))
}

const chartOption = computed(() => {
  const y = view.value.yMetric
  const meta = metricMeta(y)
  const xs = xValues.value
  const svs = seriesValues.value
  const unit = meta.unit ? ` ${meta.unit}` : ''

  const base = {
    backgroundColor: 'transparent',
    grid: { left: 10, right: 24, top: 36, bottom: 8, containLabel: true },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { show: false },
    dataZoom: xs.length > 12 ? [{ type: 'inside' }] : []
  }

  if (view.value.chart === 'bar') {
    return {
      ...base,
      xAxis: {
        type: 'category', data: xs,
        axisLabel: { color: '#9aa7b4', fontSize: 11, interval: 0, rotate: xs.length > 6 ? 26 : 0 },
        axisLine: { lineStyle: { color: '#333c4a' } }
      },
      yAxis: {
        type: 'value', name: meta.label + unit, nameTextStyle: { color: '#9aa7b4', fontSize: 11 },
        axisLabel: { color: '#9aa7b4' }, splitLine: { lineStyle: { color: '#222933' } },
        scale: true
      },
      series: svs.flatMap((sv, si) => {
        const means = xs.map((xv) => stats(groupRows(xv, sv).map((r) => r[y])))
        const color = PALETTE[si % PALETTE.length]
        const scale = Math.max(...means.map((m) => Math.abs(m.ci ?? 0)), 1e-9)
        return [
          {
            name: sv, type: 'bar', barMaxWidth: 38,
            data: means.map((m) => (m.mean === null ? null : Number(m.mean.toFixed(3)))),
            itemStyle: { color, borderRadius: [3, 3, 0, 0] },
            label: {
              show: xs.length * svs.length <= 14, position: 'top', color: '#9aa7b4', fontSize: 10,
              formatter: (p) => (p.value === null ? '' : p.value.toFixed(1))
            }
          },
          {
            name: `${sv} 95%CI`, type: 'custom', silent: true, z: 5,
            renderItem: (params, apiL) => {
              const idx = params.dataIndex
              const m = means[idx]
              if (!m || m.ci === null) return null
              const x = apiL.coord([apiL.value(0), 0])[0]
              const hi = apiL.coord([apiL.value(0), m.mean + m.ci])[1]
              const lo = apiL.coord([apiL.value(0), m.mean - m.ci])[1]
              const half = 7
              return {
                type: 'group',
                children: [
                  { type: 'line', shape: { x1: x, y1: hi, x2: x, y2: lo }, style: { stroke: '#e6edf3', lineWidth: 1 } },
                  { type: 'line', shape: { x1: x - half, y1: hi, x2: x + half, y2: hi }, style: { stroke: '#e6edf3', lineWidth: 1 } },
                  { type: 'line', shape: { x1: x - half, y1: lo, x2: x + half, y2: lo }, style: { stroke: '#e6edf3', lineWidth: 1 } }
                ]
              }
            },
            data: means.map((m) => (m.mean ?? 0)),
            dimensions: [scale]
          }
        ]
      })
    }
  }

  if (view.value.chart === 'line') {
    return {
      ...base,
      xAxis: {
        type: 'category', data: xs,
        axisLabel: { color: '#9aa7b4', fontSize: 11 },
        axisLine: { lineStyle: { color: '#333c4a' } }
      },
      yAxis: {
        type: 'value', name: meta.label + unit, nameTextStyle: { color: '#9aa7b4', fontSize: 11 },
        axisLabel: { color: '#9aa7b4' }, splitLine: { lineStyle: { color: '#222933' } }, scale: true
      },
      series: svs.map((sv, si) => ({
        name: sv, type: 'line', smooth: false, symbolSize: 6,
        connectNulls: true,
        itemStyle: { color: PALETTE[si % PALETTE.length] },
        data: xs.map((xv) => {
          const st = stats(groupRows(xv, sv).map((r) => r[y]))
          return st.mean === null ? null : Number(st.mean.toFixed(3))
        })
      }))
    }
  }

  if (view.value.chart === 'box') {
    return {
      ...base,
      tooltip: { trigger: 'item' },
      xAxis: {
        type: 'category', data: xs,
        axisLabel: { color: '#9aa7b4', fontSize: 11, interval: 0, rotate: xs.length > 6 ? 26 : 0 },
        axisLine: { lineStyle: { color: '#333c4a' } }
      },
      yAxis: {
        type: 'value', name: meta.label + unit, nameTextStyle: { color: '#9aa7b4', fontSize: 11 },
        axisLabel: { color: '#9aa7b4' }, splitLine: { lineStyle: { color: '#222933' } }, scale: true
      },
      series: svs.map((sv, si) => ({
        name: sv, type: 'boxplot',
        itemStyle: { color: PALETTE[si % PALETTE.length], borderColor: PALETTE[si % PALETTE.length] },
        data: xs.map((xv) => {
          const v = groupRows(xv, sv).map((r) => r[y])
            .filter((x) => typeof x === 'number' && isFinite(x)).sort((a, b) => a - b)
          if (!v.length) return null
          const q = (p) => {
            const i = (v.length - 1) * p
            const lo = Math.floor(i), hi = Math.ceil(i)
            return lo === hi ? v[lo] : v[lo] + (v[hi] - v[lo]) * (i - lo)
          }
          return [v[0], q(0.25), q(0.5), q(0.75), v[v.length - 1]].map((x) => Number(x.toFixed(3)))
        })
      }))
    }
  }

  // scatter：X 取"另一个指标"，Y 取所选指标，按系列着色
  const xMetric = deltaCfg.value.scatterX || 'redundant_steps'
  return {
    ...base,
    tooltip: { trigger: 'item' },
    xAxis: {
      type: 'value', name: metricMeta(xMetric).label,
      nameTextStyle: { color: '#9aa7b4', fontSize: 11 },
      axisLabel: { color: '#9aa7b4' }, splitLine: { lineStyle: { color: '#222933' } }, scale: true
    },
    yAxis: {
      type: 'value', name: meta.label + unit, nameTextStyle: { color: '#9aa7b4', fontSize: 11 },
      axisLabel: { color: '#9aa7b4' }, splitLine: { lineStyle: { color: '#222933' } }, scale: true
    },
    series: svs.map((sv, si) => ({
      name: sv, type: 'scatter', symbolSize: 9,
      itemStyle: { color: PALETTE[si % PALETTE.length], opacity: 0.75 },
      data: filtered.value
        .filter((r) => view.value.seriesDim === 'none' || fmt(r[view.value.seriesDim]) === sv)
        .filter((r) => typeof r[y] === 'number' && typeof r[xMetric] === 'number')
        .map((r) => ({
          value: [r[xMetric], r[y]],
          name: `${r.group ?? ''} · 种子 ${r.seed ?? '—'} · ${r.connectome ?? '—'}`
        }))
    }))
  }
})

// ---------------- Δ 计算器
const deltaGroups = computed(() => [...new Set(filtered.value.map((r) => r.group))].filter(Boolean))

const deltaResult = computed(() => {
  const { treatment, control, metric, pairBy } = deltaCfg.value
  if (!treatment || !control || treatment === control) return null
  const dir = metricMeta(metric).direction || 1

  const tRows = filtered.value.filter((r) => r.group === treatment)
  const cRows = filtered.value.filter((r) => r.group === control)
  const keyOf = (r) => `${r.connectome ?? ''}|${r.axis ?? ''}|${r.condition ?? ''}|${r.phase ?? ''}|${r[pairBy]}`
  const cmap = new Map(cRows.map((r) => [keyOf(r), r]))

  const pairs = []
  for (const t of tRows) {
    const c = cmap.get(keyOf(t))
    if (!c) continue
    if (typeof t[metric] !== 'number' || typeof c[metric] !== 'number') continue
    pairs.push({ key: keyOf(t), seed: t[pairBy], connectome: t.connectome,
                 t: t[metric], c: c[metric], d: t[metric] - c[metric] })
  }
  if (!pairs.length) return { pairs: [], n: 0 }
  const ds = pairs.map((p) => p.d)
  const st = stats(ds)
  return {
    pairs, n: st.n, mean: st.mean, sd: st.sd, ci: st.ci,
    low: st.ci === null ? null : st.mean - st.ci,
    high: st.ci === null ? null : st.mean + st.ci,
    significant: st.ci !== null && (st.mean - st.ci > 0 || st.mean + st.ci < 0),
    favorable: st.mean * dir > 0,
    metricLabel: metricMeta(metric).label
  }
})

// ---------------- 明细表
const tableRows = computed(() => filtered.value.slice(0, 400))

function exportCsv() {
  const cols = ['source', 'experiment', 'artifact', 'connectome', 'group', 'backend',
                'comm', 'seed', 'axis', 'condition', 'phase', 'agents', 'ticks',
                ...Object.keys(metrics.value)]
  const lines = [cols.join(',')]
  for (const r of filtered.value) {
    lines.push(cols.map((c) => {
      const v = r[c]
      if (v === null || v === undefined) return ''
      return `"${String(v).replace(/"/g, '""')}"`
    }).join(','))
  }
  const blob = new Blob(['\ufeff' + lines.join('\n')], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `flycluster-analytics-${Date.now()}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
  ElMessage.success(`已导出 ${filtered.value.length} 行`)
}

// 切到散点时，如果 X 指标和 Y 指标相同就自动换一个，避免画成一条对角线
watch(() => view.value.chart, (c) => {
  if (c === 'scatter' && deltaCfg.value.scatterX === view.value.yMetric) {
    const other = Object.keys(metrics.value).find((k) => k !== view.value.yMetric)
    if (other) deltaCfg.value.scatterX = other
  }
})

onMounted(load)
</script>

<template>
  <div class="fc-page">
    <h2 class="fc-page-title">结果分析</h2>
    <p class="fc-page-sub">
      把命令行产物与控制台运行归一化成同一张表后按任意维度切片。
      <b>命令行产物才是正式结论</b>（≥10 种子、含置信区间）；控制台的运行用于交互式验证。
    </p>

    <el-alert v-if="err" type="error" :closable="false" :title="err" style="margin-bottom: 14px" />

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title">
        <el-icon><Filter /></el-icon> 数据范围
        <span class="fc-hint">{{ filtered.length }} / {{ rows.length }} 行</span>
        <div style="flex: 1"></div>
        <el-button size="small" @click="load">重新载入</el-button>
        <el-button size="small" @click="exportCsv">导出 CSV</el-button>
      </div>
      <div class="fc-grid fc-grid-4">
        <el-form-item label="实验类型">
          <el-select v-model="filters.experiment" multiple collapse-tags placeholder="全部" style="width: 100%">
            <el-option v-for="v in dims.experiment?.values || []" :key="v" :label="v" :value="v" />
          </el-select>
        </el-form-item>
        <el-form-item label="连接组">
          <el-select v-model="filters.connectome" multiple collapse-tags placeholder="全部" style="width: 100%">
            <el-option v-for="v in dims.connectome?.values || []" :key="v" :label="v" :value="v" />
          </el-select>
        </el-form-item>
        <el-form-item label="中间结构">
          <el-select v-model="filters.backend" multiple collapse-tags placeholder="全部" style="width: 100%">
            <el-option v-for="v in dims.backend?.values || []" :key="v" :label="v" :value="v" />
          </el-select>
        </el-form-item>
        <el-form-item label="种子范围">
          <div style="display: flex; gap: 6px; align-items: center">
            <el-input-number v-model="filters.seedMin" :min="0" :max="99999" placeholder="起"
                             controls-position="right" style="width: 116px" />
            <span style="color: var(--fc-text-faint)">–</span>
            <el-input-number v-model="filters.seedMax" :min="0" :max="99999" placeholder="止"
                             controls-position="right" style="width: 116px" />
          </div>
        </el-form-item>
      </div>
      <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap">
        <el-checkbox v-model="filters.traceableOnly">
          只看可追溯（有明确连接组来源）
        </el-checkbox>
        <span v-if="data?.untraceable_artifacts?.length" class="fc-tag warn">
          {{ data.untraceable_artifacts.length }} 个旧扫描产物未记录连接组，已标为未知来源
        </span>
      </div>
    </div>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title">
        <el-icon><DataAnalysis /></el-icon> 灵活对比
        <span class="fc-hint">选好维度与指标即可，不必为每种组合单独写图</span>
      </div>
      <div class="fc-grid fc-grid-4">
        <el-form-item label="图表类型">
          <el-select v-model="view.chart">
            <el-option label="分组柱状（带 95% 置信区间）" value="bar" />
            <el-option label="折线（看趋势）" value="line" />
            <el-option label="箱线（看分布）" value="box" />
            <el-option label="散点（成本 vs 收益）" value="scatter" />
          </el-select>
        </el-form-item>
        <el-form-item :label="view.chart === 'scatter' ? 'X 轴指标（散点）' : '分组维度（X 轴）'">
          <el-select v-if="view.chart === 'scatter'" v-model="deltaCfg.scatterX">
            <el-option v-for="(m, k) in metrics" :key="k" :label="m.label" :value="k" />
          </el-select>
          <el-select v-else v-model="view.xDim">
            <el-option v-for="(d, k) in dims" :key="k" :label="d.label" :value="k" />
          </el-select>
        </el-form-item>
        <el-form-item label="指标（Y 轴）">
          <el-select v-model="view.yMetric">
            <el-option v-for="(m, k) in metrics" :key="k"
                       :label="`${m.label}${m.unit ? '（' + m.unit + '）' : ''}`" :value="k" />
          </el-select>
        </el-form-item>
        <el-form-item label="分组着色">
          <el-select v-model="view.seriesDim">
            <el-option label="不分组" value="none" />
            <el-option v-for="(d, k) in dims" :key="k" :label="d.label" :value="k" />
          </el-select>
        </el-form-item>
      </div>
      <div style="font-size: 11.5px; color: var(--fc-text-faint); margin-bottom: 10px">
        {{ metricMeta(view.yMetric).desc || '—' }}
        <span v-if="metricMeta(view.yMetric).direction === -1">（该指标越小越好）</span>
      </div>
      <div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 6px; font-size: 12px; color: var(--fc-text-secondary)">
        <span v-for="(sv, i) in seriesValues.slice(0, 8)" :key="sv"
              style="display: flex; align-items: center; gap: 5px">
          <span :style="{ width: '10px', height: '10px', borderRadius: '2px',
                          background: ['#35c9c0','#d9a441','#7c9cff','#e07bd0','#8fd14f','#e5534b','#9aa7b4','#4a5666'][i % 8] }"></span>
          {{ sv }}
        </span>
      </div>
      <v-chart v-if="filtered.length" :option="chartOption" style="height: 380px" autoresize />
      <el-empty v-else description="当前筛选没有数据" />
    </div>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title">
        <el-icon><Switch /></el-icon> Δ 计算器
        <span class="fc-hint">选两个组，按种子配对求差——这才是项目真正要看的指标</span>
      </div>
      <div class="fc-grid fc-grid-4">
        <el-form-item label="处理组">
          <el-select v-model="deltaCfg.treatment" placeholder="选择" style="width: 100%">
            <el-option v-for="g in deltaGroups" :key="g" :label="g" :value="g" />
          </el-select>
        </el-form-item>
        <el-form-item label="对照组">
          <el-select v-model="deltaCfg.control" placeholder="选择" style="width: 100%">
            <el-option v-for="g in deltaGroups" :key="g" :label="g" :value="g" />
          </el-select>
        </el-form-item>
        <el-form-item label="指标">
          <el-select v-model="deltaCfg.metric">
            <el-option v-for="(m, k) in metrics" :key="k" :label="m.label" :value="k" />
          </el-select>
        </el-form-item>
        <el-form-item label="配对键">
          <el-select v-model="deltaCfg.pairBy">
            <el-option label="随机种子" value="seed" />
            <el-option label="节点数" value="agents" />
            <el-option label="步数上限" value="ticks" />
          </el-select>
        </el-form-item>
      </div>

      <div v-if="!deltaResult" style="font-size: 12px; color: var(--fc-text-faint)">
        选择两个不同的组后，这里会按配对键把两组的样本一一对应求差。
      </div>
      <div v-else-if="!deltaResult.n" style="font-size: 12px; color: var(--fc-warn)">
        这两组在当前筛选下没有可配对的重合样本（配对键需一致：连接组 / 扫描条件 / 种子）。
        请检查筛选范围，或换一个配对键。
      </div>
      <div v-else>
        <div class="fc-grid fc-grid-4" style="margin-bottom: 12px">
          <div class="fc-card fc-metric" :class="deltaResult.significant ? (deltaResult.favorable ? 'ok' : 'danger') : ''">
            <div class="fc-metric-label">Δ 均值（{{ deltaResult.metricLabel }}）</div>
            <div class="fc-metric-value">{{ deltaResult.mean.toFixed(3) }}</div>
            <div class="fc-metric-note">配对样本 {{ deltaResult.n }} 对</div>
          </div>
          <div class="fc-card fc-metric">
            <div class="fc-metric-label">95% 置信区间</div>
            <div class="fc-metric-value" style="font-size: 16px">
              [{{ deltaResult.low.toFixed(2) }}, {{ deltaResult.high.toFixed(2) }}]
            </div>
            <div class="fc-metric-note">标准差 {{ deltaResult.sd.toFixed(3) }}</div>
          </div>
          <div class="fc-card fc-metric" :class="deltaResult.significant ? 'ok' : 'warn'">
            <div class="fc-metric-label">区间是否跨 0</div>
            <div class="fc-metric-value" style="font-size: 16px">
              {{ deltaResult.significant ? '不跨 0' : '跨 0' }}
            </div>
            <div class="fc-metric-note">
              {{ deltaResult.significant ? '可判为显著差异' : '不支持差异结论，只能表述为方法有效' }}
            </div>
          </div>
          <div class="fc-card fc-metric" :class="deltaResult.n >= 10 ? 'ok' : 'warn'">
            <div class="fc-metric-label">样本量是否够结论</div>
            <div class="fc-metric-value" style="font-size: 16px">{{ deltaResult.n }} / 10</div>
            <div class="fc-metric-note">
              {{ deltaResult.n >= 10 ? '达到协议要求' : '协议要求 ≥10 个独立种子才允许谈结论' }}
            </div>
          </div>
        </div>
        <el-table :data="deltaResult.pairs.slice(0, 40)" size="small" max-height="240">
          <el-table-column prop="seed" label="配对键" width="90" />
          <el-table-column prop="connectome" label="连接组" min-width="150" />
          <el-table-column label="处理组" width="110">
            <template #default="{ row }"><span class="fc-num">{{ row.t.toFixed(3) }}</span></template>
          </el-table-column>
          <el-table-column label="对照组" width="110">
            <template #default="{ row }"><span class="fc-num">{{ row.c.toFixed(3) }}</span></template>
          </el-table-column>
          <el-table-column label="差值" width="110">
            <template #default="{ row }">
              <span class="fc-num" :style="{ color: row.d > 0 ? 'var(--fc-accent)' : 'var(--fc-warn)' }">
                {{ row.d.toFixed(3) }}
              </span>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title">
        <el-icon><Document /></el-icon> 产物清单
        <span class="fc-hint">命令行产物的可追溯性</span>
      </div>
      <el-table :data="data?.artifacts || []" size="small" max-height="260">
        <el-table-column prop="artifact" label="文件" min-width="240" />
        <el-table-column prop="experiment" label="类型" width="80" />
        <el-table-column prop="connectome" label="连接组" min-width="150" />
        <el-table-column prop="n_rows" label="行数" width="80" />
        <el-table-column label="可追溯" width="90">
          <template #default="{ row }">
            <span class="fc-tag" :class="row.traceable ? 'ok' : 'bad'">
              {{ row.traceable ? '是' : '否' }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="note" label="备注" min-width="240" />
      </el-table>
      <div v-if="data?.errors?.length" style="margin-top: 10px">
        <el-alert type="warning" :closable="false"
                  :title="`${data.errors.length} 个产物解析失败（不会影响其他结果）`" />
      </div>
    </div>

    <div class="fc-card">
      <div class="fc-card-title">
        <el-icon><Grid /></el-icon> 明细
        <span class="fc-hint">最多显示 400 行 · 完整数据用「导出 CSV」</span>
      </div>
      <el-table :data="tableRows" size="small" max-height="420">
        <el-table-column prop="source" label="来源" width="80" />
        <el-table-column prop="experiment" label="类型" width="80" />
        <el-table-column prop="connectome" label="连接组" min-width="140" />
        <el-table-column prop="group" label="组" width="130" />
        <el-table-column prop="seed" label="种子" width="70" />
        <el-table-column prop="axis" label="扫描轴" width="110" />
        <el-table-column prop="condition" label="条件" width="70" />
        <el-table-column v-for="(m, k) in metrics" :key="k" :label="m.label" width="110" align="right">
          <template #default="{ row }">
            <span class="fc-num" style="font-size: 11.5px">
              {{ typeof row[k] === 'number' ? row[k].toFixed(2) : '—' }}
            </span>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>
