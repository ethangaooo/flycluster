<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from '../api/client'

const runs = ref([])
const err = ref('')

async function load() {
  try {
    runs.value = (await api.runs()).filter((r) => r.status === 'finished').slice().reverse()
    err.value = ''
  } catch (e) {
    err.value = e.friendly
  }
}

// 三道门的阈值与服务端 flycore/scoring.py 保持一致
const THRESHOLDS = [
  { gate: '门1 实现可信', rule: '实时比 ≥ 0.05；脉冲丢弃率 ≤ 0.001；NaN 事件 = 0；数值有界',
    why: '输入变化能引起可解释的状态变化；相同版本、种子和输入能重复结果；模拟时间与现实时间之比被记录；脉冲丢弃、数值失控和异常重启没有被隐藏。' },
  { gate: '门2 任务收益', rule: '先用 ≥3 个种子筛查且平均增益 > 0；通过后扩到 ≥10 个独立种子并保留置信区间',
    why: '不凭一条最佳曲线下结论。真实结构与重连结构若无差异，结论必须限定为「该协作方法有效」，而不是「果蝇脑有特殊优势」。' },
  { gate: '门3 值得继续花钱', rule: '配对种子数 ≥10 且 Δ 的 95% 置信区间下界 > 0',
    why: '交互效应为正也不自动等于商业优势，还要看主实验组是否优于最好的普通模型、以及多花了多少成本。' }
]

const rows = computed(() => runs.value.slice(0, 30).map((r) => {
  const g = {}
  for (const x of r.gates || []) g[x.gate] = x
  return { run_id: r.run_id, backend: r.backend, seed: r.seed, g }
}))

onMounted(load)
</script>

<template>
  <div class="fc-page">
    <h2 class="fc-page-title">三道门</h2>
    <p class="fc-page-sub">
      预算是分两次批准的：先用有限人时判断"有没有值得继续做的信号"，通过之后再批准硬件与更长周期。
      任一门不通过，都应该如实说明阻塞原因，而不是为了凑计划继续加功能。
    </p>

    <el-alert v-if="err" type="error" :closable="false" :title="err" style="margin-bottom: 14px" />

    <div class="fc-grid fc-grid-3" style="margin-bottom: 16px">
      <div v-for="t in THRESHOLDS" :key="t.gate" class="fc-card">
        <div class="fc-card-title"><el-icon><Filter /></el-icon> {{ t.gate }}</div>
        <div style="font-size: 12px; color: var(--fc-text); line-height: 1.8; margin-bottom: 8px">
          <b>判定规则</b>：{{ t.rule }}
        </div>
        <div style="font-size: 11.5px; color: var(--fc-text-dim); line-height: 1.8">{{ t.why }}</div>
      </div>
    </div>

    <div class="fc-card">
      <div class="fc-card-title"><el-icon><Tickets /></el-icon> 最近完成运行的判定
        <span class="fc-hint">门 2 与门 3 需要跨运行聚合，单次运行只给出门 1 的结论</span></div>
      <el-empty v-if="!rows.length" description="还没有已完成的运行" />
      <el-table v-else :data="rows" size="small">
        <el-table-column prop="run_id" label="run_id" min-width="190" />
        <el-table-column prop="backend" label="结构" width="100" />
        <el-table-column prop="seed" label="种子" width="70" />
        <el-table-column v-for="t in THRESHOLDS" :key="t.gate" :label="t.gate" min-width="150">
          <template #default="{ row }">
            <template v-if="row.g[t.gate]">
              <span class="fc-tag" :class="row.g[t.gate].passed ? 'ok' : 'bad'">
                {{ row.g[t.gate].passed ? '通过' : '未通过' }}
              </span>
            </template>
            <span v-else style="color: var(--fc-text-faint); font-size: 11.5px">—</span>
          </template>
        </el-table-column>
      </el-table>
      <div v-if="rows.length" style="margin-top: 16px">
        <div class="fc-card-title" style="margin-bottom: 8px">判定明细</div>
        <el-collapse>
          <el-collapse-item v-for="r in rows" :key="r.run_id" :name="r.run_id">
            <template #title>
              <span class="fc-num" style="font-size: 12px">{{ r.run_id }}</span>
            </template>
            <div v-for="(g, k) in r.g" :key="k" style="margin-bottom: 10px">
              <span class="fc-tag" :class="g.passed ? 'ok' : 'bad'" style="margin-right: 8px">
                {{ g.passed ? '通过' : '未通过' }}
              </span>
              <b style="font-size: 12px">{{ g.gate }}</b>
              <div style="font-size: 11.5px; color: var(--fc-text-dim); line-height: 1.7; margin-top: 4px">
                {{ g.detail }}
              </div>
            </div>
          </el-collapse-item>
        </el-collapse>
      </div>
    </div>
  </div>
</template>
