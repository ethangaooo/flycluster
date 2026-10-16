<script setup>
import { onMounted, ref } from 'vue'
import { api } from '../api/client'
import MetricCard from '../components/MetricCard.vue'

const cluster = ref(null)
const mem = ref(null)
const err = ref('')
const loading = ref(false)

async function load() {
  loading.value = true
  try {
    cluster.value = await api.cluster()
    err.value = ''
  } catch (e) {
    err.value = e.friendly
  }
  try {
    mem.value = await api.memoryStats()
  } catch (e) {
    mem.value = null
  }
  loading.value = false
}

const ratioTone = (v) => (v >= 0.05 ? 'ok' : 'danger')

onMounted(load)
</script>

<template>
  <div class="fc-page">
    <h2 class="fc-page-title">集群总览</h2>
    <p class="fc-page-sub">
      实时比与脉冲丢弃率必须与任何性能数字同时出现：一个"跑得快"的仿真，如果靠丢弃大量神经事件换来的，
      它跑的已经不是同一个模型。
    </p>

    <el-alert v-if="err" type="error" :closable="false" :title="err" style="margin-bottom: 14px" />

    <div class="fc-grid fc-grid-4" style="margin-bottom: 14px">
      <MetricCard label="运行模式" :value="cluster?.mode === 'docker' ? 'Docker 多容器' : '单进程'" tone="accent"
                  note="docker：脑节点为独立容器；in-process：同一进程内构造" />
      <MetricCard label="脑节点在线" :value="`${(cluster?.nodes || []).filter(n => n.reachable).length} / ${(cluster?.nodes || []).length || 0}`"
                  :tone="(cluster?.nodes || []).every(n => n.reachable) ? 'ok' : 'warn'"
                  note="每节点一个独立果蝇脑实例" />
      <MetricCard label="记忆记录" :value="mem?.records_total ?? '—'"
                  :note="`原始 ${mem?.records_primary ?? 0} · 派生 ${mem?.records_derived ?? 0} · 回声 ${mem?.echo_events ?? 0}`" />
      <MetricCard label="记忆检索计费" :value="mem?.query_cost_bytes_total ?? '—'" unit="B"
                  note="中心记忆也是一条通信通道，检索返回量必须计入预算" />
    </div>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title">
        <el-icon><Monitor /></el-icon> 脑节点
        <span class="fc-hint">连接组规模 / 放电遥测 / 神经元分组</span>
      </div>
      <el-table :data="cluster?.nodes || []" size="small" style="width: 100%" v-loading="loading">
        <el-table-column prop="node_id" label="节点" width="90" />
        <el-table-column label="状态" width="90">
          <template #default="{ row }">
            <span class="fc-dot" :class="row.reachable ? 'ok' : 'bad'"></span>
            <span style="margin-left: 6px; font-size: 11.5px">{{ row.reachable ? '在线' : '离线' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="连接组" width="150">
          <template #default="{ row }">{{ row.health?.connectome?.name || '—' }}</template>
        </el-table-column>
        <el-table-column label="神经元" width="90">
          <template #default="{ row }">
            <span class="fc-num">{{ row.health?.connectome?.neuron_count ?? '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="连接边" width="100">
          <template #default="{ row }">
            <span class="fc-num">{{ row.health?.connectome?.edge_count ?? '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="实时比" width="100">
          <template #default="{ row }">
            <span class="fc-num" :style="{ color: ratioTone(row.health?.simulation?.realtime_ratio) === 'ok' ? 'var(--fc-ok)' : 'var(--fc-danger)' }">
              {{ row.health?.simulation?.realtime_ratio ?? '—' }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="丢弃脉冲" width="100">
          <template #default="{ row }">
            <span class="fc-num">{{ row.health?.simulation?.dropped_spikes ?? '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="NaN 事件" width="90">
          <template #default="{ row }">
            <span class="fc-num" :style="{ color: (row.health?.simulation?.nan_events ?? 0) > 0 ? 'var(--fc-danger)' : 'var(--fc-text-dim)' }">
              {{ row.health?.simulation?.nan_events ?? '—' }}
            </span>
          </template>
        </el-table-column>
        <el-table-column label="学习 / 记忆范围" min-width="180">
          <template #default="{ row }">
            <span class="fc-tag">{{ row.health?.learning_mode || '—' }}</span>
            <span class="fc-tag" style="margin-left: 6px">{{ row.health?.memory_scope || '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="URL" min-width="180">
          <template #default="{ row }">
            <span class="fc-num" style="font-size: 11px; color: var(--fc-text-faint)">{{ row.url }}</span>
          </template>
        </el-table-column>
      </el-table>
      <div v-if="cluster?.mode === 'in-process'" style="margin-top: 12px; font-size: 12px; color: var(--fc-text-faint); line-height: 1.7">
        当前为单进程模式，脑节点由中枢调度在进程内直接构造，因此这里没有独立容器可展示。
        用 <code class="fc-num">docker compose up --build</code> 启动即可看到 4 个独立脑容器的遥测。
      </div>
    </div>

    <div class="fc-grid fc-grid-2">
      <div class="fc-card">
        <div class="fc-card-title"><el-icon><Coin /></el-icon> 记忆中枢统计</div>
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="记录总数">{{ mem?.records_total ?? '—' }}</el-descriptions-item>
          <el-descriptions-item label="原始证据">{{ mem?.records_primary ?? '—' }}</el-descriptions-item>
          <el-descriptions-item label="派生记录">{{ mem?.records_derived ?? '—' }}</el-descriptions-item>
          <el-descriptions-item label="独立事件源">{{ mem?.distinct_origins ?? '—' }}</el-descriptions-item>
          <el-descriptions-item label="回声事件">{{ mem?.echo_events ?? '—' }}</el-descriptions-item>
          <el-descriptions-item label="快照数">{{ mem?.snapshots ?? '—' }}</el-descriptions-item>
          <el-descriptions-item label="当前版本">{{ mem?.head_version ?? '—' }}</el-descriptions-item>
          <el-descriptions-item label="检索次数">{{ mem?.query_count ?? '—' }}</el-descriptions-item>
        </el-descriptions>
      </div>
      <div class="fc-card">
        <div class="fc-card-title"><el-icon><InfoFilled /></el-icon> 判读提示</div>
        <div style="font-size: 12px; color: var(--fc-text-dim); line-height: 1.9">
          · <b>回声事件 &gt; 0</b>：说明有节点把同一原始事件当成自己的经历回传，中心已识别并拒绝计入独立证据。<br />
          · <b>原始 / 派生</b>：只有原始记录计入独立证据；采纳他人结论的派生记录不计。<br />
          · <b>实时比</b>：模拟时间 ÷ 现实时间。低于 0.05 时"跑通"没有意义。<br />
          · <b>丢弃脉冲</b>：单步上限触发后的丢弃量。丢弃率高说明实际模型已被改变。
        </div>
      </div>
    </div>
  </div>
</template>
