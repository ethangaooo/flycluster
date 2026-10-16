<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api/client'
import MetricCard from '../components/MetricCard.vue'

const stats = ref(null)
const conflicts = ref([])
const origin = ref('')
const chain = ref(null)
const err = ref('')

async function load() {
  try {
    stats.value = await api.memoryStats()
    conflicts.value = await api.conflicts()
    err.value = ''
  } catch (e) {
    err.value = e.friendly
  }
}

async function lookup() {
  if (!origin.value.trim()) return
  try {
    chain.value = await api.provenance(origin.value.trim())
  } catch (e) {
    ElMessage.error(e.friendly)
  }
}

async function snapshot() {
  try {
    const r = await api.snapshot('控制台手动快照')
    ElMessage.success(`已创建快照 ${r.version}（序号 ${r.seq}）`)
    load()
  } catch (e) {
    ElMessage.error(e.friendly)
  }
}

onMounted(load)
</script>

<template>
  <div class="fc-page">
    <h2 class="fc-page-title">记忆中枢</h2>
    <p class="fc-page-sub">
      四层记忆的落点：即时工作记忆只在各脑运行内存；个体经历与技能留在本地；
      群体共享记忆由中心服务统一管理来源、适用条件、冲突与版本。
      物理上集中存储不等于逻辑上共享——只有明确进入共享区的内容其他节点才读得到。
    </p>

    <el-alert v-if="err" type="error" :closable="false" :title="err" style="margin-bottom: 14px" />

    <div class="fc-grid fc-grid-4" style="margin-bottom: 14px">
      <MetricCard label="记录总数" :value="stats?.records_total ?? '—'" tone="accent" />
      <MetricCard label="原始证据" :value="stats?.records_primary ?? '—'"
                  note="只有原始观察计入独立证据" />
      <MetricCard label="派生记录" :value="stats?.records_derived ?? '—'"
                  note="采纳了检索结论，不计独立证据" />
      <MetricCard label="回声事件" :value="stats?.echo_events ?? '—'"
                  :tone="(stats?.echo_events ?? 0) > 0 ? 'warn' : 'ok'"
                  note="同源转发被识别并拒绝计数的次数" />
    </div>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title"><el-icon><Search /></el-icon> 来源链追溯
        <span class="fc-hint">输入原始事件 ID（evt-…），看它在集群里被传播了多少次、被谁独立验证过</span>
      </div>
      <div style="display: flex; gap: 10px; max-width: 620px">
        <el-input v-model="origin" placeholder="evt-run-1-node-0-12" @keyup.enter="lookup" />
        <el-button @click="lookup">追溯</el-button>
        <el-button @click="snapshot">创建快照</el-button>
      </div>
      <div v-if="chain" style="margin-top: 14px">
        <div style="display: flex; gap: 20px; font-size: 12px; color: var(--fc-text-dim); margin-bottom: 10px">
          <span>记录数 <b class="fc-num">{{ chain.n_records }}</b></span>
          <span>独立来源 <b class="fc-num" style="color: var(--fc-accent)">{{ chain.independent_sources }}</b></span>
          <span>回声次数 <b class="fc-num">{{ chain.echo_events }}</b></span>
        </div>
        <el-table :data="chain.chain" size="small">
          <el-table-column prop="record_id" label="记录" min-width="200" />
          <el-table-column prop="producer_node" label="产生节点" width="110" />
          <el-table-column prop="kind" label="类型" width="90" />
          <el-table-column label="是否原始" width="90">
            <template #default="{ row }">
              <span class="fc-tag" :class="row.is_primary ? 'ok' : 'warn'">{{ row.is_primary ? '原始' : '派生' }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="observed_at" label="时间" min-width="180" />
        </el-table>
      </div>
    </div>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title"><el-icon><Warning /></el-icon> 冲突记录
        <span class="fc-hint">同一 key 下结论不一致时并列保留，不做最后写入覆盖</span>
      </div>
      <el-empty v-if="!conflicts.length" description="当前没有冲突：同一位置/条件下的结论一致，或还没有多次观测" />
      <div v-for="c in conflicts" :key="c.key" style="margin-bottom: 14px">
        <div style="font-size: 12.5px; margin-bottom: 6px">
          <b class="fc-num">{{ c.key }}</b>
          <span class="fc-tag warn" style="margin-left: 8px">{{ c.n_conclusions }} 种结论</span>
          <span class="fc-tag" style="margin-left: 6px">{{ c.n_records }} 条记录</span>
        </div>
        <el-table :data="c.records" size="small">
          <el-table-column prop="record_id" label="记录" min-width="190" />
          <el-table-column prop="producer_node" label="节点" width="100" />
          <el-table-column label="结论" width="80">
            <template #default="{ row }"><span class="fc-num">{{ JSON.stringify(row.conclusion) }}</span></template>
          </el-table-column>
          <el-table-column prop="observed_at" label="时间" min-width="180" />
          <el-table-column label="条件" min-width="200">
            <template #default="{ row }">
              <span class="fc-num" style="font-size: 11px; color: var(--fc-text-faint)">
                {{ JSON.stringify(row.conditions) }}
              </span>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <div class="fc-card">
      <div class="fc-card-title"><el-icon><InfoFilled /></el-icon> 为什么这样设计</div>
      <div style="font-size: 12px; color: var(--fc-text-dim); line-height: 1.95">
        · <b>防回声</b>：A 产生经验、B 读到后回传，中心必须认出这是同一次原始事件。否则
        "传播给 8 个节点"会被误记成"8 个节点独立验证"。<br />
        · <b>派生不计证据</b>：只有当节点真的采纳了检索到的结论，才记为派生；仅仅是"检索返回过上下文"不算，
        否则会系统性低估计数。<br />
        · <b>冲突保留</b>：上午可通行、下午有障碍，是环境变了；同一地点两个节点判断不同，是感知差异。
        两者都应按时间与条件并存，而不是覆盖成一个无条件结论。<br />
        · <b>检索计费</b>：中心记忆本身是一条通信通道。查询返回的字节数计入通信预算，
        否则"低带宽协作"会因为共享数据库而失真。
      </div>
    </div>
  </div>
</template>
