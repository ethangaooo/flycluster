<script setup>
import GridView from './GridView.vue'
import MetricCard from './MetricCard.vue'

const props = defineProps({ report: { type: Object, required: true } })

const tel = () => props.report.telemetry || {}
const gateTone = (p) => (p ? 'ok' : 'bad')
</script>

<template>
  <div>
    <div class="fc-grid fc-grid-4" style="margin-bottom: 14px">
      <MetricCard label="任务得分 S" :value="report.score.score" tone="accent"
                  :note="`目标 ${report.score.targets_found}/${report.score.targets_total} · 覆盖 ${report.score.coverage}`" />
      <MetricCard label="碰撞 / 重复探索" :value="`${report.score.collisions} / ${report.score.redundant_steps}`"
                  note="两者都直接扣分，是「看起来在动」与「真的在搜」的分界" />
      <MetricCard label="实时比（最慢节点）" :value="tel().realtime_ratio_min ?? '—'"
                  :tone="(tel().realtime_ratio_min ?? 0) >= 0.05 ? 'ok' : 'danger'"
                  :note="`均值 ${tel().realtime_ratio_mean ?? '—'}；阈值 0.05`" />
      <MetricCard label="脉冲丢弃率" :value="tel().drop_rate ?? '—'"
                  :tone="(tel().drop_rate ?? 0) > 0.001 ? 'warn' : 'ok'"
                  :note="`丢弃 ${tel().dropped_spikes ?? '—'} / 脉冲 ${tel().spikes ?? '—'}`" />
    </div>

    <div class="fc-grid fc-grid-2" style="margin-bottom: 14px">
      <div class="fc-card">
        <div class="fc-card-title"><el-icon><Grid /></el-icon> 结束时的环境</div>
        <GridView v-if="report.env_snapshot" :snapshot="report.env_snapshot" />
        <el-empty v-else description="本次运行未保存环境快照" />
      </div>
      <div class="fc-card">
        <div class="fc-card-title"><el-icon><Histogram /></el-icon> 各节点动作分布</div>
        <el-table :data="report.per_agent" size="small">
          <el-table-column prop="agent_id" label="节点" width="64" />
          <el-table-column label="上 / 下 / 左 / 右 / 停" min-width="230">
            <template #default="{ row }">
              <span class="fc-num">
                {{ row.action_histogram?.up }} / {{ row.action_histogram?.down }} /
                {{ row.action_histogram?.left }} / {{ row.action_histogram?.right }} /
                {{ row.action_histogram?.stay }}
              </span>
            </template>
          </el-table-column>
          <el-table-column prop="targets_found" label="命中" width="64" />
          <el-table-column prop="collisions" label="碰撞" width="64" />
          <el-table-column prop="redundant_steps" label="重复" width="64" />
        </el-table>
        <div style="margin-top: 14px; font-size: 12px; color: var(--fc-text-dim); line-height: 1.9">
          <b>通信预算</b>：发出 {{ report.comm_accounting.messages_issued }} 条，
          送达 {{ report.comm_accounting.messages_delivered }}，
          丢弃 {{ report.comm_accounting.messages_dropped }}；
          实测 {{ report.comm_accounting.raw_bytes_per_second }} B/s，
          文档口径 {{ report.comm_accounting.formula_bytes_per_second }} B/s。
        </div>
      </div>
    </div>

    <div class="fc-grid fc-grid-2" style="margin-bottom: 14px">
      <div class="fc-card">
        <div class="fc-card-title"><el-icon><Connection /></el-icon> 能力归因
          <span class="fc-hint">不是所有成绩都该记在脑结构账上</span></div>
        <dl class="fc-attrib">
          <template v-for="(v, k) in report.attribution" :key="k">
            <dt>{{ k }}</dt>
            <dd>{{ v }}</dd>
          </template>
        </dl>
      </div>
      <div class="fc-card">
        <div class="fc-card-title"><el-icon><Filter /></el-icon> 本次运行的三道门</div>
        <div v-for="g in report.gates" :key="g.gate" style="margin-bottom: 12px">
          <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px">
            <span class="fc-tag" :class="gateTone(g.passed)">{{ g.passed ? '通过' : '未通过' }}</span>
            <b style="font-size: 12.5px">{{ g.gate }}</b>
          </div>
          <div style="font-size: 12px; color: var(--fc-text-dim); line-height: 1.7">{{ g.detail }}</div>
        </div>
      </div>
    </div>

    <div class="fc-card">
      <div class="fc-card-title"><el-icon><Memo /></el-icon> 记忆与仿真细节</div>
      <el-tabs>
        <el-tab-pane label="记忆通道">
          <div class="fc-mono-block">{{ JSON.stringify(report.memory_accounting, null, 2) }}</div>
        </el-tab-pane>
        <el-tab-pane label="通信记账">
          <div class="fc-mono-block">{{ JSON.stringify(report.comm_accounting, null, 2) }}</div>
        </el-tab-pane>
        <el-tab-pane label="仿真遥测">
          <div class="fc-mono-block">{{ JSON.stringify(report.telemetry, null, 2) }}</div>
        </el-tab-pane>
        <el-tab-pane label="运行配置">
          <div class="fc-mono-block">{{ JSON.stringify(report.config, null, 2) }}</div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </div>
</template>
