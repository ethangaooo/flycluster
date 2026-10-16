<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api/client'

const catalog = ref(null)
const state = ref(null)
const decision = ref(null)
const pcMove = ref(null)
const busy = ref(false)
const sid = ref('')

const cfg = ref({
  game: 'gomoku',
  difficulty: 'normal',
  structure: 'brain',
  connectome: 'male-cns-gateway',
  seed: 7,
  learning: false,
  use_goal_info: true,
  limit: 24
})

const lastReward = ref(null)

// ---- 思考过程动画
const frame = ref(0)
const playing = ref(false)
let timer = null

const frames = computed(() => decision.value?.chosen_probe?.frames || [])
const maxFrame = computed(() => Math.max(0, frames.value.length - 1))

function play() {
  if (!frames.value.length) return
  stop()
  playing.value = true
  timer = setInterval(() => {
    frame.value = frame.value >= maxFrame.value ? 0 : frame.value + 1
  }, 45)
}
function stop() {
  if (timer) clearInterval(timer)
  timer = null
  playing.value = false
}
onUnmounted(stop)

// ---- 棋盘渲染：三个棋种共用同一套几何（引擎返回归一化坐标与线段）
const cells = computed(() => state.value?.cells || [])
const lines = computed(() => state.value?.lines || [])

// 棋子半径按"洞位间最小距离"推——这样一套渲染器就能适配 15×15 / 9×10 / 121 洞星形
const minDist = computed(() => {
  const c = cells.value
  if (c.length < 2) return 0.05
  const n = Math.min(c.length, 200)
  let best = 1
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = Math.hypot(c[i].x - c[j].x, c[i].y - c[j].y)
      if (d > 1e-6 && d < best) best = d
    }
  }
  return best
})
const radius = computed(() => Math.min(14, Math.max(0.9, minDist.value * 100 * 0.42)))

const SYM = { fly: '帅', pc: '将' }
// 棋子：实色填充 + 亮边 + 深色字，三件套缺一不可。
// 之前只用了半透明的背景色，在深色棋盘上直接糊成一片。
function fillOf(piece) {
  return piece === 'fly' ? 'var(--fc-piece-fly)' : 'var(--fc-piece-pc)'
}
function edgeOf(piece) {
  return piece === 'fly' ? 'var(--fc-piece-fly-edge)' : 'var(--fc-piece-pc-edge)'
}
function inkOf(piece) {
  return piece === 'fly' ? 'var(--fc-piece-fly-ink)' : 'var(--fc-piece-pc-ink)'
}
function groupOf(piece) {
  return piece === 'fly' ? 'fly' : 'pc'
}

// 候选倾向分：按值排序取前若干
const topCandidates = computed(() => {
  const cs = decision.value?.candidates || []
  const withV = cs.filter((c) => typeof c.value === 'number')
  if (!withV.length) return []
  const sorted = [...withV].sort((a, b) => b.value - a.value)
  const lo = sorted[sorted.length - 1].value
  const hi = sorted[0].value
  const span = Math.max(1e-9, hi - lo)
  return sorted.slice(0, 14).map((c) => ({
    ...c,
    norm: (c.value - lo) / span,
    chosen: c.label === decision.value.move_label
  }))
})

const raster = computed(() => {
  const f = frames.value
  if (!f.length) return { w: 120, h: 60, dots: [] }
  let maxIdx = 1
  for (const fr of f) for (const n of fr.firing) maxIdx = Math.max(maxIdx, n)
  const dots = []
  f.forEach((fr, fi) => {
    for (const n of fr.firing) dots.push({ x: fi, y: n / maxIdx })
  })
  return { w: Math.max(1, f.length - 1), h: 1, dots, maxIdx }
})

const groupSeries = computed(() => {
  const gp = decision.value?.chosen_probe?.group_per_step || {}
  const keys = Object.keys(gp)
  const COLORS = { sensor: '#35c9c0', central: '#7c9cff', action_out: '#d9a441',
                   memory: '#8fd14f', peer: '#e07bd0', message_out: '#e5534b' }
  return keys.map((k) => ({ key: k, color: COLORS[k] || '#9aa7b4',
                            values: gp[k], total: gp[k].reduce((a, b) => a + b, 0) }))
})

// ---- 交互
async function loadCatalog() {
  try {
    catalog.value = await api.gameCatalog()
  } catch (e) { ElMessage.error(e.friendly) }
}

async function newGame() {
  busy.value = true; stop(); frame.value = 0
  try {
    const r = await api.gameNew(cfg.value)
    sid.value = r.session_id
    state.value = r.state
    decision.value = null
    pcMove.value = null
  } catch (e) { ElMessage.error(e.friendly) }
  busy.value = false
}

async function ply() {
  if (!sid.value) return
  busy.value = true
  try {
    const r = await api.gamePly(sid.value, { limit: cfg.value.limit })
    state.value = r.state
    if (r.fly) { decision.value = r.fly; frame.value = 0; play() }
    pcMove.value = r.pc
    lastReward.value = r.reward || null
  } catch (e) { ElMessage.error(e.friendly) }
  busy.value = false
}

async function autoPlay(n = 6) {
  for (let i = 0; i < n; i++) {
    if (state.value?.meta?.over) break
    await ply()
  }
}

const structureNote = computed(() =>
  (catalog.value?.structures || []).find((s) => s.key === cfg.value.structure)?.note || '')

onMounted(loadCatalog)
</script>

<template>
  <div class="fc-page">
    <h2 class="fc-page-title">果蝇集群对弈</h2>
    <p class="fc-page-sub">
      让连接组派生的脉冲网络集群与电脑对手下棋，并把它每一步的神经活动画出来。
      <b>这是演示，不是在主张果蝇具备棋类认知</b>——脑只提供脉冲动力学，
      棋盘编码、候选剪枝、倾向分投影、胜负判定、电脑搜索全部是外围代码。
    </p>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title"><el-icon><Setting /></el-icon> 对局设置</div>
      <div class="fc-grid fc-grid-4">
        <el-form-item label="棋种">
          <el-select v-model="cfg.game" style="width: 100%">
            <el-option v-for="g in catalog?.games || []" :key="g.key"
                       :label="`${g.label} · ${g.desc}`" :value="g.key" />
          </el-select>
        </el-form-item>
        <el-form-item label="电脑难度">
          <el-select v-model="cfg.difficulty" style="width: 100%">
            <el-option v-for="d in catalog?.difficulties || []" :key="d.key"
                       :label="`${d.label}（搜索 ${d.depth} 层）`" :value="d.key" />
          </el-select>
        </el-form-item>
        <el-form-item label="飞蝇结构">
          <el-select v-model="cfg.structure" style="width: 100%">
            <el-option v-for="s in catalog?.structures || []" :key="s.key"
                       :label="s.label" :value="s.key" />
          </el-select>
        </el-form-item>
        <el-form-item label="连接组">
          <el-select v-model="cfg.connectome" style="width: 100%">
            <el-option label="male-cns-gateway（891 神经元）" value="male-cns-gateway" />
            <el-option label="male-cns-large（6000 神经元）" value="male-cns-large" />
          </el-select>
        </el-form-item>
      </div>
      <div class="fc-grid fc-grid-4">
        <el-form-item label="随机种子">
          <el-input-number v-model="cfg.seed" :min="0" :max="9999" controls-position="right" />
        </el-form-item>
        <el-form-item label="候选上限">
          <el-input-number v-model="cfg.limit" :min="1" :max="60" controls-position="right" />
        </el-form-item>
        <el-form-item label="读出器学习">
          <el-switch v-model="cfg.learning" />
        </el-form-item>
        <el-form-item label="知道'什么是赢'">
          <el-switch v-model="cfg.use_goal_info" />
        </el-form-item>
        <el-form-item label=" ">
          <div style="display: flex; gap: 8px">
            <el-button type="primary" :loading="busy" @click="newGame">新开一局</el-button>
            <el-button :disabled="!sid || busy || state?.meta?.over" @click="ply">走一步</el-button>
            <el-button :disabled="!sid || busy || state?.meta?.over" @click="autoPlay(6)">连走 6 步</el-button>
          </div>
        </el-form-item>
      </div>
      <div v-if="lastReward" class="fc-card" style="margin-bottom: 12px; padding: 10px 12px">
        <div style="font-size: 12px; font-weight: 500; margin-bottom: 6px">
          它因为什么被奖励（{{ lastReward.kind }}）
        </div>
        <div style="display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px">
          <span>奖励值
            <b class="fc-num" :style="{ color: lastReward.value >= 0 ? 'var(--fc-ok)' : 'var(--fc-danger)' }">
              {{ lastReward.value >= 0 ? '+' : '' }}{{ lastReward.value }}
            </b>
          </span>
          <span>我有一步取胜：<b>{{ lastReward.detail?.i_can_win_now ? '是' : '否' }}</b></span>
          <span>对方有一步取胜：<b>{{ lastReward.detail?.opp_can_win_now ? '是' : '否' }}</b></span>
        </div>
      </div>
      <div style="font-size: 11.5px; color: var(--fc-text-faint)">
        {{ structureNote }}
        <span v-if="cfg.learning">
          　·　读出器学习开启：提升只能归因于<b>外围读出器</b>，不能表述为"脑内学习"。
        </span>
      </div>
    </div>

    <div style="display: grid; grid-template-columns: minmax(320px, 1.1fr) minmax(340px, 1fr); gap: 14px">
      <div class="fc-card">
        <div class="fc-card-title">
          <el-icon><Grid /></el-icon> 棋盘
          <span v-if="state" class="fc-hint">
            第 {{ state.meta.move_no }} 步 ·
            轮到 {{ state.meta.turn === 'fly' ? '飞蝇' : '电脑' }}
          </span>
          <div style="flex: 1"></div>
          <span v-if="state" class="fc-tag" :class="state.meta.over ? 'bad' : 'ok'">
            {{ state.meta.over
              ? (state.meta.winner === 'fly' ? '飞蝇胜' : state.meta.winner === 'pc' ? '电脑胜' : '和棋')
              : '进行中' }}
          </span>
        </div>

        <svg v-if="cells.length" viewBox="-4 -4 108 108" style="width: 100%; background: var(--fc-board-bg); border-radius: 8px">
          <line v-for="(l, i) in lines" :key="'l' + i"
                :x1="l.x1 * 100" :y1="l.y1 * 100" :x2="l.x2 * 100" :y2="l.y2 * 100"
                :stroke="l.kind === 'river' ? 'var(--fc-board-line-strong)' : 'var(--fc-board-line)'"
                :stroke-width="l.width ? l.width * 0.6 : (l.kind === 'river' ? 0.7 : 0.35)"
                :stroke-dasharray="l.dash ? '1.5 1.5' : ''" />
          <circle v-for="(c, i) in cells" :key="'c' + i"
                  :cx="c.x * 100" :cy="c.y * 100" :r="radius"
                  :fill="c.piece ? fillOf(c.piece) : (c.zone && c.zone !== 'neutral' ? 'var(--fc-cell-hollow)' : 'none')"
                  :stroke="c.piece ? edgeOf(c.piece) : 'var(--fc-piece-hint)'"
                  :stroke-width="c.piece ? 0.7 : 0.3" />
          <text v-if="state?.game === 'xiangqi'"
                v-for="(c, i) in cells.filter(x => x.piece)" :key="'t' + i"
                :x="c.x * 100" :y="c.y * 100" text-anchor="middle" dominant-baseline="central"
                :font-size="radius * 1.45" :fill="inkOf(c.piece)"
                style="font-weight: 600">{{ c.symbol }}</text>
          <circle v-for="(mk, i) in (state?.marks || [])" :key="'m' + i"
                  :cx="mk.x * 100" :cy="mk.y * 100" :r="radius * 1.75"
                  fill="none" stroke="var(--fc-info)" stroke-width="0.7" />
        </svg>
        <el-empty v-else description="点「新开一局」开始" />

        <div v-if="state" style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; font-size: 12px">
          <span class="fc-tag">飞蝇 {{ state.meta.fly_in_goal ?? '—' }} 进目标区</span>
          <span v-if="pcMove" class="fc-tag">电脑上一步 {{ pcMove.label }}（{{ pcMove.elapsed_ms }}ms）</span>
          <span v-if="state.heuristic_curve?.length" class="fc-tag">
            局面评估 {{ state.heuristic_curve[state.heuristic_curve.length - 1].toFixed(3) }}
          </span>
        </div>
      </div>

      <div class="fc-card">
        <div class="fc-card-title">
          <el-icon><TrendCharts /></el-icon> 果蝇的思考过程
          <div style="flex: 1"></div>
          <span v-if="decision" class="fc-tag ok">{{ decision.elapsed_ms }} ms</span>
        </div>

        <div v-if="!decision" style="font-size: 12px; color: var(--fc-text-faint)">
          走一步之后，这里会显示它如何逐个掂量每个落点，以及选中落点时的神经活动。
        </div>

        <template v-else>
          <div style="font-size: 12px; color: var(--fc-text-dim); margin-bottom: 8px">
            在 <b>{{ decision.candidates.length }}</b> 个候选落点中，选中了
            <b style="color: var(--fc-text-info)">{{ decision.move_label }}</b>。
            每个候选都从<b>逐位相同</b>的初始状态出发（含噪声序列），唯一差别是观测。
          </div>

          <div style="font-size: 12px; font-weight: 500; margin-bottom: 6px">候选倾向分（top 14）</div>
          <div v-for="c in topCandidates" :key="c.label"
               style="display: flex; align-items: center; gap: 8px; margin-bottom: 3px">
            <span class="fc-num" style="width: 52px; font-size: 11px"
                  :style="{ color: c.chosen ? 'var(--fc-piece-fly-ink)' : 'var(--fc-text-dim)' }">
              {{ c.label }}
            </span>
            <span style="flex: 1; height: 12px; background: var(--fc-surface-2); border-radius: 2px; position: relative">
              <span :style="{ width: Math.max(2, c.norm * 100) + '%', height: '100%', display: 'block',
                              borderRadius: '2px',
                              background: c.chosen ? 'var(--fc-piece-fly)' : 'var(--fc-cell-hollow)',
                              border: c.chosen ? '0.5px solid var(--fc-info)' : 'none' }"></span>
            </span>
            <span class="fc-num" style="width: 62px; text-align: right; font-size: 11px; color: var(--fc-text-faint)">
              {{ c.value.toFixed(4) }}
            </span>
          </div>

          <div style="display: flex; align-items: center; gap: 10px; margin: 14px 0 6px">
            <span style="font-size: 12px; font-weight: 500">
              选中落点的神经活动（{{ decision.chosen_probe?.steps }} 个仿真步）
            </span>
            <div style="flex: 1"></div>
            <el-button size="small" @click="playing ? stop() : play()">
              {{ playing ? '暂停' : '播放' }}
            </el-button>
          </div>

          <svg v-if="frames.length" viewBox="0 0 100 26" style="width: 100%; background: var(--fc-surface-2); border-radius: 6px">
            <rect :x="(frame / Math.max(1, raster.w)) * 100" y="0"
                  :width="100 / Math.max(1, raster.w)" height="24"
                  fill="var(--fc-info-bg)" opacity="1" />
            <circle v-for="(d, i) in raster.dots" :key="i"
                    :cx="(d.x / Math.max(1, raster.w)) * 100" :cy="24 - d.y * 22"
                    r="0.35" fill="var(--fc-info)" />
            <line x1="0" y1="25" x2="100" y2="25" stroke="var(--fc-piece-hint)" stroke-width="0.2" />
          </svg>
          <div style="font-size: 11px; color: var(--fc-text-faint); margin-top: 4px">
            横轴 = 仿真步（当前第 {{ frame + 1 }} / {{ frames.length }} 步）；
            纵轴 = 神经元编号。共 {{ decision.chosen_probe?.spikes_total }} 次放电。
          </div>

          <div style="margin-top: 12px">
            <div style="font-size: 12px; font-weight: 500; margin-bottom: 6px">各组放电量</div>
            <div v-for="g in groupSeries" :key="g.key"
                 style="display: flex; align-items: center; gap: 8px; margin-bottom: 3px">
              <span class="fc-num" style="width: 78px; font-size: 11px; color: var(--fc-text-dim)">
                {{ g.key }}
              </span>
              <span style="flex: 1; height: 10px; background: var(--fc-surface-2); border-radius: 2px">
                <span :style="{ width: (g.total / Math.max(1, Math.max(...groupSeries.map(x => x.total)))) * 100 + '%',
                                height: '100%', display: 'block', borderRadius: '2px', background: g.color }"></span>
              </span>
              <span class="fc-num" style="width: 46px; text-align: right; font-size: 11px; color: var(--fc-text-faint)">
                {{ g.total }}
              </span>
            </div>
          </div>
        </template>
      </div>
    </div>

    <div v-if="state?.moves?.length" class="fc-card" style="margin-top: 14px">
      <div class="fc-card-title"><el-icon><Document /></el-icon> 对局记录</div>
      <el-table :data="[...state.moves].reverse()" size="small" max-height="240">
        <el-table-column label="手数" width="70">
          <template #default="{ $index }">{{ state.moves.length - $index }}</template>
        </el-table-column>
        <el-table-column label="方" width="90">
          <template #default="{ row }">
            <span class="fc-tag" :class="row.side === 'fly' ? 'ok' : ''">
              {{ row.side === 'fly' ? '飞蝇' : '电脑' }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="label" label="着法" min-width="180" />
        <el-table-column prop="structure" label="结构" width="120" />
        <el-table-column label="耗时" width="100">
          <template #default="{ row }"><span class="fc-num">{{ row.elapsed_ms }}ms</span></template>
        </el-table-column>
      </el-table>
    </div>

    <div v-if="catalog" class="fc-card" style="margin-top: 14px">
      <div class="fc-card-title"><el-icon><InfoFilled /></el-icon> 能力归因（哪些不是脑做的）</div>
      <div style="font-size: 12px; line-height: 1.95; color: var(--fc-text-dim)">
        <b>脑负责：</b>{{ catalog.attribution.brain }}<br />
        <b>外围代码负责：</b>
        <ul style="margin: 4px 0 0 18px; padding: 0">
          <li v-for="(p, i) in catalog.attribution.periphery" :key="i">{{ p }}</li>
        </ul>
        <div style="margin-top: 8px; color: var(--fc-warn)">{{ catalog.attribution.note }}</div>
        <div style="margin-top: 8px">
          <b>倾向分是怎么算出来的：</b>把网络 891 个神经元的<b>膜电位向量</b>减均值后，
          经一个固定随机投影压成标量。用膜电位而不是脉冲计数是实测决定的——
          读出组只有 48 个神经元、20 步里放电约 5 次，整数计数<b>量化太粗</b>，
          24 个候选里大半会拿到逐位相同的向量：

          <table style="margin-top: 6px; border-collapse: collapse; font-size: 11.5px">
            <tr><td style="padding: 1px 12px 1px 0; color: var(--fc-text-faint)">读出源</td>
                <td style="padding: 1px 12px 1px 0">12 个不同观测里有几个拿到不同取值</td></tr>
            <tr><td style="padding: 1px 12px 1px 0; color: var(--fc-text-faint)">脉冲计数（20 步 / 120 步）</td>
                <td style="padding: 1px 12px 1px 0">1/12 ／ 2/12</td></tr>
            <tr><td style="padding: 1px 12px 1px 0; color: var(--fc-text-faint)">膜电位（20 步）</td>
                <td style="padding: 1px 12px 1px 0; color: var(--fc-ok)">12/12</td></tr>
          </table>
          <div style="margin-top: 6px; color: var(--fc-text-faint)">
            读出权重是固定随机值（不是学出来的）。<b>"怎么读"属于外围选择</b>——
            脑只负责提供动力学，换一种读法不会改变这一点。
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
