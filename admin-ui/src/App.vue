<script setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { api, getApiKey, setApiKey } from './api/client'

const route = useRoute()
const keyInput = ref(getApiKey())
const health = ref(null)
const keyDialog = ref(false)
const err = ref('')

const nav = [
  { path: '/task', label: '任务下发', icon: 'Promotion' },
  { path: '/cluster', label: '集群总览', icon: 'Odometer' },
  { path: '/runs', label: '运行与实验', icon: 'VideoPlay' },
  { path: '/memory', label: '记忆中枢', icon: 'Coin' },
  { path: '/games', label: '果蝇对弈', icon: 'Grid' },
  { path: '/analytics', label: '结果分析', icon: 'TrendCharts' },
  { path: '/experiments', label: '对照结果', icon: 'DataAnalysis' },
  { path: '/gates', label: '三道门', icon: 'Filter' },
  { path: '/api', label: 'API 接入', icon: 'Key' }
]

// 主题：跟着本机偏好走，但允许手动固定。
// 之前界面是写死的深色，没有切换机制，用户只能在深色下忍受对比度问题。
const theme = ref(localStorage.getItem('fc-theme') || 'dark')
function applyTheme() {
  document.documentElement.classList.toggle('dark', theme.value === 'dark')
  document.documentElement.classList.toggle('light', theme.value === 'light')
  localStorage.setItem('fc-theme', theme.value)
}
function toggleTheme() {
  theme.value = theme.value === 'dark' ? 'light' : 'dark'
  applyTheme()
}

async function refresh() {
  try {
    health.value = await api.health()
    err.value = ''
  } catch (e) {
    health.value = null
    err.value = e.friendly
  }
}

function saveKey() {
  setApiKey(keyInput.value.trim())
  keyDialog.value = false
  refresh()
}

onMounted(() => { applyTheme(); refresh() })
</script>

<template>
  <el-container style="height: 100%">
    <el-aside width="212px" style="border-right: 1px solid var(--fc-border); background: var(--fc-surface)">
      <div style="padding: 18px 18px 14px">
        <div style="display: flex; align-items: center; gap: 9px">
          <div style="width: 24px; height: 24px; border-radius: 7px; background: linear-gradient(135deg, var(--fc-accent), #1f6f6a); display:flex;align-items:center;justify-content:center; font-size: 13px">🧭</div>
          <div>
            <div style="font-size: 13.5px; font-weight: 600; letter-spacing: .3px">FlyCluster</div>
            <div style="font-size: 10.5px; color: var(--fc-text-faint)">连接组约束多智能体验证</div>
          </div>
        </div>
      </div>

      <el-menu :default-active="route.path" router class="fc-nav-menu">
        <el-menu-item v-for="n in nav" :key="n.path" :index="n.path">
          <el-icon><component :is="n.icon" /></el-icon>
          <span>{{ n.label }}</span>
        </el-menu-item>
      </el-menu>

      <div style="position: absolute; bottom: 0; width: 212px; padding: 14px 18px; border-top: 1px solid var(--fc-border)">
        <div style="display: flex; align-items: center; gap: 7px; font-size: 11.5px; color: var(--fc-text-dim)">
          <span class="fc-dot" :class="health ? 'ok' : 'bad'"></span>
          <span>{{ health ? '中枢在线' : '中枢不可达' }}</span>
        </div>
        <div v-if="health" class="fc-num" style="font-size: 10.5px; color: var(--fc-text-faint); margin-top: 6px">
          运行 {{ health.runs_total }} 次 · {{ health.brain_urls ? 'docker' : 'in-process' }}
        </div>
        <div style="display: flex; gap: 10px; margin-top: 8px">
          <el-button link size="small" style="color: var(--fc-text-dim)" @click="keyDialog = true">
            设置 API Key
          </el-button>
          <el-button link size="small" style="color: var(--fc-text-dim)" @click="toggleTheme">
            {{ theme === 'dark' ? '切换到亮色' : '切换到暗色' }}
          </el-button>
        </div>
      </div>
    </el-aside>

    <el-container>
      <el-header height="46px" style="border-bottom: 1px solid var(--fc-border); background: var(--fc-surface); display: flex; align-items: center; justify-content: space-between; padding: 0 22px">
        <div style="font-size: 12.5px; color: var(--fc-text-dim)">{{ route.meta.title || '' }}</div>
        <div style="display: flex; align-items: center; gap: 12px">
          <el-tag v-if="err" type="danger" size="small" effect="dark">{{ err }}</el-tag>
          <el-button size="small" :icon="'Refresh'" @click="refresh">刷新状态</el-button>
        </div>
      </el-header>

      <el-main style="padding: 0; overflow: auto">
        <router-view />
      </el-main>
    </el-container>

    <el-dialog v-model="keyDialog" title="开放 API 密钥" width="420px">
      <div style="font-size: 12px; color: var(--fc-text-dim); line-height: 1.7; margin-bottom: 10px">
        密钥只保存在浏览器本地存储。服务端的取值来自环境变量
        <code class="fc-num">FLYCLUSTER_API_KEYS</code>（格式 <code class="fc-num">name:key[:每分钟上限]</code>）。
      </div>
      <el-input v-model="keyInput" placeholder="X-API-Key" show-password />
      <template #footer>
        <el-button @click="keyDialog = false">取消</el-button>
        <el-button type="primary" @click="saveKey">保存并重连</el-button>
      </template>
    </el-dialog>
  </el-container>
</template>
