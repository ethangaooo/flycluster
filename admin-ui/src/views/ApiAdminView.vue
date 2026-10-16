<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api } from '../api/client'

const endpoints = ref(null)
const keys = ref([])
const stats = ref({})
const caller = ref(null)
const loading = ref(false)
const newKey = ref(null)          // 只在创建/轮换后出现一次
const createDialog = ref(false)
const showKeyDialog = ref(false)
const activeGroup = ref('接入管理')

const draft = ref({ name: '', rpm: 120, scopes: ['run', 'read'], note: '' })

async function load() {
  loading.value = true
  try {
    const [e, k] = await Promise.all([api.adminEndpoints(), api.adminKeys()])
    endpoints.value = e
    keys.value = k.keys
    stats.value = k.stats
    caller.value = { name: k.caller, scopes: k.caller_scopes }
  } catch (err) {
    ElMessage.error(err.friendly)
  }
  loading.value = false
}

async function create() {
  if (!draft.value.name.trim()) return ElMessage.warning('请填写名称')
  if (!draft.value.scopes.length) return ElMessage.warning('至少选择一个作用域')
  try {
    const r = await api.createKey(draft.value)
    newKey.value = r
    showKeyDialog.value = true
    createDialog.value = false
    draft.value = { name: '', rpm: 120, scopes: ['run', 'read'], note: '' }
    await load()
  } catch (e) {
    ElMessage.error(e.friendly)
  }
}

async function rotate(row) {
  try {
    await ElMessageBox.confirm(
      `轮换后旧密钥<b>立即失效</b>，正在使用它的外部系统会马上报 401。确认轮换「${row.name}」？`,
      '轮换密钥', { dangerouslyUseHTMLString: true, type: 'warning' })
  } catch { return }
  try {
    newKey.value = await api.rotateKey(row.key_id)
    showKeyDialog.value = true
    await load()
  } catch (e) {
    ElMessage.error(e.friendly)
  }
}

async function revoke(row) {
  const isEnv = row.source === 'env'
  try {
    await ElMessageBox.confirm(
      isEnv
        ? `「${row.name}」由环境变量注入，<b>不能删除，只能吊销</b>。<br/>吊销状态会落库，重启后依然有效。确认吊销？`
        : `确认吊销「${row.name}」？吊销后该密钥立即失效，记录会保留以便审计。`,
      '吊销密钥', { dangerouslyUseHTMLString: true, type: 'warning' })
  } catch { return }
  try {
    await api.revokeKey(row.key_id, false)
    ElMessage.success('已吊销')
    await load()
  } catch (e) {
    ElMessage.error(e.friendly)
  }
}

async function copyKey() {
  try {
    await navigator.clipboard.writeText(newKey.value.key)
    ElMessage.success('已复制到剪贴板')
  } catch {
    ElMessage.warning('浏览器拒绝了剪贴板访问，请手动选中复制')
  }
}

function curlFor(e) {
  const base = endpoints.value?.base_url || 'http://127.0.0.1:8800'
  const h = e.auth ? ` \\\n  -H 'X-API-Key: <你的密钥>'` : ''
  const path = e.path.replace('{run_id}', 'YOUR_RUN_ID').replace('{origin_event_id}', 'evt-1')
                  .replace('{key_id}', 'KEY_ID')
  const method = e.method === 'GET' ? '' : ` -X ${e.method}`
  return `curl${method}${h} \\\n  '${base}${path}'`
}

const groups = computed(() => (endpoints.value?.groups)
  ? Object.keys(endpoints.value.groups) : [])
const scopesOf = (s) => (s || '').split(',').filter(Boolean)

const tone = (row) => (row.revoked_at ? 'bad' : (row.source === 'env' ? 'warn' : 'ok'))
const toneText = (row) => (row.revoked_at ? '已吊销' : (row.source === 'env' ? '环境变量' : '页面创建'))

onMounted(load)
</script>

<template>
  <div class="fc-page">
    <h2 class="fc-page-title">API 接入管理</h2>
    <p class="fc-page-sub">
      外部系统通过 <code class="fc-num">X-API-Key</code> 调用开放接口。
      密钥只在创建时显示一次明文，服务端只保存哈希——数据库泄漏不等于凭据泄漏。
    </p>

    <div class="fc-grid fc-grid-4" style="margin-bottom: 14px">
      <div class="fc-card fc-metric accent">
        <div class="fc-metric-label">密钥总数</div>
        <div class="fc-metric-value">{{ stats.total ?? '—' }}</div>
        <div class="fc-metric-note">活跃 {{ stats.active ?? '—' }} · 已吊销 {{ stats.revoked ?? '—' }}</div>
      </div>
      <div class="fc-card fc-metric">
        <div class="fc-metric-label">来源分布</div>
        <div class="fc-metric-value">{{ stats.db_sourced ?? 0 }}<span class="fc-metric-unit">页面创建</span></div>
        <div class="fc-metric-note">另有 {{ stats.env_sourced ?? 0 }} 把由环境变量注入（不可删除）</div>
      </div>
      <div class="fc-card fc-metric">
        <div class="fc-metric-label">累计调用</div>
        <div class="fc-metric-value">{{ stats.calls_total ?? '—' }}</div>
        <div class="fc-metric-note">所有密钥合计</div>
      </div>
      <div class="fc-card fc-metric">
        <div class="fc-metric-label">当前操作身份</div>
        <div class="fc-metric-value" style="font-size: 15px">{{ caller?.name ?? '—' }}</div>
        <div class="fc-metric-note">
          作用域 {{ (caller?.scopes || []).join(' / ') || '—' }}
        </div>
      </div>
    </div>

    <div class="fc-card" style="margin-bottom: 14px">
      <div class="fc-card-title">
        <el-icon><Key /></el-icon> 密钥管理
        <span class="fc-hint">环境变量注入的密钥由部署流程管理，页面只能吊销</span>
        <div style="flex: 1"></div>
        <el-button type="primary" size="small" @click="createDialog = true">创建密钥</el-button>
      </div>
      <el-table :data="keys" size="small" v-loading="loading">
        <el-table-column prop="name" label="名称" min-width="130" />
        <el-table-column label="密钥前缀" width="120">
          <template #default="{ row }">
            <span class="fc-num" style="font-size: 11.5px">{{ row.prefix }}</span>
          </template>
        </el-table-column>
        <el-table-column label="来源" width="130">
          <template #default="{ row }">
            <span class="fc-tag" :class="tone(row)">{{ toneText(row) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="作用域" width="180">
          <template #default="{ row }">
            <span v-for="s in row.scopes" :key="s" class="fc-tag" style="margin-right: 4px">{{ s }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="rpm" label="限额/分" width="80" />
        <el-table-column prop="calls_total" label="调用" width="80" />
        <el-table-column label="最近使用" min-width="150">
          <template #default="{ row }">
            <span class="fc-num" style="font-size: 11px; color: var(--fc-text-faint)">
              {{ row.last_used_at || '—' }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="note" label="备注" min-width="120" />
        <el-table-column label="操作" width="140" fixed="right">
          <template #default="{ row }">
            <el-button link size="small" @click="rotate(row)">轮换</el-button>
            <el-button link size="small" :disabled="!!row.revoked_at" @click="revoke(row)">吊销</el-button>
          </template>
        </el-table-column>
      </el-table>

      <div style="margin-top: 14px; font-size: 12px; color: var(--fc-text-dim); line-height: 1.9">
        <b>作用域</b>：
        <span class="fc-tag">run</span> 提交/中止运行、写记忆快照　
        <span class="fc-tag">read</span> 查询类接口　
        <span class="fc-tag">admin</span> 管理密钥与接入
        <br />
        <b>轮换 vs 吊销</b>：轮换会立刻作废旧密钥并生成新的（用于定期换密钥）；
        吊销只停用不删除（保留记录便于审计）。两者都不会删除历史调用统计。
      </div>
    </div>

    <div class="fc-card">
      <div class="fc-card-title">
        <el-icon><Connection /></el-icon> 开放接口清单
        <span class="fc-hint">这份清单由后端提供，避免页面与代码不同步</span>
      </div>
      <el-tabs v-model="activeGroup">
        <el-tab-pane v-for="g in groups" :key="g" :label="g" :name="g">
          <div v-for="e in endpoints.groups[g]" :key="e.method + e.path"
               style="border: 1px solid var(--fc-border); border-radius: 8px; padding: 12px 14px; margin-bottom: 10px">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px">
              <span class="fc-tag" :class="e.method === 'GET' ? '' : (e.method === 'DELETE' ? 'bad' : 'ok')">
                {{ e.method }}
              </span>
              <span class="fc-num" style="font-size: 12.5px">{{ e.path }}</span>
              <span v-if="e.scope !== '—'" class="fc-tag">scope: {{ e.scope }}</span>
              <span v-if="!e.auth" class="fc-tag warn">免鉴权</span>
            </div>
            <div style="font-size: 12px; color: var(--fc-text-dim); margin-bottom: 8px">{{ e.desc }}</div>
            <div class="fc-mono-block" style="max-height: 120px">{{ curlFor(e) }}</div>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>

    <!-- 创建密钥 -->
    <el-dialog v-model="createDialog" title="创建密钥" width="480px">
      <el-form :model="draft" label-width="86px" label-position="left">
        <el-form-item label="名称">
          <el-input v-model="draft.name" placeholder="例如 third-party-a" />
        </el-form-item>
        <el-form-item label="作用域">
          <el-checkbox-group v-model="draft.scopes">
            <el-checkbox value="run">run（提交运行）</el-checkbox>
            <el-checkbox value="read">read（查询）</el-checkbox>
            <el-checkbox value="admin">admin（管理密钥）</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
        <el-form-item label="限额/分">
          <el-input-number v-model="draft.rpm" :min="1" :max="100000" controls-position="right" />
        </el-form-item>
        <el-form-item label="备注">
          <el-input v-model="draft.note" placeholder="接入方、用途" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createDialog = false">取消</el-button>
        <el-button type="primary" @click="create">创建</el-button>
      </template>
    </el-dialog>

    <!-- 明文只显示一次 -->
    <el-dialog v-model="showKeyDialog" title="请立即保存这个密钥" width="560px" :close-on-click-modal="false">
      <el-alert type="warning" :closable="false" style="margin-bottom: 12px"
                title="这是唯一一次显示明文。服务端只保存哈希，关闭后无法再取回，只能重新轮换。" />
      <div class="fc-mono-block" style="font-size: 13px; word-break: break-all; max-height: none">
        {{ newKey?.key }}
      </div>
      <div style="margin-top: 10px; font-size: 12px; color: var(--fc-text-dim); line-height: 1.8">
        名称 <b>{{ newKey?.record?.name }}</b>　
        作用域 <b>{{ (newKey?.record?.scopes || []).join(', ') }}</b>　
        限额 <b>{{ newKey?.record?.rpm }}</b> 次/分
      </div>
      <template #footer>
        <el-button @click="copyKey">复制</el-button>
        <el-button type="primary" @click="showKeyDialog = false">我已保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>
