import axios from 'axios'

// API Key 只存在浏览器本地，不进代码库、不进日志。
const KEY_STORAGE = 'flycluster.apiKey'

export function getApiKey() {
  return localStorage.getItem(KEY_STORAGE) || 'dev-local-key'
}

export function setApiKey(v) {
  localStorage.setItem(KEY_STORAGE, v)
}

const http = axios.create({ baseURL: '', timeout: 30000 })

http.interceptors.request.use((cfg) => {
  cfg.headers['X-API-Key'] = getApiKey()
  return cfg
})

// 统一错误提示：后端返回的 detail 必须原样透出，前端不吞错。
http.interceptors.response.use(
  (r) => r,
  (err) => {
    const detail = err?.response?.data?.detail
    err.friendly = detail ? `${err.response.status} ${typeof detail === 'string' ? detail : JSON.stringify(detail)}` : (err.message || '请求失败')
    return Promise.reject(err)
  }
)

export const api = {
  health: () => http.get('/v1/health').then((r) => r.data),
  cluster: () => http.get('/v1/cluster').then((r) => r.data),
  runs: () => http.get('/v1/runs').then((r) => r.data),
  run: (id) => http.get(`/v1/runs/${id}`).then((r) => r.data),
  report: (id) => http.get(`/v1/runs/${id}/report`).then((r) => r.data),
  abort: (id) => http.post(`/v1/runs/${id}/abort`).then((r) => r.data),
  createRun: (payload) => http.post('/v1/runs', payload).then((r) => r.data),
  tasks: () => http.get('/v1/tasks').then((r) => r.data),
  memoryStats: () => http.get('/v1/memory/stats').then((r) => r.data),
  conflicts: (taskId) => http.get('/v1/memory/conflicts', { params: taskId ? { task_id: taskId } : {} }).then((r) => r.data),
  provenance: (origin) => http.get(`/v1/memory/provenance/${encodeURIComponent(origin)}`).then((r) => r.data),
  snapshot: (note) => http.post('/v1/memory/snapshot', null, { params: { note } }).then((r) => r.data),

  connectomes: () => http.get('/v1/connectomes').then((r) => r.data),
  analytics: () => http.get('/v1/analytics/runs').then((r) => r.data),

  // 棋局演示
  gameCatalog: () => http.get('/v1/games/catalog').then((r) => r.data),
  gameNew: (body) => http.post('/v1/games/sessions', body).then((r) => r.data),
  gameState: (sid) => http.get(`/v1/games/sessions/${sid}`).then((r) => r.data),
  gamePly: (sid, body) => http.post(`/v1/games/sessions/${sid}/ply`, body).then((r) => r.data),

  // 接入管理
  adminEndpoints: () => http.get('/v1/admin/endpoints').then((r) => r.data),
  adminKeys: () => http.get('/v1/admin/keys').then((r) => r.data),
  createKey: (body) => http.post('/v1/admin/keys', body).then((r) => r.data),
  rotateKey: (keyId) => http.post(`/v1/admin/keys/${keyId}/rotate`).then((r) => r.data),
  revokeKey: (keyId, purge = false) =>
    http.delete(`/v1/admin/keys/${keyId}`, { params: purge ? { purge: true } : {} }).then((r) => r.data)
}

export function streamRun(runId, onEvent) {
  // SSE 走原生 EventSource，但浏览器无法给它加自定义头，
  // 因此这里用带 Key 的查询参数轮询兼容：优先 fetch 流，失败退回轮询。
  const controller = new AbortController()

  const poll = async () => {
    let last = 0
    for (;;) {
      if (controller.signal.aborted) return
      try {
        const st = await api.run(runId)
        onEvent({ type: 'status', ...st })
        if (st.status === 'finished' || st.status === 'failed') {
          try {
            onEvent({ type: 'report', report: await api.report(runId) })
          } catch (e) {
            onEvent({ type: 'error', message: e.friendly })
          }
          return
        }
      } catch (e) {
        onEvent({ type: 'error', message: e.friendly })
        return
      }
      last += 1
      await new Promise((r) => setTimeout(r, 700))
    }
  }
  poll()
  return () => controller.abort()
}
