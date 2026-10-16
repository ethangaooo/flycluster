import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import 'element-plus/dist/index.css'
// Element Plus 官方深色变量。**必须导入**：不导入的话所有组件都用浅色主题的 token，
// 深色侧栏上的 el-menu 就是"深灰字压深色底"。之前是在 styles.css 里逐个组件打补丁，
// 结果漏了 el-menu —— 这类补丁式做法必然会有漏网的，改用官方机制一次覆盖全部组件。
import 'element-plus/theme-chalk/dark/css-vars.css'
import * as Icons from '@element-plus/icons-vue'
import axios from 'axios'
import { createRouter, createWebHashHistory } from 'vue-router'

import App from './App.vue'
import './styles.css'
import TaskConsoleView from './views/TaskConsoleView.vue'
import ClusterView from './views/ClusterView.vue'
import RunsView from './views/RunsView.vue'
import MemoryView from './views/MemoryView.vue'
import ExperimentsView from './views/ExperimentsView.vue'
import GatesView from './views/GatesView.vue'
import ApiAdminView from './views/ApiAdminView.vue'
import AnalyticsView from './views/AnalyticsView.vue'
import GameView from './views/GameView.vue'

// 用 hash 路由：dist 可以被任意静态服务器/挂载点直接托管，不需要服务端 rewrite。
const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', redirect: '/task' },
    { path: '/task', component: TaskConsoleView, meta: { title: '任务下发' } },
    { path: '/cluster', component: ClusterView, meta: { title: '集群总览' } },
    { path: '/runs', component: RunsView, meta: { title: '运行与实验' } },
    { path: '/memory', component: MemoryView, meta: { title: '记忆中枢' } },
    { path: '/games', component: GameView, meta: { title: '果蝇对弈' } },
    { path: '/analytics', component: AnalyticsView, meta: { title: '结果分析' } },
    { path: '/experiments', component: ExperimentsView, meta: { title: '对照结果' } },
    { path: '/gates', component: GatesView, meta: { title: '三道门' } },
    { path: '/api', component: ApiAdminView, meta: { title: 'API 接入管理' } }
  ]
})

const app = createApp(App)
app.use(ElementPlus, { locale: zhCn })
app.use(router)
Object.entries(Icons).forEach(([name, comp]) => app.component(name, comp))
app.config.globalProperties.$http = axios
app.mount('#app')
