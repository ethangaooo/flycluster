import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// base:'./' 让 dist 既可由中枢调度器直接挂载，也可放在任意子路径下。
// dev 时把 /v1 代理到本机中枢调度，避免联调还要改代码。
export default defineConfig({
  plugins: [vue()],
  base: './',
  server: {
    port: 5173,
    proxy: {
      '/v1': { target: 'http://127.0.0.1:8800', changeOrigin: true }
    }
  },
  build: { outDir: 'dist', emptyOutDir: true, chunkSizeWarningLimit: 1600 }
})
