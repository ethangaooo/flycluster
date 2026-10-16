<script setup>
import { ref, watch, onMounted } from 'vue'

const props = defineProps({
  snapshot: { type: Object, default: null },   // { grid, visited, agents, targets_remaining }
  cell: { type: Number, default: 11 }
})

const el = ref(null)
const palette = {
  0: '#1b2129',   // 可通行
  1: '#39424f',   // 墙
  2: '#d9a441',   // 目标
  3: '#35c9c0',   // 智能体 0
  4: '#7c9cff',
  5: '#e07bd0',
  6: '#8fd14f'
}

function draw() {
  const canvas = el.value
  if (!canvas || !props.snapshot) return
  const grid = props.snapshot.grid
  const n = grid.length
  const size = n * props.cell
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, size, size)
  const visited = props.snapshot.visited

  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const v = grid[y][x]
      ctx.fillStyle = palette[v] || '#1b2129'
      ctx.fillRect(x * props.cell, y * props.cell, props.cell, props.cell)
      if (visited && visited[y] && visited[y][x] && v === 0) {
        ctx.fillStyle = 'rgba(53,201,192,.13)'
        ctx.fillRect(x * props.cell, y * props.cell, props.cell, props.cell)
      }
      ctx.strokeStyle = 'rgba(255,255,255,.03)'
      ctx.strokeRect(x * props.cell, y * props.cell, props.cell, props.cell)
    }
  }
  // 智能体画成实心方块 + 方向缺口，便于区分
  for (const a of props.snapshot.agents || []) {
    ctx.fillStyle = palette[3 + a.id] || '#35c9c0'
    ctx.fillRect(a.x * props.cell + 1, a.y * props.cell + 1, props.cell - 2, props.cell - 2)
    ctx.fillStyle = '#0e1116'
    ctx.fillRect(a.x * props.cell + props.cell / 2 - 1, a.y * props.cell + props.cell / 2 - 1, 2, 2)
  }
}

watch(() => props.snapshot, draw, { deep: true })
onMounted(draw)
</script>

<template>
  <div>
    <canvas ref="el" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--fc-border)"></canvas>
    <div style="display: flex; gap: 14px; margin-top: 10px; font-size: 11px; color: var(--fc-text-faint); flex-wrap: wrap">
      <span><i style="display:inline-block;width:9px;height:9px;background:#1b2129;border:1px solid #39424f;margin-right:5px"></i>可通行</span>
      <span><i style="display:inline-block;width:9px;height:9px;background:#39424f;margin-right:5px"></i>墙</span>
      <span><i style="display:inline-block;width:9px;height:9px;background:#d9a441;margin-right:5px"></i>未拾取目标</span>
      <span><i style="display:inline-block;width:9px;height:9px;background:rgba(53,201,192,.4);margin-right:5px"></i>已探索</span>
    </div>
  </div>
</template>
