<script setup>
defineProps({
  label: { type: String, required: true },
  value: { type: [String, Number], default: '—' },
  unit: { type: String, default: '' },
  note: { type: String, default: '' },
  tone: { type: String, default: '' } // '' | ok | warn | danger | accent
})
</script>

<template>
  <div class="fc-card fc-metric" :class="tone">
    <div class="fc-metric-label">{{ label }}</div>
    <div class="fc-metric-value">
      {{ value }}<span v-if="unit" class="fc-metric-unit">{{ unit }}</span>
    </div>
    <div v-if="note" class="fc-metric-note">{{ note }}</div>
  </div>
</template>
