/* Shaping helpers for the showdown document.
   Everything the UI renders comes from public/data/showdown.json — there are no
   hard-coded temperatures here, only presentation rules. */

/** CSS custom property carrying each method's identity colour. */
const COLOR_VAR = {
  lstm: '--m-lstm',
  unet: '--m-unet',
  cnn_lstm: '--m-cnnlstm',
  ecmwf_aifs: '--m-ref-1',
  gfs_analysis: '--m-ref-2',
  hrrr_interpolated: '--m-ref-3',
}

/** Short source label shown under each method name. */
const SOURCE_LABEL = {
  gfs: 'GFS 0.25°',
  hrrr: 'HRRR 3 km',
  ecmwf_aifs: 'ECMWF AIFS',
}

export function colorOf(method) {
  return `var(${COLOR_VAR[method.id] || '--m-ref-2'})`
}

export function sourceLabel(method) {
  if (method.group === 'reference') return 'raw model output'
  return SOURCE_LABEL[method.base_source] || method.base_source || ''
}

/** Absolute error -> status colour role. Thresholds are in °C. */
export function errorTone(absErr) {
  if (absErr == null) return 'none'
  if (absErr < 1) return 'good'
  if (absErr < 2.5) return 'warn'
  return 'bad'
}

export function toneVars(tone) {
  if (tone === 'good') return { '--c': 'var(--good)', '--c-bg': 'var(--good-bg)' }
  if (tone === 'warn') return { '--c': 'var(--warn)', '--c-bg': 'var(--warn-bg)' }
  if (tone === 'bad') return { '--c': 'var(--bad)', '--c-bg': 'var(--bad-bg)' }
  return { '--c': 'var(--text-3)', '--c-bg': 'transparent' }
}

export function fmtTemp(v, digits = 1) {
  return v == null ? '—' : v.toFixed(digits)
}

export function fmtDelta(d) {
  if (d == null) return '—'
  const s = d >= 0 ? '+' : '−'
  return `${s}${Math.abs(d).toFixed(1)}`
}

/** "Mon 2 PM" from an ISO instant, rendered in the station's local clock. */
export function fmtHour(iso, opts = {}) {
  const d = new Date(iso)
  return d.toLocaleString('en-US', {
    weekday: opts.weekday === false ? undefined : 'short',
    hour: 'numeric',
    hour12: true,
    timeZone: 'America/Toronto',
  })
}

export function fmtDay(iso) {
  return new Date(iso).toLocaleString('en-US', {
    weekday: 'long', month: 'short', day: 'numeric', timeZone: 'America/Toronto',
  })
}

export function fmtInit(iso) {
  const d = new Date(iso)
  return `${d.toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC',
  })} · ${String(d.getUTCHours()).padStart(2, '0')}Z`
}

/** A crude sky descriptor so the hourly rows carry an icon like a weather app. */
export function skyOf(tempC, hourUTC) {
  const localHour = (hourUTC - 4 + 24) % 24 // EDT in July
  const night = localHour < 6 || localHour >= 20
  if (tempC == null) return night ? 'night' : 'day'
  if (tempC >= 22) return night ? 'night' : 'sun'
  if (tempC >= 15) return night ? 'night' : 'partly'
  return night ? 'night' : 'cloud'
}

/**
 * Rows for the hourly list: one per hour of the horizon, carrying the selected
 * method's value and the verifying observation.
 */
export function hourRows(doc, method) {
  const truth = new Map(doc.truth.steps.map((s) => [s.lead_hours, s.value]))
  const series = new Map(method.series.map((s) => [s.lead_hours, s]))
  const rows = []
  for (let L = 0; L <= doc.horizon_hours; L++) {
    const s = series.get(L)
    const actual = truth.get(L) ?? null
    rows.push({
      lead: L,
      validTime: s?.valid_time || null,
      value: s?.value ?? null,
      actual,
      delta: s?.delta ?? null,
    })
  }
  return rows
}

/** Min/max across the selected method and truth, for chart scaling. */
export function extentOf(rows) {
  const vals = []
  for (const r of rows) {
    if (r.value != null) vals.push(r.value)
    if (r.actual != null) vals.push(r.actual)
  }
  if (!vals.length) return [0, 1]
  const lo = Math.min(...vals)
  const hi = Math.max(...vals)
  const pad = Math.max(1, (hi - lo) * 0.18)
  return [lo - pad, hi + pad]
}

export function methodById(doc, id) {
  return doc.methods.find((m) => m.id === id) || doc.methods[0]
}

/** Per-lead ranking rows, joined back to method metadata for display. */
export function rankingFor(doc, lead) {
  const entries = doc.per_lead_ranking[String(lead)] || []
  return entries.map((e) => {
    const m = doc.methods.find((x) => x.key === e.method)
    return { ...e, meta: m, color: m ? colorOf(m) : 'var(--text-3)' }
  })
}
