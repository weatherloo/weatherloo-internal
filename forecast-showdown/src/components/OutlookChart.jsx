import { useMemo, useRef, useState } from 'react'
import { colorOf, extentOf, fmtDelta, fmtHour, fmtTemp } from '../lib/data.js'

const W = 580
const H = 210
const PAD = { t: 26, r: 14, b: 26, l: 14 }

/**
 * Selected method against the verifying observation.
 *
 * Only two series are ever drawn, so identity is a legend plus the truth line's
 * dashed treatment — the method keeps its categorical colour, and truth is ink
 * rather than a third hue, because it is the reference, not another category.
 */
export default function OutlookChart({ doc, method, onPick, selectedLead }) {
  const [hover, setHover] = useState(null)
  const [table, setTable] = useState(false)
  const svgRef = useRef(null)

  const color = colorOf(method)
  const horizon = doc.horizon_hours

  const truth = useMemo(
    () => doc.truth.steps.map((s) => ({ lead: s.lead_hours, v: s.value })),
    [doc],
  )
  const fcst = useMemo(
    () => method.series.map((s) => ({ lead: s.lead_hours, v: s.value, d: s.delta })),
    [method],
  )

  const [lo, hi] = useMemo(
    () => extentOf(method.series.map((s, i) => ({ value: s.value, actual: truth[i]?.v ?? null }))),
    [method, truth],
  )

  const x = (lead) => PAD.l + (lead / horizon) * (W - PAD.l - PAD.r)
  const y = (v) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b)

  /**
   * Connect consecutive available points, breaking only on a *real* hole.
   * A 6-hourly method has nothing at +7h by design, so treating every null as a
   * break would leave the line as a row of disconnected dots; the line is only
   * lifted when the step to the next value exceeds the method's own cadence.
   */
  const linePath = (pts, cadence) => {
    let d = ''
    let prev = null
    for (const p of pts) {
      if (p.v == null) continue
      const broken = prev != null && p.lead - prev > cadence
      d += `${prev == null || broken ? 'M' : 'L'}${x(p.lead).toFixed(1)} ${y(p.v).toFixed(1)} `
      prev = p.lead
    }
    return d.trim()
  }

  const marks = fcst.filter((p) => p.v != null)
  // Direct-label a handful of points rather than every one.
  const labelEvery = Math.max(1, Math.round(marks.length / 6))
  const labelled = new Set(marks.filter((_, i) => i % labelEvery === 0).map((p) => p.lead))

  const onMove = (e) => {
    const rect = svgRef.current.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    const lead = Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * horizon)
    if (lead < 0 || lead > horizon) return setHover(null)
    setHover(lead)
  }

  const hoverF = hover == null ? null : fcst.find((p) => p.lead === hover)
  const hoverT = hover == null ? null : truth.find((p) => p.lead === hover)

  const ticks = [0, 12, 24, 36, 48].filter((t) => t <= horizon)

  return (
    <section className="card">
      <div className="card-h">
        <h2 className="card-t">48-hour outlook</h2>
        <span className="card-sub">{method.label} vs station</span>
      </div>

      <div className="chart-wrap">
        <svg
          ref={svgRef}
          className="chart"
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`${method.label} forecast against observed temperature over ${horizon} hours`}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          {/* recessive baseline grid */}
          {ticks.map((t) => (
            <line key={t} className="grid" x1={x(t)} y1={PAD.t - 8} x2={x(t)} y2={H - PAD.b} />
          ))}

          {/* observed */}
          <path
            d={linePath(truth, doc.truth.cadence_hours)}
            fill="none"
            stroke="var(--truth)"
            strokeWidth="2"
            strokeDasharray="5 4"
            strokeLinecap="round"
            opacity="0.5"
          />

          {/* forecast */}
          <path d={linePath(fcst, method.cadence_hours)} fill="none" stroke={color}
                strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />

          {/* markers — 2px surface ring so overlaps stay readable */}
          {marks.map((p) => (
            <g key={p.lead}>
              <circle
                cx={x(p.lead)} cy={y(p.v)} r="4.6"
                fill={color} stroke="var(--surface)" strokeWidth="2"
                opacity={selectedLead === p.lead ? 1 : 0.95}
              />
              {selectedLead === p.lead && (
                <circle cx={x(p.lead)} cy={y(p.v)} r="8.5" fill="none" stroke={color} strokeWidth="1.6" opacity="0.5" />
              )}
              {labelled.has(p.lead) && (
                <text className="pt-lbl" x={x(p.lead)} y={y(p.v) - 11} textAnchor="middle">
                  {p.v.toFixed(0)}°
                </text>
              )}
            </g>
          ))}

          {/* crosshair */}
          {hover != null && (
            <line className="cross" x1={x(hover)} y1={PAD.t - 8} x2={x(hover)} y2={H - PAD.b} />
          )}

          {/* x labels */}
          {ticks.map((t) => (
            <text key={t} className="ax" x={x(t)} y={H - PAD.b + 15} textAnchor="middle">
              {t === 0 ? 'Init' : `+${t}h`}
            </text>
          ))}

          {/* generous hit targets */}
          {Array.from({ length: horizon + 1 }, (_, L) => (
            <rect
              key={L} className="hit"
              x={x(L) - (W / horizon) / 2} y={PAD.t - 10}
              width={W / horizon} height={H - PAD.t - PAD.b + 10}
              onClick={() => onPick?.(L)}
            />
          ))}
        </svg>

        {hover != null && (hoverF?.v != null || hoverT?.v != null) && (
          <div className="footnote" style={{ paddingTop: 6 }}>
            <strong>{hoverT?.v != null || hoverF?.v != null
              ? fmtHour(new Date(Date.parse(doc.init) + hover * 3600e3).toISOString())
              : ''}</strong>
            {' · '}
            {method.label} {fmtTemp(hoverF?.v)}°C
            {' · '}observed {fmtTemp(hoverT?.v)}°C
            {hoverF?.d != null && <> · off by {fmtDelta(hoverF.d)}°C</>}
          </div>
        )}
      </div>

      <div className="legend">
        <span className="legend-i">
          <span className="legend-k" style={{ '--c': color }} /> {method.label}
        </span>
        <span className="legend-i">
          <span className="legend-k dash" style={{ '--c': 'var(--truth)' }} /> Observed (station)
        </span>
        <button className="table-toggle" style={{ marginLeft: 'auto' }}
                onClick={() => setTable((v) => !v)} aria-expanded={table}>
          {table ? 'Hide table' : 'View as table'}
        </button>
      </div>

      {table && (
        <div className="dtable-wrap">
          <table className="dtable">
            <caption className="footnote" style={{ captionSide: 'top', textAlign: 'left' }}>
              {method.label} against the station observation, by lead hour.
            </caption>
            <thead>
              <tr><th>Lead</th><th>{method.label} °C</th><th>Observed °C</th><th>Difference</th></tr>
            </thead>
            <tbody>
              {fcst.filter((p) => p.v != null).map((p) => (
                <tr key={p.lead}>
                  <td>+{p.lead}h</td>
                  <td>{fmtTemp(p.v)}</td>
                  <td>{fmtTemp(truth.find((t) => t.lead === p.lead)?.v)}</td>
                  <td>{fmtDelta(p.d)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
