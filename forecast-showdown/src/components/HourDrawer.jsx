import { useEffect } from 'react'
import { sourceLabel, fmtDelta, fmtDay, fmtHour, fmtTemp, rankingFor, errorTone, toneVars } from '../lib/data.js'

const MEDAL = { 1: '🥇', 2: '🥈', 3: '🥉' }

/**
 * Bottom sheet for one hour: every method that produced a value, ranked by how
 * close it landed to the station observation for that hour.
 *
 * This is a per-hour ranking, deliberately not the overall leaderboard — a
 * model can win a single hour and still lose across the run, and seeing that
 * happen is the interesting part.
 */
export default function HourDrawer({ doc, lead, selectedId, onClose, onSelect }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [onClose])

  if (lead == null) return null

  const entries = rankingFor(doc, lead)
  const truth = doc.truth.steps.find((s) => s.lead_hours === lead)?.value ?? null
  const validTime = new Date(Date.parse(doc.init) + lead * 3600e3).toISOString()

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <div className="drawer" role="dialog" aria-modal="true"
           aria-label={`Method comparison for ${fmtHour(validTime)}`}>
        <div className="drawer-inner">
          <div className="grabber" />

          <div className="drawer-h">
            <div>
              <div className="drawer-t">{fmtHour(validTime)}</div>
              <div className="drawer-s">{fmtDay(validTime)} · +{lead}h from the run</div>
            </div>
            <div className="drawer-truth">
              <div className="l">Observed</div>
              <div className="v">{fmtTemp(truth)}°C</div>
            </div>
          </div>

          {entries.length === 0 ? (
            <div className="footnote" style={{ paddingTop: 14 }}>
              No method has a value at this hour.
            </div>
          ) : (
            <div className="rank-list">
              {entries.map((e) => {
                const tone = errorTone(e.abs_error)
                const isFirst = e.rank === 1
                return (
                  <button
                    key={e.method}
                    className={`rank${isFirst ? ' is-first' : ''}${e.meta?.id === selectedId ? ' is-sel' : ''}`}
                    style={{ '--c': e.color, '--c-bg': `color-mix(in srgb, ${e.color} 10%, transparent)` }}
                    onClick={() => { onSelect(e.meta.id); onClose() }}
                  >
                    <span className="rank-n">
                      {MEDAL[e.rank] ? <span className="medal">{MEDAL[e.rank]}</span> : e.rank}
                    </span>
                    <span className="swatch" style={{ '--c': e.color, width: 22, height: 22, borderRadius: '50%' }} />
                    <span>
                      <div className="rank-name">{e.meta?.label}</div>
                      <div className="rank-sub">
                        {e.meta ? sourceLabel(e.meta) : ''}
                        {e.meta?.group === 'ours' ? ' · Weatherloo' : ''}
                      </div>
                    </span>
                    <span className="rank-v">{fmtTemp(e.value)}°</span>
                    <span className="delta" style={toneVars(tone)}>{fmtDelta(e.delta)}</span>
                  </button>
                )
              })}
            </div>
          )}

          <div className="footnote" style={{ paddingTop: 14 }}>
            Ranked by absolute difference from the station observation at this hour.
            Selecting a row switches the page to that method.
          </div>
        </div>
      </div>
    </>
  )
}
