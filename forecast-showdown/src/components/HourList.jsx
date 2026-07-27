import { SKY } from './Icons.jsx'
import {
  colorOf, errorTone, fmtDelta, fmtHour, fmtTemp, hourRows, skyOf, toneVars,
} from '../lib/data.js'

/**
 * The hourly strip. Each row carries the selected method's value and, in the
 * corner, how far that value landed from the station — the badge is the whole
 * point of the page, so it gets the status colour rather than the method's.
 *
 * Rows the selected method cannot fill are still rendered (the observation
 * exists for every hour); a blank badge says "this model has no value here"
 * instead of quietly dropping the hour.
 */
export default function HourList({ doc, method, onPick, selectedLead }) {
  // Only the hours this method actually produced. Padding the list out to every
  // hour would fill most of it with blanks for the 6-hourly models and read as
  // broken rather than as "this model runs on a coarser grid".
  const rows = hourRows(doc, method).filter((r) => r.value != null)
  const color = colorOf(method)

  const vals = rows.map((r) => r.value)
  const lo = Math.min(...vals)
  const hi = Math.max(...vals)
  const span = Math.max(0.5, hi - lo)

  // A hole is a lead on the method's own grid where it still produced nothing.
  const cadence = method.cadence_hours || 1
  const expected = Math.floor(doc.horizon_hours / cadence)
  const gaps = Math.max(0, expected - rows.filter((r) => r.lead > 0).length)

  return (
    <section className="card">
      <div className="card-h">
        <h2 className="card-t">Hourly forecast</h2>
        <span className="card-sub">
          {method.cadence_hours === 1 ? 'hourly' : `every ${method.cadence_hours} h`} · tap an hour to compare all methods
        </span>
      </div>

      <div className="hours">
        {rows.map((r) => {
          const Sky = SKY[skyOf(r.value ?? r.actual, (Date.parse(doc.init) / 3600e3 + r.lead) % 24)]
          const tone = errorTone(r.delta == null ? null : Math.abs(r.delta))
          const width = r.value == null ? 0 : ((r.value - lo) / span) * 82 + 18
          return (
            <button
              key={r.lead}
              className={`hour${selectedLead === r.lead ? ' is-open' : ''}`}
              onClick={() => onPick(r.lead)}
              aria-label={`${fmtHour(r.validTime)}, ${method.label} ${fmtTemp(r.value)} degrees, observed ${fmtTemp(r.actual)}`}
            >
              <span>
                <div className="hour-time">
                  {r.lead === 0 ? 'Init' : fmtHour(r.validTime, { weekday: false })}
                </div>
                <div className="hour-lead">+{r.lead}h</div>
              </span>

              <Sky className="hour-ico" />

              <span className="hour-bar">
                <span className="hour-fill" style={{ '--c': color, width: `${width}%` }} />
              </span>

              <span>
                <div className="hour-temp" style={{ color: r.value == null ? 'var(--text-3)' : undefined }}>
                  {fmtTemp(r.value)}°
                </div>
                <div className="hour-truth">obs {fmtTemp(r.actual)}°</div>
              </span>

              <span className={`delta${r.delta == null ? ' is-empty' : ''}`} style={toneVars(tone)}>
                {r.delta == null ? '—' : fmtDelta(r.delta)}
              </span>
            </button>
          )
        })}
      </div>

      <div className="footnote" style={{ padding: '0 20px 16px' }}>
        {rows.length} step{rows.length === 1 ? '' : 's'} on {method.label}&rsquo;s{' '}
        {cadence === 1 ? 'hourly' : `${cadence}-hourly`} grid
        {cadence > 1 && ', the leads it has a trained model for'}. Nothing is interpolated between
        them, and the badge is measured against the station observation at that exact hour.
        {gaps > 0 && ` ${gaps} lead${gaps === 1 ? '' : 's'} on that grid produced no value.`}
      </div>
    </section>
  )
}
