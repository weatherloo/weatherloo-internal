import { Clock, Grid, Info, Pin, Ruler, Station, Target, Warn, Wind } from './Icons.jsx'
import { colorOf, fmtInit, sourceLabel } from '../lib/data.js'

/** One card per method: what it is, what it ran on, and how it scored. */
export function Methods({ doc, selectedId, onSelect }) {
  const order = ['ours', 'reference']
  return (
    <div className="methods">
      {order.map((group) => (
        <div key={group} className="methods">
          <div className="picker-group" style={{ padding: '4px 4px 0' }}>
            {group === 'ours' ? 'Weatherloo methods' : 'Raw model baselines'}
          </div>
          {doc.methods.filter((m) => m.group === group).map((m) => {
            const color = colorOf(m)
            return (
              <button
                key={m.key}
                className="card method-card"
                style={{ textAlign: 'left', width: '100%' }}
                onClick={() => onSelect(m.id)}
                aria-current={m.id === selectedId ? 'true' : undefined}
              >
                <div className="method-head">
                  <span className="swatch" style={{ '--c': color, width: 30, height: 30, borderRadius: '50%' }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="method-name">{m.label}</div>
                    <div className="method-src">{sourceLabel(m)} · {m.cadence_hours}-hourly</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div className="board-rmse">{m.scores.rmse?.toFixed(2)}°</div>
                    <div className="board-mae">rank {m.rank} of {doc.ranking.length}</div>
                  </div>
                </div>

                <p className="method-body">{m.blurb}</p>

                <div className="method-grid">
                  <div className="chip">
                    <div className="chip-l">RMSE</div>
                    <div className="chip-v">{m.scores.rmse?.toFixed(2)}°</div>
                  </div>
                  <div className="chip">
                    <div className="chip-l">MAE</div>
                    <div className="chip-v">{m.scores.mae?.toFixed(2)}°</div>
                  </div>
                  <div className="chip">
                    <div className="chip-l">Bias</div>
                    <div className="chip-v">
                      {m.scores.bias >= 0 ? '+' : '−'}{Math.abs(m.scores.bias).toFixed(2)}°
                    </div>
                  </div>
                  <div className="chip">
                    <div className="chip-l">Native grid</div>
                    <div className="chip-v">{m.scores_native.n} pts</div>
                  </div>
                </div>

                {m.provenance?.checkpoint && (
                  <div className="footnote" style={{ padding: '12px 0 0' }}>
                    <code>{m.provenance.checkpoint}</code>
                  </div>
                )}
              </button>
            )
          })}
        </div>
      ))}
    </div>
  )
}

/** Run metadata, station facts, and what could not be sourced. */
export function StationPanel({ doc }) {
  const t = doc.truth.steps.filter((s) => s.value != null).map((s) => s.value)
  const rows = [
    { Icon: Pin, l: 'Coordinates', v: `${doc.station.lat.toFixed(4)}°N, ${Math.abs(doc.station.lon).toFixed(4)}°W` },
    { Icon: Station, l: 'Station', v: doc.station.name },
    { Icon: Clock, l: 'Initialization', v: fmtInit(doc.init) },
    { Icon: Target, l: 'Horizon', v: `${doc.horizon_hours} hours` },
    { Icon: Ruler, l: 'Observed high', v: `${Math.max(...t).toFixed(1)}°C` },
    { Icon: Ruler, l: 'Observed low', v: `${Math.min(...t).toFixed(1)}°C` },
    { Icon: Grid, l: 'Truth cadence', v: `${doc.truth.cadence_hours}-hourly (${t.length} pts)` },
    { Icon: Wind, l: 'Variable', v: '2 m air temperature' },
  ]

  return (
    <>
      <section className="card">
        <div className="card-h">
          <h2 className="card-t">This run</h2>
          <span className="card-sub">{doc.station.place}</span>
        </div>
        <div className="cond">
          {rows.map((r) => (
            <div className="cond-row" key={r.l}>
              <r.Icon />
              <span className="cond-l">{r.l}</span>
              <span className="cond-v">{r.v}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <div className="card-h"><h2 className="card-t">Ground truth</h2></div>
        <p className="method-body" style={{ padding: '0 20px 18px', marginTop: 8 }}>
          {doc.truth.source}. {doc.truth.time_note}
        </p>
      </section>

      {doc.unavailable?.map((u) => (
        <div className="notice" key={u.id}>
          <Warn />
          <div>
            <b>{u.label} could not be included.</b> {u.reason}
            <div style={{ marginTop: 6 }}>{u.unblock}</div>
          </div>
        </div>
      ))}

      <div className="notice" style={{ background: 'var(--surface-2)' }}>
        <Info style={{ color: 'var(--text-3)' }} />
        <div>
          <b>How to read this.</b> {doc.scoring.note}
        </div>
      </div>
    </>
  )
}
