import { Trophy } from './Icons.jsx'
import { colorOf, sourceLabel } from '../lib/data.js'

const MEDAL = { 1: '🥇', 2: '🥈', 3: '🥉' }

/**
 * Overall standings on the common grid. The meter is scaled to the worst RMSE
 * on screen, so the bars compare methods to each other rather than to an
 * arbitrary ceiling.
 */
export default function Leaderboard({ doc, selectedId, onSelect }) {
  const ranked = doc.ranking
    .map((key) => doc.methods.find((m) => m.key === key))
    .filter(Boolean)
  const worst = Math.max(...ranked.map((m) => m.scores.rmse))
  const winner = ranked[0]

  return (
    <>
      <section className="card hero" style={{ padding: '24px 28px' }}>
        <div className="hero-coords" style={{ marginTop: 0 }}>
          <Trophy style={{ width: 13, height: 13 }} /> Most accurate over 48 hours
        </div>
        <div className="hero-temp" style={{ color: colorOf(winner), fontSize: 58, marginTop: 8 }}>
          {winner.label}
        </div>
        <p className="hero-desc" style={{ marginTop: 12 }}>
          {winner.label} finished at <strong>{winner.scores.rmse.toFixed(2)}°C RMSE</strong> across the{' '}
          {doc.scoring.common_grid_leads.length} verifying hours — against{' '}
          <strong>{ranked[1].scores.rmse.toFixed(2)}°C</strong> for {ranked[1].label} in second.
        </p>
      </section>

      <section className="card">
        <div className="card-h">
          <h2 className="card-t">Standings</h2>
          <span className="card-sub">RMSE, lower is better</span>
        </div>

        <div className="board">
          {ranked.map((m) => {
            const color = colorOf(m)
            return (
              <button
                key={m.key}
                className={`board-row${m.rank === 1 ? ' is-first' : ''}`}
                style={{ '--c': color, '--c-bg': `color-mix(in srgb, ${color} 10%, transparent)` }}
                onClick={() => onSelect(m.id)}
                aria-current={m.id === selectedId ? 'true' : undefined}
              >
                <span className="rank-n">
                  {MEDAL[m.rank] ? <span className="medal">{MEDAL[m.rank]}</span> : m.rank}
                </span>
                <span className="swatch" style={{ '--c': color, width: 22, height: 22, borderRadius: '50%' }} />
                <span>
                  <div className="rank-name">{m.label}</div>
                  <div className="rank-sub">
                    {m.group === 'ours' ? 'Weatherloo · ' : 'Baseline · '}{sourceLabel(m)}
                  </div>
                </span>
                <span className="board-meter">
                  <span style={{ width: `${(m.scores.rmse / worst) * 100}%` }} />
                </span>
                <span>
                  <div className="board-rmse">{m.scores.rmse.toFixed(2)}°</div>
                  <div className="board-mae">MAE {m.scores.mae.toFixed(2)}°</div>
                </span>
              </button>
            )
          })}
        </div>

        <div className="footnote" style={{ padding: '0 20px 18px' }}>
          Scored on leads {doc.scoring.common_grid_leads.map((l) => `+${l}h`).join(', ')} — the hours
          every method produced a value for, so no model is graded on a different set of points
          than its rivals.
        </div>
      </section>
    </>
  )
}
