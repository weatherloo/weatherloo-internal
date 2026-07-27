import { Pin, Trophy, Target, Warn } from './Icons.jsx'
import { colorOf, errorTone, fmtTemp, toneVars } from '../lib/data.js'

/** Soft mascot blob, tinted with the selected method's identity colour. */
function Mascot({ color, mood }) {
  return (
    <svg className="hero-art" viewBox="0 0 120 100" role="img" aria-hidden="true">
      <defs>
        <linearGradient id="blob" x1="0" y1="0" x2="0.3" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.95" />
          <stop offset="100%" stopColor={color} stopOpacity="0.55" />
        </linearGradient>
        <filter id="soft" x="-30%" y="-30%" width="160%" height="180%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="b" />
          <feBlend in="SourceGraphic" in2="b" />
        </filter>
      </defs>

      <ellipse cx="60" cy="86" rx="30" ry="5.5" fill={color} opacity="0.16" />

      <g filter="url(#soft)">
        <path
          d="M34 66c-11 0-19-8-19-17.5S23 31 34 31c2.2 0 4.3.3 6.2 1C44.4 21.4 54.3 14 66 14c15.2 0 27.5 12.1 27.5 27 0 1.3-.1 2.6-.3 3.8 6.9 2.2 11.8 8.4 11.8 15.7C105 69.6 97.4 66 88 66Z"
          fill="url(#blob)"
        />
      </g>

      {/* face */}
      <ellipse cx="52" cy="46" rx="3.1" ry="3.6" fill="#0d1220" opacity="0.82" />
      <ellipse cx="70" cy="46" rx="3.1" ry="3.6" fill="#0d1220" opacity="0.82" />
      <ellipse cx="53.1" cy="44.8" rx="1.05" ry="1.2" fill="#fff" opacity="0.95" />
      <ellipse cx="71.1" cy="44.8" rx="1.05" ry="1.2" fill="#fff" opacity="0.95" />
      {mood === 'good' ? (
        <path d="M55 55c2.4 2.6 9.6 2.6 12 0" stroke="#0d1220" strokeOpacity="0.68"
              strokeWidth="2.1" fill="none" strokeLinecap="round" />
      ) : mood === 'bad' ? (
        <path d="M55 57c2.4-2.6 9.6-2.6 12 0" stroke="#0d1220" strokeOpacity="0.68"
              strokeWidth="2.1" fill="none" strokeLinecap="round" />
      ) : (
        <path d="M55.5 56h11" stroke="#0d1220" strokeOpacity="0.62"
              strokeWidth="2.1" fill="none" strokeLinecap="round" />
      )}
      <ellipse cx="45" cy="53" rx="4" ry="2.6" fill="#ff8fa3" opacity="0.3" />
      <ellipse cx="77" cy="53" rx="4" ry="2.6" fill="#ff8fa3" opacity="0.3" />
    </svg>
  )
}

export default function Hero({ doc, method }) {
  const color = colorOf(method)
  const rank = method.rank
  const total = doc.ranking.length
  const rmse = method.scores.rmse
  const mae = method.scores.mae
  const bias = method.scores.bias

  const vals = method.series.filter((s) => s.value != null).map((s) => s.value)
  const high = vals.length ? Math.max(...vals) : null
  const low = vals.length ? Math.min(...vals) : null

  const tone = errorTone(mae)
  const mood = tone === 'good' ? 'good' : tone === 'bad' ? 'bad' : 'flat'

  const verdict =
    rank === 1 ? 'Most accurate' : rank === total ? 'Least accurate' : `${ordinal(rank)} of ${total}`
  const VerdictIcon = rank === 1 ? Trophy : rank === total ? Warn : Target

  const worst = method.series
    .filter((s) => s.delta != null)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))[0]

  return (
    <section className="card hero">
      <div className="hero-top">
        <Mascot color={color} mood={mood} />

        <div className="hero-main">
          <div className="hero-temp" style={{ color }}>
            {fmtTemp(high, 1)}<sup>°C</sup>
          </div>
          <h1 className="hero-name">{method.label} — forecast high</h1>
          <div className="hero-coords">
            <Pin />
            {doc.station.name} · {doc.station.lat.toFixed(4)}°N {Math.abs(doc.station.lon).toFixed(4)}°W
          </div>
          <p className="hero-desc">
            Over the 48&nbsp;hours after this run, {method.label} sat{' '}
            <strong>{mae?.toFixed(2)}°C</strong> from the station on average
            {bias != null && (
              <> and ran <strong>{Math.abs(bias).toFixed(2)}°C {bias >= 0 ? 'warm' : 'cold'}</strong></>
            )}
            {worst && (
              <>. Its worst hour was <strong>{Math.abs(worst.delta).toFixed(1)}°C</strong> off</>
            )}.
          </p>

          <div className="chips">
            <div className="chip">
              <div className="chip-l">RMSE</div>
              <div className="chip-v">{rmse?.toFixed(2)}°</div>
            </div>
            <div className="chip">
              <div className="chip-l">Mean err</div>
              <div className="chip-v">{mae?.toFixed(2)}°</div>
            </div>
            <div className="chip">
              <div className="chip-l">Bias</div>
              <div className="chip-v">{bias >= 0 ? '+' : '−'}{Math.abs(bias).toFixed(2)}°</div>
            </div>
            <div className="chip">
              <div className="chip-l">High / Low</div>
              <div className="chip-v">{fmtTemp(high, 0)}° / {fmtTemp(low, 0)}°</div>
            </div>
            <div className="verdict" style={toneVars(tone)}>
              <VerdictIcon />
              {verdict}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function ordinal(n) {
  if (n == null) return '—'
  const s = ['th', 'st', 'nd', 'rd']
  const v = n % 100
  return n + (s[(v - 20) % 10] || s[v] || s[0])
}
