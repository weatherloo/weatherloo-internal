import { useEffect, useRef, useState } from 'react'
import { Check, Chevron } from './Icons.jsx'
import { colorOf, sourceLabel } from '../lib/data.js'

/**
 * The pill dropdown in the top bar. Selecting a method swaps which forecast the
 * whole page shows — the raw baselines sit in their own group so they read as
 * the comparison set rather than as more of our models.
 */
export default function MethodPicker({ doc, selected, onSelect }) {
  const [open, setOpen] = useState(false)
  const root = useRef(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e) => { if (!root.current?.contains(e.target)) setOpen(false) }
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const groups = [
    { key: 'ours', title: 'Weatherloo methods' },
    { key: 'reference', title: 'Raw model baselines' },
  ]

  return (
    <div className="picker" ref={root}>
      <button
        className="picker-trigger"
        aria-expanded={open}
        aria-haspopup="listbox"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="swatch" style={{ '--c': colorOf(selected) }} />
        <span>{selected.label}</span>
        <span className="rmse">{selected.scores.rmse?.toFixed(2)}°</span>
        <Chevron className="chev" />
      </button>

      {open && (
        <div className="picker-menu" role="listbox" aria-label="Forecast method">
          {groups.map((g) => {
            const items = doc.methods.filter((m) => m.group === g.key)
            if (!items.length) return null
            return (
              <div key={g.key}>
                <div className="picker-group">{g.title}</div>
                {items.map((m) => (
                  <button
                    key={m.id}
                    role="option"
                    aria-selected={m.id === selected.id}
                    className="picker-item"
                    onClick={() => { onSelect(m.id); setOpen(false) }}
                  >
                    <span className="swatch" style={{ '--c': colorOf(m) }}>
                      {m.id === selected.id && <Check />}
                    </span>
                    <span className="meta">
                      <span className="name">
                        {m.label}
                        {m.rank === 1 && <span aria-label="most accurate">🏆</span>}
                      </span>
                      <span className="sub">{sourceLabel(m)}</span>
                    </span>
                    <span className="score">
                      <span className="score-v">{m.scores.rmse?.toFixed(2)}°</span>
                      <span className="score-l">RMSE</span>
                    </span>
                  </button>
                ))}
              </div>
            )
          })}

          {doc.unavailable?.map((u) => (
            <div key={u.id}>
              <div className="picker-group">Unavailable</div>
              <div className="picker-item is-off" title={u.reason}>
                <span className="swatch" style={{ '--c': 'var(--text-3)' }} />
                <span className="meta">
                  <span className="name">{u.label}</span>
                  <span className="sub">blocked by network policy</span>
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
