import { useEffect, useMemo, useState } from 'react'
import Sidebar from './components/Sidebar.jsx'
import MethodPicker from './components/MethodPicker.jsx'
import Hero from './components/Hero.jsx'
import OutlookChart from './components/OutlookChart.jsx'
import HourList from './components/HourList.jsx'
import HourDrawer from './components/HourDrawer.jsx'
import Leaderboard from './components/Leaderboard.jsx'
import { Methods, StationPanel } from './components/Panels.jsx'
import { Moon2, Pin, SunSmall } from './components/Icons.jsx'
import { fmtInit, methodById } from './lib/data.js'

const TABS = [
  { id: 'forecast', label: 'Forecast' },
  { id: 'leaderboard', label: 'Accuracy' },
  { id: 'methods', label: 'Methods' },
  { id: 'station', label: 'Station' },
]

function useTheme() {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('wl-theme')
    if (saved) return saved
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('wl-theme', theme)
  }, [theme])
  return [theme, setTheme]
}

export default function App() {
  const [doc, setDoc] = useState(null)
  const [err, setErr] = useState(null)
  const [methodId, setMethodId] = useState(null)
  const [view, setView] = useState('forecast')
  const [lead, setLead] = useState(null)
  const [theme, setTheme] = useTheme()

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/showdown.json`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((d) => {
        setDoc(d)
        // Open on the winner — the page's headline answer.
        setMethodId(d.methods.find((m) => m.rank === 1)?.id ?? d.methods[0].id)
      })
      .catch((e) => setErr(e.message))
  }, [])

  const method = useMemo(() => (doc ? methodById(doc, methodId) : null), [doc, methodId])

  if (err) {
    return <div style={{ padding: 40, fontFamily: 'system-ui' }}>Could not load the run: {err}</div>
  }
  if (!doc || !method) {
    return <div style={{ padding: 40, color: '#8b95a8', fontFamily: 'system-ui' }}>Loading run…</div>
  }

  return (
    <div className="app">
      <Sidebar view={view} onView={setView} station={doc.station} />

      <div className="main">
        <div className="topbar">
          <MethodPicker doc={doc} selected={method} onSelect={setMethodId} />

          <div className="tabs" role="tablist" aria-label="View">
            {TABS.map((t) => (
              <button
                key={t.id}
                className="tab"
                role="tab"
                aria-selected={view === t.id}
                onClick={() => setView(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>

          <div className="topbar-right">
            <span className="topbar-loc">
              <Pin /> {doc.station.place}
            </span>
            <span title={`Run initialized ${fmtInit(doc.init)}`}>{fmtInit(doc.init)}</span>
            <button
              className="icon-btn"
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
            >
              {theme === 'dark' ? <SunSmall /> : <Moon2 />}
            </button>
          </div>
        </div>

        <main className="col">
          {view === 'forecast' && (
            <>
              <Hero doc={doc} method={method} />
              <OutlookChart doc={doc} method={method} onPick={setLead} selectedLead={lead} />
              <HourList doc={doc} method={method} onPick={setLead} selectedLead={lead} />
            </>
          )}

          {view === 'leaderboard' && (
            <Leaderboard doc={doc} selectedId={methodId} onSelect={(id) => { setMethodId(id); setView('forecast') }} />
          )}

          {view === 'methods' && (
            <Methods doc={doc} selectedId={methodId} onSelect={(id) => { setMethodId(id); setView('forecast') }} />
          )}

          {view === 'station' && <StationPanel doc={doc} />}

          <div className="footnote">
            Weatherloo · {doc.title} · run {fmtInit(doc.init)} · generated {doc.generated_at}
          </div>
        </main>
      </div>

      {lead != null && (
        <HourDrawer
          doc={doc}
          lead={lead}
          selectedId={methodId}
          onClose={() => setLead(null)}
          onSelect={setMethodId}
        />
      )}
    </div>
  )
}
