import { Bolt, Chart, Layers, Station, Trophy, Pin } from './Icons.jsx'

const NAV = [
  { id: 'forecast', label: 'Forecast', Icon: Chart },
  { id: 'leaderboard', label: 'Accuracy', Icon: Trophy },
  { id: 'methods', label: 'Methods', Icon: Layers },
  { id: 'station', label: 'Station', Icon: Station },
]

export default function Sidebar({ view, onView, station }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <Bolt className="brand-mark" />
        <span>Weatherloo</span>
      </div>

      <nav className="nav" aria-label="Sections">
        {NAV.map(({ id, label, Icon }) => (
          <button
            key={id}
            className="nav-item"
            aria-current={view === id ? 'page' : undefined}
            onClick={() => onView(id)}
          >
            <Icon />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-foot">
        <div className="station-chip" title={station.place}>
          <Pin />
          <span>Waterloo</span>
        </div>
      </div>
    </aside>
  )
}
