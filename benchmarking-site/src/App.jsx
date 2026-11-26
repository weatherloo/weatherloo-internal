import { useCallback, useState } from "react";
import DetailPanel from "./components/DetailPanel.jsx";
import StationMap from "./components/StationMap.jsx";
import { LOCATION_LABELS } from "./constants.js";

export default function App() {
  const [locationId, setLocationId] = useState(null);

  const selectLocation = useCallback((id) => {
    setLocationId(id);
  }, []);

  const selectedLabel = locationId
    ? `Selected: ${LOCATION_LABELS[locationId] ?? locationId}`
    : "Select a location on the map.";

  return (
    <>
      <header className="topbar">
        <div className="brand-row">
          <div className="brand-lockup" aria-label="Weatherloo benchmark brand">
            <div className="brand-mark" aria-hidden="true">W</div>
            <div>
              <p className="eyebrow">weatherloo</p>
              <h1>Forecast benchmark</h1>
            </div>
          </div>

          <div className="header-meta" aria-label="Benchmark summary highlights">
            <span className="meta-pill">2025</span>
            <span className="meta-pill">t2m + wind</span>
            <span className="meta-pill">6–72h leads</span>
          </div>
        </div>

        <p className="subtitle">
          Internal forecast skill scores for 2025 — 2 m temperature and 10 m
          wind speed, averaged over loaded initializations (target: 1460 inits at
          00/06/12/18 UTC).
        </p>
      </header>

      <main>
        <section id="map-panel" aria-label="Station map">
          <div className="panel-header panel-header--dark">
            <h2>Locations</h2>
            <span className="status-badge">active station</span>
          </div>
          <StationMap
            selectedLocationId={locationId}
            onSelectLocation={selectLocation}
          />
          <p id="selected-location" className="selected-location">
            {selectedLabel}
          </p>
        </section>

        {locationId ? <DetailPanel locationId={locationId} /> : null}
      </main>
    </>
  );
}
