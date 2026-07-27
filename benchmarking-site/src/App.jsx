import { useCallback, useState } from "react";
import DetailPanel from "./components/DetailPanel.jsx";
import HindcastPanel from "./components/HindcastPanel.jsx";
import StationMap from "./components/StationMap.jsx";
import { LOCATION_LABELS } from "./constants.js";

/** Whole-day leads only -- the public-forecast archive has no sub-daily offsets. */
const HINDCAST_LEADS = [24, 48, 72];
const HINDCAST_METHOD = "ecmwf_aifs";

export default function App() {
  const [locationId, setLocationId] = useState(null);
  const [lead, setLead] = useState(48);

  const selectLocation = useCallback((id) => {
    setLocationId(id);
  }, []);

  const selectedLabel = locationId
    ? `Selected: ${LOCATION_LABELS[locationId] ?? locationId}`
    : "Select a location on the map.";

  return (
    <>
      <header>
        <h1>Weather Forecast Benchmark</h1>
        <p className="subtitle">
          Internal forecast skill scores for 2025 — 2 m temperature and 10 m
          wind speed, averaged over loaded initializations (target: 1460 inits at
          00/06/12/18 UTC).
        </p>
      </header>

      <main>
        <section id="map-panel" aria-label="Station map">
          <h2>Locations</h2>
          <StationMap
            selectedLocationId={locationId}
            onSelectLocation={selectLocation}
          />
          <p id="selected-location">{selectedLabel}</p>
        </section>

        {locationId ? (
          <>
            <div className="hindcast-controls">
              <label htmlFor="hindcast-lead">Lead time</label>
              <select
                id="hindcast-lead"
                value={lead}
                onChange={(e) => setLead(Number(e.target.value))}
              >
                {HINDCAST_LEADS.map((h) => (
                  <option key={h} value={h}>
                    {h}h
                  </option>
                ))}
              </select>
            </div>
            <HindcastPanel
              locationId={locationId}
              method={HINDCAST_METHOD}
              lead={lead}
            />
            <DetailPanel locationId={locationId} />
          </>
        ) : null}
      </main>
    </>
  );
}
