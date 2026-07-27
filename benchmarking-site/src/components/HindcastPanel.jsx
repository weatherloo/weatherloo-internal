import { useEffect, useState } from "react";
import HindcastChart from "./HindcastChart.jsx";
import { DATA_DIR, HINDCAST_SERIES, LOCATION_LABELS } from "../constants.js";

const RMSE_LABELS = {
  reference: "Forecast you'd have seen",
  ours: "What we'd have predicted",
  raw_model: "Raw model",
};

/**
 * Windows are counted in 6-hourly initializations. The default is deliberately
 * short: the point of this view is watching three forecasts diverge from the
 * observation, which is unreadable once a full season is on one axis. Whole-
 * period skill is what the benchmark panel below is for.
 */
const WINDOWS = [
  { value: 28, label: "Last 7 days" },
  { value: 56, label: "Last 14 days" },
  { value: 120, label: "Last 30 days" },
  { value: 0, label: "All data" },
];

/** RMSE over whatever subset is on screen, so the tiles match the chart. */
function rmseOf(records, key) {
  const errs = records
    .filter((r) => r[key] !== null && r.actual !== null)
    .map((r) => (r[key] - r.actual) ** 2);
  if (errs.length === 0) return null;
  return Math.sqrt(errs.reduce((a, b) => a + b, 0) / errs.length);
}

export default function HindcastPanel({ locationId, variable = "t2m", method, lead }) {
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState(null);
  const [windowSize, setWindowSize] = useState(56);

  useEffect(() => {
    let cancelled = false;
    setDoc(null);
    setError(null);

    const url = `${DATA_DIR}/hindcast/${locationId}/${variable}_${method}_${lead}h.json`;
    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then((json) => {
        if (!cancelled) setDoc(json);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });

    return () => {
      cancelled = true;
    };
  }, [locationId, variable, method, lead]);

  if (error) {
    return (
      <section className="hindcast-panel">
        <h3>What you'd have seen vs. what we'd have said</h3>
        <p className="hindcast-empty">
          No hindcast built for {LOCATION_LABELS[locationId] ?? locationId} at{" "}
          {lead}h ({error}). Generate it with{" "}
          <code>scripts/build_hindcast.py --lead {lead}</code>.
        </p>
      </section>
    );
  }

  if (!doc) return <p>Loading hindcast…</p>;

  const shown =
    windowSize > 0 ? doc.records.slice(-windowSize) : doc.records;
  const shownDoc = { ...doc, records: shown };
  const hasReference = shown.some((r) => r.reference !== null);
  const span = shown.length
    ? `${shown[0].valid_time.slice(0, 10)} to ${shown[shown.length - 1].valid_time.slice(0, 10)}`
    : "no records";

  return (
    <section className="hindcast-panel viz-root">
      <h3>What you'd have seen vs. what we'd have said</h3>
      <p className="subtitle">
        {LOCATION_LABELS[locationId] ?? locationId} — {lead}h lead, {span}. Each
        point is a forecast issued {lead} hours before the plotted valid time.
      </p>

      <div className="hindcast-controls hindcast-controls--inline">
        <label htmlFor="hindcast-window">Window</label>
        <select
          id="hindcast-window"
          value={windowSize}
          onChange={(e) => setWindowSize(Number(e.target.value))}
        >
          {WINDOWS.map((w) => (
            <option key={w.value} value={w.value}>
              {w.label}
            </option>
          ))}
        </select>
        <span className="hindcast-count">
          {shown.length} of {doc.records.length} initializations
        </span>
      </div>

      <ul className="hindcast-summary">
        {Object.keys(doc.summary_rmse).map((key) => {
          const value = rmseOf(shown, key);
          return (
            <li key={key}>
              <span className="label">{RMSE_LABELS[key] ?? key} RMSE</span>
              <span className="value">
                {value === null ? "—" : `${value.toFixed(2)} ${doc.units}`}
              </span>
            </li>
          );
        })}
      </ul>

      <HindcastChart doc={shownDoc} />

      {!hasReference ? (
        <p className="hindcast-caveat">
          The public-forecast series is empty — run{" "}
          <code>scripts/fetch_openmeteo_previous.py</code> to populate it. The
          other three series are unaffected.
        </p>
      ) : null}

      <p className="hindcast-caveat">
        {HINDCAST_SERIES.find((s) => s.key === "raw_model").label} is
        reconstructed as observation + stored bias, so an initialization whose
        verifying observation is missing cannot appear at all. Corrected values
        use only biases that had already verified by issue time.
      </p>
    </section>
  );
}
