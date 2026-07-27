import { useEffect, useState } from "react";
import HindcastChart from "./HindcastChart.jsx";
import { DATA_DIR, HINDCAST_SERIES, LOCATION_LABELS } from "../constants.js";

const RMSE_LABELS = {
  reference: "Forecast you'd have seen",
  ours: "What we'd have predicted",
  raw_model: "Raw model",
};

export default function HindcastPanel({ locationId, variable = "t2m", method, lead }) {
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState(null);

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

  const hasReference = doc.records.some((r) => r.reference !== null);

  return (
    <section className="hindcast-panel viz-root">
      <h3>What you'd have seen vs. what we'd have said</h3>
      <p className="subtitle">
        {LOCATION_LABELS[locationId] ?? locationId} — {lead}h lead,{" "}
        {doc.period.start} to {doc.period.end}. Each point is a forecast issued{" "}
        {lead} hours before the plotted valid time.
      </p>

      <ul className="hindcast-summary">
        {Object.entries(doc.summary_rmse).map(([key, value]) => (
          <li key={key}>
            <span className="label">{RMSE_LABELS[key] ?? key} RMSE</span>
            <span className="value">
              {value === null ? "—" : `${value.toFixed(2)} ${doc.units}`}
            </span>
          </li>
        ))}
      </ul>

      <HindcastChart doc={doc} />

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
