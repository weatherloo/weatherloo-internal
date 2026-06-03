import { useEffect, useState } from "react";
import { LOCATION_LABELS, METHODS } from "../constants.js";
import { aggregateRuns, loadMethodRuns } from "../lib/benchmarkData.js";
import VariableCharts from "./VariableCharts.jsx";

export default function DetailPanel({ locationId }) {
  const [methodId, setMethodId] = useState(METHODS[0].id);
  const [loadStatus, setLoadStatus] = useState("");
  const [t2mAgg, setT2mAgg] = useState(null);
  const [windAgg, setWindAgg] = useState(null);

  useEffect(() => {
    if (!locationId) return;

    let cancelled = false;

    async function load() {
      setLoadStatus("Loading…");
      setT2mAgg(null);
      setWindAgg(null);

      const runs = await loadMethodRuns(methodId);
      if (cancelled) return;

      const n = runs.length;
      if (n === 0) {
        setLoadStatus(
          `No JSON found for method "${methodId}". Add files under data/${methodId}/.`,
        );
        return;
      }

      setLoadStatus(
        n === 1
          ? "Showing 1 initialization (placeholder). Production: average over 365 runs."
          : `Averaged over ${n} initializations.`,
      );
      setT2mAgg(aggregateRuns(runs, locationId, "t2m"));
      setWindAgg(aggregateRuns(runs, locationId, "wind_speed"));
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [locationId, methodId]);

  const title = LOCATION_LABELS[locationId] ?? locationId;
  const noData = loadStatus.startsWith("No JSON");

  return (
    <section id="detail-panel" aria-label="Skill score charts">
      <div className="panel-header">
        <h2 id="detail-title">{title}</h2>
        <label>
          Method
          <select
            id="method-select"
            value={methodId}
            onChange={(e) => setMethodId(e.target.value)}
          >
            {METHODS.map(({ id, label }) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <p id="load-status" className="status">
        {loadStatus}
      </p>

      {noData ? (
        <p>—</p>
      ) : (
        <>
          <div className="variable-block">
            <h3>2 m temperature (t2m)</h3>
            <VariableCharts agg={t2mAgg} />
          </div>

          <div className="variable-block">
            <h3>10 m wind speed</h3>
            <VariableCharts agg={windAgg} />
          </div>
        </>
      )}
    </section>
  );
}
