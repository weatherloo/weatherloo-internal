import { useEffect, useState } from "react";
import {
  LOCATION_LABELS,
  METHODS,
  TARGET_INIT_COUNT,
} from "../constants.js";
import { loadMethodData } from "../lib/benchmarkData.js";
import VariableCharts from "./VariableCharts.jsx";

function formatLoadStatus(nInits, source) {
  const via = source === "npz" ? " (consolidated NPZ)" : "";
  if (nInits === TARGET_INIT_COUNT) {
    return `Averaged over all ${TARGET_INIT_COUNT} initializations (00/06/12/18 UTC)${via}.`;
  }
  if (nInits === 1) {
    return `Showing 1 initialization. Full-year target: ${TARGET_INIT_COUNT} inits (365 days × 4 cycles)${via}.`;
  }
  return `Averaged over ${nInits} of ${TARGET_INIT_COUNT} initializations (00/06/12/18 UTC)${via}.`;
}

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

      const data = await loadMethodData(methodId, locationId);
      if (cancelled) return;

      if (!data) {
        setLoadStatus(
          `No data for method "${methodId}". Add per-init JSON or export ${methodId}_2025.npz under data/${methodId}/.`,
        );
        return;
      }

      setLoadStatus(formatLoadStatus(data.nInits, data.source));
      setT2mAgg(data.t2m);
      setWindAgg(data.wind_speed);
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [locationId, methodId]);

  const title = LOCATION_LABELS[locationId] ?? locationId;
  const noData = loadStatus.startsWith("No data");

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
