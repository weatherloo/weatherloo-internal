import { useEffect, useState } from "react";
import {
  INIT_CYCLES,
  LOCATION_LABELS,
  METHODS,
  TARGET_INIT_COUNT,
} from "../constants.js";
import { loadMethodData } from "../lib/benchmarkData.js";
import VariableCharts from "./VariableCharts.jsx";

function formatLoadStatus(nInits, source, selectedCycles) {
  const via = source === "npz" ? " (consolidated NPZ)" : "";
  const allSelected = selectedCycles.length === INIT_CYCLES.length;
  const cycleLabel = allSelected
    ? "00/06/12/18 UTC"
    : selectedCycles.map((h) => `${String(h).padStart(2, "0")}Z`).join(", ");
  const target = allSelected ? TARGET_INIT_COUNT : 365 * selectedCycles.length;

  if (nInits === 0) return `No initializations match the selected cycles (${cycleLabel}).`;
  if (nInits === target) return `Averaged over all ${nInits} initializations (${cycleLabel})${via}.`;
  return `Averaged over ${nInits} of ${target} initializations (${cycleLabel})${via}.`;
}

export default function DetailPanel({ locationId }) {
  const [methodId, setMethodId] = useState(METHODS[0].id);
  const [cycleFilter, setCycleFilter] = useState("all");
  const [loadStatus, setLoadStatus] = useState("");
  const [t2mAgg, setT2mAgg] = useState(null);
  const [windAgg, setWindAgg] = useState(null);

  const selectedCycles =
    cycleFilter === "all" ? INIT_CYCLES : [parseInt(cycleFilter, 10)];

  useEffect(() => {
    if (!locationId) return;

    let cancelled = false;

    async function load() {
      setLoadStatus("Loading…");
      setT2mAgg(null);
      setWindAgg(null);

      if (selectedCycles.length === 0) {
        setLoadStatus("No cycles selected. Select at least one cycle.");
        return;
      }

      const filters =
        selectedCycles.length === INIT_CYCLES.length
          ? {}
          : { cycles: selectedCycles.join(",") };

      const data = await loadMethodData(methodId, locationId, filters);
      if (cancelled) return;

      if (!data) {
        setLoadStatus(
          `No data for method "${methodId}". Add per-init JSON or export ${methodId}_2025.npz under data/${methodId}/.`,
        );
        return;
      }

      setLoadStatus(formatLoadStatus(data.nInits, data.source, selectedCycles));
      setT2mAgg(data.t2m);
      setWindAgg(data.wind_speed);
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [locationId, methodId, cycleFilter]);

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
        <label>
          Init cycle
          <select
            id="cycle-select"
            value={cycleFilter}
            onChange={(e) => setCycleFilter(e.target.value)}
          >
            <option value="all">All cycles</option>
            {INIT_CYCLES.map((hour) => (
              <option key={hour} value={String(hour)}>
                {String(hour).padStart(2, "0")}Z
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
