import { useEffect, useState } from "react";
import {
  INIT_CYCLES,
  LOCATION_LABELS,
  METHODS,
  TARGET_INIT_COUNT,
  TIME_PRESETS,
} from "../constants.js";
import { loadMethodData } from "../lib/benchmarkData.js";
import VariableCharts from "./VariableCharts.jsx";
import SummaryTable from "./SummaryTable.jsx";

function formatLoadStatus(nInits, source, selectedCycles, timePreset, customFrom, customTo) {
  const via = source === "npz" ? " (consolidated NPZ)" : "";
  const allCycles = selectedCycles.length === INIT_CYCLES.length;
  const cycleLabel = allCycles
    ? null
    : selectedCycles.map((h) => `${String(h).padStart(2, "0")}Z`).join("/");

  let timeLabel = null;
  if (timePreset !== "all") {
    const preset = TIME_PRESETS.find((p) => p.value === timePreset);
    if (preset && timePreset !== "custom") {
      timeLabel = preset.label;
    } else if (timePreset === "custom" && (customFrom || customTo)) {
      timeLabel = `${customFrom || "…"} – ${customTo || "…"}`;
    }
  }

  const parts = [timeLabel, cycleLabel].filter(Boolean);
  const rangeDesc = parts.length > 0 ? `${parts.join(", ")} — ` : "";

  if (nInits === 0) return `No initializations match the selected filters.`;
  return `${rangeDesc}${nInits} inits${via}.`;
}

export default function DetailPanel({ locationId }) {
  const [methodId, setMethodId] = useState(METHODS[0].id);
  const [cycleFilter, setCycleFilter] = useState("all");
  const [timePreset, setTimePreset] = useState("all");
  const [customFrom, setCustomFrom] = useState("");
  const [customTo, setCustomTo] = useState("");
  const [loadStatus, setLoadStatus] = useState("");
  const [t2mAgg, setT2mAgg] = useState(null);
  const [windAgg, setWindAgg] = useState(null);

  const selectedCycles =
    cycleFilter === "all" ? INIT_CYCLES : [parseInt(cycleFilter, 10)];

  const filters = {};
  if (selectedCycles.length !== INIT_CYCLES.length) filters.cycles = selectedCycles.join(",");

  if (timePreset !== "all") {
    const preset = TIME_PRESETS.find((p) => p.value === timePreset);
    if (preset && timePreset !== "custom") {
      filters.init_from = preset.from;
      filters.init_to = preset.to;
    } else if (timePreset === "custom") {
      if (customFrom) filters.init_from = `${customFrom}T00:00:00Z`;
      if (customTo) filters.init_to = `${customTo}T18:00:00Z`;
    }
  }

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

      const filters = {};
      if (selectedCycles.length !== INIT_CYCLES.length)
        filters.cycles = selectedCycles.join(",");

      if (timePreset !== "all") {
        const preset = TIME_PRESETS.find((p) => p.value === timePreset);
        if (preset && timePreset !== "custom") {
          filters.init_from = preset.from;
          filters.init_to = preset.to;
        } else if (timePreset === "custom") {
          if (customFrom) filters.init_from = `${customFrom}T00:00:00Z`;
          if (customTo) filters.init_to = `${customTo}T18:00:00Z`;
        }
      }

      const data = await loadMethodData(methodId, locationId, filters);
      if (cancelled) return;

      if (!data) {
        setLoadStatus(
          `No data for method "${methodId}". Add per-init JSON or export ${methodId}_2025.npz under data/${methodId}/.`,
        );
        return;
      }

      setLoadStatus(formatLoadStatus(data.nInits, data.source, selectedCycles, timePreset, customFrom, customTo));
      setT2mAgg(data.t2m);
      setWindAgg(data.wind_speed);
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [locationId, methodId, cycleFilter, timePreset, customFrom, customTo]);

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
        <label>
          Time range
          <select
            id="time-preset-select"
            value={timePreset}
            onChange={(e) => setTimePreset(e.target.value)}
          >
            <option value="all">All data</option>
            <optgroup label="Quarters">
              {TIME_PRESETS.filter((p) => p.value.startsWith("Q")).map((p) => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </optgroup>
            <optgroup label="Months">
              {TIME_PRESETS.filter((p) => p.value.length === 3 && p.value !== "all" && !p.value.startsWith("Q")).map((p) => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </optgroup>
            <option value="custom">Custom…</option>
          </select>
        </label>
        {timePreset === "custom" && (
          <>
            <label>
              From
              <input
                type="date"
                value={customFrom}
                min="2025-01-01"
                max="2025-12-31"
                onChange={(e) => setCustomFrom(e.target.value)}
              />
            </label>
            <label>
              To
              <input
                type="date"
                value={customTo}
                min="2025-01-01"
                max="2025-12-31"
                onChange={(e) => setCustomTo(e.target.value)}
              />
            </label>
          </>
        )}
      </div>

      <p id="load-status" className="status">
        {loadStatus}
      </p>

      <SummaryTable locationId={locationId} filters={filters} />

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
