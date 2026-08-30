import { useEffect, useMemo, useRef, useState } from "react";
import {
  INIT_CYCLES,
  LOCATION_LABELS,
  METHODS,
  TIME_PRESETS,
} from "../constants.js";
import { loadMethodData } from "../lib/benchmarkData.js";
import VariableCharts from "./VariableCharts.jsx";
import SummaryTable from "./SummaryTable.jsx";

function buildFilters(cycleFilter, timePreset, customFrom, customTo) {
  const selectedCycles =
    cycleFilter === "all" ? INIT_CYCLES : [parseInt(cycleFilter, 10)];

  const filters = {};
  if (selectedCycles.length !== INIT_CYCLES.length) {
    filters.cycles = selectedCycles.join(",");
  }

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

  return filters;
}

function formatLoadStatus(nInits, source, selectedCycles, timePreset, customFrom, customTo) {
  const via = source === "aggregate" ? " (precomputed NPZ aggregate)" : "";
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
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const pickerRef = useRef(null);
  const [cycleFilter, setCycleFilter] = useState("all");
  const [timePreset, setTimePreset] = useState("all");
  const [customFrom, setCustomFrom] = useState("");
  const [customTo, setCustomTo] = useState("");
  const [loadStatus, setLoadStatus] = useState("");
  const [methodResults, setMethodResults] = useState([]);

  const selectedCycles =
    cycleFilter === "all" ? INIT_CYCLES : [parseInt(cycleFilter, 10)];
  const filters = useMemo(
    () => buildFilters(cycleFilter, timePreset, customFrom, customTo),
    [cycleFilter, timePreset, customFrom, customTo],
  );

  function toggleMethod(id) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  useEffect(() => {
    function handleOutsideClick(e) {
      if (!pickerRef.current?.contains(e.target)) setDropdownOpen(false);
    }
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, []);

  useEffect(() => {
    if (!locationId) return;

    let cancelled = false;

    async function load() {
      if (selectedIds.size === 0) {
        setLoadStatus("");
        setMethodResults([]);
        return;
      }

      setLoadStatus("Loading…");
      setMethodResults([]);

      if (selectedCycles.length === 0) {
        setLoadStatus("No cycles selected. Select at least one cycle.");
        return;
      }

      const ids = [...selectedIds];
      const results = await Promise.all(
        ids.map((id) => loadMethodData(id, locationId, filters))
      );

      if (cancelled) return;

      const entries = ids
        .map((id, i) => ({
          id,
          label: METHODS.find((m) => m.id === id)?.label ?? id,
          ...(results[i] ?? {}),
        }))
        .filter((_, i) => results[i] !== null);

      if (entries.length === 0) {
        setLoadStatus(
          `No data for selected method(s). Add per-init JSON or export NPZ under data/<method>/.`
        );
        return;
      }

      setLoadStatus(
        entries.length === 1
          ? formatLoadStatus(entries[0].nInits, entries[0].source, selectedCycles, timePreset, customFrom, customTo)
          : `${entries.length} methods loaded.`
      );
      setMethodResults(entries);
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [locationId, selectedIds, cycleFilter, timePreset, customFrom, customTo, filters]);

  const title = LOCATION_LABELS[locationId] ?? locationId;
  const noData = selectedIds.size > 0 && methodResults.length === 0 && !loadStatus.startsWith("Loading");

  return (
    <section id="detail-panel" aria-label="Skill score charts">
      <div className="panel-header">
        <h2 id="detail-title">{title}</h2>
        <div className="method-picker" ref={pickerRef}>
          <span className="picker-label">Methods</span>
          <button
            className="picker-trigger"
            onClick={() => setDropdownOpen((o) => !o)}
          >
            {selectedIds.size === 0 ? "None selected" : `${selectedIds.size} selected`}
            <span className="picker-caret">{dropdownOpen ? "▲" : "▼"}</span>
          </button>
          {dropdownOpen && (
            <div className="method-dropdown">
              <button
                className="clear-all-btn"
                onClick={() => setSelectedIds(new Set())}
              >
                Clear all
              </button>
              {METHODS.map(({ id, label }) => (
                <label key={id} className="method-option">
                  <input
                    type="checkbox"
                    checked={selectedIds.has(id)}
                    onChange={() => toggleMethod(id)}
                  />
                  {label}
                </label>
              ))}
            </div>
          )}
        </div>
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

      {selectedIds.size === 0 ? (
        <p className="status">Select one or more methods above to see charts.</p>
      ) : noData ? (
        <p>—</p>
      ) : (
        <>
          <div className="variable-block">
            <h3>2 m temperature (t2m)</h3>
            <VariableCharts entries={methodResults.map((r) => ({ id: r.id, label: r.label, agg: r.t2m }))} />
          </div>

          <div className="variable-block">
            <h3>10 m wind speed</h3>
            <VariableCharts entries={methodResults.map((r) => ({ id: r.id, label: r.label, agg: r.wind_speed }))} />
          </div>
        </>
      )}
    </section>
  );
}
