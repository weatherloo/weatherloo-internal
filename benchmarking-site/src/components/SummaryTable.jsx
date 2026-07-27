import { useEffect, useState } from "react";
import {
  METHODS,
  METRICS,
  METRIC_TITLES,
  LEAD_TIMES_HOURS,
  VARIABLES,
} from "../constants.js";
import { loadMethodData } from "../lib/benchmarkData.js";

/** Match linear_regression MIN_PAIRS — low-n scores must not enter the ranking. */
const MIN_SAMPLES_FOR_RANK = 30;

function formatValue(metric, value) {
  if (value === null || value === undefined) return "—";
  if (metric === "acc") return (value === null ? "—" : value.toFixed(2));
  return Number.isFinite(value) ? value.toFixed(2) : "—";
}

function statusLabel(status) {
  if (status === "insufficient") return "Insufficient data";
  if (status === "unavailable") return "Unavailable";
  return null;
}

export default function SummaryTable({ locationId, filters = {} }) {
  const [metric, setMetric] = useState(METRICS[0]); // default rmse
  const [lead, setLead] = useState(LEAD_TIMES_HOURS[3]); // default 24h
  const [variable, setVariable] = useState(VARIABLES[0]);
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!locationId) return;
    let cancelled = false;
    async function loadAll() {
      setLoading(true);
      const promises = METHODS.map(async (m) => {
        try {
          const d = await loadMethodData(m.id, locationId, filters);
          return { id: m.id, label: m.label, data: d };
        } catch (e) {
          return { id: m.id, label: m.label, data: null };
        }
      });

      const results = await Promise.all(promises);
      if (cancelled) return;

      const leadIndex = LEAD_TIMES_HOURS.indexOf(lead);

      const prepared = results.map((r) => {
        const md = r.data;
        let value = null;
        let sample = null;
        let nInits = md?.nInits ?? null;
        if (md && md[variable]) {
          const agg = md[variable];
          const arr = agg[metric];
          if (arr && typeof arr[leadIndex] === "number") value = arr[leadIndex];
          if (agg.n_samples && Array.isArray(agg.n_samples)) sample = agg.n_samples[leadIndex] ?? null;
        }
        // Prefer per-lead n_samples; fall back to nInits (same column as Samples).
        const sampleCount = typeof sample === "number" ? sample : nInits;
        let status = "unavailable";
        if (typeof value === "number") {
          status =
            typeof sampleCount === "number" && sampleCount < MIN_SAMPLES_FOR_RANK
              ? "insufficient"
              : "ok";
        }
        return { id: r.id, label: r.label, value, sample, nInits, status };
      });

      // Sorting per metric semantics
      const cmp = (a, b) => {
        const aAvail = a.status === "ok";
        const bAvail = b.status === "ok";
        // Non-rankable (unavailable / insufficient data) go last
        if (aAvail && !bAvail) return -1;
        if (!aAvail && bAvail) return 1;
        if (!aAvail && !bAvail) return a.label.localeCompare(b.label);

        const av = a.value;
        const bv = b.value;
        if (metric === "rmse" || metric === "mae") {
          return av - bv;
        }
        if (metric === "acc") {
          return bv - av; // higher better
        }
        if (metric === "bias") {
          return Math.abs(av) - Math.abs(bv);
        }
        return 0;
      };

      prepared.sort(cmp);
      if (!cancelled) setRows(prepared);
      setLoading(false);
    }
    loadAll();
    return () => { cancelled = true; };
  }, [locationId, filters, metric, lead, variable]);

  return (
    <div className="summary-table">
      <div className="summary-controls">
        <label>
          Variable
          <select value={variable} onChange={(e) => setVariable(e.target.value)}>
            {VARIABLES.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>

        <label>
          Metric
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {METRICS.map((m) => (
              <option key={m} value={m}>{METRIC_TITLES[m]}</option>
            ))}
          </select>
        </label>

        <label>
          Lead
          <select value={String(lead)} onChange={(e) => setLead(Number(e.target.value))}>
            {LEAD_TIMES_HOURS.map((h) => (
              <option key={h} value={String(h)}>{String(h)}h</option>
            ))}
          </select>
        </label>
      </div>

      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Method</th>
            <th>{METRIC_TITLES[metric]}</th>
            <th>Samples</th>
          </tr>
        </thead>
        <tbody>
          {loading && (
            <tr><td colSpan={4}>Loading…</td></tr>
          )}
          {!loading && rows.map((r, i) => (
            <tr key={r.id} className={r.status === "ok" ? "available" : "unavailable"}>
              <td>{r.status === "ok" ? i + 1 : "—"}</td>
              <td>{r.label}</td>
              <td>{r.status === "ok" ? formatValue(metric, r.value) : statusLabel(r.status)}</td>
              <td>{r.sample ?? r.nInits ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
