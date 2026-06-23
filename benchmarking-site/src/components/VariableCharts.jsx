import { METRICS, METRIC_TITLES } from "../constants.js";
import MetricChart from "./MetricChart.jsx";

const COLORS = [
  "#2563eb", "#dc2626", "#16a34a", "#d97706", "#9333ea",
  "#0891b2", "#db2777", "#65a30d", "#ea580c", "#0284c7",
  "#7c3aed", "#059669",
];

// entries: [{ id, label, agg }]
export default function VariableCharts({ entries }) {
  const valid = entries.filter((e) => e.agg);

  if (valid.length === 0) {
    return <p>No data for this variable (placeholder JSON missing).</p>;
  }

  const labels = valid[0].agg.lead_times_hours;

  let sampleSummary = null;
  if (valid.length === 1) {
    const nSamples = valid[0].agg.n_samples;
    if (nSamples?.length > 0) {
      const min = Math.min(...nSamples);
      const max = Math.max(...nSamples);
      sampleSummary = min === max
        ? `n = ${min} per lead — valid forecast/obs pairs contributing to RMSE, MAE & bias (out of ${max} loaded inits)`
        : `n = ${min}–${max} per lead — valid forecast/obs pairs contributing to RMSE, MAE & bias (out of 1460 target inits; varies by lead due to observation gaps)`;
    }
  }

  return (
    <>
      {sampleSummary && <p className="sample-summary">{sampleSummary}</p>}
      <div className="chart-grid">
        {METRICS.map((metric) => (
          <MetricChart
            key={metric}
            title={METRIC_TITLES[metric]}
            labels={labels}
            yLabel={METRIC_TITLES[metric]}
            datasets={valid.map((e, i) => ({
              label: e.label,
              values: e.agg[metric],
              nSamples: e.agg.n_samples ?? null,
              color: COLORS[i % COLORS.length],
            }))}
          />
        ))}
      </div>
    </>
  );
}
