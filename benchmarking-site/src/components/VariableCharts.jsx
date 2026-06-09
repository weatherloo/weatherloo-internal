import { METRICS, METRIC_TITLES } from "../constants.js";
import MetricChart from "./MetricChart.jsx";

export default function VariableCharts({ agg }) {
  if (!agg) {
    return <p>No data for this variable (placeholder JSON missing).</p>;
  }

  const labels = agg.lead_times_hours;
  const nSamples = agg.n_samples ?? null;

  let sampleSummary = null;
  if (nSamples && nSamples.length > 0) {
    const min = Math.min(...nSamples);
    const max = Math.max(...nSamples);
    sampleSummary = min === max
      ? `n = ${min} per lead — valid forecast/obs pairs contributing to RMSE, MAE & bias (out of ${max} loaded inits)`
      : `n = ${min}–${max} per lead — valid forecast/obs pairs contributing to RMSE, MAE & bias (out of 1460 target inits; varies by lead due to observation gaps)`;
  }

  return (
    <>
      {sampleSummary && (
        <p className="sample-summary">{sampleSummary}</p>
      )}
      <div className="chart-grid">
        {METRICS.map((metric) => (
          <MetricChart
            key={metric}
            title={METRIC_TITLES[metric]}
            labels={labels}
            values={agg[metric]}
            yLabel={METRIC_TITLES[metric]}
            nSamples={nSamples}
          />
        ))}
      </div>
    </>
  );
}
