import { METRICS, METRIC_TITLES } from "../constants.js";
import MetricChart from "./MetricChart.jsx";

export default function VariableCharts({ agg }) {
  if (!agg) {
    return <p>No data for this variable (placeholder JSON missing).</p>;
  }

  const labels = agg.lead_times_hours;

  return (
    <div className="chart-grid">
      {METRICS.map((metric) => (
        <MetricChart
          key={metric}
          title={METRIC_TITLES[metric]}
          labels={labels}
          values={agg[metric]}
          yLabel={METRIC_TITLES[metric]}
        />
      ))}
    </div>
  );
}
