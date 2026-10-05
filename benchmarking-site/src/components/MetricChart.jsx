import { useEffect, useRef } from "react";
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend,
);

// datasets: [{ label, values, nSamples, color }]
export default function MetricChart({ title, labels, datasets, yLabel }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const showLegend = datasets.length > 1;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    chartRef.current?.destroy();
    chartRef.current = new Chart(canvas, {
      type: "line",
      data: {
        labels: labels.map((h) => `${h}h`),
        datasets: datasets.map(({ label, values, color }) => ({
          label,
          data: values,
          borderColor: color,
          backgroundColor: color + "22",
          tension: 0.15,
          pointRadius: 3,
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: showLegend, position: "top" },
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const n = datasets[ctx.datasetIndex]?.nSamples?.[ctx.dataIndex];
                return typeof n === "number" ? `n = ${n}` : "";
              },
            },
          },
        },
        scales: {
          x: { title: { display: true, text: "Lead time" } },
          y: { title: { display: true, text: yLabel } },
        },
      },
    });

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [title, labels, datasets, yLabel, showLegend]);

  return (
    <div className={`chart-card${showLegend ? " chart-card--tall" : ""}`}>
      <h4>{title}</h4>
      <div className="chart-wrap">
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}
