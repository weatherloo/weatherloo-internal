import { useEffect, useMemo, useRef, useState } from "react";
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Legend,
  Title,
  Tooltip,
} from "chart.js";
import { HINDCAST_SERIES } from "../constants.js";

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Legend,
  Title,
  Tooltip,
);

/** Read a CSS custom property off the chart's own root so theming stays in CSS. */
function cssVar(el, name, fallback) {
  if (!el) return fallback;
  const v = getComputedStyle(el).getPropertyValue(name).trim();
  return v || fallback;
}

function shortTime(iso) {
  // 2026-07-24T00:00:00Z -> Jul 24 00Z
  const d = new Date(iso);
  const month = d.toLocaleString("en-US", { month: "short", timeZone: "UTC" });
  const day = String(d.getUTCDate()).padStart(2, "0");
  const hour = String(d.getUTCHours()).padStart(2, "0");
  return `${month} ${day} ${hour}Z`;
}

export default function HindcastChart({ doc }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const rootRef = useRef(null);
  const [showTable, setShowTable] = useState(false);

  const records = useMemo(
    () => (doc?.records ?? []).filter((r) => r.actual !== null),
    [doc],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const root = rootRef.current;
    if (!canvas || records.length === 0) return;

    const grid = cssVar(root, "--viz-grid", "rgba(0,0,0,0.08)");
    const ink = cssVar(root, "--viz-text-primary", "#0b0b0b");
    const inkMuted = cssVar(root, "--viz-text-secondary", "#52514e");
    const surface = cssVar(root, "--viz-surface", "#fcfcfb");

    const datasets = HINDCAST_SERIES.map(({ key, label, varName, fallback, emphasis }) => ({
      label,
      data: records.map((r) => r[key]),
      borderColor: cssVar(root, varName, fallback),
      backgroundColor: cssVar(root, varName, fallback),
      borderWidth: emphasis ? 3 : 2,
      borderDash: emphasis ? [] : undefined,
      tension: 0.15,
      pointRadius: 0,
      pointHoverRadius: 5,
      pointHoverBorderColor: surface,
      pointHoverBorderWidth: 2,
      hitRadius: 12,
      spanGaps: false,
      order: emphasis ? 0 : 1,
    }));

    chartRef.current?.destroy();
    chartRef.current = new Chart(canvas, {
      type: "line",
      data: { labels: records.map((r) => shortTime(r.valid_time)), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            display: true,
            position: "top",
            align: "start",
            labels: {
              color: inkMuted,
              boxWidth: 10,
              boxHeight: 10,
              usePointStyle: true,
              pointStyle: "line",
            },
          },
          tooltip: {
            callbacks: {
              title: (items) =>
                items.length ? `valid ${records[items[0].dataIndex].valid_time}` : "",
              label: (ctx) => {
                const v = ctx.parsed.y;
                if (v === null || v === undefined) return `${ctx.dataset.label}: no data`;
                return `${ctx.dataset.label}: ${v.toFixed(2)} ${doc.units}`;
              },
              afterBody: (items) => {
                if (!items.length) return "";
                return `init ${records[items[0].dataIndex].init}`;
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Valid time (UTC)", color: inkMuted },
            ticks: { color: inkMuted, maxTicksLimit: 10, autoSkip: true },
            grid: { color: grid },
          },
          y: {
            title: {
              display: true,
              text: doc.variable === "t2m" ? "Temperature (°C)" : "Wind speed (km/h)",
              color: inkMuted,
            },
            ticks: { color: inkMuted },
            grid: { color: grid },
          },
        },
      },
    });

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [records, doc]);

  if (!doc) return null;
  if (records.length === 0) {
    return (
      <p className="hindcast-empty">
        No verified initializations in this range — every record is missing its
        verifying observation.
      </p>
    );
  }

  return (
    <div className="viz-root hindcast" ref={rootRef}>
      <div className="chart-wrap hindcast-wrap">
        <canvas ref={canvasRef} />
      </div>

      <button
        type="button"
        className="hindcast-toggle"
        onClick={() => setShowTable((v) => !v)}
        aria-expanded={showTable}
      >
        {showTable ? "Hide" : "Show"} table view ({records.length} rows)
      </button>

      {showTable ? (
        <div className="hindcast-table-wrap">
          <table className="hindcast-table">
            <caption>
              All values in {doc.units}. Blank cells are initializations that
              series does not cover.
            </caption>
            <thead>
              <tr>
                <th scope="col">Valid time (UTC)</th>
                {HINDCAST_SERIES.map((s) => (
                  <th scope="col" key={s.key}>
                    {s.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {records.map((r) => (
                <tr key={r.valid_time}>
                  <th scope="row">{r.valid_time}</th>
                  {HINDCAST_SERIES.map((s) => (
                    <td key={s.key}>
                      {r[s.key] === null ? "—" : r[s.key].toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
