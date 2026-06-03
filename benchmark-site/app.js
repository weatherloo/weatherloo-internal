/**
 * Minimal benchmark dashboard. Expects JSON files in the issue #1 format:
 * one object per method + initialization, with locations / variables / metrics.
 */

import OlMap from "https://esm.sh/ol@9.2.4/Map.js";
import View from "https://esm.sh/ol@9.2.4/View.js";
import TileLayer from "https://esm.sh/ol@9.2.4/layer/Tile.js";
import VectorLayer from "https://esm.sh/ol@9.2.4/layer/Vector.js";
import XYZ from "https://esm.sh/ol@9.2.4/source/XYZ.js";
import VectorSource from "https://esm.sh/ol@9.2.4/source/Vector.js";
import Feature from "https://esm.sh/ol@9.2.4/Feature.js";
import Point from "https://esm.sh/ol@9.2.4/geom/Point.js";
import { fromLonLat, transformExtent } from "https://esm.sh/ol@9.2.4/proj.js";
import Style from "https://esm.sh/ol@9.2.4/style/Style.js";
import CircleStyle from "https://esm.sh/ol@9.2.4/style/Circle.js";
import Fill from "https://esm.sh/ol@9.2.4/style/Fill.js";
import Stroke from "https://esm.sh/ol@9.2.4/style/Stroke.js";

const LEAD_TIMES_HOURS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72];
const VARIABLES = ["t2m", "wind_speed"];
const METRICS = ["rmse", "mae", "bias", "acc"];

/** Southern Ontario extent [minLon, minLat, maxLon, maxLat] in WGS84. */
const SOUTH_ONTARIO_EXTENT_4326 = [-84.85, 41.85, -77.75, 44.95];
const SOUTH_ONTARIO_EXTENT = transformExtent(
  SOUTH_ONTARIO_EXTENT_4326,
  "EPSG:4326",
  "EPSG:3857",
);

/** Station coords; should match lat/lon in benchmark JSON. */
const STATIONS = [
  {
    id: "eric_d_soulis",
    label: "Eric D. Soulis station",
    lat: 43.4668,
    lon: -80.5164,
  },
  {
    id: "cyyz",
    label: "Toronto Pearson (CYYZ)",
    lat: 43.6777,
    lon: -79.6248,
  },
];

const LOCATION_LABELS = Object.fromEntries(
  STATIONS.map((s) => [s.id, s.label]),
);

/** Method id → display label. Data files use the method id as filename stem. */
const METHODS = [
  { id: "climatology", label: "Climatology baseline" },
  { id: "persistence", label: "Persistence baseline" },
  { id: "mean_bias_correction", label: "Mean bias correction" },
  { id: "linear_regression", label: "Linear regression" },
  { id: "mos", label: "Model Output Statistics (MOS)" },
  { id: "gefs_mean", label: "GEFS ensemble mean" },
  { id: "gfs_interpolated", label: "GFS interpolated at station" },
  { id: "gfs_analysis", label: "GFS past-hour analysis" },
  { id: "ecmwf_aifs", label: "ECMWF AIFS" },
  { id: "graphcast", label: "GraphCast" },
  { id: "pangu", label: "Pangu-Weather" },
];

const DATA_DIR = "./data";

const state = {
  locationId: null,
  methodId: METHODS[0].id,
  /** @type {Map<string, object[]>} methodId → array of run JSON objects */
  runsByMethod: new Map(),
  charts: [],
  olMap: null,
  stationsLayer: null,
};

const els = {
  map: document.getElementById("map"),
  selectedLocation: document.getElementById("selected-location"),
  detailPanel: document.getElementById("detail-panel"),
  detailTitle: document.getElementById("detail-title"),
  methodSelect: document.getElementById("method-select"),
  loadStatus: document.getElementById("load-status"),
  chartsT2m: document.getElementById("charts-t2m"),
  chartsWind: document.getElementById("charts-wind_speed"),
};

function initMethodSelect() {
  for (const { id, label } of METHODS) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = label;
    els.methodSelect.appendChild(opt);
  }
  els.methodSelect.value = state.methodId;
  els.methodSelect.addEventListener("change", () => {
    state.methodId = els.methodSelect.value;
    renderCharts();
  });
}

function stationStyle(feature) {
  const selected = feature.get("id") === state.locationId;
  return new Style({
    image: new CircleStyle({
      radius: selected ? 9 : 7,
      fill: new Fill({ color: selected ? "#f0f6fc" : "#58a6ff" }),
      stroke: new Stroke({ color: "#0d1117", width: 2 }),
    }),
  });
}

function initMap() {
  const stationFeatures = STATIONS.map(
    (station) =>
      new Feature({
        geometry: new Point(fromLonLat([station.lon, station.lat])),
        id: station.id,
        label: station.label,
      }),
  );

  const stationsSource = new VectorSource({ features: stationFeatures });

  state.stationsLayer = new VectorLayer({
    source: stationsSource,
    style: stationStyle,
  });

  const darkBasemap = new TileLayer({
    source: new XYZ({
      url: "https://{a-d}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
      attributions:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
      maxZoom: 19,
    }),
  });

  state.olMap = new OlMap({
    target: els.map,
    layers: [darkBasemap, state.stationsLayer],
    view: new View({
      extent: SOUTH_ONTARIO_EXTENT,
      constrainOnlyCenter: false,
      minZoom: 7,
      maxZoom: 13,
    }),
    controls: [],
  });

  state.olMap.getView().fit(SOUTH_ONTARIO_EXTENT, {
    padding: [24, 24, 24, 24],
    maxZoom: 9,
  });

  state.olMap.on("click", (evt) => {
    const feature = state.olMap.forEachFeatureAtPixel(evt.pixel, (f) => f, {
      layerFilter: (layer) => layer === state.stationsLayer,
    });
    if (feature) selectLocation(feature.get("id"));
  });

  state.olMap.on("pointermove", (evt) => {
    const hit = state.olMap.hasFeatureAtPixel(evt.pixel, {
      layerFilter: (layer) => layer === state.stationsLayer,
    });
    state.olMap.getTargetElement().style.cursor = hit ? "pointer" : "";
  });

  setTimeout(() => state.olMap?.updateSize(), 0);
}

function updateMarkerSelection() {
  state.stationsLayer?.changed();
}

function selectLocation(locationId) {
  state.locationId = locationId;
  updateMarkerSelection();
  els.selectedLocation.textContent = `Selected: ${LOCATION_LABELS[locationId] ?? locationId}`;
  els.detailPanel.hidden = false;
  els.detailTitle.textContent = LOCATION_LABELS[locationId] ?? locationId;
  state.olMap?.updateSize();
  renderCharts();
}

/**
 * Average metric arrays across multiple run objects for one location + variable.
 * @param {object[]} runs
 * @param {string} locationId
 * @param {string} variableKey
 */
function aggregateRuns(runs, locationId, variableKey) {
  const slices = runs
    .map((run) => run.locations?.[locationId]?.variables?.[variableKey])
    .filter(Boolean);

  if (slices.length === 0) return null;

  const leadTimes = slices[0].lead_times_hours ?? LEAD_TIMES_HOURS;
  const out = { lead_times_hours: leadTimes };

  for (const metric of METRICS) {
    out[metric] = leadTimes.map((_, i) => {
      const values = slices
        .map((s) => s[metric]?.[i])
        .filter((v) => typeof v === "number");
      if (values.length === 0) return null;
      return values.reduce((a, b) => a + b, 0) / values.length;
    });
  }

  return out;
}

async function loadMethodRuns(methodId) {
  if (state.runsByMethod.has(methodId)) {
    return state.runsByMethod.get(methodId);
  }

  const indexUrl = `${DATA_DIR}/${methodId}/index.json`;
  let files = [`${methodId}_sample.json`];

  try {
    const res = await fetch(indexUrl);
    if (res.ok) {
      const index = await res.json();
      if (Array.isArray(index.files) && index.files.length > 0) {
        files = index.files;
      }
    }
  } catch {
    /* use default sample filename */
  }

  const runs = [];
  for (const file of files) {
    const path = `${DATA_DIR}/${methodId}/${file}`;
    try {
      const res = await fetch(path);
      if (!res.ok) continue;
      const data = await res.json();
      if (data.method && data.locations) runs.push(data);
    } catch {
      /* skip missing placeholder files */
    }
  }

  state.runsByMethod.set(methodId, runs);
  return runs;
}

function destroyCharts() {
  for (const chart of state.charts) chart.destroy();
  state.charts = [];
}

function renderMetricChart(container, title, labels, values, yLabel) {
  const card = document.createElement("div");
  card.className = "chart-card";
  const heading = document.createElement("h4");
  heading.textContent = title;
  const wrap = document.createElement("div");
  wrap.className = "chart-wrap";
  const canvas = document.createElement("canvas");
  wrap.appendChild(canvas);
  card.append(heading, wrap);
  container.appendChild(card);

  const chart = new Chart(canvas, {
    type: "line",
    data: {
      labels: labels.map((h) => `${h}h`),
      datasets: [
        {
          label: yLabel,
          data: values,
          borderColor: "#222",
          backgroundColor: "rgba(0,0,0,0.05)",
          tension: 0.15,
          pointRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "Lead time" } },
        y: { title: { display: true, text: yLabel } },
      },
    },
  });
  state.charts.push(chart);
}

function renderVariableSection(container, variableKey, agg) {
  container.replaceChildren();
  if (!agg) {
    container.textContent = "No data for this variable (placeholder JSON missing).";
    return;
  }

  const labels = agg.lead_times_hours;
  const metricTitles = {
    rmse: "RMSE",
    mae: "MAE",
    bias: "Bias",
    acc: "ACC",
  };

  for (const metric of METRICS) {
    renderMetricChart(
      container,
      metricTitles[metric],
      labels,
      agg[metric],
      metricTitles[metric],
    );
  }
}

async function renderCharts() {
  if (!state.locationId) return;

  destroyCharts();
  els.chartsT2m.replaceChildren();
  els.chartsWind.replaceChildren();
  els.loadStatus.textContent = "Loading…";

  const runs = await loadMethodRuns(state.methodId);
  const n = runs.length;

  if (n === 0) {
    els.loadStatus.textContent =
      `No JSON found for method "${state.methodId}". Add files under data/${state.methodId}/.`;
    els.chartsT2m.textContent = "—";
    els.chartsWind.textContent = "—";
    return;
  }

  els.loadStatus.textContent =
    n === 1
      ? `Showing 1 initialization (placeholder). Production: average over 365 runs.`
      : `Averaged over ${n} initializations.`;

  const t2m = aggregateRuns(runs, state.locationId, "t2m");
  const wind = aggregateRuns(runs, state.locationId, "wind_speed");

  renderVariableSection(els.chartsT2m, "t2m", t2m);
  renderVariableSection(els.chartsWind, "wind_speed", wind);
}

initMethodSelect();
initMap();
