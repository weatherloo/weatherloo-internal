/** Full-year 2025 target: 365 days × 4 cycles (00/06/12/18 UTC). See AGENTS.md. */
export const TARGET_INIT_COUNT = 1460;

export const LEAD_TIMES_HOURS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72];
export const VARIABLES = ["t2m", "wind_speed"];
export const METRICS = ["rmse", "mae", "bias", "acc"];

export const STATIONS = [
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

export const LOCATION_LABELS = Object.fromEntries(
  STATIONS.map((s) => [s.id, s.label]),
);

export const METHODS = [
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

export const METRIC_TITLES = {
  rmse: "RMSE",
  mae: "MAE",
  bias: "Bias",
  acc: "ACC",
};

/** Southern Ontario extent [minLon, minLat, maxLon, maxLat] in WGS84. */
export const SOUTH_ONTARIO_EXTENT_4326 = [-84.85, 41.85, -77.75, 44.95];

export const DATA_DIR = "/data";
export const API_DIR = "/api/benchmark";
