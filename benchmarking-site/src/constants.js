/** Full-year 2025 target: 365 days × 4 cycles (00/06/12/18 UTC). See AGENTS.md. */
export const TARGET_INIT_COUNT = 1460;
export const INIT_CYCLES = [0, 6, 12, 18];

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
  { id: "gfs_analysis", label: "GFS analysis" },
  { id: "hrdps_analysis", label: "HRDPS analysis" },
  { id: "hrrr_interpolated", label: "HRRR interpolated at station" },
  { id: "cnn_lstm_bias_correction", label: "CNN-LSTM bias correction (HRRR)" },
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

export const TIME_PRESETS = [
  { value: "all", label: "All data" },
  { value: "Q1", label: "Q1 (Jan–Mar)", from: "2025-01-01T00:00:00Z", to: "2025-03-31T18:00:00Z" },
  { value: "Q2", label: "Q2 (Apr–Jun)", from: "2025-04-01T00:00:00Z", to: "2025-06-30T18:00:00Z" },
  { value: "Q3", label: "Q3 (Jul–Sep)", from: "2025-07-01T00:00:00Z", to: "2025-09-30T18:00:00Z" },
  { value: "Q4", label: "Q4 (Oct–Dec)", from: "2025-10-01T00:00:00Z", to: "2025-12-31T18:00:00Z" },
  { value: "Jan", label: "January",   from: "2025-01-01T00:00:00Z", to: "2025-01-31T18:00:00Z" },
  { value: "Feb", label: "February",  from: "2025-02-01T00:00:00Z", to: "2025-02-28T18:00:00Z" },
  { value: "Mar", label: "March",     from: "2025-03-01T00:00:00Z", to: "2025-03-31T18:00:00Z" },
  { value: "Apr", label: "April",     from: "2025-04-01T00:00:00Z", to: "2025-04-30T18:00:00Z" },
  { value: "May", label: "May",       from: "2025-05-01T00:00:00Z", to: "2025-05-31T18:00:00Z" },
  { value: "Jun", label: "June",      from: "2025-06-01T00:00:00Z", to: "2025-06-30T18:00:00Z" },
  { value: "Jul", label: "July",      from: "2025-07-01T00:00:00Z", to: "2025-07-31T18:00:00Z" },
  { value: "Aug", label: "August",    from: "2025-08-01T00:00:00Z", to: "2025-08-31T18:00:00Z" },
  { value: "Sep", label: "September", from: "2025-09-01T00:00:00Z", to: "2025-09-30T18:00:00Z" },
  { value: "Oct", label: "October",   from: "2025-10-01T00:00:00Z", to: "2025-10-31T18:00:00Z" },
  { value: "Nov", label: "November",  from: "2025-11-01T00:00:00Z", to: "2025-11-30T18:00:00Z" },
  { value: "Dec", label: "December",  from: "2025-12-01T00:00:00Z", to: "2025-12-31T18:00:00Z" },
  { value: "custom", label: "Custom…" },
];
