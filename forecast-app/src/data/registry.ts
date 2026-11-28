/**
 * Add a model here and make its forecast records available from the source.
 */
export interface ModelInfo {
  id: string; // matches ForecastRecord.model_id
  label: string;
  family: string;
  /** Where the records come from: "fixture" today, "api" once the read API is live. */
  source: "fixture" | "api";
  /** One or two sentences: who makes it and what it is. Shown on the compare screen. */
  description?: string;
  /** Benchmark site method id, when a directly comparable historical score exists. */
  benchmarkMethodId?: string;
  /** Built or trained by this team (starred in the model picker). */
  ours?: boolean;
  /** Longer write-up shown under the forecast: short titled sections. */
  about?: { title: string; body: string }[];
}

export const MODELS: ModelInfo[] = [
  {
    id: "hrrr_interpolated",
    label: "HRRR",
    family: "hrrr",
    source: "fixture",
    benchmarkMethodId: "hrrr_interpolated",
    description: "NOAA's High-Resolution Rapid Refresh: a physics-based model on a 3 km grid over North America, re-run every hour. Interpolated to the station.",
    about: [
      { title: "What it is", body: "The High-Resolution Rapid Refresh, run by NOAA. A physics-based model that simulates the atmosphere on a 3 km grid over the continental US and southern Canada, and starts a new run every hour from the latest radar, satellite and surface observations." },
      { title: "Strengths", body: "Fine enough to resolve individual thunderstorms, and refreshed hourly, so it is one of the best short-range guides available for North America." },
      { title: "Limits at one station", body: "Each grid cell averages 9 km² of terrain, land use and buildings, so the forecast at a single thermometer carries steady, repeatable errors. We interpolate it to the station; that is the baseline our correction starts from." },
    ],
  },
  {
    id: "hrrr_cnn",
    label: "HRRR CNN",
    family: "HRRR station correction",
    source: "fixture",
    description: "Our model. A neural network that reads HRRR's local forecast and recent nearby observations, and predicts HRRR's error at this station.",
    ours: true,
    about: [
      { title: "What we built", body: "A correction layer we trained on top of NOAA's HRRR. HRRR still makes the forecast; our convolutional neural network predicts how wrong HRRR will be at this station for each hour ahead, and adds that back. We train one per station, for temperature and wind (and precipitation at Soulis)." },
      { title: "How we trained it", body: "Target: the observed value minus HRRR's forecast at the station. Inputs, all available when the forecast is issued: a 17×17 map of 15 HRRR fields around the station (about 50 km), the station's full 0–48 h HRRR trajectory, the three previous HRRR runs and their spread, and recent observations and HRRR errors at the station and its 30 nearest neighbours. Trained on Dec 2020–2023, tuned on 2024, three seeds averaged." },
      { title: "Why it works", body: "HRRR's misses at one site are mostly systematic, not random. They come from what a 3 km cell can't see (the instrument's exact siting, nearby surfaces) and from weather regimes HRRR handles badly. Those errors repeat whenever the surrounding pattern repeats, so a network that sees the local map can learn them. Recent errors also tell it how HRRR is doing today. Learning the correction rather than the weather is a far smaller problem: physics comes from HRRR, and the network only learns the residual." },
      { title: "Results (2025, never seen in training)", body: "At Eric D. Soulis, RMSE falls from 2.18 to 1.77 °C for temperature (19% lower) and from 10.86 to 2.73 km/h for wind (75% lower). On matched cases it also beats ECMWF AIFS and GraphCast through about 36 h; at Pearson, AIFS is better at 42–48 h. The forecast shown above is a demo fixture; these scores are from the real evaluation." },
    ],
  },
  {
    id: "ecmwf_aifs",
    label: "ECMWF AIFS",
    family: "global NWP",
    source: "fixture",
    benchmarkMethodId: "ecmwf_aifs",
    description: "AIFS Single, the machine-learned global model from the European Centre for Medium-Range Weather Forecasts (ECMWF). 0.25° grid (~28 km), 6-hourly steps.",
    about: [
      { title: "What it is", body: "The Artificial Intelligence Forecasting System from ECMWF, the European weather centre behind the IFS, one of the world's leading physics models. AIFS Single has been run operationally since 2025." },
      { title: "How it works", body: "A neural network trained on decades of ECMWF's reanalysis (its best reconstruction of past weather). From ECMWF's analysis of the current atmosphere, it steps the whole globe forward 6 hours at a time on a 0.25° grid (about 28 km)." },
      { title: "Limits at one station", body: "Very strong over days, but a 28 km grid can't see local effects at a single site, and it only updates four times a day." },
    ],
  },
  {
    id: "graphcast",
    label: "GraphCast",
    family: "AI global",
    source: "fixture",
    benchmarkMethodId: "graphcast",
    description: "Google DeepMind's graph-neural-network global model, as run by NOAA from GFS starting conditions (GraphCastGFS). 0.25° grid, 6-hourly steps.",
    about: [
      { title: "What it is", body: "A global machine-learning model from Google DeepMind, published in 2023. The version here is GraphCastGFS, which NOAA runs starting from its own GFS analysis." },
      { title: "How it works", body: "A graph neural network trained on ECMWF reanalysis. It treats the globe as a mesh of connected points and predicts the next 6 hours from the last two states, repeating that step to reach 48 h and beyond, on a 0.25° grid." },
      { title: "Limits at one station", body: "Competitive with physics models at large scales, but like AIFS it is coarse for a single station and has no knowledge of local siting." },
    ],
  },
  {
    id: "persistence",
    label: "Persistence",
    family: "baseline",
    source: "fixture",
    benchmarkMethodId: "persistence",
    description: "Assumes nothing changes: the value observed when the forecast is issued is used for every hour ahead. Any useful model should beat it.",
    about: [
      { title: "What it is", body: "The simplest possible forecast: take what the station observed when the forecast was issued, and predict that same value for every hour ahead." },
      { title: "Why it's a baseline", body: "Weather has memory, so persistence is hard to beat an hour out. It quickly gets worse as the day warms and cools or a front passes. It sets the floor: a model that can't beat persistence adds nothing." },
    ],
  },
];

export const DEFAULT_MODEL_ID = "hrrr_cnn";
export const getModel = (id: string) => MODELS.find((m) => m.id === id);
