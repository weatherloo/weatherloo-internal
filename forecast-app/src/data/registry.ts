/**
 * Add a model here and make its forecast records available from the source.
 */
export interface ModelInfo {
  id: string; // matches ForecastRecord.model_id
  label: string;
  family: string;
  /** Where the records come from: "fixture" today, "api" once the read API is live. */
  source: "fixture" | "api";
  description?: string;
  /** Benchmark site method id, when a directly comparable historical score exists. */
  benchmarkMethodId?: string;
}

export const MODELS: ModelInfo[] = [
  {
    id: "hrrr_interpolated",
    label: "HRRR",
    family: "hrrr",
    source: "fixture",
    benchmarkMethodId: "hrrr_interpolated",
    description: "NOAA HRRR forecast interpolated to the station.",
  },
  {
    id: "hrrr_cnn",
    label: "HRRR CNN",
    family: "HRRR station correction",
    source: "fixture",
    description: "HRRR station-correction CNN (models/hrrr_cnn): temperature, wind, precipitation, 0-48 h.",
  },
  {
    id: "ecmwf_aifs",
    label: "ECMWF AIFS",
    family: "global NWP",
    source: "fixture",
    benchmarkMethodId: "ecmwf_aifs",
    description: "ECMWF AIFS Single global forecast.",
  },
  {
    id: "graphcast",
    label: "GraphCast",
    family: "AI global",
    source: "fixture",
    benchmarkMethodId: "graphcast",
    description: "GraphCastGFS global forecast.",
  },
  {
    id: "persistence",
    label: "Persistence",
    family: "baseline",
    source: "fixture",
    benchmarkMethodId: "persistence",
    description: "Simple persistence baseline.",
  },
];

export const DEFAULT_MODEL_ID = "hrrr_cnn";
export const getModel = (id: string) => MODELS.find((m) => m.id === id);
