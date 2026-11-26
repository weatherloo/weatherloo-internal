/**
 * Model registry. Add a model here (and its fixtures/source) to make it available
 * to the app; nothing else needs to change. A switcher / compare view reads MODELS
 * and calls setActiveModelId from useForecast().
 */
export interface ModelInfo {
  id: string; // matches ForecastRecord.model_id
  label: string;
  family: string;
  /** Where the records come from: "fixture" today, "api" once the read API is live. */
  source: "fixture" | "api";
  description?: string;
}

export const MODELS: ModelInfo[] = [
  {
    id: "hrrr_cnn",
    label: "HRRR CNN",
    family: "hrrr",
    source: "fixture",
    description: "HRRR station-correction CNN (models/hrrr_cnn): temperature, wind, precipitation, 0-48 h.",
  },
];

export const DEFAULT_MODEL_ID = "hrrr_cnn";
export const getModel = (id: string) => MODELS.find((m) => m.id === id);
