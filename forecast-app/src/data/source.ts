import type { ForecastRecord } from "./contract";
import { validateRecord } from "./contract";
import { FIXTURE_NOW, FIXTURE_RECORDS } from "./fixtures";

/**
 * The only seam between the UI and where forecasts come from. To go live, implement
 * this against the read API (GET /forecast) and change `forecastSource` below.
 */
export interface ForecastSource {
  load(modelId: string, stationId: string): Promise<ForecastRecord[]>;
  /** The clock the report is written against. */
  now(): Date;
}

export const fixtureSource: ForecastSource = {
  async load(modelId, stationId) {
    const records = FIXTURE_RECORDS.filter((r) => r.model_id === modelId && r.station_id === stationId);
    for (const r of records) {
      const issues = validateRecord(r);
      if (issues.length) throw new Error(`fixture ${r.variable}: ${issues.join("; ")}`);
    }
    return records;
  },
  now: () => new Date(FIXTURE_NOW),
};

// Live example (not wired up):
// export const apiSource: ForecastSource = {
//   load: async (modelId, stationId) =>
//     (await fetch(`${API_URL}/forecast?station=${stationId}&model=${modelId}`)).json(),
//   now: () => new Date(),
// };

export const forecastSource: ForecastSource = fixtureSource;
