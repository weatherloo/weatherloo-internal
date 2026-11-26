/**
 * Forecast record contract v1, mirrored from weatherloo-forecast
 * (contract/forecast_record.v1.schema.json, frontend/src/contract/types.ts).
 *
 * Extension: the v1 schema's `unit` enum only lists "degC" | "m/s". The CNN also
 * forecasts precipitation (chance as a "fraction", amounts in "mm"), so those two
 * units are included here. The schema needs a contract bump before the live API
 * can carry them; see README "Swapping in live data".
 */
export type ForecastUnit = "degC" | "m/s" | "fraction" | "mm";

export interface ForecastPoint {
  lead_hours: number;
  valid_time: string; // UTC, YYYY-MM-DDTHH:MM:SSZ
  value: number;
  lower?: number;
  upper?: number;
}

export interface ModelMeta {
  family?: string;
  version?: string;
  tags?: string[];
  [key: string]: unknown;
}

export interface ForecastRecord {
  schema_version: 1;
  model_id: string;
  model?: ModelMeta;
  station_id: string;
  variable: string;
  unit: ForecastUnit;
  init_time: string;
  issued_at: string;
  points: ForecastPoint[];
}

const TS = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;

/** Semantic checks from the contract README. Returns human-readable issues. */
export function validateRecord(r: ForecastRecord): string[] {
  const issues: string[] = [];
  if (r.schema_version !== 1) issues.push("schema_version must be 1");
  if (!TS.test(r.init_time) || !TS.test(r.issued_at)) issues.push("timestamps must be UTC with trailing Z");
  if (Date.parse(r.issued_at) < Date.parse(r.init_time)) issues.push("issued_at precedes init_time");
  if (r.points.length === 0) issues.push("points is empty");
  const seen = new Set<number>();
  r.points.forEach((p, i) => {
    if (seen.has(p.lead_hours)) issues.push(`points[${i}]: duplicate lead_hours`);
    seen.add(p.lead_hours);
    if (Date.parse(p.valid_time) !== Date.parse(r.init_time) + p.lead_hours * 3600e3)
      issues.push(`points[${i}]: valid_time must equal init_time + lead_hours`);
    if ((p.lower !== undefined && p.lower > p.value) || (p.upper !== undefined && p.upper < p.value))
      issues.push(`points[${i}]: lower <= value <= upper violated`);
  });
  return issues;
}
