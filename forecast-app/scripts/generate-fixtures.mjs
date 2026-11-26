// Generates PLACEHOLDER forecast records in the forecast-record contract v1 shape
// (weatherloo-forecast/contract/forecast_record.v1.schema.json), mimicking the HRRR
// station-correction CNN (models/hrrr_cnn): hourly leads 1-48 h at eric_d_soulis for
// t2m (degC), wind_speed (m/s), pop1 (hourly chance of >=0.2 mm), plus 6-hour blocks
// pop6 / amt6 (mm) at leads 6,12,...,48. Deterministic; rerun: npm run fixtures.
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const out = join(dirname(fileURLToPath(import.meta.url)), "../src/data/fixtures");
mkdirSync(out, { recursive: true });

const INIT = Date.UTC(2026, 10, 27, 6); // 2026-11-27T06:00Z = 01:00 EST (HRRR 06 UTC run)
const ISSUED = Date.UTC(2026, 10, 27, 10, 0); // CNN site export available ~4 h later
const iso = (ms) => new Date(ms).toISOString().replace(/\.\d{3}Z$/, "Z");
const r1 = (x) => Math.round(x * 10) / 10;
const r3 = (x) => Math.round(x * 1000) / 1000;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
let s = 7;
const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647) - 0.5; // deterministic noise

const base = {
  schema_version: 1,
  model_id: "hrrr_cnn",
  model: { family: "hrrr", version: "cnn_v1", tags: ["station_corrected", "cnn", "placeholder"] },
  station_id: "eric_d_soulis",
  init_time: iso(INIT),
  issued_at: iso(ISSUED),
};
const point = (lead, value, extra = {}) => ({
  lead_hours: lead, valid_time: iso(INIT + lead * 3600e3), value, ...extra });

const hours = Array.from({ length: 48 }, (_, i) => i + 1);
const localHour = (lead) => (1 + lead) % 24; // EST = UTC-5; init is 01:00 local
// Cold clear Friday, cloud and light snow arriving Saturday evening.
const t2m = hours.map((h) => {
  const diurnal = 3.2 * Math.cos(((localHour(h) - 15) / 24) * 2 * Math.PI);
  const trend = 0.4 + 0.025 * h + (h > 30 ? -0.04 * (h - 30) : 0);
  const v = trend + diurnal + rnd() * 0.5;
  const w = 0.9 + 0.035 * h;
  return point(h, r1(v), { lower: r1(v - w), upper: r1(v + w) });
});
const wind = hours.map((h) => {
  const v = clamp(2.2 + 1.3 * Math.sin((h - 8) / 7) + (h > 30 ? 0.12 * (h - 30) : 0) + rnd() * 0.5, 0.4, 12);
  const w = 0.5 + 0.03 * h;
  return point(h, r1(v), { lower: r1(Math.max(0, v - w)), upper: r1(v + w) });
});
const pop1 = hours.map((h) => {
  const v = clamp(0.03 + (h > 28 ? 0.62 * (1 - Math.exp(-(h - 28) / 6)) : 0) + rnd() * 0.04, 0, 0.97);
  return point(h, r3(v));
});
const blocks = [6, 12, 18, 24, 30, 36, 42, 48];
const blockMean = (arr, end) => arr.slice(end - 6, end).reduce((a, p) => a + p.value, 0) / 6;
const pop6 = blocks.map((e) => point(e, r3(clamp(1 - Math.pow(1 - blockMean(pop1, e), 2.2), 0, 0.99))));
const amt6 = blocks.map((e) => {
  const m = blockMean(pop1, e);
  const v = Math.max(0, m - 0.12) * 3.4; // mm / 6 h, water equivalent
  return point(e, r1(v), { lower: 0, upper: r1(v * 2.1 + 0.2) });
});

const files = {
  t2m: [t2m, "degC"], wind_speed: [wind, "m/s"], pop1: [pop1, "fraction"],
  pop6: [pop6, "fraction"], amt6: [amt6, "mm"],
};
for (const [variable, [points, unit]] of Object.entries(files)) {
  const rec = { ...base, variable, unit, points };
  writeFileSync(join(out, `eric_d_soulis.${variable}.hrrr_cnn.json`), JSON.stringify(rec, null, 2) + "\n");
}
console.log("wrote", Object.keys(files).length, "fixtures to", out);
