import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const appDir = join(dirname(fileURLToPath(import.meta.url)), "..");
const fixturesDir = join(appDir, "src/data/fixtures");
const aggregatesDir = join(appDir, "../benchmarking-site/data/aggregates");
const summaryPath = join(appDir, "src/data/benchmark-summary.json");
const expectedModelIds = new Set([
  "hrrr_cnn",
  "ecmwf_aifs",
  "graphcast",
  "hrrr_interpolated",
  "persistence",
]);
const fixtureFiles = readdirSync(fixturesDir).filter((file) => file.endsWith(".json"));
const modelsWithFixtures = new Set();

assert.ok(fixtureFiles.length > 0, "forecast fixtures must exist");
for (const file of fixtureFiles) {
  const record = JSON.parse(readFileSync(join(fixturesDir, file), "utf8"));
  assert.equal(record.schema_version, 1, `${file}: schema version`);
  assert.equal(record.station_id, "eric_d_soulis", `${file}: station`);
  assert.ok(expectedModelIds.has(record.model_id), `${file}: registered model`);
  assert.ok(record.model.tags.includes("synthetic_demo"), `${file}: synthetic-data tag`);
  assert.ok(Array.isArray(record.points) && record.points.length > 0, `${file}: forecast points`);
  modelsWithFixtures.add(record.model_id);

  const leads = new Set();
  for (const point of record.points) {
    assert.ok(Number.isFinite(point.value), `${file}: finite value`);
    assert.ok(Number.isInteger(point.lead_hours) && point.lead_hours > 0, `${file}: positive integer lead`);
    assert.ok(!leads.has(point.lead_hours), `${file}: duplicate lead ${point.lead_hours}`);
    leads.add(point.lead_hours);
    assert.equal(
      Date.parse(point.valid_time),
      Date.parse(record.init_time) + point.lead_hours * 3600e3,
      `${file}: valid time must match init and lead`,
    );
  }
}
assert.deepEqual(modelsWithFixtures, expectedModelIds, "each selectable model must have demo fixtures");

const summary = JSON.parse(readFileSync(summaryPath, "utf8"));
assert.equal(summary.year, 2025);
assert.equal(summary.station_id, "eric_d_soulis");
assert.equal(summary.lead_time_hours, 24);
assert.equal(summary.metric, "RMSE");
assert.equal(summary.aggregation, "sqrt(mean(per-initialization signed forecast-observation error squared))");
assert.equal(summary.source, "benchmarking-site/data/aggregates");
assert.deepEqual(
  new Set(Object.keys(summary.methods)),
  new Set(["ecmwf_aifs", "graphcast", "hrrr_interpolated", "persistence"]),
  "only exact historical benchmark method mappings may have scores",
);
assert.ok(!("matched_comparison" in summary), "pooled CNN comparison must not be included");

for (const [methodId, method] of Object.entries(summary.methods)) {
  const aggregate = JSON.parse(readFileSync(join(aggregatesDir, `${methodId}.json`), "utf8"));
  assert.equal(aggregate.method_id, methodId, `${methodId}: exact method mapping`);
  assert.equal(aggregate.metric, "bias", `${methodId}: source stores signed pair errors`);
  const initIndexes = aggregate.initializations
    .map((initialization, index) => initialization.startsWith(`${summary.year}-`) ? index : -1)
    .filter((index) => index >= 0);
  const leadIndex = aggregate.lead_times_hours.indexOf(summary.lead_time_hours);
  assert.ok(leadIndex >= 0, `${methodId}: exact lead exists`);
  assert.equal(method.n_inits, initIndexes.length, `${methodId}: initialization count`);
  for (const variable of ["t2m", "wind_speed"]) {
    const score = method[variable];
    assert.ok(Number.isFinite(score?.rmse) && score.rmse >= 0, `${methodId}/${variable}: RMSE`);
    assert.ok(Number.isInteger(score.n_samples) && score.n_samples > 0, `${methodId}/${variable}: valid pairs`);

    const errors = initIndexes
      .map((index) => aggregate.bias?.[summary.station_id]?.[variable]?.[index]?.[leadIndex])
      .filter(Number.isFinite);
    const expectedRmse = Number(
      Math.sqrt(errors.reduce((sum, error) => sum + error ** 2, 0) / errors.length).toFixed(2),
    );
    assert.equal(score.rmse, expectedRmse, `${methodId}/${variable}: pooled f024 RMSE`);
    assert.equal(score.n_samples, errors.length, `${methodId}/${variable}: f024 valid-pair count`);
  }
}

console.log(`Validated ${fixtureFiles.length} synthetic forecast fixtures and exact 2025 f024 benchmark summaries.`);
