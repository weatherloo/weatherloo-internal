import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const appDir = join(dirname(fileURLToPath(import.meta.url)), "..");
const aggregatesDir = join(appDir, "../benchmarking-site/data/aggregates");
const outputPath = join(appDir, "src/data/benchmark-summary.json");
const methods = ["ecmwf_aifs", "graphcast", "hrrr_interpolated", "persistence"];
const variables = ["t2m", "wind_speed"];
const targetLead = 24;
const stationId = "eric_d_soulis";
const year = 2025;

const summary = {
  schema_version: 1,
  year,
  station_id: stationId,
  lead_time_hours: targetLead,
  metric: "RMSE",
  aggregation: "sqrt(mean(per-initialization signed forecast-observation error squared))",
  source: "benchmarking-site/data/aggregates",
  methods: {},
};

for (const methodId of methods) {
  const path = join(aggregatesDir, `${methodId}.json`);
  const aggregate = JSON.parse(readFileSync(path, "utf8"));
  if (!Array.isArray(aggregate.initializations) || !Array.isArray(aggregate.lead_times_hours)) {
    throw new Error(`Invalid benchmark aggregate contract for ${methodId}`);
  }
  if (aggregate.method_id !== methodId) {
    throw new Error(`Expected benchmark method ${methodId}, got ${aggregate.method_id}`);
  }
  if (aggregate.metric !== "bias") {
    throw new Error(`Expected per-initialization signed errors for ${methodId}, got ${aggregate.metric}`);
  }
  const initIndexes = aggregate.initializations
    .map((initialization, index) => initialization.startsWith(`${year}-`) ? index : -1)
    .filter((index) => index >= 0);
  const leadIndex = aggregate.lead_times_hours.indexOf(targetLead);
  if (leadIndex < 0 || initIndexes.length === 0) {
    throw new Error(`No ${year} ${targetLead} h benchmark data for ${methodId}`);
  }

  const method = { n_inits: initIndexes.length };
  for (const variable of variables) {
    const rows = aggregate.bias?.[stationId]?.[variable];
    if (!Array.isArray(rows)) {
      throw new Error(`No ${stationId} ${variable} bias data for ${methodId}`);
    }
    if (rows.length !== aggregate.initializations.length) {
      throw new Error(`Mismatched initialization count for ${methodId}/${stationId}/${variable}`);
    }
    const errors = initIndexes.map((index) => rows[index]?.[leadIndex]).filter(Number.isFinite);
    if (errors.length === 0) {
      throw new Error(`No valid ${year} ${targetLead} h pairs for ${methodId}/${stationId}/${variable}`);
    }
    // Each entry is one matched pair; pool signed errors as RMSE, not mean bias.
    method[variable] = {
      rmse: Number(Math.sqrt(errors.reduce((sum, error) => sum + error ** 2, 0) / errors.length).toFixed(2)),
      n_samples: errors.length,
    };
  }
  summary.methods[methodId] = method;
}

writeFileSync(outputPath, `${JSON.stringify(summary, null, 2)}\n`);
console.log(`Wrote ${Object.keys(summary.methods).length} aggregate summaries to ${outputPath}`);
