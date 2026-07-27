import {
  DATA_DIR,
  INIT_CYCLES,
  LEAD_TIMES_HOURS,
  METRICS,
} from "../constants.js";
import { combineFromPartials, resolveMonths } from "./aggregateCombine.js";

const runsCache = new Map();
const methodDataCache = new Map();
const aggregateDocCache = new Map();

const ALL_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

/**
 * @typedef {object} AggregateResult
 * @property {number[]} lead_times_hours
 * @property {(number|null)[]} rmse
 * @property {(number|null)[]} mae
 * @property {(number|null)[]} bias
 * @property {(number|null)[]} acc
 */

/**
 * @typedef {object} MethodData
 * @property {"npz"|"json"} source
 * @property {number} nInits
 * @property {AggregateResult|null} t2m
 * @property {AggregateResult|null} wind_speed
 */

/**
 * Average metric arrays across multiple run objects for one location + variable.
 * @param {object[]} runs
 * @param {string} locationId
 * @param {string} variableKey
 * @returns {AggregateResult|null}
 */
export function aggregateRuns(runs, locationId, variableKey) {
  const slices = runs
    .map((run) => run.locations?.[locationId]?.variables?.[variableKey])
    .filter(Boolean);

  if (slices.length === 0) return null;

  const leadTimes = slices[0].lead_times_hours ?? LEAD_TIMES_HOURS;
  const out = { lead_times_hours: leadTimes };

  for (const metric of METRICS) {
    out[metric] = leadTimes.map((_, i) => {
      const values = slices
        .map((s) => s[metric]?.[i])
        .filter((v) => typeof v === "number");
      if (values.length === 0) return null;
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      // Each per-init record holds one forecast/obs pair, so its stored "rmse"
      // is just |error|. Averaging those gives MAE; a pooled RMSE has to square
      // first and take the root last.
      if (metric === "rmse") {
        const ms = values.reduce((a, b) => a + b * b, 0) / values.length;
        return Math.sqrt(ms);
      }
      // ACC over single-sample records degenerates to +/-1; averaging is meaningless.
      if (metric === "acc") return null;
      return mean;
    });
  }

  out.n_samples = leadTimes.map((_, i) =>
    slices.filter((s) => typeof s.rmse?.[i] === "number").length
  );

  return out;
}

/**
 * @param {object|undefined} entry — aggregates[station][variable] from aggregate.json
 * @param {object} doc — parsed aggregate.json
 * @returns {AggregateResult|null}
 */
function aggregateEntryToResult(entry, doc) {
  if (!entry) return null;
  const out = { lead_times_hours: doc.lead_times_hours ?? LEAD_TIMES_HOURS };
  for (const metric of METRICS) {
    out[metric] = entry[metric] ?? [];
  }
  out.n_samples = entry.n_samples ?? null;
  return out;
}

/**
 * Filter runs to those whose initialization UTC hour is in the cycles list.
 * @param {object[]} runs
 * @param {number[]} cycles — e.g. [0, 6, 12, 18]; empty means keep all
 * @returns {object[]}
 */
export function filterRunsByCycle(runs, cycles) {
  if (!cycles || cycles.length === 0) return runs;
  return runs.filter((run) => {
    const init = run.initialization;
    if (!init) return cycles.includes(0);
    const hour = new Date(init).getUTCHours();
    return cycles.includes(hour);
  });
}

/**
 * Filter runs to those whose initialization falls within [initFrom, initTo] (inclusive).
 * @param {object[]} runs
 * @param {string|null} initFrom — ISO8601 UTC string
 * @param {string|null} initTo   — ISO8601 UTC string
 * @returns {object[]}
 */
export function filterRunsByDateRange(runs, initFrom, initTo) {
  if (!initFrom && !initTo) return runs;
  const from = initFrom ? new Date(initFrom).getTime() : -Infinity;
  const to = initTo ? new Date(initTo).getTime() : Infinity;
  return runs.filter((run) => {
    if (!run.initialization) return false;
    const t = new Date(run.initialization).getTime();
    return t >= from && t <= to;
  });
}

const aggregateSourceCache = new Map();

/**
 * One compact per-method file of per-init bias values, built by
 * scripts/build_site_aggregates.py. This is what makes the site work on static
 * hosting: one request per method instead of a backend call or thousands of
 * per-init fetches.
 * @param {string} methodId
 */
async function loadAggregateSource(methodId) {
  if (aggregateSourceCache.has(methodId)) {
    return aggregateSourceCache.get(methodId);
  }
  let doc = null;
  try {
    const res = await fetch(`${DATA_DIR}/aggregates/${methodId}.json`);
    if (res.ok) doc = await res.json();
  } catch {
    /* fall through to the other sources */
  }
  aggregateSourceCache.set(methodId, doc);
  return doc;
}

/**
 * Pool per-init bias into skill scores for one station/variable.
 * @param {object} source
 * @returns {AggregateResult|null}
 */
export function aggregateFromBias(source, locationId, variableKey, filters = {}) {
  const rows = source?.bias?.[locationId]?.[variableKey];
  if (!Array.isArray(rows)) return null;

  const leadTimes = source.lead_times_hours ?? LEAD_TIMES_HOURS;
  const inits = source.initializations ?? [];

  const cycles = filters.cycles
    ? filters.cycles.split(",").map((c) => parseInt(c, 10)).filter((n) => !Number.isNaN(n))
    : [];
  const from = filters.init_from ? new Date(filters.init_from).getTime() : -Infinity;
  const to = filters.init_to ? new Date(filters.init_to).getTime() : Infinity;

  const keep = [];
  for (let i = 0; i < rows.length; i += 1) {
    const iso = inits[i];
    if (!iso) continue;
    const t = new Date(iso).getTime();
    if (t < from || t > to) continue;
    if (cycles.length && !cycles.includes(new Date(iso).getUTCHours())) continue;
    keep.push(rows[i]);
  }
  if (keep.length === 0) return null;

  const out = { lead_times_hours: leadTimes };
  const nSamples = [];
  const rmse = [];
  const mae = [];
  const bias = [];

  leadTimes.forEach((_, i) => {
    let n = 0;
    let sum = 0;
    let sumAbs = 0;
    let sumSq = 0;
    for (const row of keep) {
      const b = row?.[i];
      if (typeof b !== "number") continue;
      n += 1;
      sum += b;
      sumAbs += Math.abs(b);
      sumSq += b * b;
    }
    nSamples.push(n);
    if (n === 0) {
      rmse.push(null);
      mae.push(null);
      bias.push(null);
      return;
    }
    rmse.push(Math.sqrt(sumSq / n));
    mae.push(sumAbs / n);
    bias.push(sum / n);
  });

  out.rmse = rmse;
  out.mae = mae;
  out.bias = bias;
  out.acc = leadTimes.map(() => null);
  out.n_samples = nSamples;
  out.n_inits = keep.length;
  return out;
}

/**
 * Fetch (and cache) the static aggregate.json for a method.
 * Built by scripts/build_static_aggregates.py; null when the method has none.
 * @param {string} methodId
 * @returns {Promise<object|null>}
 */
async function loadAggregateDoc(methodId) {
  if (aggregateDocCache.has(methodId)) {
    return aggregateDocCache.get(methodId);
  }
  let doc = null;
  try {
    const res = await fetch(`${DATA_DIR}/${methodId}/aggregate.json`);
    if (res.ok) doc = await res.json();
  } catch {
    /* no static aggregate — fall back to per-init JSON */
  }
  aggregateDocCache.set(methodId, doc);
  return doc;
}

/**
 * Serve the request from the static aggregate document when it can reproduce
 * the NPZ API's masked aggregation exactly: no filters -> precomputed overall
 * mean; cycle subsets and month-aligned date ranges -> recombined partials.
 * Returns null when the filters need per-init JSON (non-month-aligned range).
 * @param {object} doc
 * @param {string} locationId
 * @param {Record<string, string>} filters
 * @returns {MethodData|null}
 */
function tryFromStaticAggregate(doc, locationId, filters) {
  const hasCycles = Boolean(filters.cycles);
  const hasRange = Boolean(filters.init_from || filters.init_to);

  if (!hasCycles && !hasRange) {
    const perVariable = doc.aggregates?.[locationId];
    if (!perVariable) return null;
    return {
      source: "aggregate",
      nInits: doc.n_inits ?? 0,
      t2m: aggregateEntryToResult(perVariable.t2m, doc),
      wind_speed: aggregateEntryToResult(perVariable.wind_speed, doc),
    };
  }

  const cycleHours = hasCycles
    ? filters.cycles.split(",").map((c) => parseInt(c, 10))
    : INIT_CYCLES;
  const months = hasRange
    ? resolveMonths(filters.init_from, filters.init_to, doc.year)
    : ALL_MONTHS;
  if (!months) return null;

  const t2m = combineFromPartials(doc, locationId, "t2m", months, cycleHours);
  const wind = combineFromPartials(doc, locationId, "wind_speed", months, cycleHours);
  if (!t2m || !wind) return null;

  return {
    source: "aggregate",
    nInits: t2m.nInits,
    t2m: t2m.result,
    wind_speed: wind.result,
  };
}

export async function loadMethodRuns(methodId) {
  if (runsCache.has(methodId)) {
    return runsCache.get(methodId);
  }

  const indexUrl = `${DATA_DIR}/${methodId}/index.json`;
  let files = [`${methodId}_sample.json`];

  try {
    const res = await fetch(indexUrl);
    if (res.ok) {
      const index = await res.json();
      if (Array.isArray(index.files) && index.files.length > 0) {
        files = index.files;
      }
    }
  } catch {
    /* use default sample filename */
  }

  const runs = [];
  for (const file of files) {
    const path = `${DATA_DIR}/${methodId}/${file}`;
    try {
      const res = await fetch(path);
      if (!res.ok) continue;
      const data = await res.json();
      if (data.method && data.locations) runs.push(data);
    } catch {
      /* skip missing placeholder files */
    }
  }

  runsCache.set(methodId, runs);
  return runs;
}

/**
 * Load aggregated skill scores for a method at a location.
 * Prefers the static aggregate.json (precomputed from consolidated NPZ);
 * falls back to per-init JSON when there is none or the filters need it.
 *
 * @param {string} methodId
 * @param {string} locationId
 * @param {Record<string, string>} [filters] — init_from, init_to, cycles
 * @returns {Promise<MethodData|null>}
 */
export async function loadMethodData(methodId, locationId, filters = {}) {
  const cacheKey = `${methodId}:${locationId}:${JSON.stringify(filters)}`;
  if (methodDataCache.has(cacheKey)) {
    return methodDataCache.get(cacheKey);
  }

  // Static aggregates first: they work on any host, need one request, and pool
  // RMSE correctly. The API and the per-init walk stay as fallbacks.
  const source = await loadAggregateSource(methodId);
  if (source) {
    const t2m = aggregateFromBias(source, locationId, "t2m", filters);
    const wind = aggregateFromBias(source, locationId, "wind_speed", filters);
    if (t2m || wind) {
      const result = {
        source: "aggregates",
        nInits: t2m?.n_inits ?? wind?.n_inits ?? 0,
        t2m,
        wind_speed: wind,
      };
      methodDataCache.set(cacheKey, result);
      return result;
    }
  }

  try {
    const doc = await loadAggregateDoc(methodId);
    if (doc) {
      const fromStatic = tryFromStaticAggregate(doc, locationId, filters);
      if (fromStatic) {
        methodDataCache.set(cacheKey, fromStatic);
        return fromStatic;
      }
    }
  } catch (err) {
    console.warn(`Static aggregate unusable for ${methodId}, falling back to JSON:`, err);
  }

  const allRuns = await loadMethodRuns(methodId);
  if (allRuns.length === 0) return null;

  const cycleNums = filters.cycles
    ? filters.cycles.split(",").map((c) => parseInt(c, 10))
    : [];
  const afterCycle = filterRunsByCycle(allRuns, cycleNums);
  const runs = filterRunsByDateRange(afterCycle, filters.init_from ?? null, filters.init_to ?? null);

  if (runs.length === 0) return null;

  const result = {
    source: "json",
    nInits: runs.length,
    t2m: aggregateRuns(runs, locationId, "t2m"),
    wind_speed: aggregateRuns(runs, locationId, "wind_speed"),
  };
  methodDataCache.set(cacheKey, result);
  return result;
}
