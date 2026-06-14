import {
  API_DIR,
  DATA_DIR,
  LEAD_TIMES_HOURS,
  METRICS,
  VARIABLES,
} from "../constants.js";

const runsCache = new Map();
const methodDataCache = new Map();

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
      return values.reduce((a, b) => a + b, 0) / values.length;
    });
  }

  out.n_samples = leadTimes.map((_, i) =>
    slices.filter((s) => typeof s.rmse?.[i] === "number").length
  );

  return out;
}

/**
 * @param {object} payload
 * @returns {AggregateResult}
 */
function aggregatePayloadToResult(payload) {
  const out = { lead_times_hours: payload.lead_times_hours ?? LEAD_TIMES_HOURS };
  for (const metric of METRICS) {
    out[metric] = payload[metric] ?? [];
  }
  out.n_samples = payload.n_samples ?? null;
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

/**
 * @param {string} methodId
 * @param {Record<string, string>} [filters]
 */
async function tryLoadFromNpzApi(methodId, locationId, filters = {}) {
  const statusRes = await fetch(`${API_DIR}/${methodId}`);
  if (!statusRes.ok) return null;

  const status = await statusRes.json();
  if (!status.npz_available) return null;

  const baseParams = new URLSearchParams({ location: locationId, ...filters });

  const fetchVariable = async (variable) => {
    const params = new URLSearchParams(baseParams);
    params.set("variable", variable);
    const res = await fetch(`${API_DIR}/${methodId}/aggregate?${params}`);
    if (!res.ok) {
      throw new Error(`NPZ aggregate failed for ${variable}: ${res.status}`);
    }
    return res.json();
  };

  const [t2mPayload, windPayload] = await Promise.all(
    VARIABLES.map((variable) => fetchVariable(variable)),
  );

  return {
    source: "npz",
    nInits: t2mPayload.n_inits ?? status.n_inits ?? 0,
    t2m: aggregatePayloadToResult(t2mPayload),
    wind_speed: aggregatePayloadToResult(windPayload),
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
 * Prefers consolidated NPZ via /api when available; falls back to per-init JSON.
 *
 * @param {string} methodId
 * @param {string} locationId
 * @param {Record<string, string>} [filters] — forwarded to NPZ API (init_from, init_to, cycles)
 * @returns {Promise<MethodData|null>}
 */
export async function loadMethodData(methodId, locationId, filters = {}) {
  const cacheKey = `${methodId}:${locationId}:${JSON.stringify(filters)}`;
  if (methodDataCache.has(cacheKey)) {
    return methodDataCache.get(cacheKey);
  }

  try {
    const fromApi = await tryLoadFromNpzApi(methodId, locationId, filters);
    if (fromApi) {
      methodDataCache.set(cacheKey, fromApi);
      return fromApi;
    }
  } catch (err) {
    console.warn(`NPZ API unavailable for ${methodId}, falling back to JSON:`, err);
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
