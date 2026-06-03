import {
  DATA_DIR,
  LEAD_TIMES_HOURS,
  METRICS,
} from "../constants.js";

const runsCache = new Map();

/**
 * Average metric arrays across multiple run objects for one location + variable.
 * @param {object[]} runs
 * @param {string} locationId
 * @param {string} variableKey
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

  return out;
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
