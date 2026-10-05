import { METRICS } from "../constants.js";

/**
 * Pure helpers for reading static aggregate.json documents
 * (built by scripts/build_static_aggregates.py from consolidated NPZ).
 *
 * "partials" hold per-(month, cycle) nansum + non-NaN count, so any
 * month-aligned date range x cycle subset recombines exactly as
 * sum(sums)/sum(counts) — identical to the NPZ API's masked nanmean.
 */

const ALL_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

function lastDayOfMonth(year, month) {
  return new Date(Date.UTC(year, month, 0)).getUTCDate();
}

/**
 * Resolve an init_from/init_to range to a list of whole months, or null when
 * the range is not month-aligned (then the caller must use per-init JSON).
 * @param {string|null|undefined} initFrom — ISO8601 UTC
 * @param {string|null|undefined} initTo — ISO8601 UTC
 * @param {number} year — the year the aggregate document covers
 * @returns {number[]|null}
 */
export function resolveMonths(initFrom, initTo, year) {
  let startMonth = 1;
  let endMonth = 12;

  if (initFrom) {
    const from = new Date(initFrom);
    if (
      Number.isNaN(from.getTime()) ||
      from.getUTCFullYear() !== year ||
      from.getUTCDate() !== 1 ||
      from.getUTCHours() !== 0 ||
      from.getUTCMinutes() !== 0 ||
      from.getUTCSeconds() !== 0
    ) {
      return null;
    }
    startMonth = from.getUTCMonth() + 1;
  }

  if (initTo) {
    const to = new Date(initTo);
    if (
      Number.isNaN(to.getTime()) ||
      to.getUTCFullYear() !== year ||
      to.getUTCDate() !== lastDayOfMonth(year, to.getUTCMonth() + 1) ||
      to.getUTCHours() < 18
    ) {
      return null;
    }
    endMonth = to.getUTCMonth() + 1;
  }

  if (startMonth > endMonth) return [];
  return ALL_MONTHS.filter((m) => m >= startMonth && m <= endMonth);
}

/**
 * Recombine partial sums/counts for a set of months and init cycles.
 * @param {object} doc — parsed aggregate.json
 * @param {string} locationId
 * @param {string} variableKey
 * @param {number[]} months — e.g. [1, 2, 3]
 * @param {number[]} cycleHours — e.g. [0, 6, 12, 18]
 * @returns {{result: object, nInits: number}|null} null when partials are absent
 */
export function combineFromPartials(doc, locationId, variableKey, months, cycleHours) {
  if (!doc.partials) return null;
  const leadTimes = doc.lead_times_hours;
  const n = leadTimes.length;
  const sums = {};
  const counts = {};
  for (const metric of METRICS) {
    sums[metric] = new Array(n).fill(0);
    counts[metric] = new Array(n).fill(0);
  }

  let nInits = 0;
  for (const month of months) {
    const byCycle = doc.partials[String(month)];
    if (!byCycle) continue;
    for (const hour of cycleHours) {
      const cell = byCycle[String(hour)];
      if (!cell) continue;
      nInits += cell.n_inits ?? 0;
      const slice = cell.cells?.[locationId]?.[variableKey];
      if (!slice) continue;
      for (const metric of METRICS) {
        const entry = slice[metric];
        if (!entry) continue;
        for (let i = 0; i < n; i++) {
          sums[metric][i] += entry.sum[i] ?? 0;
          counts[metric][i] += entry.count[i] ?? 0;
        }
      }
    }
  }

  const result = { lead_times_hours: leadTimes };
  for (const metric of METRICS) {
    result[metric] = sums[metric].map((s, i) =>
      counts[metric][i] > 0 ? s / counts[metric][i] : null,
    );
  }
  result.n_samples = leadTimes.map((_, i) =>
    Math.max(...METRICS.map((metric) => counts[metric][i])),
  );

  return { result, nInits };
}
