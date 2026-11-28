// Ported from the benchmarking site's demo (demoChecks.js).
// Tap-time checks for the training demo: is this spot one the hrrr_cnn pipeline
// could train on? Mirrors models/hrrr_cnn/points.py so the browser and the
// pipeline agree, without reading any HRRR data.

// HRRR native grid: Lambert conformal, tangent at 38.5N, centred on 97.5W,
// 1799 x 1059 points at 3 km (NCEP grid definition).
const R = 6371229;
const PHI1 = (38.5 * Math.PI) / 180;
const LON0 = -97.5;
const NX = 1799;
const NY = 1059;
const DX = 3000;
const LL = { lat: 21.138123, lon: -122.719528 }; // centre of grid point (0, 0)

// dynamical.org stores the grid in 265 (y) x 300 (x) chunks ("tiles"); the
// 17 x 17 patch around the target must sit inside one (points.ContextReader).
const CHUNK_Y = 265;
const CHUNK_X = 300;
const PATCH_HALF = 8;

export interface LatLon { lat: number; lon: number }
export interface Station extends LatLon { id: string; name: string; begin: string; network?: string }
export interface CheckStep { id: string; label: string; status: "pass" | "fail" | "skip"; detail: string }
export interface CheckResult { steps: CheckStep[]; station: Station | null; km: number; ok: boolean }

export const STATION_RADIUS_KM = 125;
// hrrr_cnn trains from 2020-12-03; station_labels.inventory drops ASOS stations that start later.
export const HISTORY_FROM = "2020-12-01";

const N = Math.sin(PHI1);
const F = (Math.cos(PHI1) * Math.tan(Math.PI / 4 + PHI1 / 2) ** N) / N;
const RHO0 = (R * F) / Math.tan(Math.PI / 4 + PHI1 / 2) ** N;

function toLcc(lat: number, lon: number): [number, number] {
  const phi = (lat * Math.PI) / 180;
  const rho = (R * F) / Math.tan(Math.PI / 4 + phi / 2) ** N;
  let dLon = lon - LON0;
  dLon = ((dLon + 540) % 360) - 180;
  const theta = N * ((dLon * Math.PI) / 180);
  return [rho * Math.sin(theta), RHO0 - rho * Math.cos(theta)];
}

function fromLcc(x: number, y: number): [number, number] {
  const rho = Math.hypot(x, RHO0 - y);
  const theta = Math.atan2(x, RHO0 - y);
  const phi = 2 * Math.atan(((R * F) / rho) ** (1 / N)) - Math.PI / 2;
  return [LON0 + (theta / N) * (180 / Math.PI), (phi * 180) / Math.PI];
}

const [X0, Y0] = toLcc(LL.lat, LL.lon);

/** Fractional grid position, i from the west edge and j from the south edge. */
export function gridPosition(lat: number, lon: number): [number, number] {
  const [x, y] = toLcc(lat, lon);
  return [(x - X0) / DX, (y - Y0) / DX];
}

export function insideHrrr(lat: number, lon: number) {
  const [i, j] = gridPosition(lat, lon);
  return i >= 0 && i <= NX - 1 && j >= 0 && j <= NY - 1;
}

function patchFits(index: number, n: number, chunk: number) {
  const start = Math.floor(index / chunk) * chunk;
  const size = Math.min(start + chunk, n) - start;
  const local = index - start;
  return local >= PATCH_HALF && local < size - PATCH_HALF;
}

/** Same rule as points.ContextReader: the 17x17 patch must not cross a tile edge. */
export function patchInsideTile(lat: number, lon: number) {
  const [i, j] = gridPosition(lat, lon);
  const xi = Math.round(i);
  // ponytail: assumes the store's y axis runs north to south (ContextReader
  // handles both). With NY = 4 * 265 - 1 the other order shifts tile edges by
  // one grid point (3 km), so at most a site within ~3 km of the cut-off flips.
  const yi = NY - 1 - Math.round(j);
  return patchFits(xi, NX, CHUNK_X) && patchFits(yi, NY, CHUNK_Y);
}

/** HRRR domain edge as a GeoJSON polygon (clockwise, as d3-geo expects). */
export function hrrrDomain(step = 40): GeoJSON.Polygon {
  const ring: [number, number][] = [];
  const push = (i: number, j: number) => ring.push(fromLcc(X0 + i * DX, Y0 + j * DX));
  for (let i = 0; i < NX; i += step) push(i, 0);
  for (let j = 0; j < NY; j += step) push(NX - 1, j);
  for (let i = NX - 1; i > 0; i -= step) push(i, NY - 1);
  for (let j = NY - 1; j > 0; j -= step) push(0, j);
  ring.push(ring[0]);
  return { type: "Polygon", coordinates: [ring.reverse()] };
}

export function distanceKm(a: LatLon, b: LatLon) {
  const rad = Math.PI / 180;
  const dLat = (b.lat - a.lat) * rad;
  const dLon = (b.lon - a.lon) * rad;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
}

/**
 * Run the tap-time checks in order and stop at the first failure.
 */
export function runChecks(tap: LatLon, stations: Station[]): CheckResult {
  const steps: CheckStep[] = [];
  let failed = false;
  const add = (id: string, label: string, test: () => [boolean, string]) => {
    if (failed) {
      steps.push({ id, label, status: "skip", detail: "Not checked" });
      return;
    }
    const [ok, detail] = test();
    steps.push({ id, label, status: ok ? "pass" : "fail", detail });
    failed = !ok;
  };

  let station: Station | null = null;
  let km = Infinity;

  add("coverage", "Inside HRRR coverage", () =>
    insideHrrr(tap.lat, tap.lon)
      ? [true, "HRRR forecasts this spot on its 3 km grid"]
      : [false, "HRRR only covers the continental US and southern Canada"],
  );

  add("station", `Weather station within ${STATION_RADIUS_KM} km`, () => {
    for (const s of stations) {
      const d = distanceKm(tap, s);
      if (d < km) {
        km = d;
        station = s;
      }
    }
    return km <= STATION_RADIUS_KM
      ? [true, `${station!.name} (${station!.id}), ${km.toFixed(0)} km away`]
      : [false, `Nearest station is ${km.toFixed(0)} km away; there is nothing nearby to learn from or score against`];
  });

  add("history", "Station has records back to Dec 2020", () =>
    station!.begin <= HISTORY_FROM
      ? [true, `Records start ${station!.begin}`]
      : [false, `Records only start ${station!.begin}`],
  );

  add("tile", "Local 50 km map fits one HRRR tile", () =>
    patchInsideTile(station!.lat, station!.lon)
      ? [true, "The 17x17 grid square around the station sits inside one stored tile"]
      : [false, "The station sits on an HRRR tile edge; pick a spot a little further in"],
  );

  return { steps, station: failed ? null : station, km, ok: !failed };
}
