import raw from "./hrrr-map.json";

/**
 * One real NOAA HRRR run around the station, exported by pipelines/download/export_hrrr_map.py.
 * Unlike the station fixtures this is genuine model output: native 3 km cells, rows north to south
 * in grid orientation (grid "up" is `northDeg` from true north), uint8-quantized per field.
 */
export type MapField = "t2m" | "wind_speed" | "precip_rate";

interface RawField { unit: string; min: number; max: number; transform: string | null; data: string; every?: number }
interface RawMap {
  source: string; init_time: string; leads: number[];
  grid: { cells: number; spacing_km: number; north_deg: number };
  site: { lat: number; lon: number; col: number; row: number };
  places: { name: string; col: number; row: number; water: boolean }[];
  fields: Record<MapField | "wind_toward", RawField>;
}

export interface HrrrMap {
  source: string;
  initTime: Date;
  leads: number[];
  cells: number;
  spacingKm: number;
  northDeg: number;
  site: { col: number; row: number };
  places: RawMap["places"];
  /** Value of a field at [frame][row * cells + col], in display units (°C, km/h, mm/h). */
  value: (field: MapField, frame: number, index: number) => number;
  /** Grid-relative direction the wind blows toward, on a coarser lattice of `windEvery` cells. */
  windToward: (frame: number, row: number, col: number) => number;
  windEvery: number;
  /** Bilinear sample at a fractional cell position. */
  sample: (field: MapField, frame: number, col: number, row: number) => number;
  /** Data range over all frames, in display units. */
  range: (field: MapField) => [number, number];
}

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
function decodeBase64(text: string): Uint8Array {
  const lookup = new Uint8Array(128);
  for (let i = 0; i < B64.length; i++) lookup[B64.charCodeAt(i)] = i;
  const clean = text.replace(/=+$/, "");
  const out = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let o = 0;
  for (let i = 0; i < clean.length; i += 4) {
    const a = lookup[clean.charCodeAt(i)], b = lookup[clean.charCodeAt(i + 1)];
    const c = lookup[clean.charCodeAt(i + 2)] ?? 0, d = lookup[clean.charCodeAt(i + 3)] ?? 0;
    const n = (a << 18) | (b << 12) | (c << 6) | d;
    if (o < out.length) out[o++] = (n >> 16) & 255;
    if (o < out.length) out[o++] = (n >> 8) & 255;
    if (o < out.length) out[o++] = n & 255;
  }
  return out;
}

/** Display conversion: wind m/s -> km/h; temperature and precipitation stay as stored. */
const DISPLAY_SCALE: Record<MapField, number> = { t2m: 1, wind_speed: 3.6, precip_rate: 1 };

function dequantizer(f: RawField, scale: number) {
  const sqrt = f.transform === "sqrt";
  const lo = sqrt ? Math.sqrt(f.min) : f.min, hi = sqrt ? Math.sqrt(f.max) : f.max;
  const table = new Float32Array(256);
  for (let q = 0; q < 255; q++) {
    const v = lo + (q / 254) * (hi - lo);
    table[q] = (sqrt ? v * v : v) * scale;
  }
  table[255] = NaN;
  return table;
}

function build(map: RawMap): HrrrMap {
  const n = map.grid.cells, frameSize = n * n;
  const fields = (["t2m", "wind_speed", "precip_rate"] as MapField[]).map((k) => {
    const f = map.fields[k];
    return [k, { bytes: decodeBase64(f.data), table: dequantizer(f, DISPLAY_SCALE[k]), range: [f.min * DISPLAY_SCALE[k], f.max * DISPLAY_SCALE[k]] as [number, number] }] as const;
  });
  const lookup = Object.fromEntries(fields) as Record<MapField, { bytes: Uint8Array; table: Float32Array; range: [number, number] }>;
  const wind = map.fields.wind_toward, every = wind.every ?? 1, coarse = Math.floor(n / every);
  const windBytes = decodeBase64(wind.data), windTable = dequantizer(wind, 1);

  const value = (field: MapField, frame: number, index: number) => {
    const f = lookup[field];
    return f.table[f.bytes[frame * frameSize + index]];
  };
  const clamp = (v: number) => Math.max(0, Math.min(n - 1, v));
  return {
    source: map.source,
    initTime: new Date(map.init_time),
    leads: map.leads,
    cells: n,
    spacingKm: map.grid.spacing_km,
    northDeg: map.grid.north_deg,
    site: map.site,
    places: map.places,
    value,
    windEvery: every,
    windToward: (frame, row, col) => windTable[windBytes[frame * coarse * coarse + row * coarse + col]],
    sample: (field, frame, col, row) => {
      const c0 = Math.floor(clamp(col)), r0 = Math.floor(clamp(row));
      const c1 = Math.min(n - 1, c0 + 1), r1 = Math.min(n - 1, r0 + 1);
      const fc = clamp(col) - c0, fr = clamp(row) - r0;
      const at = (r: number, c: number) => value(field, frame, r * n + c);
      return (at(r0, c0) * (1 - fc) + at(r0, c1) * fc) * (1 - fr) + (at(r1, c0) * (1 - fc) + at(r1, c1) * fc) * fr;
    },
    range: (field) => lookup[field].range,
  };
}

let cached: HrrrMap | undefined;
export function hrrrMap(): HrrrMap {
  return (cached ??= build(raw as unknown as RawMap));
}
