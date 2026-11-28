/**
 * Colour scales for the HRRR map, as (value, colour) stops in display units.
 * Temperature is anchored to absolute values (0 °C sits at the blue/teal break) so frames
 * and days compare honestly; wind is a viridis-like ramp; precipitation follows radar conventions.
 */
export type Stop = [number, string];

export const TEMPERATURE: Stop[] = [
  [-30, "#2a1d6b"], [-20, "#2b3f9e"], [-10, "#2f63b8"], [-4, "#3487c4"], [0, "#35a3bf"],
  [3, "#3fae96"], [6, "#79bd5c"], [9, "#c8bf3f"], [12, "#e69a35"], [16, "#dd642e"],
  [22, "#bf3a3a"], [30, "#8a1d4f"],
];
export const WIND_KMH: Stop[] = [
  [0, "#1d1b4f"], [8, "#2c4a8c"], [18, "#21808c"], [30, "#3fb36d"], [45, "#b6dc3c"], [65, "#fde725"],
];
export const PRECIP_MMH: Stop[] = [
  [0, "#161c4a"], [0.1, "#1f3d7d"], [0.3, "#2a6cc2"], [1, "#2fb0d8"], [2.5, "#56d28a"],
  [5, "#f2d14b"], [10, "#f07d3c"], [20, "#c2307a"],
];

const rgb = (hex: string): [number, number, number] => [
  parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16),
];
const cache = new WeakMap<Stop[], [number, [number, number, number]][]>();

/** Colour at `value`, linearly interpolated between stops and clamped at the ends. */
export function colorAt(stops: Stop[], value: number): [number, number, number] {
  let parsed = cache.get(stops);
  if (!parsed) cache.set(stops, (parsed = stops.map(([v, c]) => [v, rgb(c)])));
  if (!(value > parsed[0][0])) return parsed[0][1];
  for (let i = 1; i < parsed.length; i++) {
    const [v1, c1] = parsed[i];
    if (value <= v1) {
      const [v0, c0] = parsed[i - 1], f = (value - v0) / (v1 - v0);
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return parsed[parsed.length - 1][1];
}

export const toHex = ([r, g, b]: [number, number, number]) =>
  `#${[r, g, b].map((x) => Math.round(x).toString(16).padStart(2, "0")).join("")}`;
