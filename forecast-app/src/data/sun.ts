/** Sunrise/sunset from the NOAA general solar position equations (good to ~1 min). */
export interface SunTimes { sunrise: Date; sunset: Date; daylightMinutes: number }

export function sunTimes(year: number, month: number, day: number, lat: number, lon: number): SunTimes {
  const rad = Math.PI / 180;
  const n = Math.round((Date.UTC(year, month - 1, day) - Date.UTC(year, 0, 0)) / 86400e3);
  const g = ((2 * Math.PI) / 365) * (n - 1);
  const eqtime = 229.18 * (0.000075 + 0.001868 * Math.cos(g) - 0.032077 * Math.sin(g)
    - 0.014615 * Math.cos(2 * g) - 0.040849 * Math.sin(2 * g));
  const decl = 0.006918 - 0.399912 * Math.cos(g) + 0.070257 * Math.sin(g) - 0.006758 * Math.cos(2 * g)
    + 0.000907 * Math.sin(2 * g) - 0.002697 * Math.cos(3 * g) + 0.00148 * Math.sin(3 * g);
  const cosHa = Math.cos(90.833 * rad) / (Math.cos(lat * rad) * Math.cos(decl)) - Math.tan(lat * rad) * Math.tan(decl);
  const ha = Math.acos(Math.min(1, Math.max(-1, cosHa))) / rad;
  const midnight = Date.UTC(year, month - 1, day);
  const rise = 720 - 4 * (lon + ha) - eqtime;
  const set = 720 - 4 * (lon - ha) - eqtime;
  return {
    sunrise: new Date(midnight + rise * 60e3),
    sunset: new Date(midnight + set * 60e3),
    daylightMinutes: Math.round(set - rise),
  };
}
