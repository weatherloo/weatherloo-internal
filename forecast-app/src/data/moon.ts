/** Moon phase from the mean synodic month (accurate to about a day). */
const SYNODIC = 29.530588853;
const REF_NEW_MOON = Date.UTC(2000, 0, 6, 18, 14);

export interface MoonInfo { age: number; illumination: number; name: string; waxing: boolean; daysToFull: number }

export function moonInfo(d: Date): MoonInfo {
  const age = (((d.getTime() - REF_NEW_MOON) / 86400e3) % SYNODIC + SYNODIC) % SYNODIC;
  const illumination = (1 - Math.cos((2 * Math.PI * age) / SYNODIC)) / 2;
  const f = age / SYNODIC;
  const name = f < 0.03 || f > 0.97 ? "New moon" : f < 0.22 ? "Waxing crescent" : f < 0.28 ? "First quarter"
    : f < 0.47 ? "Waxing gibbous" : f < 0.53 ? "Full moon" : f < 0.72 ? "Waning gibbous" : f < 0.78 ? "Last quarter" : "Waning crescent";
  return { age, illumination, name, waxing: f < 0.5, daysToFull: Math.round((((SYNODIC / 2 - age) % SYNODIC) + SYNODIC) % SYNODIC) };
}
