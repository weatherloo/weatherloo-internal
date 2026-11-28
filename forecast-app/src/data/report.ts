import type { ForecastRecord } from "./contract";
import { getModel } from "./registry";
import type { Station } from "./stations";
import { sunTimes, type SunTimes } from "./sun";
import { dayKey, hourLabel, localParts, capitalize } from "./time";

/** Display-unit conversions. Canonical (contract) units are SI. */
export const msToKmh = (v: number) => v * 3.6;

/** Environment Canada wind chill; applies at T <= 10 degC and wind > 4.8 km/h. */
export function windChill(tC: number, kmh: number) {
  if (tC > 10 || kmh <= 4.8) return tC;
  const v = Math.pow(kmh, 0.16);
  return 13.12 + 0.6215 * tC - 11.37 * v + 0.3965 * tC * v;
}

export type Sky = "dry" | "rain" | "snow";
export const POP_WET = 0.4; // hourly chance at which an hour counts as wet

export interface HourEntry {
  time: Date; label: string; hour: number;
  temp: number; feels: number; windKmh: number; pop: number; sky: Sky; isDay: boolean;
}
export interface DayEntry {
  key: string; weekday: string; dateLabel: string; isToday: boolean;
  high: number; low: number; maxPop: number; sky: Sky; windKmh: number; note: string; hours: HourEntry[];
}
export interface Report {
  station: Station; modelId: string; modelLabel: string;
  now: Date; initTime: Date; issuedAt: Date;
  /** Weather at `now`. */
  current: HourEntry;
  today: DayEntry; days: DayEntry[]; hours: HourEntry[]; // hours: now .. +24h
  allHours: HourEntry[];
  sun: SunTimes; sunTomorrow: SunTimes;
  /** Total expected precipitation over the forecast, from the 6 h amount blocks. */
  precipMm: number; precipSky: Sky;
}

type Series = { ms: number[]; v: number[] };
function seriesOf(records: ForecastRecord[], variable: string): Series {
  const rec = records.find((r) => r.variable === variable);
  if (!rec) throw new Error(`no ${variable} record for the active model`);
  const pts = [...rec.points].sort((a, b) => a.lead_hours - b.lead_hours);
  return { ms: pts.map((p) => Date.parse(p.valid_time)), v: pts.map((p) => p.value) };
}
function at(s: Series, ms: number): number {
  const { ms: t, v } = s;
  if (ms <= t[0]) return v[0];
  if (ms >= t[t.length - 1]) return v[v.length - 1];
  let i = t.findIndex((x) => x >= ms);
  if (t[i] === ms) return v[i];
  const f = (ms - t[i - 1]) / (t[i] - t[i - 1]);
  return v[i - 1] + f * (v[i] - v[i - 1]);
}

function skyOf(pop: number, temp: number): Sky {
  return pop >= POP_WET ? (temp <= 0.5 ? "snow" : "rain") : "dry";
}

export function buildReport(records: ForecastRecord[], station: Station, now: Date): Report {
  const tz = station.timeZone;
  const t2m = seriesOf(records, "t2m");
  const wind = seriesOf(records, "wind_speed");
  const pop = seriesOf(records, "pop1");
  const meta = records.find((r) => r.variable === "t2m")!;

  const sunCache = new Map<string, SunTimes>();
  const sunFor = (d: Date) => {
    const p = localParts(d, tz);
    const k = `${p.year}-${p.month}-${p.day}`;
    if (!sunCache.has(k)) sunCache.set(k, sunTimes(p.year, p.month, p.day, station.lat, station.lon));
    return sunCache.get(k)!;
  };
  const entry = (ms: number): HourEntry => {
    const time = new Date(ms);
    const temp = at(t2m, ms), kmh = msToKmh(at(wind, ms)), p = at(pop, ms);
    const s = sunFor(time);
    return {
      time, label: hourLabel(time, tz), hour: localParts(time, tz).hour, temp, windKmh: kmh,
      feels: windChill(temp, kmh), pop: p, sky: skyOf(p, temp),
      isDay: time >= s.sunrise && time <= s.sunset,
    };
  };

  const allHours = t2m.ms.map(entry);
  const nowMs = now.getTime();
  const first = Math.max(0, allHours.findIndex((h) => h.time.getTime() >= nowMs));
  const hours = allHours.slice(first, first + 24);

  const byDay = new Map<string, HourEntry[]>();
  for (const h of allHours) {
    const k = dayKey(h.time, tz);
    byDay.set(k, [...(byDay.get(k) ?? []), h]);
  }
  const todayKey = dayKey(now, tz);
  const days: DayEntry[] = [...byDay.entries()]
    .filter(([, hs]) => hs.length >= 12) // skip the sliver at the end of the 48 h window
    .map(([key, hs]) => {
      const p0 = localParts(hs[0].time, tz);
      const temps = hs.map((h) => h.temp);
      const maxPop = Math.max(...hs.map((h) => h.pop));
      const wet = hs.filter((h) => h.pop >= POP_WET);
      const sky: Sky = wet.length ? (wet.filter((h) => h.sky === "snow").length >= wet.length / 2 ? "snow" : "rain") : "dry";
      return {
        key, hours: hs, weekday: key === todayKey ? "today" : p0.weekday.slice(0, 3),
        dateLabel: `${p0.monthShort} ${p0.day}`, isToday: key === todayKey,
        high: Math.max(...temps), low: Math.min(...temps), maxPop, sky,
        windKmh: Math.max(...hs.map((h) => h.windKmh)), note: "",
      };
    });
  for (const d of days) d.note = dayNote(d, tz);

  const amt = records.find((r) => r.variable === "amt6");
  const precipMm = amt ? amt.points.reduce((a, p) => a + p.value, 0) : 0;
  const wetHours = allHours.filter((h) => h.pop >= POP_WET);
  const precipSky: Sky = wetHours.length
    ? wetHours.filter((h) => h.sky === "snow").length >= wetHours.length / 2 ? "snow" : "rain" : "dry";

  const tomorrow = new Date(nowMs + 86400e3);
  return {
    station, modelId: meta.model_id, modelLabel: getModel(meta.model_id)?.label ?? meta.model_id,
    now, initTime: new Date(meta.init_time), issuedAt: new Date(meta.issued_at),
    current: entry(nowMs), today: days.find((d) => d.isToday) ?? days[0], days, hours, allHours,
    sun: sunFor(now), sunTomorrow: sunFor(tomorrow), precipMm, precipSky,
  };
}

function dayNote(d: DayEntry, tz: string): string {
  const wetStart = d.hours.find((h) => h.pop >= POP_WET);
  if (wetStart) {
    const w = d.sky === "snow" ? "snow" : "rain";
    return `${Math.round(d.maxPop * 100)}% ${w} chance, from about ${wetStart.label.replace(/[ap]$/, (m) => (m === "a" ? "am" : "pm"))}.`;
  }
  if (d.windKmh >= 25) return `windy, gusts of ${Math.round(d.windKmh)} km/h.`;
  return d.maxPop >= 0.15 ? `a ${Math.round(d.maxPop * 100)}% chance of a flurry.` : "dry all day.";
}
export { capitalize };
