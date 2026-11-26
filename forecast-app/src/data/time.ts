/** Display-time helpers. Data stays UTC; only presentation uses the station's zone. */
export interface LocalParts {
  year: number; month: number; day: number; hour: number; minute: number; weekday: string; monthShort: string;
}

const cache = new Map<string, Intl.DateTimeFormat>();
function fmt(tz: string, opts: Intl.DateTimeFormatOptions) {
  const key = tz + JSON.stringify(opts);
  let f = cache.get(key);
  if (!f) cache.set(key, (f = new Intl.DateTimeFormat("en-CA", { timeZone: tz, ...opts })));
  return f;
}

export function localParts(d: Date, tz: string): LocalParts {
  const parts = fmt(tz, {
    year: "numeric", month: "numeric", day: "numeric", hour: "numeric", minute: "numeric",
    hourCycle: "h23", weekday: "long",
  }).formatToParts(d);
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
  const monthShort = fmt(tz, { month: "short" }).format(d).replace(".", "").toLowerCase();
  return {
    year: +get("year"), month: +get("month"), day: +get("day"),
    hour: +get("hour") % 24, minute: +get("minute"), weekday: get("weekday").toLowerCase(), monthShort,
  };
}

export const dayKey = (d: Date, tz: string) => {
  const p = localParts(d, tz);
  return `${p.year}-${String(p.month).padStart(2, "0")}-${String(p.day).padStart(2, "0")}`;
};

/** 7a, 12p, 1p */
export function hourLabel(d: Date, tz: string) {
  const h = localParts(d, tz).hour;
  return `${h % 12 === 0 ? 12 : h % 12}${h < 12 ? "a" : "p"}`;
}

/** 7am, 4pm (bare hour for narrative text) */
export function hourWord(h: number) {
  const hh = ((h % 24) + 24) % 24;
  return `${hh % 12 === 0 ? 12 : hh % 12}${hh < 12 ? "am" : "pm"}`;
}

/** 6:45am */
export function clock(d: Date, tz: string) {
  const p = localParts(d, tz);
  return `${p.hour % 12 === 0 ? 12 : p.hour % 12}:${String(p.minute).padStart(2, "0")}${p.hour < 12 ? "am" : "pm"}`;
}

export const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
