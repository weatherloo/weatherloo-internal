import { POP_WET, type Report } from "./report";

/** Typographic minus, rounded. */
export const deg = (n: number) => `${Math.round(n) < 0 ? "−" : ""}${Math.abs(Math.round(n))}`;

const SKY_WORD = { dry: "Dry", rain: "Rain", snow: "Snow" } as const;

/** Headline condition under the big temperature. */
export function conditionLabel(r: Report) {
  const c = r.current;
  return c.sky === "dry" ? (c.windKmh >= 25 ? "Dry & windy" : "Dry") : SKY_WORD[c.sky];
}

/** One or two plain sentences for the summary card above the hourly strip. */
export function summaryText(r: Report) {
  const next = r.allHours.find((h) => h.time > r.now && h.pop >= POP_WET);
  const winds = r.hours.map((h) => h.windKmh);
  const wind = `Wind up to ${Math.round(Math.max(...winds))} km/h.`;
  if (!next) return `Dry conditions expected for the next 24 hours. ${wind}`;
  const day = r.days.find((d) => d.hours.some((h) => h.time.getTime() === next.time.getTime()));
  const when = `${day && !day.isToday ? `${day.weekday.charAt(0).toUpperCase()}${day.weekday.slice(1)} ` : ""}around ${next.label.replace(/([ap])$/, (m) => (m === "a" ? "AM" : "PM"))}`;
  return `${SKY_WORD[next.sky]} expected ${when}. ${wind}`;
}
