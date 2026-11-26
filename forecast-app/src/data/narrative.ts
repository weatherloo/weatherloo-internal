import { clock, hourWord } from "./time";
import { POP_WET, type HourEntry, type Report } from "./report";

/** A run of text; `accent` renders italic terracotta, `em` italic only. */
export interface Run { text: string; accent?: boolean; em?: boolean }
export interface Chip { icon: "umbrella" | "shirt" | "sun"; label: string; text: string; accent?: boolean }

/** Typographic minus, rounded. */
export const deg = (n: number) => `${Math.round(n) < 0 ? "−" : ""}${Math.abs(Math.round(n))}`;

function dryWord(r: Report) {
  const sky = r.today.sky;
  return sky === "snow" ? "snowy" : sky === "rain" ? "wet" : "dry";
}

function hourText(h: HourEntry) {
  return hourWord(h.hour).replace(":00", "");
}

export function buildNarrative(r: Report) {
  const tz = r.station.timeZone;
  const tomorrow = r.days.find((d) => !d.isToday);
  const hi = r.today.high;
  const peak = r.today.hours.reduce((a, b) => (b.temp > a.temp ? b : a));
  const laterWet = r.allHours.find((h) => h.time > r.now && h.pop >= POP_WET);

  const headline: Run[] = [{ text: `${deg(hi)}° & ${dryWord(r)}, ` }];
  headline.push(
    laterWet
      ? { text: `${laterWet.sky} by ${r.allHours.length ? weekdayOf(laterWet, r) : ""}`, accent: true }
      : { text: "a quiet, steady day", accent: true },
  );

  const todayHrs = r.today.hours.filter((h) => h.time >= r.now);
  const winds = todayHrs.map((h) => h.windKmh);
  const lo = Math.round(Math.min(...winds)), hiW = Math.round(Math.max(...winds));
  const windWord = hiW < 15 ? "light" : hiW < 30 ? "breezy" : "strong";
  const pops = todayHrs.map((h) => h.pop);
  const rainLine = Math.max(...pops) < 0.1 ? "0% rain every hour" : `rain chance tops out at ${Math.round(Math.max(...pops) * 100)}%`;
  const night = r.today.hours.filter((h) => h.hour >= 21 || h.hour <= 3).map((h) => h.temp);

  const lede: Run[] = [
    { text: `about ${deg(r.current.temp)} at ${hourText(r.current)}, feels ${deg(r.current.feels)}. ` },
    { text: `${dryWord(r) === "dry" ? "dry all day" : `${dryWord(r)} at times`}, ${rainLine}, ${windWord} wind ${lo === hiW ? lo : `${lo} to ${hiW}`} km/h. ` },
    { text: `warms to ${deg(peak.temp)} around ${hourText(peak)}, sundown about ${clock(r.sun.sunset, tz).replace(/am|pm/, "")}, then ${night.length && Math.min(...night) < -5 ? "a cold" : "a chilly"} night near ${deg(Math.min(...(night.length ? night : [r.today.low])))}. ` },
  ];
  if (tomorrow) {
    lede.push({ text: `${tomorrow.weekday} ${deg(tomorrow.high)}`, em: true });
    lede.push({ text: tomorrow.sky === "dry" ? ", staying dry." : `, with ${tomorrow.sky} chances ${tomorrow.note.replace(/^.*from about /, "from about ").replace(/\.$/, "")}.` });
  }

  const dryHrs = todayHrs.filter((h) => h.pop < 0.1).length;
  const chips: Chip[] = [
    {
      icon: "umbrella", label: "umbrella",
      text: Math.max(...pops) < POP_WET ? `no, ${Math.round(Math.max(...pops) * 100)}% at most` : `yes, ${Math.round(Math.max(...pops) * 100)}% peak`,
    },
    { icon: "shirt", label: "wear", text: wear(r) },
  ];
  const win = bestWindow(todayHrs);
  if (win) {
    chips.push({
      icon: "sun", label: "best window", accent: true,
      text: `${hourWord(win.from)} to ${hourWord(win.to)}, ${deg(win.lo) === deg(win.hi) ? `${deg(win.lo)}°` : `${deg(win.lo)} to ${deg(win.hi)}°`}`,
    });
  } else if (!dryHrs) chips.push({ icon: "sun", label: "best window", text: "none today", accent: true });
  return { headline, lede, chips };
}

function weekdayOf(h: HourEntry, r: Report) {
  const d = r.days.find((x) => x.hours.some((y) => y.time.getTime() === h.time.getTime()));
  const part = h.hour < 12 ? "morning" : h.hour < 17 ? "afternoon" : "evening";
  return d ? (d.isToday ? `${part}` : `${d.weekday === "sat" ? "saturday" : d.weekday} ${part}`) : part;
}

function wear(r: Report) {
  const t = r.current.feels, noon = r.hours.find((h) => h.hour === 12)?.feels ?? t;
  const coat = (x: number) => x <= -15 ? "a parka, toque and mitts" : x <= -3 ? "a winter coat, toque and gloves" : x <= 5 ? "a warm jacket" : x <= 14 ? "a light jacket" : "a t-shirt";
  return coat(t) === coat(noon) ? coat(t) : `${coat(t)} this morning, ${coat(noon)} by noon`;
}

/** Longest daylight run (9am-6pm) with low rain chance, preferring warmth. */
function bestWindow(hours: HourEntry[]) {
  const day = hours.filter((h) => h.hour >= 9 && h.hour <= 17 && h.isDay && h.pop < 0.25);
  let best: HourEntry[] = [];
  for (let i = 0; i < day.length; i++) {
    const run = [day[i]];
    while (i + run.length < day.length && day[i + run.length].hour === run[run.length - 1].hour + 1 && run.length < 4) run.push(day[i + run.length]);
    const score = (x: HourEntry[]) => x.length * 10 + x.reduce((a, h) => a + h.temp, 0) / x.length;
    if (run.length >= 2 && (!best.length || score(run) > score(best))) best = run;
  }
  if (!best.length) return null;
  const temps = best.map((h) => h.temp);
  return { from: best[0].hour, to: best[best.length - 1].hour + 1, lo: Math.min(...temps), hi: Math.max(...temps) };
}
