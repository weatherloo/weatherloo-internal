import { StyleSheet, Text, View } from "react-native";
import Svg, { Circle, Defs, G, Line, LinearGradient, Path, Stop } from "react-native-svg";
import { Card } from "@/components/Card";
import { moonInfo } from "@/data/moon";
import { deg } from "@/data/narrative";
import { POP_WET, type Report } from "@/data/report";
import { clock } from "@/data/time";
import { fonts, useTheme } from "@/theme/tokens";

/**
 * Detail tiles. Only quantities the model forecasts, or that are computed from the
 * station location/date, appear here (no humidity, pressure, UV, visibility).
 */
export function Tiles({ report }: { report: Report }) {
  return (
    <View style={s.grid}>
      <View style={s.half}><FeelsLike report={report} /></View>
      <View style={s.half}><Precip report={report} /></View>
      <View style={s.full}><Wind report={report} /></View>
      <View style={s.half}><Sunrise report={report} /></View>
      <View style={s.half}><Daylight report={report} /></View>
      <View style={s.full}><Moon report={report} /></View>
    </View>
  );
}

function Big({ children }: { children: string }) {
  const t = useTheme();
  return <Text style={[s.big, { color: t.text }]}>{children}</Text>;
}
function Note({ children }: { children: string }) {
  const t = useTheme();
  return <Text style={[s.note, { color: t.text }]}>{children}</Text>;
}

function FeelsLike({ report }: { report: Report }) {
  const c = report.current, diff = c.feels - c.temp;
  return (
    <Card title="Feels like" icon="thermometer" style={s.tile}>
      <Big>{`${deg(c.feels)}°`}</Big>
      <View style={{ flex: 1 }} />
      <Note>{diff < -1 ? "Wind is making it feel colder." : "Similar to the actual temperature."}</Note>
    </Card>
  );
}

function Precip({ report }: { report: Report }) {
  const t = useTheme();
  const today = report.today;
  const next = report.allHours.find((h) => h.time > report.now && h.pop >= POP_WET);
  const peak = Math.round(Math.max(...report.allHours.map((h) => h.pop)) * 100);
  const unit = report.precipSky === "snow" ? "snow" : "precip";
  return (
    <Card title="Precipitation" icon="drop" style={s.tile}>
      <Big>{`${Math.round(today.maxPop * 100)}%`}</Big>
      <Text style={[s.sub, { color: t.text }]}>Chance today</Text>
      <View style={{ flex: 1 }} />
      <Note>
        {next
          ? `${report.precipMm.toFixed(1)} mm ${unit} expected over 48 h.`
          : `Peak chance ${peak}% over the next 48 h.`}
      </Note>
    </Card>
  );
}

function Wind({ report }: { report: Report }) {
  const t = useTheme();
  const w = report.hours.map((h) => h.windKmh);
  const now = Math.round(report.current.windKmh);
  const rows: [string, string][] = [
    ["Wind", `${now} km/h`],
    ["Strongest, 24 h", `${Math.round(Math.max(...w))} km/h`],
    ["Lightest, 24 h", `${Math.round(Math.min(...w))} km/h`],
  ];
  return (
    <Card title="Wind" icon="wind">
      <View style={{ flexDirection: "row", alignItems: "center" }}>
        <View style={{ flex: 1, paddingRight: 16 }}>
          {rows.map(([k, v], i) => (
            <View key={k} style={[s.kv, i > 0 && { borderTopWidth: 1, borderTopColor: t.hairline }]}>
              <Text style={[s.kvKey, { color: t.text }]}>{k}</Text>
              <Text style={[s.kvVal, { color: t.muted }]}>{v}</Text>
            </View>
          ))}
        </View>
        <Dial value={now} />
      </View>
    </Card>
  );
}

/** Tick ring with the speed in the middle (the CNN has no wind direction, so no compass). */
function Dial({ value }: { value: number }) {
  const t = useTheme();
  const size = 140, c = size / 2;
  return (
    <View style={{ width: size, height: size, alignItems: "center", justifyContent: "center" }}>
      <Svg width={size} height={size} style={{ position: "absolute" }}>
        {Array.from({ length: 72 }, (_, i) => {
          const a = (i / 72) * 2 * Math.PI;
          return <Line key={i} x1={c + Math.sin(a) * 60} y1={c - Math.cos(a) * 60} x2={c + Math.sin(a) * 68} y2={c - Math.cos(a) * 68}
            stroke={t.muted} strokeWidth={1} opacity={0.6} />;
        })}
      </Svg>
      <Text style={{ fontFamily: fonts.semi, fontSize: 28, color: t.text }}>{value}</Text>
      <Text style={{ fontFamily: fonts.regular, fontSize: 15, color: t.text }}>km/h</Text>
    </View>
  );
}

function Sunrise({ report }: { report: Report }) {
  const tz = report.station.timeZone, t = useTheme();
  const { sunrise, sunset } = report.sun;
  const frac = Math.min(1, Math.max(0, (report.now.getTime() - sunrise.getTime()) / (sunset.getTime() - sunrise.getTime())));
  const before = report.now < sunrise, after = report.now > sunset;
  // Curve: horizon at y=40, peak y=8; dot rides the daylight arc, below the line at night.
  const px = 4 + frac * 112, py = 40 - Math.sin(frac * Math.PI) * 32;
  return (
    <Card title={before ? "Sunrise" : after ? "Sunset" : "Sunset"} icon="sunrise" style={s.tile}>
      <Big>{clock(before ? sunrise : sunset, tz)}</Big>
      <Svg width="100%" height={54} viewBox="0 0 120 54" style={{ marginTop: 6 }}>
        <Line x1="0" y1="40" x2="120" y2="40" stroke={t.muted} strokeWidth={1} />
        <Path d="M4 40 C 30 -6, 90 -6, 116 40" stroke={t.text} strokeWidth={2} fill="none" opacity={0.55} />
        {!before && !after && <Circle cx={px} cy={py} r={4.5} fill="#fff" />}
      </Svg>
      <Note>{before ? `Sunset: ${clock(sunset, tz)}` : `Sunrise: ${clock(sunrise, tz)}`}</Note>
    </Card>
  );
}

function Daylight({ report }: { report: Report }) {
  const m = report.sun.daylightMinutes, diff = report.sunTomorrow.daylightMinutes - m;
  return (
    <Card title="Daylight" icon="clock" style={s.tile}>
      <Big>{`${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`}</Big>
      <View style={{ flex: 1 }} />
      <Note>{diff < 0 ? "Days are getting shorter until Dec 21." : "Days are getting longer."}</Note>
    </Card>
  );
}

function Moon({ report }: { report: Report }) {
  const t = useTheme();
  const m = moonInfo(report.now);
  const rows: [string, string][] = [
    ["Illumination", `${Math.round(m.illumination * 100)}%`],
    ["Next full moon", m.daysToFull === 0 ? "Today" : `${m.daysToFull} days`],
  ];
  return (
    <Card title={m.name} icon="moon">
      <View style={{ flexDirection: "row", alignItems: "center" }}>
        <View style={{ flex: 1, paddingRight: 16 }}>
          {rows.map(([k, v], i) => (
            <View key={k} style={[s.kv, i > 0 && { borderTopWidth: 1, borderTopColor: t.hairline }]}>
              <Text style={[s.kvKey, { color: t.text }]}>{k}</Text>
              <Text style={[s.kvVal, { color: t.muted }]}>{v}</Text>
            </View>
          ))}
        </View>
        <MoonDisc illumination={m.illumination} waxing={m.waxing} />
      </View>
    </Card>
  );
}

function MoonDisc({ illumination, waxing }: { illumination: number; waxing: boolean }) {
  const r = 60, rx = r * Math.abs(1 - 2 * illumination), crescent = illumination < 0.5;
  const edgeSweep = waxing ? 1 : 0;
  const termSweep = waxing ? (crescent ? 0 : 1) : (crescent ? 1 : 0);
  const lit = `M0 ${-r} A${r} ${r} 0 0 ${edgeSweep} 0 ${r} A${rx} ${r} 0 0 ${termSweep} 0 ${-r} Z`;
  return (
    <Svg width={130} height={130} viewBox="-65 -65 130 130">
      <Defs>
        <LinearGradient id="moonlit" x1="0" y1="0" x2="1" y2="1"><Stop offset="0" stopColor="#f3f4fb" /><Stop offset="1" stopColor="#b9bdd6" /></LinearGradient>
      </Defs>
      <Circle r={r} fill="#232850" />
      <G><Path d={lit} fill="url(#moonlit)" /></G>
    </Svg>
  );
}

const s = StyleSheet.create({
  grid: { flexDirection: "row", flexWrap: "wrap", gap: 12 },
  half: { width: "48.4%", flexGrow: 1 },
  full: { width: "100%" },
  tile: { minHeight: 176 },
  big: { fontFamily: fonts.regular, fontSize: 38, color: "#fff" },
  sub: { fontFamily: fonts.semi, fontSize: 17, marginTop: 2 },
  note: { fontFamily: fonts.regular, fontSize: 16, lineHeight: 21 },
  kv: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 13 },
  kvKey: { fontFamily: fonts.regular, fontSize: 17 },
  kvVal: { fontFamily: fonts.regular, fontSize: 17 },
});
