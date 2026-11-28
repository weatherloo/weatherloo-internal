import { StyleSheet, Text, View, type StyleProp, type ViewStyle } from "react-native";
import Svg, { Circle, Line, Path } from "react-native-svg";
import { Card } from "@/components/Card";
import { deg } from "@/data/narrative";
import { POP_WET, type Report } from "@/data/report";
import { clock } from "@/data/time";
import { fonts, useTheme } from "@/theme/tokens";

/**
 * Detail tiles. Only quantities the model forecasts, or that are computed from the
 * station location/date, appear here (no humidity, pressure, UV, visibility).
 */
export function Tiles({ report }: { report: Report }) {
  // Phone layout: precipitation and sunrise side by side, wind full width (compact labels fit 375 px).
  return (
    <View style={s.grid}>
      <View style={s.half}><Precip report={report} style={s.fill} /></View>
      <View style={s.half}><Sunrise report={report} style={s.fill} /></View>
      <View style={s.full}><Wind report={report} compact /></View>
    </View>
  );
}

/** `compact` shrinks the dial/disc and row text for the narrower desktop columns. */
type TileProps = { report: Report; style?: StyleProp<ViewStyle>; compact?: boolean };

function Big({ children }: { children: string }) {
  const t = useTheme();
  return <Text style={[s.big, { color: t.text }]}>{children}</Text>;
}
function Note({ children }: { children: string }) {
  const t = useTheme();
  return <Text style={[s.note, { color: t.text }]}>{children}</Text>;
}

export function Precip({ report, style }: TileProps) {
  const t = useTheme();
  const today = report.today;
  const next = report.allHours.find((h) => h.time > report.now && h.pop >= POP_WET);
  const peak = Math.round(Math.max(...report.allHours.map((h) => h.pop)) * 100);
  const unit = report.precipSky === "snow" ? "snow" : "precip";
  return (
    <Card title="Precipitation" icon="drop" style={[s.tile, style]}>
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

export function Wind({ report, style, compact }: TileProps) {
  const w = report.hours.map((h) => h.windKmh);
  const now = Math.round(report.current.windKmh);
  const rows: [string, string][] = [
    [compact ? "Now" : "Wind", `${now} km/h`],
    [compact ? "Max, 24 h" : "Strongest, 24 h", `${Math.round(Math.max(...w))} km/h`],
    [compact ? "Min, 24 h" : "Lightest, 24 h", `${Math.round(Math.min(...w))} km/h`],
  ];
  return (
    <Card title="Wind" icon="wind" style={style}>
      <View style={{ flexDirection: "row", alignItems: "center" }}>
        <KeyValues rows={rows} compact={compact} />
        <Dial value={now} size={compact ? 108 : 140} />
      </View>
    </Card>
  );
}

function KeyValues({ rows, compact }: { rows: [string, string][]; compact?: boolean }) {
  const t = useTheme();
  const text = compact && s.kvCompact;
  return (
    <View style={{ flex: 1, paddingRight: compact ? 12 : 16 }}>
      {rows.map(([k, v], i) => (
        <View key={k} style={[s.kv, compact && s.kvRowCompact, i > 0 && { borderTopWidth: 1, borderTopColor: t.hairline }]}>
          <Text style={[s.kvKey, text, { color: t.text }]}>{k}</Text>
          <Text style={[s.kvVal, text, { color: t.muted }]}>{v}</Text>
        </View>
      ))}
    </View>
  );
}

/** Tick ring with the speed in the middle (the CNN has no wind direction, so no compass). */
function Dial({ value, size }: { value: number; size: number }) {
  const t = useTheme();
  const c = 70;
  return (
    <View style={{ width: size, height: size, alignItems: "center", justifyContent: "center" }}>
      <Svg width={size} height={size} viewBox="0 0 140 140" style={{ position: "absolute" }}>
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

export function Sunrise({ report, style }: TileProps) {
  const tz = report.station.timeZone, t = useTheme();
  const { sunrise, sunset } = report.sun;
  const frac = Math.min(1, Math.max(0, (report.now.getTime() - sunrise.getTime()) / (sunset.getTime() - sunrise.getTime())));
  const before = report.now < sunrise, after = report.now > sunset;
  // Curve: horizon at y=40, peak y=8; dot rides the daylight arc, below the line at night.
  const px = 4 + frac * 112, py = 40 - Math.sin(frac * Math.PI) * 32;
  return (
    <Card title={before ? "Sunrise" : after ? "Sunset" : "Sunset"} icon="sunrise" style={[s.tile, style]}>
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

const s = StyleSheet.create({
  grid: { flexDirection: "row", flexWrap: "wrap", gap: 12 },
  // Two per row: 2 x 47% plus the 12 px gap fits a 343 px phone column.
  half: { width: "47%", flexGrow: 1 },
  full: { width: "100%" },
  fill: { flex: 1 },
  tile: { minHeight: 176 },
  big: { fontFamily: fonts.regular, fontSize: 38, color: "#fff" },
  sub: { fontFamily: fonts.semi, fontSize: 17, marginTop: 2 },
  note: { fontFamily: fonts.regular, fontSize: 16, lineHeight: 21 },
  kv: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 13 },
  kvKey: { fontFamily: fonts.regular, fontSize: 17 },
  kvVal: { fontFamily: fonts.regular, fontSize: 17 },
  kvCompact: { fontSize: 15 },
  kvRowCompact: { paddingVertical: 11, gap: 8 },
});
