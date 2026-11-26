import { StyleSheet, Text, View } from "react-native";
import { SunArc } from "@/components/SunArc";
import { Meta } from "@/components/Type";
import { deg } from "@/data/narrative";
import type { Report } from "@/data/report";
import { clock } from "@/data/time";
import { colors, fonts } from "@/theme/tokens";

interface Stat { label: string; value: string; note: string }

/** Only values the model actually forecasts (no empty humidity / pressure placeholders). */
export function statsFor(r: Report): Stat[] {
  const tz = r.station.timeZone;
  const h = Math.floor(r.sun.daylightMinutes / 60), m = r.sun.daylightMinutes % 60;
  const diff = r.sunTomorrow.daylightMinutes - r.sun.daylightMinutes;
  const winds = r.allHours.slice(0, 24).map((x) => x.windKmh);
  const maxPop = Math.max(...r.allHours.map((x) => x.pop));
  const mm = r.precipMm;
  return [
    { label: "sunrise", value: clock(r.sun.sunrise, tz), note: `sunset ${clock(r.sun.sunset, tz)}` },
    { label: "daylight", value: `${h}h ${String(m).padStart(2, "0")}m`, note: diff < 0 ? "getting shorter till dec 21" : "getting longer" },
    { label: "wind", value: `${Math.round(r.current.windKmh)} km/h`, note: `${Math.round(Math.min(...winds))} to ${Math.round(Math.max(...winds))} km/h next 24 h` },
    { label: "feels like", value: `${deg(r.current.feels)}°`, note: r.current.feels < r.current.temp - 1 ? "wind chill" : "close to the air temp" },
    { label: "precip chance", value: `${Math.round(maxPop * 100)}%`, note: "peak hourly chance, next 48 h" },
    { label: r.precipSky === "snow" ? "snow, 48 h" : "precip, 48 h", value: `${mm.toFixed(1)} mm`, note: mm < 0.5 ? "next to nothing" : "water equivalent" },
  ];
}

export function FinePrint({ report }: { report: Report }) {
  const stats = statsFor(report);
  const tz = report.station.timeZone;
  return (
    <View>
      <View style={{ marginTop: 28 }}>
        <SunArc />
        <View style={s.arcLabels}>
          <Meta style={s.arcText}>{clock(report.sun.sunrise, tz)}</Meta>
          <Meta style={s.arcText}>{clock(report.sun.sunset, tz)}</Meta>
        </View>
      </View>
      <View style={s.rule} />
      <View style={s.grid}>
        {stats.map((x) => (
          <View key={x.label} style={s.cell}>
            <Meta style={{ fontSize: 15 }}>{x.label}</Meta>
            <Text style={s.value}>{x.value}</Text>
            <Text style={s.note}>{x.note}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  arcLabels: { flexDirection: "row", justifyContent: "space-between", paddingHorizontal: 6 },
  arcText: { fontSize: 15, letterSpacing: 0.4 },
  rule: { height: 1.5, backgroundColor: colors.strongRule, marginTop: 28 },
  grid: { flexDirection: "row", flexWrap: "wrap" },
  cell: { width: "50%", paddingVertical: 26, paddingRight: 12, borderBottomWidth: 1, borderBottomColor: colors.rule },
  value: { fontFamily: fonts.serif, fontSize: 38, color: colors.text, marginTop: 12 },
  note: { fontFamily: fonts.sans, fontSize: 16, color: colors.muted, marginTop: 10, lineHeight: 24 },
});
