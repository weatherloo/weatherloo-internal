import { StyleSheet, Text, View } from "react-native";
import Svg, { Defs, LinearGradient, Rect, Stop } from "react-native-svg";
import { deg } from "@/data/narrative";
import type { DayEntry } from "@/data/report";
import { fonts, useTheme } from "@/theme/tokens";
import { Icon } from "./Icon";

/** One forecast row: day, sky (+ chance), low, range bar, high. */
export function DayRow({ day, min, max, first }: { day: DayEntry; min: number; max: number; first?: boolean }) {
  const t = useTheme();
  const span = Math.max(max - min, 1);
  const l = ((day.low - min) / span) * 100, w = Math.max(((day.high - day.low) / span) * 100, 4);
  const name = day.weekday === "today" ? "Today" : day.weekday.charAt(0).toUpperCase() + day.weekday.slice(1);
  return (
    <View style={[s.row, !first && { borderTopWidth: 1, borderTopColor: t.hairline }]}>
      <Text style={[s.day, { color: t.text }]}>{name}</Text>
      <View style={s.sky}>
        <Icon name={day.sky === "rain" ? "rain" : day.sky === "snow" ? "snow" : "sun"} size={30} />
        {day.maxPop >= 0.2 && <Text style={[s.pop, { color: t.rain }]}>{Math.round(day.maxPop * 100)}%</Text>}
      </View>
      <Text style={[s.lo, { color: t.muted }]}>{deg(day.low)}°</Text>
      <View style={s.bar}>
        <Svg width="100%" height="5" viewBox="0 0 100 5" preserveAspectRatio="none">
          <Defs>
            <LinearGradient id={`g${day.key}`} x1="0" y1="0" x2="1" y2="0">
              <Stop offset="0" stopColor="#5ac8fa" /><Stop offset="1" stopColor="#cfe67a" />
            </LinearGradient>
          </Defs>
          <Rect width="100" height="5" rx="2.5" fill="rgba(0,0,0,0.25)" />
          <Rect x={l} width={w} height="5" rx="2.5" fill={`url(#g${day.key})`} />
        </Svg>
      </View>
      <Text style={[s.hi, { color: t.text }]}>{deg(day.high)}°</Text>
    </View>
  );
}

const s = StyleSheet.create({
  row: { flexDirection: "row", alignItems: "center", height: 62 },
  day: { width: 74, fontFamily: fonts.medium, fontSize: 20 },
  sky: { width: 56, alignItems: "center" },
  pop: { fontFamily: fonts.semi, fontSize: 12, marginTop: -2 },
  lo: { width: 42, fontFamily: fonts.medium, fontSize: 19, textAlign: "right" },
  bar: { flex: 1, marginHorizontal: 10 },
  hi: { width: 42, fontFamily: fonts.medium, fontSize: 19, textAlign: "right" },
});
