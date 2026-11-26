import { StyleSheet, Text, View } from "react-native";
import Svg, { Defs, LinearGradient, Rect, Stop } from "react-native-svg";
import { colors, fonts } from "@/theme/tokens";
import { deg } from "@/data/narrative";
import type { DayEntry } from "@/data/report";
import { Icon } from "./Icon";

/** One row of "The days ahead": day, sky, low, range bar, high, note. */
export function DayRow({ day, min, max }: { day: DayEntry; min: number; max: number }) {
  const span = Math.max(max - min, 1);
  const l = ((day.low - min) / span) * 100, w = Math.max(((day.high - day.low) / span) * 100, 3);
  return (
    <View style={s.row}>
      <View style={s.top}>
        <View style={{ width: 96 }}>
          <Text style={s.day}>{day.weekday === "today" ? "Today" : day.weekday.charAt(0).toUpperCase() + day.weekday.slice(1)}</Text>
          <Text style={s.date}>{day.dateLabel}</Text>
        </View>
        <View style={s.sky}>
          <Icon name={day.sky === "rain" ? "rain" : day.sky === "snow" ? "snow" : "sun"} size={34} />
          {day.maxPop >= 0.2 && <Text style={s.pop}>{Math.round(day.maxPop * 100)}%</Text>}
        </View>
        <Text style={s.lo}>{deg(day.low)}°</Text>
        <View style={s.bar}>
          <Svg width="100%" height="8" viewBox="0 0 100 8" preserveAspectRatio="none">
            <Defs>
              <LinearGradient id={`g${day.key}`} x1="0" y1="0" x2="1" y2="0">
                <Stop offset="0" stopColor={colors.cold} /><Stop offset="1" stopColor={colors.accent} />
              </LinearGradient>
            </Defs>
            <Rect x="0" y="0" width="100" height="8" rx="4" fill={colors.track} />
            <Rect x={l} y="0" width={w} height="8" rx="4" fill={`url(#g${day.key})`} />
          </Svg>
        </View>
        <Text style={s.hi}>{deg(day.high)}°</Text>
      </View>
      <Text style={s.note}>{day.note}</Text>
    </View>
  );
}

const s = StyleSheet.create({
  row: { borderBottomWidth: 1, borderBottomColor: colors.rule, paddingVertical: 18 },
  top: { flexDirection: "row", alignItems: "center" },
  day: { fontFamily: fonts.serif, fontSize: 24, color: colors.text },
  date: { fontFamily: fonts.sans, fontSize: 15, letterSpacing: 1, color: colors.muted, marginTop: 2 },
  sky: { width: 80, flexDirection: "row", alignItems: "center", gap: 8 },
  pop: { fontFamily: fonts.sansMedium, fontSize: 16, color: colors.rain },
  lo: { width: 44, fontFamily: fonts.sans, fontSize: 18, color: colors.muted, textAlign: "right" },
  bar: { flex: 1, marginHorizontal: 12 },
  hi: { width: 48, fontFamily: fonts.sansMedium, fontSize: 18, color: colors.text, textAlign: "right" },
  note: { fontFamily: fonts.sans, fontSize: 16, color: colors.muted, marginTop: 8, lineHeight: 24 },
});
