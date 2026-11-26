import { ScrollView, StyleSheet, Text, View } from "react-native";
import Svg, { Circle, Defs, LinearGradient, Path, Stop } from "react-native-svg";
import { colors, fonts } from "@/theme/tokens";
import { deg } from "@/data/narrative";
import type { HourEntry } from "@/data/report";
import { Icon } from "./Icon";

const COL = 70, CHART_H = 120, PAD_T = 36, PAD_B = 18;

/** Horizontally scrolling temperature line with sky icon, rain chance and hour per column. */
export function HourlyChart({ hours }: { hours: HourEntry[] }) {
  const temps = hours.map((h) => h.temp);
  const lo = Math.min(...temps), hi = Math.max(...temps);
  const span = Math.max(hi - lo, 4);
  const x = (i: number) => COL / 2 + i * COL;
  const y = (t: number) => PAD_T + (1 - (t - lo) / span) * (CHART_H - PAD_T - PAD_B);
  const W = hours.length * COL;
  const line = hours.map((h, i) => `${i ? "L" : "M"}${x(i)} ${y(h.temp)}`).join(" ");
  const area = `${line} L${x(hours.length - 1)} ${CHART_H} L${x(0)} ${CHART_H} Z`;

  return (
    <View style={s.wrap}>
      <ScrollView horizontal showsHorizontalScrollIndicator={false}>
        <View style={{ width: W }}>
          <Svg width={W} height={CHART_H}>
            <Defs>
              <LinearGradient id="fill" x1="0" y1="0" x2="0" y2="1">
                <Stop offset="0" stopColor={colors.accent} stopOpacity="0.22" />
                <Stop offset="1" stopColor={colors.accent} stopOpacity="0.04" />
              </LinearGradient>
            </Defs>
            <Path d={area} fill="url(#fill)" />
            <Path d={line} stroke={colors.accent} strokeWidth={2} fill="none" strokeLinejoin="round" />
            {hours.map((h, i) => <Circle key={i} cx={x(i)} cy={y(h.temp)} r={4.5} fill={colors.bg} stroke={colors.accent} strokeWidth={2} />)}
          </Svg>
          {hours.map((h, i) => (
            <Text key={i} style={[s.temp, { left: x(i) - COL / 2, top: y(h.temp) - 30 }]}>{deg(h.temp)}°</Text>
          ))}
          <View style={s.row}>
            {hours.map((h, i) => (
              <View key={i} style={s.col}>
                <Icon name={h.sky === "rain" ? "rain" : h.sky === "snow" ? "snow" : h.isDay ? "sun" : "moon"} size={26} />
                <Text style={[s.pop, h.pop >= 0.1 && { color: colors.rain }]}>{Math.round(h.pop * 100)}%</Text>
                <Text style={[s.hour, i === 0 && s.hourNow]}>{h.label}</Text>
              </View>
            ))}
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const s = StyleSheet.create({
  wrap: { borderTopWidth: 1.5, borderTopColor: colors.strongRule, marginTop: 20, paddingTop: 0 },
  temp: { position: "absolute", width: COL, textAlign: "center", fontFamily: fonts.sansMedium, fontSize: 18, color: colors.text },
  row: { flexDirection: "row", paddingTop: 12 },
  col: { width: COL, alignItems: "center", gap: 14 },
  pop: { fontFamily: fonts.sans, fontSize: 15, color: colors.muted },
  hour: { fontFamily: fonts.sans, fontSize: 16, color: colors.muted },
  hourNow: { fontFamily: fonts.sansSemi, color: colors.accent },
});
