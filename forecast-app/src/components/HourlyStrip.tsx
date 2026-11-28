import { ScrollView, StyleSheet, Text, View } from "react-native";
import { deg } from "@/data/narrative";
import type { HourEntry } from "@/data/report";
import { fonts, useTheme } from "@/theme/tokens";
import { Icon } from "./Icon";

/** Horizontally scrolling hours: label, sky icon, chance of precipitation (if any), temperature. */
export function HourlyStrip({ hours }: { hours: HourEntry[] }) {
  const t = useTheme();
  return (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ paddingRight: 8 }}>
      {hours.map((h, i) => (
        <View key={i} style={s.col}>
          <Text style={[s.label, { color: t.text }]}>{i === 0 ? "Now" : h.label.replace(/([ap])$/, (m) => (m === "a" ? "AM" : "PM"))}</Text>
          <Icon name={h.sky === "rain" ? "rain" : h.sky === "snow" ? "snow" : h.isDay ? "sun" : "moon"} size={30} />
          <Text style={[s.pop, { color: t.rain, opacity: h.pop >= 0.1 ? 1 : 0 }]}>{Math.round(h.pop * 100)}%</Text>
          <Text style={[s.temp, { color: t.text }]}>{deg(h.temp)}°</Text>
        </View>
      ))}
    </ScrollView>
  );
}

const s = StyleSheet.create({
  col: { width: 62, alignItems: "center", gap: 10 },
  label: { fontFamily: fonts.medium, fontSize: 15 },
  pop: { fontFamily: fonts.semi, fontSize: 12, marginTop: -6 },
  temp: { fontFamily: fonts.medium, fontSize: 19 },
});
