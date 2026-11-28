import { useRef, useState, type ComponentRef } from "react";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { deg } from "@/data/narrative";
import type { HourEntry } from "@/data/report";
import { glassPill } from "@/theme/glass";
import { fonts, useTheme } from "@/theme/tokens";
import { Icon } from "./Icon";

const COL = 62;

/**
 * Horizontally scrolling hours: label, sky icon, chance of precipitation (if any), temperature.
 * With `arrows` (desktop, where a mouse wheel can't scroll sideways) it adds page buttons at the edges.
 */
export function HourlyStrip({ hours, arrows }: { hours: HourEntry[]; arrows?: boolean }) {
  const t = useTheme();
  const ref = useRef<ComponentRef<typeof ScrollView>>(null);
  const [view, setView] = useState({ x: 0, width: 0, content: 0 });
  const page = (dir: 1 | -1) => {
    const step = Math.max(COL, Math.floor((view.width - COL) / COL) * COL);
    ref.current?.scrollTo({ x: Math.max(0, view.x + dir * step), animated: true });
  };
  const canBack = view.x > 4, canForward = view.x + view.width < view.content - 4;

  return (
    <View>
      <ScrollView
        ref={ref}
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={{ paddingRight: 8 }}
        scrollEventThrottle={32}
        onLayout={(e) => { const width = e.nativeEvent.layout.width; setView((v) => ({ ...v, width })); }}
        onContentSizeChange={(content) => setView((v) => ({ ...v, content }))}
        onScroll={(e) => { const x = e.nativeEvent.contentOffset.x; setView((v) => ({ ...v, x })); }}
      >
        {hours.map((h, i) => (
          <View key={i} style={s.col}>
            <Text style={[s.label, { color: t.text }]}>{i === 0 ? "Now" : h.label.replace(/([ap])$/, (m) => (m === "a" ? "AM" : "PM"))}</Text>
            <Icon name={h.sky === "rain" ? "rain" : h.sky === "snow" ? "snow" : h.isDay ? "sun" : "moon"} size={30} />
            <Text style={[s.pop, { color: t.rain, opacity: h.pop >= 0.1 ? 1 : 0 }]}>{Math.round(h.pop * 100)}%</Text>
            <Text style={[s.temp, { color: t.text }]}>{deg(h.temp)}°</Text>
          </View>
        ))}
      </ScrollView>
      {arrows && canBack ? <Arrow side="left" onPress={() => page(-1)} /> : null}
      {arrows && canForward ? <Arrow side="right" onPress={() => page(1)} /> : null}
    </View>
  );
}

function Arrow({ side, onPress }: { side: "left" | "right"; onPress: () => void }) {
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={side === "left" ? "Earlier hours" : "Later hours"}
      // `hovered` is react-native-web only.
      style={(state) => [s.arrow, glassPill, side === "left" ? s.left : s.right, (state as { hovered?: boolean }).hovered && s.arrowHover]}
    >
      <Text style={s.arrowText}>{side === "left" ? "‹" : "›"}</Text>
    </Pressable>
  );
}

const s = StyleSheet.create({
  col: { width: COL, alignItems: "center", gap: 10 },
  label: { fontFamily: fonts.medium, fontSize: 15 },
  pop: { fontFamily: fonts.semi, fontSize: 12, marginTop: -6 },
  temp: { fontFamily: fonts.medium, fontSize: 19 },
  arrow: {
    position: "absolute", top: "50%", marginTop: -16, width: 32, height: 32, borderRadius: 16,
    alignItems: "center", justifyContent: "center", backgroundColor: "rgba(40,48,110,0.55)",
  },
  arrowHover: { backgroundColor: "rgba(70,80,150,0.75)" },
  left: { left: -6 },
  right: { right: -6 },
  arrowText: { color: "#fff", fontFamily: fonts.medium, fontSize: 22, lineHeight: 24, marginTop: -2 },
});
