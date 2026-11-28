import { StyleSheet, View } from "react-native";
import Svg, { Circle, Defs, LinearGradient, Rect, Stop } from "react-native-svg";
import { useTheme } from "@/theme/tokens";

// Deterministic star field (x, y in 0-100, radius, opacity).
const STARS = Array.from({ length: 70 }, (_, i) => {
  const r = (n: number) => { const x = Math.sin(i * 12.9898 + n * 78.233) * 43758.5453; return x - Math.floor(x); };
  return { x: r(1) * 100, y: r(2) * 70, r: 0.5 + r(3) * 1.1, o: 0.35 + r(4) * 0.65 };
});

/** Fixed full-bleed sky behind the scrolling report: stars and a moon at night, plain gradient by day. */
export function SkyBackground() {
  const t = useTheme();
  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      <Svg width="100%" height="100%" preserveAspectRatio="none" viewBox="0 0 100 100">
        <Defs>
          <LinearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0" stopColor={t.sky[0]} /><Stop offset="0.55" stopColor={t.sky[1]} /><Stop offset="1" stopColor={t.sky[2]} />
          </LinearGradient>
        </Defs>
        <Rect width="100" height="100" fill="url(#sky)" />
      </Svg>
      {t.night && (
        <Svg style={StyleSheet.absoluteFill} width="100%" height="100%" preserveAspectRatio="xMidYMid slice" viewBox="0 0 100 100">
          {STARS.map((s, i) => <Circle key={i} cx={s.x} cy={s.y} r={s.r * 0.09} fill="#fff" opacity={s.o} />)}
        </Svg>
      )}
    </View>
  );
}
