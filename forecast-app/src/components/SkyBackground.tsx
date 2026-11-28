import { Image, StyleSheet, View } from "react-native";
import Svg, { Circle, Defs, LinearGradient, Rect, Stop } from "react-native-svg";
import { useTheme } from "@/theme/tokens";
import { useSvgId } from "./svgId";

// Davis Centre bridge, University of Waterloo. Photo: Victor Vucicevich, CC BY-SA 3.0 (see README).
// Pre-toned to a dark-blue duotone; the sky gradient on top keeps it faint.
const CAMPUS = require("../../assets/images/campus-davis-centre.jpg");

// Deterministic star field (x, y in 0-100, radius, opacity).
const STARS = Array.from({ length: 70 }, (_, i) => {
  const r = (n: number) => { const x = Math.sin(i * 12.9898 + n * 78.233) * 43758.5453; return x - Math.floor(x); };
  return { x: r(1) * 100, y: r(2) * 55, r: 0.5 + r(3) * 1.1, o: 0.35 + r(4) * 0.65 };
});

/** Fixed full-bleed sky behind the scrolling content: a faint campus photo under the night/day gradient, plus stars at night. */
export function SkyBackground() {
  const t = useTheme();
  const skyId = useSvgId("sky");
  return (
    <View style={[StyleSheet.absoluteFill, { backgroundColor: t.sky[0] }]} pointerEvents="none">
      <Image source={CAMPUS} resizeMode="cover" style={[StyleSheet.absoluteFill, s.photo, { opacity: t.night ? 0.26 : 0.22 }]} />
      <Svg style={StyleSheet.absoluteFill} width="100%" height="100%" preserveAspectRatio="none" viewBox="0 0 100 100">
        <Defs>
          <LinearGradient id={skyId} x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0" stopColor={t.sky[0]} stopOpacity={0.96} />
            <Stop offset="0.5" stopColor={t.sky[1]} stopOpacity={0.84} />
            <Stop offset="1" stopColor={t.sky[2]} stopOpacity={0.88} />
          </LinearGradient>
        </Defs>
        <Rect width="100" height="100" fill={`url(#${skyId})`} />
      </Svg>
      {t.night && (
        <Svg style={StyleSheet.absoluteFill} width="100%" height="100%" preserveAspectRatio="xMidYMid slice" viewBox="0 0 100 100">
          {STARS.map((s, i) => <Circle key={i} cx={s.x} cy={s.y} r={s.r * 0.09} fill="#fff" opacity={s.o} />)}
        </Svg>
      )}
    </View>
  );
}

const s = StyleSheet.create({
  photo: { width: "100%", height: "100%" },
});
