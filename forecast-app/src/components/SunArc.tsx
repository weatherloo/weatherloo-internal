import Svg, { Line, Path } from "react-native-svg";
import { colors } from "@/theme/tokens";

/** Daylight arc. `progress` (0-1) is not drawn; the arc is static in the reference look. */
export function SunArc() {
  return (
    <Svg width="100%" height={150} viewBox="0 0 340 150" preserveAspectRatio="xMidYMid meet">
      <Line x1="10" y1="120" x2="330" y2="120" stroke={colors.rule} strokeWidth={1.5} />
      <Path d="M30 120 C 30 20, 310 20, 310 120" stroke={colors.accent} strokeWidth={1.5} fill="none" />
    </Svg>
  );
}
