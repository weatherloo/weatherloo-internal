import Svg, { Circle, G, Path } from "react-native-svg";
import { colors } from "@/theme/tokens";

export type IconName = "sun" | "moon" | "rain" | "snow" | "umbrella" | "shirt" | "cloud";

/** Thin line icons, 24x24 grid. */
export function Icon({ name, size = 24, color = colors.accent, stroke = 1.5 }: {
  name: IconName; size?: number; color?: string; stroke?: number;
}) {
  const p = { stroke: color, strokeWidth: stroke, fill: "none", strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  const cloud = "M7 16a4 4 0 0 1-.5-7.97A5.5 5.5 0 0 1 17 9.5a3.25 3.25 0 0 1 0 6.5H7z";
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      <G {...p}>
        {name === "sun" && (<>
          <Circle cx="12" cy="12" r="4.2" />
          <Path d="M12 2.5v2.4M12 19.1v2.4M2.5 12h2.4M19.1 12h2.4M5.3 5.3l1.7 1.7M17 17l1.7 1.7M5.3 18.7L7 17M17 7l1.7-1.7" />
        </>)}
        {name === "moon" && <Path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z" />}
        {name === "cloud" && <Path d={cloud} />}
        {name === "rain" && (<><Path d={cloud} transform="translate(0 -3)" /><Path d="M8 18.5l-1 2M12 18.5l-1 2M16 18.5l-1 2" /></>)}
        {name === "snow" && (<><Path d={cloud} transform="translate(0 -3)" /><Path d="M8 19h.01M12 20.5h.01M16 19h.01M10 22h.01M14 22h.01" strokeWidth={2.2} /></>)}
        {name === "umbrella" && (<><Path d="M3 12a9 9 0 0 1 18 0H3z" /><Path d="M12 12v6.5a2 2 0 0 1-4 0" /></>)}
        {name === "shirt" && <Path d="M8.5 3.5L3 7l2.2 3.4L7.5 9.5V20.5h9V9.5l2.3.9L21 7l-5.5-3.5a3.6 3.6 0 0 1-7 0z" />}
      </G>
    </Svg>
  );
}
