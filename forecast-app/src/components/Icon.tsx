import Svg, { Circle, G, Path } from "react-native-svg";

export type IconName =
  | "sun" | "moon" | "rain" | "snow" | "cloud" | "thermometer" | "wind" | "sunrise" | "drop" | "calendar" | "clock" | "nav";

const SUN = "#ffd43b", CLOUD = "#f2f4fb", RAIN = "#6ec6ff";

/** Colored weather glyphs (sun/moon/rain/snow/cloud) and monochrome card-header glyphs. */
export function Icon({ name, size = 24, color = "currentColor" }: { name: IconName; size?: number; color?: string }) {
  const line = { stroke: color, strokeWidth: 1.8, fill: "none", strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  const cloudPath = "M7 17a4.2 4.2 0 0 1-.6-8.36A5.6 5.6 0 0 1 17.2 9.6 3.7 3.7 0 0 1 17 17H7z";
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      {name === "sun" && (<>
        <Circle cx="12" cy="12" r="4.6" fill={SUN} />
        <G stroke={SUN} strokeWidth={2} strokeLinecap="round">
          <Path d="M12 2.5v2.6M12 18.9v2.6M2.5 12h2.6M18.9 12h2.6M5.3 5.3l1.8 1.8M16.9 16.9l1.8 1.8M5.3 18.7l1.8-1.8M16.9 7.1l1.8-1.8" />
        </G>
      </>)}
      {name === "moon" && (<>
        <Path d="M19.5 14.2A8 8 0 0 1 9.8 4.5a8 8 0 1 0 9.7 9.7z" fill={CLOUD} />
        <Path d="M17.5 3v3M16 4.5h3" stroke={CLOUD} strokeWidth={1.4} strokeLinecap="round" />
      </>)}
      {name === "cloud" && <Path d={cloudPath} fill={CLOUD} />}
      {name === "rain" && (<>
        <Path d={cloudPath} fill={CLOUD} transform="translate(0 -3)" />
        <Path d="M8 17.5l-1 2.5M12 17.5l-1 2.5M16 17.5l-1 2.5" stroke={RAIN} strokeWidth={1.8} strokeLinecap="round" />
      </>)}
      {name === "snow" && (<>
        <Path d={cloudPath} fill={CLOUD} transform="translate(0 -3)" />
        <G fill={RAIN}><Circle cx="8" cy="18.5" r="1.2" /><Circle cx="12" cy="20.5" r="1.2" /><Circle cx="16" cy="18.5" r="1.2" /></G>
      </>)}
      {name === "thermometer" && (<G {...line}><Path d="M10 14.5V5a2 2 0 0 1 4 0v9.5a4 4 0 1 1-4 0z" /></G>)}
      {name === "wind" && (<G {...line}><Path d="M3 9h10a2.5 2.5 0 1 0-2.5-2.5M3 14h14a2.5 2.5 0 1 1-2.5 2.5M3 19h6" /></G>)}
      {name === "sunrise" && (<G {...line}><Path d="M3 18h18M7 18a5 5 0 0 1 10 0M12 6v4M8.5 8l1.2 1.2M15.5 8l-1.2 1.2" /></G>)}
      {name === "drop" && (<G {...line}><Path d="M12 3.5s6 6.2 6 10.5a6 6 0 0 1-12 0c0-4.3 6-10.5 6-10.5z" /></G>)}
      {name === "calendar" && (<G {...line}><Path d="M4 6.5h16V20H4zM4 10.5h16M8 4v4M16 4v4" /></G>)}
      {name === "clock" && (<G {...line}><Circle cx="12" cy="12" r="8.5" /><Path d="M12 7v5l3 2" /></G>)}
      {name === "nav" && (<G fill={color}><Path d="M21 3L3 10.5l7.2 2.3L12.5 21z" /></G>)}
    </Svg>
  );
}
