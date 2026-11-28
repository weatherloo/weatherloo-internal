import { useWindowDimensions, type ViewStyle } from "react-native";
import { MAX_WIDTH } from "./tokens";

/** Width at which screens switch from the single phone column to the desktop grid. */
export const WIDE_BREAKPOINT = 960;
export const WIDE_MAX_WIDTH = 1180;

export function useWide() {
  return useWindowDimensions().width >= WIDE_BREAKPOINT;
}

/** True once the desktop page has reached its full width (room for denser rows). */
export function useRoomy() {
  return useWindowDimensions().width >= WIDE_MAX_WIDTH;
}

/** The centred page column: phone width by default, wider with more gutter on desktop. */
export function pageStyle(wide: boolean): ViewStyle {
  return wide
    ? { width: "100%", maxWidth: WIDE_MAX_WIDTH, paddingHorizontal: 32, gap: 16 }
    : { width: "100%", maxWidth: MAX_WIDTH, paddingHorizontal: 16, gap: 12 };
}
