import { Platform, type ViewStyle } from "react-native";

/**
 * Liquid-glass surface: a frosted blur of the sky photo behind it, a bright top rim and a soft drop shadow.
 * The blur is web-only (backdrop-filter); native keeps the tinted translucent fill from the theme.
 */
export const glass: ViewStyle = {
  borderWidth: 1,
  borderColor: "rgba(255,255,255,0.14)",
  boxShadow: "inset 0 1px 0 rgba(255,255,255,0.22), inset 0 -1px 0 rgba(255,255,255,0.04), 0 12px 32px rgba(3,6,25,0.32)",
  ...(Platform.OS === "web"
    ? ({
        backdropFilter: "blur(22px) saturate(165%)",
        WebkitBackdropFilter: "blur(22px) saturate(165%)",
        backgroundImage: "linear-gradient(155deg, rgba(255,255,255,0.13) 0%, rgba(255,255,255,0.03) 38%, rgba(255,255,255,0) 60%)",
      } as ViewStyle)
    : null),
};

/** Smaller glass for pills and buttons. */
export const glassPill: ViewStyle = {
  ...glass,
  boxShadow: "inset 0 1px 0 rgba(255,255,255,0.25), 0 4px 14px rgba(3,6,25,0.25)",
};
