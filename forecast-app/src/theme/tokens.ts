import { createContext, useContext } from "react";

/** Design tokens: iOS-Weather-style frosted cards over a night/day sky. */
export interface Theme {
  night: boolean;
  sky: [string, string, string];
  card: string;
  hairline: string;
  text: string;
  muted: string;
  rain: string;
}

export const nightTheme: Theme = {
  night: true,
  sky: ["#0c0f2b", "#1b2252", "#2f3a73"],
  card: "rgba(70, 78, 140, 0.42)",
  hairline: "rgba(255,255,255,0.16)",
  text: "#ffffff",
  muted: "rgba(225,228,255,0.58)",
  rain: "#6ec6ff",
};
export const dayTheme: Theme = {
  night: false,
  sky: ["#2f6fb8", "#4f93d6", "#8cc3ee"],
  card: "rgba(40, 90, 160, 0.34)",
  hairline: "rgba(255,255,255,0.22)",
  text: "#ffffff",
  muted: "rgba(235,245,255,0.7)",
  rain: "#bfe6ff",
};

export const ThemeContext = createContext<Theme>(nightTheme);
export const useTheme = () => useContext(ThemeContext);

export const fonts = {
  thin: "Inter_200ExtraLight",
  light: "Inter_300Light",
  regular: "Inter_400Regular",
  medium: "Inter_500Medium",
  semi: "Inter_600SemiBold",
};
export const MAX_WIDTH = 520;
