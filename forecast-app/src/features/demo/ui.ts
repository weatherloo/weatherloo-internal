import { StyleSheet } from "react-native";
import { fonts } from "@/theme/tokens";

/** Shared text styles for the training demo (night theme, white on the sky). */
export const ui = StyleSheet.create({
  h3: { color: "#fff", fontFamily: fonts.semi, fontSize: 20 },
  h4: { color: "#fff", fontFamily: fonts.semi, fontSize: 16, marginTop: 4 },
  body: { color: "#fff", fontFamily: fonts.regular, fontSize: 15, lineHeight: 21 },
  muted: { color: "rgba(225,228,255,0.7)", fontFamily: fonts.regular, fontSize: 14, lineHeight: 20 },
  note: { color: "rgba(225,228,255,0.62)", fontFamily: fonts.regular, fontSize: 13, lineHeight: 18 },
  num: { color: "#fff", fontFamily: fonts.medium, fontSize: 14, fontVariant: ["tabular-nums"] },
});
