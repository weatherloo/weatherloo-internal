import { Text, type TextProps, type TextStyle, StyleSheet } from "react-native";
import { colors, fonts } from "@/theme/tokens";
import type { Run } from "@/data/narrative";

/** Small lowercase letterspaced sans line (meta, captions). */
export function Meta({ style, ...p }: TextProps) {
  return <Text {...p} style={[s.meta, style]} />;
}

/** Serif section heading ("Hour by hour"). */
export function SectionTitle({ children, aside }: { children: string; aside?: string }) {
  return (
    <>
      <Text style={s.h2}>{children}</Text>
      {aside ? <Meta style={{ marginTop: 14 }}>{aside}</Meta> : null}
    </>
  );
}

/** Renders narrative runs: plain, em (italic) or accent (italic terracotta). */
export function Runs({ runs, style }: { runs: Run[]; style: TextStyle }) {
  return (
    <Text style={style}>
      {runs.map((r, i) => (
        <Text key={i} style={r.accent ? { color: colors.accent, fontFamily: fonts.serifItalic } : r.em ? { fontFamily: fonts.serifItalic, color: colors.text } : undefined}>
          {r.text}
        </Text>
      ))}
    </Text>
  );
}

const s = StyleSheet.create({
  meta: { fontFamily: fonts.sans, fontSize: 15, letterSpacing: 1.2, color: colors.muted, lineHeight: 24 },
  h2: { fontFamily: fonts.serif, fontSize: 36, color: colors.text, letterSpacing: -0.3 },
});
