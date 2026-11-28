import { StyleSheet, Text, View, type StyleProp, type ViewStyle } from "react-native";
import { glass } from "@/theme/glass";
import { fonts, useTheme } from "@/theme/tokens";
import { Icon, type IconName } from "./Icon";

/** Liquid-glass card with an optional sentence-case icon header. */
export function Card({ title, icon, children, style }: { title?: string; icon?: IconName; children: React.ReactNode; style?: StyleProp<ViewStyle> }) {
  const t = useTheme();
  return (
    <View style={[s.card, glass, { backgroundColor: t.card }, style]}>
      {title ? (
        <View style={s.head}>
          {icon ? <Icon name={icon} size={16} color={t.muted} /> : null}
          <Text style={[s.title, { color: t.muted }]}>{title}</Text>
        </View>
      ) : null}
      {children}
    </View>
  );
}

const s = StyleSheet.create({
  card: { borderRadius: 22, padding: 16, overflow: "hidden" },
  head: { flexDirection: "row", alignItems: "center", gap: 7, marginBottom: 12 },
  title: { fontFamily: fonts.medium, fontSize: 14 },
});
