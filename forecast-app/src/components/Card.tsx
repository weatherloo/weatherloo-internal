import { StyleSheet, Text, View, type ViewStyle } from "react-native";
import { fonts, useTheme } from "@/theme/tokens";
import { Icon, type IconName } from "./Icon";

/** Frosted card with an uppercase icon header, as in iOS Weather. */
export function Card({ title, icon, children, style }: { title?: string; icon?: IconName; children: React.ReactNode; style?: ViewStyle }) {
  const t = useTheme();
  return (
    <View style={[s.card, { backgroundColor: t.card }, style]}>
      {title ? (
        <View style={s.head}>
          {icon ? <Icon name={icon} size={16} color={t.muted} /> : null}
          <Text style={[s.title, { color: t.muted }]}>{title.toUpperCase()}</Text>
        </View>
      ) : null}
      {children}
    </View>
  );
}

const s = StyleSheet.create({
  card: { borderRadius: 22, padding: 16, overflow: "hidden" },
  head: { flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 12 },
  title: { fontFamily: fonts.medium, fontSize: 13, letterSpacing: 0.6 },
});
