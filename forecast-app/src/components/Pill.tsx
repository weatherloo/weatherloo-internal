import { StyleSheet, Text, View } from "react-native";
import { colors, fonts } from "@/theme/tokens";
import { Icon, type IconName } from "./Icon";

export function Pill({ icon, label, text, accent }: { icon: IconName; label: string; text: string; accent?: boolean }) {
  return (
    <View style={[s.pill, accent && { backgroundColor: colors.surfaceAccent }]}>
      <Icon name={icon} size={20} color={accent ? colors.accent : colors.muted} />
      <Text style={s.text}>
        <Text style={[s.label, accent && { color: colors.accent }]}>{label} </Text>
        <Text style={{ color: accent ? colors.body : colors.muted }}>{text}</Text>
      </Text>
    </View>
  );
}

const s = StyleSheet.create({
  pill: { flexDirection: "row", alignItems: "center", gap: 14, alignSelf: "flex-start", maxWidth: "100%", backgroundColor: colors.surface, borderRadius: 22, paddingVertical: 14, paddingHorizontal: 18 },
  text: { fontFamily: fonts.sans, fontSize: 16, flexShrink: 1, lineHeight: 22 },
  label: { fontFamily: fonts.sansSemi, color: colors.text },
});
