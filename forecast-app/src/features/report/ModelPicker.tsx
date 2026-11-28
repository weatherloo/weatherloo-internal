import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { MODELS } from "@/data/registry";
import { fonts, useTheme } from "@/theme/tokens";

export function ModelPicker({ activeModelId, onSelect }: { activeModelId: string; onSelect: (id: string) => void }) {
  const theme = useTheme();
  return (
    <View style={s.wrap}>
      <View style={s.heading}>
        <Text style={[s.label, { color: theme.muted }]}>FORECAST MODEL</Text>
        <Text style={[s.demo, { color: theme.muted }]}>DEMO FIXTURES</Text>
      </View>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={s.options}>
        {MODELS.map((model) => {
          const selected = model.id === activeModelId;
          return (
            <Pressable
              key={model.id}
              onPress={() => onSelect(model.id)}
              accessibilityRole="button"
              accessibilityState={{ selected }}
              style={[
                s.option,
                { borderColor: theme.hairline, backgroundColor: selected ? "rgba(255,255,255,0.22)" : "rgba(255,255,255,0.07)" },
              ]}
            >
              <Text style={[s.optionText, { color: theme.text, opacity: selected ? 1 : 0.72 }]}>{model.label}</Text>
            </Pressable>
          );
        })}
      </ScrollView>
    </View>
  );
}

const s = StyleSheet.create({
  wrap: { gap: 8, marginBottom: 8 },
  heading: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 2 },
  label: { fontFamily: fonts.semi, fontSize: 11, letterSpacing: 1.1 },
  demo: { fontFamily: fonts.medium, fontSize: 10, letterSpacing: 0.8 },
  options: { gap: 8, paddingRight: 4 },
  option: { borderWidth: 1, borderRadius: 18, paddingHorizontal: 14, paddingVertical: 9 },
  optionText: { fontFamily: fonts.medium, fontSize: 13 },
});
