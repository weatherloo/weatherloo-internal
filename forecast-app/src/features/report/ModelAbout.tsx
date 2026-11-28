import { StyleSheet, Text, View } from "react-native";
import { Card } from "@/components/Card";
import type { ModelInfo } from "@/data/registry";
import { fonts, useTheme } from "@/theme/tokens";

/** Write-up of the active model under the forecast: what it is, how it works, and its limits or results. */
export function ModelAbout({ model, wide }: { model: ModelInfo; wide: boolean }) {
  const t = useTheme();
  if (!model.about?.length) return null;
  return (
    <Card>
      <View style={s.head}>
        <Text style={[s.title, { color: t.text }]}>About {model.label}</Text>
        {model.ours ? <Text style={s.ours}>★ Built by our team</Text> : null}
      </View>
      <View style={[s.sections, wide && s.sectionsWide]}>
        {model.about.map((section) => (
          <View key={section.title} style={[s.section, wide && s.sectionWide, { borderTopColor: t.hairline }]}>
            <Text style={[s.sectionTitle, { color: t.text }]}>{section.title}</Text>
            <Text style={[s.body, { color: t.text }]}>{section.body}</Text>
          </View>
        ))}
      </View>
    </Card>
  );
}

const s = StyleSheet.create({
  head: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 },
  title: { fontFamily: fonts.semi, fontSize: 20 },
  ours: {
    color: "#ffd43b", fontFamily: fonts.medium, fontSize: 12, paddingVertical: 3, paddingHorizontal: 9,
    borderRadius: 999, backgroundColor: "rgba(255,212,59,0.12)", overflow: "hidden",
  },
  sections: { marginTop: 8 },
  sectionsWide: { flexDirection: "row", flexWrap: "wrap", columnGap: 28 },
  section: { borderTopWidth: 1, paddingTop: 12, marginTop: 12, gap: 4 },
  sectionWide: { flexBasis: "45%", flexGrow: 1 },
  sectionTitle: { fontFamily: fonts.semi, fontSize: 15 },
  body: { fontFamily: fonts.regular, fontSize: 14, lineHeight: 21, opacity: 0.85 },
});
