import { useEffect, useMemo, useRef, useState, type ComponentRef } from "react";
import { Platform, Pressable, StyleSheet, Text, TextInput, View } from "react-native";
import { MODELS, getModel } from "@/data/registry";
import { glass, glassPill } from "@/theme/glass";
import { fonts, useTheme } from "@/theme/tokens";

/** Searchable model dropdown: a glass button showing the active model, opening a filterable list below it. */
export function ModelPicker({ activeModelId, onSelect }: { activeModelId: string; onSelect: (id: string) => void }) {
  const theme = useTheme();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const root = useRef<ComponentRef<typeof View>>(null);
  const active = getModel(activeModelId);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return MODELS;
    return MODELS.filter((m) => [m.label, m.family, m.id, m.description ?? ""].some((f) => f.toLowerCase().includes(q)));
  }, [query]);

  const close = () => { setOpen(false); setQuery(""); };
  const choose = (id: string) => { onSelect(id); close(); };

  // Web: close on Escape or a click outside the picker.
  useEffect(() => {
    if (!open || Platform.OS !== "web") return;
    const node = root.current as unknown as HTMLElement | null;
    const onDown = (e: MouseEvent) => { if (node && !node.contains(e.target as Node)) close(); };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDown); document.removeEventListener("keydown", onKey); };
  }, [open]);

  return (
    <View ref={root} style={s.wrap}>
      <Pressable
        onPress={() => (open ? close() : setOpen(true))}
        accessibilityRole="button"
        accessibilityLabel={`Forecast model: ${active?.label ?? activeModelId}. Change model`}
        accessibilityState={{ expanded: open }}
        style={[s.trigger, glassPill, { backgroundColor: "rgba(255,255,255,0.1)" }]}
      >
        <Text style={[s.triggerLabel, { color: theme.text }]}>
          {active?.label ?? activeModelId}{active?.ours ? <Text style={s.star}> ★</Text> : null}
        </Text>
        <Text style={[s.triggerFamily, { color: theme.muted }]} numberOfLines={1}>{active?.family}</Text>
        <View style={s.demoTag}><Text style={[s.demoText, { color: theme.muted }]}>Demo</Text></View>
        <Text style={[s.chevron, { color: theme.text }]}>{open ? "⌃" : "⌄"}</Text>
      </Pressable>

      {open ? (
        <View style={[s.panel, glass, { backgroundColor: "rgba(28,34,82,0.82)" }]}>
          <TextInput
            autoFocus
            value={query}
            onChangeText={setQuery}
            placeholder="Search models"
            placeholderTextColor={theme.muted}
            style={[s.search, { color: theme.text, borderColor: theme.hairline }]}
            onSubmitEditing={() => matches[0] && choose(matches[0].id)}
            accessibilityLabel="Search forecast models"
          />
          {matches.length ? (
            matches.map((m) => {
              const selected = m.id === activeModelId;
              return (
                <Pressable
                  key={m.id}
                  onPress={() => choose(m.id)}
                  accessibilityRole="button"
                  accessibilityState={{ selected }}
                  // `hovered` is react-native-web only.
                  style={(state) => [s.option, ((state as { hovered?: boolean }).hovered || selected) && s.optionActive]}
                >
                  <View style={s.optionText}>
                    <Text style={[s.optionLabel, { color: theme.text }]}>
                      {m.label}{m.ours ? <Text style={s.star} accessibilityLabel="built by our team"> ★</Text> : null}
                    </Text>
                    <Text style={[s.optionFamily, { color: theme.muted }]}>{m.family}</Text>
                  </View>
                  {selected ? <Text style={[s.check, { color: theme.text }]}>✓</Text> : null}
                </Pressable>
              );
            })
          ) : (
            <Text style={[s.empty, { color: theme.muted }]}>No models match “{query}”.</Text>
          )}
          <Text style={[s.footnote, { color: theme.muted }]}><Text style={s.star}>★</Text> Built by our team · all forecasts are demo fixtures.</Text>
        </View>
      ) : null}
    </View>
  );
}

const s = StyleSheet.create({
  // zIndex keeps the open panel above the cards that follow.
  wrap: { zIndex: 20, alignSelf: "flex-start" },
  trigger: { flexDirection: "row", alignItems: "center", gap: 8, borderRadius: 18, paddingLeft: 14, paddingRight: 12, paddingVertical: 9 },
  triggerLabel: { fontFamily: fonts.semi, fontSize: 14 },
  triggerFamily: { fontFamily: fonts.regular, fontSize: 12, maxWidth: 160 },
  demoTag: { borderRadius: 8, borderWidth: 1, borderColor: "rgba(255,255,255,0.2)", paddingHorizontal: 7, paddingVertical: 1 },
  demoText: { fontFamily: fonts.medium, fontSize: 11 },
  chevron: { fontFamily: fonts.medium, fontSize: 14, marginLeft: 2 },
  panel: { position: "absolute", top: "100%", left: 0, marginTop: 8, width: 300, borderRadius: 18, padding: 8, gap: 2 },
  search: { fontFamily: fonts.regular, fontSize: 14, borderWidth: 1, borderRadius: 12, paddingHorizontal: 12, paddingVertical: 9, marginBottom: 6 },
  option: { flexDirection: "row", alignItems: "center", borderRadius: 12, paddingHorizontal: 10, paddingVertical: 8 },
  optionActive: { backgroundColor: "rgba(255,255,255,0.12)" },
  optionText: { flex: 1, gap: 1 },
  optionLabel: { fontFamily: fonts.medium, fontSize: 14 },
  optionFamily: { fontFamily: fonts.regular, fontSize: 12 },
  check: { fontFamily: fonts.semi, fontSize: 14 },
  star: { color: "#ffd43b" },
  empty: { fontFamily: fonts.regular, fontSize: 13, padding: 10 },
  footnote: { fontFamily: fonts.regular, fontSize: 11, paddingHorizontal: 10, paddingTop: 6, paddingBottom: 4 },
});
