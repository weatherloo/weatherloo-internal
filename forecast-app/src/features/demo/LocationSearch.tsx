import { useMemo, useState } from "react";
import { Pressable, StyleSheet, Text, TextInput, View } from "react-native";
import { glass } from "@/theme/glass";
import { fonts } from "@/theme/tokens";

export interface SearchEntry {
  key: string;
  title: string;
  subtitle: string;
  lat: number;
  lon: number;
  /** Set for locations with a finished run to replay. */
  preparedId?: string;
}

const LIMIT = 6;
const COORDS = /^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$/;

/** Search box over the globe: finds stations by name, ID or state, or takes "lat, lon". */
export function LocationSearch({ entries, onPick }: { entries: SearchEntry[]; onPick: (entry: SearchEntry) => void }) {
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const coords = q.match(COORDS);
    if (coords) {
      const lat = Number(coords[1]), lon = Number(coords[2]);
      if (Math.abs(lat) <= 90 && Math.abs(lon) <= 180) {
        return [{ key: "coords", title: `${lat.toFixed(2)}°, ${lon.toFixed(2)}°`, subtitle: "Coordinates", lat, lon }];
      }
    }
    const rank = (e: SearchEntry) => {
      const title = e.title.toLowerCase(), sub = e.subtitle.toLowerCase();
      if (!title.includes(q) && !sub.includes(q)) return -1;
      return (e.preparedId ? 4 : 0) + (title.startsWith(q) ? 2 : 0) + (title.includes(q) ? 1 : 0);
    };
    return entries
      .map((e) => [rank(e), e] as const)
      .filter(([r]) => r >= 0)
      .sort((a, b) => b[0] - a[0] || a[1].title.localeCompare(b[1].title))
      .slice(0, LIMIT)
      .map(([, e]) => e);
  }, [entries, query]);

  const pick = (entry: SearchEntry) => {
    onPick(entry);
    setQuery("");
  };
  const showList = focused && query.trim().length > 0;

  return (
    <View style={s.wrap}>
      <TextInput
        value={query}
        onChangeText={setQuery}
        onFocus={() => setFocused(true)}
        // Delay so a click on a result lands before the list closes.
        onBlur={() => setTimeout(() => setFocused(false), 150)}
        onSubmitEditing={() => results[0] && pick(results[0])}
        placeholder="Search a station, city or lat, lon"
        placeholderTextColor="rgba(225,228,255,0.5)"
        style={s.input}
        accessibilityLabel="Search for a location"
      />
      {showList ? (
        <View style={[s.list, glass]}>
          {results.length ? results.map((r) => (
            <Pressable
              key={r.key}
              onPress={() => pick(r)}
              accessibilityRole="button"
              // `hovered` is react-native-web only.
              style={(state) => [s.row, (state as { hovered?: boolean }).hovered && s.rowHover]}
            >
              <View style={s.flex}>
                <Text style={s.title} numberOfLines={1}>{r.title}</Text>
                <Text style={s.subtitle} numberOfLines={1}>{r.subtitle}</Text>
              </View>
              {r.preparedId ? <Text style={s.tag}>Finished run</Text> : null}
            </Pressable>
          )) : <Text style={s.empty}>No stations match.</Text>}
        </View>
      ) : null}
    </View>
  );
}

const s = StyleSheet.create({
  // zIndex keeps the results above the globe.
  wrap: { zIndex: 10, marginBottom: 10 },
  input: {
    color: "#fff", fontFamily: fonts.regular, fontSize: 15, paddingHorizontal: 14, paddingVertical: 10,
    borderRadius: 14, borderWidth: 1, borderColor: "rgba(255,255,255,0.16)", backgroundColor: "rgba(8,12,40,0.45)",
  },
  list: {
    position: "absolute", top: "100%", left: 0, right: 0, marginTop: 6, borderRadius: 14, padding: 6,
    backgroundColor: "rgba(24,30,78,0.92)",
  },
  row: { flexDirection: "row", alignItems: "center", gap: 10, paddingHorizontal: 10, paddingVertical: 8, borderRadius: 10 },
  rowHover: { backgroundColor: "rgba(255,255,255,0.1)" },
  flex: { flex: 1, minWidth: 0 },
  title: { color: "#fff", fontFamily: fonts.medium, fontSize: 14 },
  subtitle: { color: "rgba(225,228,255,0.6)", fontFamily: fonts.regular, fontSize: 12 },
  tag: {
    color: "#04241a", backgroundColor: "#6ee7b7", fontFamily: fonts.semi, fontSize: 11,
    paddingVertical: 2, paddingHorizontal: 8, borderRadius: 999, overflow: "hidden",
  },
  empty: { color: "rgba(225,228,255,0.6)", fontFamily: fonts.regular, fontSize: 13, padding: 10 },
});
