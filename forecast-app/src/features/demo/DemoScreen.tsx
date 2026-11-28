import { useMemo, useState } from "react";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { Link } from "expo-router";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Card } from "@/components/Card";
import { SkyBackground } from "@/components/SkyBackground";
import stationData from "@/data/demo/stations.json";
import demo from "@/data/demo/locations.json";
import { fonts, MAX_WIDTH, nightTheme, ThemeContext } from "@/theme/tokens";
import { STATION_RADIUS_KM, distanceKm, hrrrDomain, runChecks, type LatLon, type Station } from "./checks";
import { Globe } from "./Globe";
import { TrainRun } from "./TrainRun";
import { ui } from "./ui";
import { Walkthrough, type DemoLocation, type DemoMeta } from "./Walkthrough";

const DOMAIN = hrrrDomain();
const PREPARED = demo.locations as unknown as DemoLocation[];
const META = demo as unknown as DemoMeta;
const PIN_LABELS: Record<string, string> = { eric_d_soulis: "Waterloo", cyyz: "Pearson", sfo: "SFO" };
type Row = [string, string, number, number, string, string];
const STATIONS: Station[] = [
  ...PREPARED.map((l) => ({ id: l.station_id, name: l.station_name, lat: l.lat, lon: l.lon, begin: l.begin })),
  ...(stationData.stations as Row[]).map(([id, name, lat, lon, begin, network]) => ({ id, name, lat, lon, begin, network })),
];

// A station within 2 km of a prepared target is the same site (e.g. IEM's CYYZ vs ECCC 51459).
const preparedFor = (station: Station | null) => station && PREPARED.find((l) => distanceKm(l, station) < 2);

const ICON = { pass: "✓", fail: "✕", skip: "–" } as const;
const ICON_BG = { pass: "#10b981", fail: "#dc2626", skip: "rgba(255,255,255,0.2)" } as const;

/** Training demo: pick a spot on the globe, check it, then replay or plan an hrrr_cnn training run. */
export function DemoScreen() {
  const insets = useSafeAreaInsets();
  const [tap, setTap] = useState<LatLon | null>(null);
  const [focus, setFocus] = useState<LatLon | null>(null);
  const [open, setOpen] = useState<{ loc: DemoLocation; km: number | null } | null>(null);

  const result = useMemo(() => (tap ? runChecks(tap, STATIONS) : null), [tap]);
  const prepared = result?.ok ? preparedFor(result.station) : null;

  const pickPrepared = (id: string) => {
    const loc = PREPARED.find((l) => l.id === id)!;
    setTap({ lat: loc.lat, lon: loc.lon });
    setFocus({ lat: loc.lat, lon: loc.lon });
    setOpen(null);
  };

  return (
    <ThemeContext.Provider value={nightTheme}>
      <View style={s.root}>
        <SkyBackground />
        <ScrollView contentContainerStyle={[s.content, { paddingTop: insets.top + 20, paddingBottom: insets.bottom + 40 }]}>
          <View style={s.page}>
            <View style={s.topBar}>
              <Link href="/" asChild>
                <Text style={s.backLink}>‹  Forecast</Text>
              </Link>
            </View>
            <Text style={s.title}>Training demo</Text>
            <Text style={s.subtitle}>
              Pick a spot inside HRRR coverage and step through how we train a station forecast correction: download,
              preprocessing, training and validation.
            </Text>

            <Card title="Pick a location" icon="nav">
              <Globe
                domain={DOMAIN}
                pins={PREPARED.map((l) => ({
                  id: l.id, label: PIN_LABELS[l.id] ?? l.label, lat: l.lat, lon: l.lon,
                  left: l.id === "eric_d_soulis", // Waterloo and Pearson are 75 km apart; keep labels apart
                }))}
                tap={tap}
                station={result?.station ?? null}
                radiusKm={STATION_RADIUS_KM}
                tapOk={result?.ok ?? true}
                focus={focus}
                onTap={(p) => { setTap(p); setOpen(null); }}
                onPin={pickPrepared}
              />
              <View style={s.legend}>
                <Legend color="#67e8f9" label="HRRR coverage" />
                <Legend color="#10b981" label="Prepared run" />
                <Legend color="#10b981" dashed label={`${STATION_RADIUS_KM} km station search`} />
              </View>
            </Card>

            {open ? (
              <Walkthrough loc={open.loc} km={open.km} meta={META} onClose={() => setOpen(null)} />
            ) : (
              <Card title="Train a forecast correction">
                <Text style={ui.muted}>
                  Tap anywhere on the globe. We check that the spot can be trained on, then step through the pipeline:
                  download, preprocessing, training and validation.
                </Text>

                {result && tap ? (
                  <View style={s.checks}>
                    <Text style={ui.h4}>{tap.lat.toFixed(2)}°, {tap.lon.toFixed(2)}°</Text>
                    {result.steps.map((st) => (
                      <View key={st.id} style={[s.check, st.status === "skip" && s.skip]}
                        accessibilityLabel={`${st.label}: ${st.status}. ${st.detail}`}>
                        <Text style={[s.checkIcon, { backgroundColor: ICON_BG[st.status] }]}>{ICON[st.status]}</Text>
                        <View style={s.flex}>
                          <Text style={[ui.body, s.strong]}>{st.label}</Text>
                          <Text style={ui.muted}>{st.detail}</Text>
                        </View>
                      </View>
                    ))}
                    {!result.ok ? (
                      <Text style={[ui.body, s.bad]}>This spot can&apos;t be trained on. Try another.</Text>
                    ) : prepared ? (
                      <>
                        <Text style={[ui.body, s.good]}>Ready: {prepared.label} has a prepared run.</Text>
                        <Pressable accessibilityRole="button" style={s.primary}
                          onPress={() => setOpen({ loc: prepared, km: distanceKm(tap, prepared) < 1 ? null : result.km })}>
                          <Text style={s.primaryText}>Start training walkthrough</Text>
                        </Pressable>
                      </>
                    ) : (
                      <TrainRun station={result.station!} stations={STATIONS} />
                    )}
                  </View>
                ) : null}

                <Text style={[ui.h4, s.subhead]}>Prepared runs</Text>
                {PREPARED.map((l) => (
                  <Pressable key={l.id} accessibilityRole="button" style={s.preparedRow} onPress={() => pickPrepared(l.id)}>
                    <Text style={[ui.body, s.strong]}>{l.label}</Text>
                    <Text style={ui.muted}>{l.lat.toFixed(2)}°, {l.lon.toFixed(2)}°</Text>
                  </Pressable>
                ))}
              </Card>
            )}
          </View>
        </ScrollView>
      </View>
    </ThemeContext.Provider>
  );
}

function Legend({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <View style={s.legendItem}>
      <View style={[s.legendKey, { borderColor: color, backgroundColor: dashed ? "transparent" : color }, dashed && s.dashed]} />
      <Text style={ui.note}>{label}</Text>
    </View>
  );
}

const s = StyleSheet.create({
  root: { flex: 1 },
  content: { alignItems: "center" },
  page: { width: "100%", maxWidth: MAX_WIDTH, paddingHorizontal: 16, gap: 12 },
  topBar: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  backLink: { color: "#fff", fontFamily: fonts.medium, fontSize: 14, paddingVertical: 6 },
  title: { color: "#fff", fontFamily: fonts.semi, fontSize: 34, letterSpacing: -0.5, marginTop: 4 },
  subtitle: { color: "rgba(225,228,255,0.7)", fontFamily: fonts.regular, fontSize: 15, lineHeight: 21, marginTop: -8 },
  legend: { flexDirection: "row", flexWrap: "wrap", gap: 12, marginTop: 10 },
  legendItem: { flexDirection: "row", alignItems: "center", gap: 6 },
  legendKey: { width: 12, height: 12, borderRadius: 6, borderWidth: 1.5 },
  dashed: { borderStyle: "dashed" },
  checks: { gap: 10, marginTop: 14 },
  check: { flexDirection: "row", gap: 10, alignItems: "flex-start" },
  skip: { opacity: 0.55 },
  checkIcon: {
    width: 22, height: 22, borderRadius: 11, overflow: "hidden", textAlign: "center", lineHeight: 22,
    color: "#fff", fontFamily: fonts.semi, fontSize: 12,
  },
  flex: { flex: 1 },
  strong: { fontFamily: fonts.semi },
  good: { color: "#6ee7b7", fontFamily: fonts.semi },
  bad: { color: "#ffb4ab", fontFamily: fonts.semi },
  primary: { alignSelf: "flex-start", paddingVertical: 10, paddingHorizontal: 16, borderRadius: 999, backgroundColor: "#10b981" },
  primaryText: { color: "#04241a", fontFamily: fonts.semi, fontSize: 15 },
  subhead: { marginTop: 16 },
  preparedRow: {
    flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingVertical: 10,
    borderTopWidth: 1, borderTopColor: "rgba(255,255,255,0.12)",
  },
});
