import { useEffect, useRef, useState, type ComponentRef, type ReactNode } from "react";
import { AccessibilityInfo, Animated, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { Card } from "@/components/Card";
import { glassPill } from "@/theme/glass";
import { fonts } from "@/theme/tokens";
import { Button, LinkButton } from "./TrainRun";
import { ui } from "./ui";

/** Pipeline stages, replayed in order; each one adds a card to the carousel. */
const STAGES = [
  { id: "location", label: "Location", short: "Location", ms: 1400 },
  { id: "download", label: "Download", short: "Download", ms: 2600 },
  { id: "preprocess", label: "Preprocess", short: "Prep", ms: 2000 },
  { id: "train", label: "Train", short: "Train", ms: 2600 },
  { id: "validate", label: "Validate", short: "Validate", ms: 1800 },
] as const;
const PAUSE_MS = 450;

const METHOD_LABELS: Record<string, string> = {
  hrrr_cnn: "Ours (HRRR + CNN)",
  hrrr_interpolated: "Raw HRRR",
  ecmwf_aifs: "ECMWF AIFS",
  graphcast: "GraphCast",
};
const GREEN = "#10b981", MINT = "#6ee7b7", GREY = "#94a3b8";

interface VarResult {
  unit: string; cnn: number; raw_hrrr: number; leads: number[];
  cnn_by_lead: (number | null)[]; raw_by_lead: (number | null)[] | null;
}
export interface DemoLocation {
  id: string; label: string; station_id: string; station_name: string; lat: number; lon: number; begin: string;
  region: string; region_stations: number; obs_source: string;
  results: { t2m: VarResult; wind_speed: VarResult };
  matched_6h: { t2m: Record<string, number>; wind_speed: Record<string, number> } | null;
}
export interface DemoMeta {
  split: { train: [string, string]; validate: [string, string]; test: [string, string] };
  months_downloaded: number;
}

/**
 * Replays a precomputed training run: a progress bar across the five stages, and a carousel that gains a card
 * per stage and moves to it automatically. Once every stage is done the run shows as ready.
 */
export function Walkthrough({ loc, km, meta, onClose }: { loc: DemoLocation; km: number | null; meta: DemoMeta; onClose: () => void }) {
  const [stage, setStage] = useState(0); // stage being replayed; STAGES.length once finished
  const [progress, setProgress] = useState(0);
  const [view, setView] = useState(0);
  const [width, setWidth] = useState(0);
  const scroller = useRef<ComponentRef<typeof ScrollView>>(null);
  const done = stage >= STAGES.length;
  const cards = Math.min(stage + 1, STAGES.length);

  useEffect(() => {
    if (done) return;
    let frame = 0, cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    AccessibilityInfo.isReduceMotionEnabled().catch(() => false).then((reduce) => {
      if (cancelled) return;
      if (reduce) { setProgress(1); setStage(STAGES.length); return; }
      const start = Date.now(), ms = STAGES[stage].ms;
      const tick = () => {
        const p = Math.min(1, (Date.now() - start) / ms);
        setProgress(p);
        if (p < 1) frame = requestAnimationFrame(tick);
        else timer = setTimeout(() => { setProgress(0); setStage((v) => v + 1); }, PAUSE_MS);
      };
      frame = requestAnimationFrame(tick);
    });
    return () => { cancelled = true; cancelAnimationFrame(frame); if (timer) clearTimeout(timer); };
  }, [stage, done]);

  const goTo = (i: number, animated = true) => {
    setView(i);
    scroller.current?.scrollTo({ x: i * width, animated });
  };
  // Follow the run: each new stage's card slides into view.
  useEffect(() => { if (!done && width) goTo(stage); }, [stage, width]); // eslint-disable-line react-hooks/exhaustive-deps

  const stageProgress = (i: number) => (i < stage ? 1 : i === stage ? progress : 0);
  const status = done ? `Ready · ${loc.label}` : `${STAGES[stage].label}…`;

  const body = (i: number) => {
    const p = stageProgress(i);
    switch (STAGES[i].id) {
      case "location": return <LocationCard loc={loc} km={km} />;
      case "download": return <DownloadCard months={meta.months_downloaded} p={p} />;
      case "preprocess": return <PreprocessCard split={meta.split} p={p} />;
      case "train": return <TrainCard p={p} />;
      default: return <ValidateCard loc={loc} p={p} />;
    }
  };

  return (
    <Card>
      <View style={s.top}>
        <View style={s.flex}>
          <Text style={s.replayTag}>Replay of a completed run</Text>
          <Text style={ui.h3} accessibilityLiveRegion="polite">{done ? "✓ " : ""}{status}</Text>
        </View>
        <LinkButton label="Back to globe" onPress={onClose} />
      </View>

      <View style={s.stepper} accessibilityLabel="Pipeline progress">
        {STAGES.map((st, i) => {
          const p = stageProgress(i), finished = p >= 1;
          return (
            <Pressable key={st.id} style={s.step} disabled={i >= cards} onPress={() => goTo(i)}
              accessibilityRole="button" accessibilityLabel={`${st.label}: ${finished ? "done" : i === stage ? "running" : "waiting"}`}>
              <View style={s.track}><View style={[s.fill, { width: `${p * 100}%` }]} /></View>
              <Text style={[s.stepLabel, i < cards && s.stepOn, i === view && s.stepView]} numberOfLines={1}>
                {st.short}
              </Text>
            </Pressable>
          );
        })}
      </View>

      <View onLayout={(e) => setWidth(e.nativeEvent.layout.width)}>
        {width ? (
          <ScrollView
            ref={scroller}
            horizontal
            pagingEnabled
            showsHorizontalScrollIndicator={false}
            scrollEventThrottle={32}
            onScroll={(e) => setView(Math.round(e.nativeEvent.contentOffset.x / width))}
          >
            {Array.from({ length: cards }, (_, i) => (
              <View key={STAGES[i].id} style={{ width, paddingRight: 1 }}>
                <StageCard index={i} label={STAGES[i].label} done={stageProgress(i) >= 1}>{body(i)}</StageCard>
              </View>
            ))}
          </ScrollView>
        ) : null}
      </View>

      <View style={s.nav}>
        <View style={s.dots}>
          {Array.from({ length: cards }, (_, i) => (
            <Pressable key={i} onPress={() => goTo(i)} accessibilityRole="button" accessibilityLabel={`Show ${STAGES[i].label}`}
              hitSlop={6} style={[s.dot, i === view && s.dotOn]} />
          ))}
        </View>
        <View style={s.navButtons}>
          <ArrowButton label="‹" disabled={view === 0} onPress={() => goTo(view - 1)} name="Previous stage" />
          <ArrowButton label="›" disabled={view >= cards - 1} onPress={() => goTo(view + 1)} name="Next stage" />
          {done ? <Button label="Done" onPress={onClose} /> : null}
        </View>
      </View>
    </Card>
  );
}

function ArrowButton({ label, name, disabled, onPress }: { label: string; name: string; disabled: boolean; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} disabled={disabled} accessibilityRole="button" accessibilityLabel={name}
      style={[s.arrow, glassPill, disabled && s.disabled]}>
      <Text style={s.arrowText}>{label}</Text>
    </Pressable>
  );
}

/** One carousel card; it fades and slides up when it is added. */
function StageCard({ index, label, done, children }: { index: number; label: string; done: boolean; children: ReactNode }) {
  const appear = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(appear, { toValue: 1, duration: 380, useNativeDriver: false }).start();
  }, [appear]);
  return (
    <Animated.View style={[s.stageCard, {
      opacity: appear,
      transform: [{ translateY: appear.interpolate({ inputRange: [0, 1], outputRange: [14, 0] }) }],
    }]}>
      <View style={s.cardHead}>
        <Text style={s.cardStep}>{index + 1} / {STAGES.length}</Text>
        <Text style={s.cardTitle}>{label}</Text>
        <Text style={[s.cardStatus, done && s.cardDone]}>{done ? "✓ Done" : "Running"}</Text>
      </View>
      {children}
    </Animated.View>
  );
}

function Stat({ value, label }: { value: string | number; label: string }) {
  return (
    <View style={s.stat}>
      <Text style={s.statValue} numberOfLines={1}>{value}</Text>
      <Text style={s.statLabel}>{label}</Text>
    </View>
  );
}
/** Two decimals, rounding half up (1.765 -> 1.77; toFixed gives 1.76 from binary error). */
const round2 = (v: number) => (Math.round((v + Number.EPSILON) * 100) / 100).toFixed(2);
const Chips = ({ items }: { items: string[] }) => (
  <View style={s.chips}>{items.map((c) => <Text key={c} style={s.chip}>{c}</Text>)}</View>
);
const Bar = ({ p, color = GREEN }: { p: number; color?: string }) => (
  <View style={s.barTrack}><View style={[s.bar, { width: `${Math.max(0, Math.min(1, p)) * 100}%`, backgroundColor: color }]} /></View>
);

function LocationCard({ loc, km }: { loc: DemoLocation; km: number | null }) {
  return (
    <>
      <Text style={s.lead}>{loc.label}</Text>
      <View style={s.stats}>
        <Stat value={loc.region_stations} label="stations in region" />
        <Stat value={30} label="neighbours as inputs" />
        <Stat value={km == null ? "—" : `${km.toFixed(0)} km`} label="from your tap" />
      </View>
      <Chips items={[`Obs: ${loc.obs_source}`, `${loc.lat.toFixed(2)}°, ${loc.lon.toFixed(2)}°`]} />
    </>
  );
}

function DownloadCard({ months, p }: { months: number; p: number }) {
  return (
    <>
      <Text style={s.counter}>
        <Text style={s.counterBig}>{Math.round(p * months)}</Text> / {months} months of HRRR
      </Text>
      <Bar p={p} />
      <Chips items={["4 runs / day", "0–48 h ahead", "Dec 2020 → Dec 2025", "One tile per run"]} />
    </>
  );
}

function PreprocessCard({ split, p }: { split: DemoMeta["split"]; p: number }) {
  return (
    <>
      <View style={[s.split, { opacity: 0.35 + 0.65 * p }]} accessibilityLabel="Date split">
        <Text numberOfLines={1} style={[s.splitPart, { flex: 2.4, backgroundColor: "#3b5bdb" }]}>Train {split.train[0].slice(0, 4)}–{split.train[1].slice(0, 4)}</Text>
        <Text numberOfLines={1} style={[s.splitPart, { flex: 1.3, backgroundColor: "#0e7490" }]}>Tune {split.validate[0].slice(0, 4)}</Text>
        <Text numberOfLines={1} style={[s.splitPart, { flex: 1.3, backgroundColor: "#047857" }]}>Test {split.test[0].slice(0, 4)}</Text>
      </View>
      <View style={s.stats}>
        <Stat value={15} label="HRRR fields" />
        <Stat value="17×17" label="local map (~50 km)" />
        <Stat value="0" label="future data used" />
      </View>
      <Text style={ui.note}>Target: HRRR&apos;s error at the station.</Text>
    </>
  );
}

function TrainCard({ p }: { p: number }) {
  const inputs = ["Local map", "Station point", "0–48 h trajectory", "Single values"];
  return (
    <>
      <View style={s.arch} accessibilityLabel="Four inputs feed one head that predicts the correction">
        <View style={s.inputs}>{inputs.map((x) => <Text key={x} style={s.input}>{x}</Text>)}</View>
        <Text style={s.arrowGlyph}>→</Text>
        <Text style={s.headBox}>Correction{"\n"}temp + wind</Text>
      </View>
      <View style={s.seeds}>
        {[0, 1, 2].map((k) => (
          <View key={k} style={s.seedRow}>
            <Text style={s.seedLabel}>Seed {k + 1}</Text>
            <Bar p={p * 3 - k} />
          </View>
        ))}
      </View>
      <Chips items={["AdamW", "≤ 40 epochs, early stop", "L4 / A10 GPU"]} />
    </>
  );
}

function ValidateCard({ loc, p }: { loc: DemoLocation; p: number }) {
  const vars = [["t2m", "Temperature"], ["wind_speed", "Wind"]] as const;
  const matched = loc.matched_6h;
  return (
    <>
      <Text style={ui.note}>RMSE on 2025, a year the model never saw. Lower is better.</Text>
      {vars.map(([key, name]) => {
        const r = loc.results[key];
        return (
          <View key={key} style={s.result}>
            <View style={s.resultHead}>
              <Text style={s.resultName}>{name}</Text>
              <Text style={s.badge}>{Math.round((1 - r.cnn / r.raw_hrrr) * 100)}% lower</Text>
            </View>
            {([["Ours", r.cnn, GREEN], ["HRRR", r.raw_hrrr, GREY]] as const).map(([label, v, color]) => (
              <View key={label} style={s.barRow}>
                <Text style={s.barLabel}>{label}</Text>
                <Bar p={(v / r.raw_hrrr) * p} color={color} />
                <Text style={s.barNum}>{round2(v)} {r.unit}</Text>
              </View>
            ))}
          </View>
        );
      })}
      {matched ? (
        <View style={s.matched} accessibilityLabel="Matched 6-hourly comparison with other forecasts">
          <Text style={s.matchedHead}>Same cases, 6-hourly to 48 h · °C / km/h</Text>
          {Object.keys(matched.t2m).map((m) => (
            <View key={m} style={s.matchedRow}>
              <Text style={[s.matchedCell, s.matchedName, m === "hrrr_cnn" && s.matchedOurs]}>{METHOD_LABELS[m] ?? m}</Text>
              <Text style={[s.matchedCell, m === "hrrr_cnn" && s.matchedOurs]}>{matched.t2m[m]?.toFixed(2) ?? "—"}</Text>
              <Text style={[s.matchedCell, m === "hrrr_cnn" && s.matchedOurs]}>{matched.wind_speed[m]?.toFixed(2) ?? "—"}</Text>
            </View>
          ))}
        </View>
      ) : null}
    </>
  );
}

const s = StyleSheet.create({
  flex: { flex: 1 },
  top: { flexDirection: "row", alignItems: "flex-start", gap: 8 },
  replayTag: { color: "rgba(225,228,255,0.6)", fontFamily: fonts.medium, fontSize: 12, marginBottom: 2 },
  stepper: { flexDirection: "row", gap: 6, marginTop: 14, marginBottom: 14 },
  step: { flex: 1, gap: 6 },
  track: { height: 5, borderRadius: 3, backgroundColor: "rgba(255,255,255,0.16)", overflow: "hidden" },
  fill: { height: 5, backgroundColor: GREEN },
  stepLabel: { color: "rgba(225,228,255,0.38)", fontFamily: fonts.medium, fontSize: 11 },
  stepOn: { color: "rgba(225,228,255,0.8)" },
  stepView: { color: "#fff", fontFamily: fonts.semi },
  stageCard: {
    minHeight: 236, borderRadius: 18, padding: 16, gap: 12,
    backgroundColor: "rgba(255,255,255,0.07)", borderWidth: 1, borderColor: "rgba(255,255,255,0.12)",
  },
  cardHead: { flexDirection: "row", alignItems: "center", gap: 10 },
  cardStep: { color: "rgba(225,228,255,0.55)", fontFamily: fonts.medium, fontSize: 12 },
  cardTitle: { flex: 1, color: "#fff", fontFamily: fonts.semi, fontSize: 17 },
  cardStatus: {
    color: "rgba(225,228,255,0.8)", fontFamily: fonts.semi, fontSize: 11, paddingVertical: 3, paddingHorizontal: 9,
    borderRadius: 999, backgroundColor: "rgba(255,255,255,0.1)", overflow: "hidden",
  },
  cardDone: { color: "#04241a", backgroundColor: MINT },
  lead: { color: "#fff", fontFamily: fonts.semi, fontSize: 19 },
  stats: { flexDirection: "row", gap: 10 },
  stat: { flex: 1, padding: 10, borderRadius: 12, backgroundColor: "rgba(255,255,255,0.06)", gap: 2 },
  statValue: { color: "#fff", fontFamily: fonts.light, fontSize: 26 },
  statLabel: { color: "rgba(225,228,255,0.65)", fontFamily: fonts.regular, fontSize: 12 },
  chips: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  chip: {
    color: "#fff", fontFamily: fonts.regular, fontSize: 12, paddingVertical: 4, paddingHorizontal: 10,
    borderRadius: 999, backgroundColor: "rgba(255,255,255,0.1)", overflow: "hidden",
  },
  counter: { color: "rgba(225,228,255,0.8)", fontFamily: fonts.regular, fontSize: 15 },
  counterBig: { color: "#fff", fontFamily: fonts.light, fontSize: 44 },
  barTrack: { flex: 1, height: 8, borderRadius: 4, backgroundColor: "rgba(255,255,255,0.12)", overflow: "hidden" },
  bar: { height: 8, borderRadius: 4 },
  split: { flexDirection: "row", borderRadius: 10, overflow: "hidden" },
  splitPart: { color: "#fff", fontFamily: fonts.semi, fontSize: 11, paddingVertical: 8, paddingHorizontal: 7 },
  arch: { flexDirection: "row", alignItems: "center", gap: 10 },
  inputs: { flex: 1, gap: 6 },
  input: {
    color: "#fff", fontFamily: fonts.medium, fontSize: 12, paddingVertical: 6, paddingHorizontal: 10,
    borderRadius: 10, backgroundColor: "rgba(59,91,219,0.35)", overflow: "hidden",
  },
  arrowGlyph: { color: "rgba(225,228,255,0.7)", fontFamily: fonts.medium, fontSize: 20 },
  headBox: {
    color: "#fff", fontFamily: fonts.semi, fontSize: 13, textAlign: "center", paddingVertical: 14, paddingHorizontal: 12,
    borderRadius: 12, backgroundColor: "rgba(16,185,129,0.3)", overflow: "hidden",
  },
  seeds: { gap: 6 },
  seedRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  seedLabel: { width: 52, color: "rgba(225,228,255,0.7)", fontFamily: fonts.medium, fontSize: 12 },
  result: { gap: 6 },
  resultHead: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  resultName: { color: "#fff", fontFamily: fonts.semi, fontSize: 15 },
  badge: {
    color: "#04241a", backgroundColor: MINT, fontFamily: fonts.semi, fontSize: 12,
    paddingVertical: 3, paddingHorizontal: 9, borderRadius: 999, overflow: "hidden",
  },
  barRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  barLabel: { width: 44, color: "rgba(225,228,255,0.7)", fontFamily: fonts.medium, fontSize: 12 },
  barNum: { width: 84, textAlign: "right", color: "#fff", fontFamily: fonts.medium, fontSize: 13, fontVariant: ["tabular-nums"] },
  matched: { borderRadius: 12, padding: 10, backgroundColor: "rgba(255,255,255,0.05)", gap: 4 },
  matchedHead: { color: "rgba(225,228,255,0.55)", fontFamily: fonts.medium, fontSize: 11, marginBottom: 2 },
  matchedRow: { flexDirection: "row" },
  matchedCell: { flex: 1, color: "rgba(245,247,255,0.8)", fontFamily: fonts.regular, fontSize: 12, textAlign: "right", fontVariant: ["tabular-nums"] },
  matchedName: { flex: 2, textAlign: "left" },
  matchedOurs: { color: MINT, fontFamily: fonts.semi },
  nav: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginTop: 14 },
  dots: { flexDirection: "row", gap: 8 },
  dot: { width: 8, height: 8, borderRadius: 4, backgroundColor: "rgba(255,255,255,0.25)" },
  dotOn: { width: 22, backgroundColor: "#fff" },
  navButtons: { flexDirection: "row", alignItems: "center", gap: 8 },
  arrow: { width: 34, height: 34, borderRadius: 17, alignItems: "center", justifyContent: "center", backgroundColor: "rgba(255,255,255,0.1)" },
  arrowText: { color: "#fff", fontFamily: fonts.medium, fontSize: 20, lineHeight: 22, marginTop: -2 },
  disabled: { opacity: 0.35 },
});
