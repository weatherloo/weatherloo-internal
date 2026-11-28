import { Fragment, useEffect, useState } from "react";
import { AccessibilityInfo, Pressable, StyleSheet, Text, View } from "react-native";
import { Card } from "@/components/Card";
import { fonts } from "@/theme/tokens";
import { Button, LinkButton } from "./TrainRun";
import { ui } from "./ui";

const STAGES = ["Location", "Download", "Preprocessing", "Training", "Validation"];
const REPLAY_MS = 1800;

// The 15 HRRR fields on the local map (models/hrrr_cnn/points.py PATCH_VARIABLES).
const PATCH_FIELDS = [
  "Temperature (2 m)", "Dew point (2 m)", "Relative humidity (2 m)", "Wind east-west (10 m)",
  "Wind north-south (10 m)", "Wind east-west (80 m)", "Wind north-south (80 m)", "Wind gust",
  "Surface pressure", "Cloud cover", "Incoming sunlight", "Incoming heat radiation",
  "Snow cover", "Water vapour in the air", "Precipitation",
];

const METHOD_LABELS: Record<string, string> = {
  hrrr_cnn: "HRRR + CNN (ours)",
  hrrr_interpolated: "Raw HRRR",
  ecmwf_aifs: "ECMWF AIFS",
  graphcast: "GraphCast",
};

const GREEN = "#10b981", GREY = "#94a3b8";

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

function useReplay(key: string) {
  const [progress, setProgress] = useState(0);
  useEffect(() => {
    let frame = 0, cancelled = false;
    setProgress(0);
    AccessibilityInfo.isReduceMotionEnabled().catch(() => false).then((reduce) => {
      if (cancelled) return;
      if (reduce) return setProgress(1);
      const start = Date.now();
      const tick = () => {
        const p = Math.min(1, (Date.now() - start) / REPLAY_MS);
        setProgress(p);
        if (p < 1) frame = requestAnimationFrame(tick);
      };
      frame = requestAnimationFrame(tick);
    });
    return () => {
      cancelled = true;
      cancelAnimationFrame(frame);
    };
  }, [key]);
  return progress;
}

function Facts({ rows }: { rows: [string, string | number][] }) {
  return (
    <View style={s.facts}>
      {rows.map(([k, v]) => (
        <View key={k} style={s.fact}>
          <Text style={[ui.muted, s.factKey]}>{k}</Text>
          <Text style={[ui.body, s.factVal]}>{String(v)}</Text>
        </View>
      ))}
    </View>
  );
}

const pct = (ours: number, base: number) => `${Math.round((1 - ours / base) * 100)}% lower`;
const fmt = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(2));
const B = ({ children }: { children: React.ReactNode }) => <Text style={s.strong}>{children}</Text>;

function LocationStage({ loc, km }: { loc: DemoLocation; km: number | null }) {
  return (
    <>
      <Text style={ui.body}>
        The tap snaps to the nearest weather station with a good record. That station is the <B>target</B>: the one
        place the model forecasts for, and the thermometer it learns from and is scored against.
      </Text>
      <Facts rows={[
        ["Target station", loc.station_name],
        ["Distance from tap", km == null ? "Selected directly" : `${km.toFixed(0)} km`],
        ["Observations from", loc.obs_source],
        ["Stations in the region table", loc.region_stations],
        ["Neighbours used as inputs", "30 nearest"],
        ["HRRR tile read", "≈ 800 × 900 km around the target"],
      ]} />
      <Text style={ui.note}>
        Neighbouring stations supply recent observations (what actually happened nearby before the forecast was issued).
      </Text>
    </>
  );
}

function DownloadStage({ months, progress }: { months: number; progress: number }) {
  return (
    <>
      <Text style={ui.body}>
        <B>HRRR</B> (High-Resolution Rapid Refresh) is NOAA&apos;s weather forecast model for North America. It covers
        the continental US and southern Canada on a 3 km grid and issues a new forecast every hour; the 00, 06, 12 and
        18 UTC runs reach 48 hours ahead.
      </Text>
      <Text style={s.counter}>
        <Text style={s.counterBig}>{Math.round(progress * months)}</Text> / {months} months fetched
      </Text>
      <Facts rows={[
        ["Forecast source", "dynamical.org cloud copy of NOAA HRRR (48-hour runs)"],
        ["Runs used", "4 per day (00 / 06 / 12 / 18 UTC)"],
        ["Forecast hours per run", "0 to 48"],
        ["Period", "Dec 2020 to Dec 2025"],
        ["Observations", "IEM ASOS (US) and Environment Canada hourly"],
        ["Stored on", "Modal data volume, one folder per month"],
      ]} />
      <Text style={ui.note}>
        Only the one tile around the target is read for each run, not the whole continent. Every month leaves a
        completion file, so an interrupted download picks up where it stopped.
      </Text>
    </>
  );
}

function PreprocessStage({ split }: { split: DemoMeta["split"] }) {
  const steps: [string, string][] = [
    ["Station table.", "Keep stations with at least 50% hourly coverage; merge duplicates within 2 km."],
    ["HRRR at each station.", "Interpolate the forecast from the four surrounding grid points."],
    ["Local map.", "Cut a 17 × 17 grid square (about 50 km) around the target."],
    ["What the model learns.", "The gap between what was observed and what HRRR said: HRRR's mistake."],
    ["Split by date.", "So the final score is on a year the model never saw."],
  ];
  return (
    <>
      {steps.map(([h, t], i) => <Text key={h} style={ui.body}>{i + 1}. <B>{h}</B> {t}</Text>)}
      <View style={s.split} accessibilityLabel="Date split">
        <Text style={[s.splitPart, { flex: 3, backgroundColor: "#3b5bdb" }]}>
          Train {split.train[0].slice(0, 7)} → {split.train[1].slice(0, 4)}
        </Text>
        <Text style={[s.splitPart, { flex: 1, backgroundColor: "#0e7490" }]}>Choose {split.validate[0].slice(0, 4)}</Text>
        <Text style={[s.splitPart, { flex: 1, backgroundColor: "#047857" }]}>Test {split.test[0].slice(0, 4)}</Text>
      </View>
      <Text style={ui.h4}>15 HRRR fields kept on the local map</Text>
      <View style={s.chips}>
        {PATCH_FIELDS.map((f) => <Text key={f} style={s.chip}>{f}</Text>)}
      </View>
      <Text style={ui.note}>
        Also kept: the same fields at the station for every forecast hour 0–48, the three earlier HRRR runs, and recent
        observations at the target and its neighbours. Twelve more fields (snow depth, precipitation type, sea-level
        pressure, cloud ceiling, radar) were tested and dropped because they did not help. Rule: nothing from after the
        forecast was issued.
      </Text>
    </>
  );
}

function TrainingStage() {
  const branches: [string, string, string][] = [
    ["Local map", "17 × 17 × 15 fields", "5 convolution layers"],
    ["Station point", "3 × 3 around the site", "1 dense layer"],
    ["Forecast trajectory", "0–48 h at the site", "2 1-D convolution layers"],
    ["Single values", "~60 numbers", "2 dense layers"],
  ];
  return (
    <>
      <Text style={ui.body}>
        One network with four inputs. Each branch summarises one kind of information; the head combines them and
        predicts HRRR&apos;s error for temperature and wind.
      </Text>
      <View style={s.arch} accessibilityLabel="Model architecture">
        {branches.map(([a, b, c]) => (
          <View key={a} style={s.branch}>
            <Text style={[ui.body, s.strong]}>{a}</Text>
            <Text style={ui.muted}>{b}</Text>
            <Text style={ui.note}>{c}</Text>
          </View>
        ))}
      </View>
      <Text style={[ui.body, s.center]}>↓</Text>
      <View style={s.head}>
        <Text style={ui.body}><B>Head</B> 256 units → temperature and wind correction</Text>
      </View>
      <Facts rows={[
        ["Optimiser", "AdamW, learning rate 0.001"],
        ["Batch size", "512"],
        ["Epochs", "Up to 40; stops after 5 with no gain on 2024"],
        ["Copies", "3 random seeds, averaged"],
        ["Hardware", "One L4 / A10 GPU on Modal"],
      ]} />
      <Text style={ui.note}>
        Training curve: will appear once the hrrr_cnn run files (cnn.json) are added. The settings above are one fixed
        recipe; San Francisco used it unchanged.
      </Text>
    </>
  );
}

function ResultTable({ header, rows }: { header: string[]; rows: (string | number)[][] }) {
  return (
    <View style={s.table}>
      {[header, ...rows].map((row, r) => (
        <View key={r} style={[s.tr, r === 0 && s.thRow]}>
          {row.map((cell, c) => (
            <Text key={c} style={[r === 0 ? ui.muted : ui.num, s.td, c === 0 && s.tdFirst]}>{String(cell)}</Text>
          ))}
        </View>
      ))}
    </View>
  );
}

function ValidationStage({ loc }: { loc: DemoLocation }) {
  const vars = [["t2m", "Temperature"], ["wind_speed", "Wind speed"]] as const;
  const matched = loc.matched_6h;
  return (
    <>
      <Text style={ui.body}>
        Scored on <B>2025</B>, a year the model never trained on or made choices with. The metric is RMSE
        (root-mean-square error): the typical size of a forecast miss. Lower is better.
      </Text>
      {vars.map(([key, name]) => {
        const r = loc.results[key];
        return (
          <View key={key} style={s.result}>
            <View style={s.resultHead}>
              <Text style={ui.h4}>{name}</Text>
              <Text style={s.badge}>{pct(r.cnn, r.raw_hrrr)} than raw HRRR</Text>
            </View>
            {([["Ours", r.cnn, GREEN], ["Raw HRRR", r.raw_hrrr, GREY]] as const).map(([label, v, color]) => (
              <View key={label} style={s.barRow}>
                <Text style={[ui.muted, s.barLabel]}>{label}</Text>
                <View style={s.barTrack}>
                  <View style={[s.bar, { width: `${(v / r.raw_hrrr) * 100}%`, backgroundColor: color }]} />
                </View>
                <Text style={[ui.num, s.barNum]}>{v.toFixed(2)} {r.unit}</Text>
              </View>
            ))}
            <ResultTable
              header={["Forecast hour", ...r.leads.map((l) => `+${l} h`)]}
              rows={[
                ["Ours", ...r.cnn_by_lead.map(fmt)],
                ...(r.raw_by_lead ? [["Raw HRRR", ...r.raw_by_lead.map(fmt)]] : []),
              ]}
            />
          </View>
        );
      })}
      {matched ? (
        <>
          <Text style={ui.h4}>Against other forecasts (same forecasts, 6-hourly to 48 h)</Text>
          <ResultTable
            header={["Method", "Temp (°C)", "Wind (km/h)"]}
            rows={Object.keys(matched.t2m).map((m) => [
              METHOD_LABELS[m] ?? m, fmt(matched.t2m[m]), fmt(matched.wind_speed[m]),
            ])}
          />
        </>
      ) : null}
    </>
  );
}

export function Walkthrough({ loc, km, meta, onClose }: { loc: DemoLocation; km: number | null; meta: DemoMeta; onClose: () => void }) {
  const [stage, setStage] = useState(0);
  const progress = useReplay(`${loc.id}-${stage}`);
  const last = STAGES.length - 1;

  const body = [
    <LocationStage key="l" loc={loc} km={km} />,
    <DownloadStage key="d" months={meta.months_downloaded} progress={progress} />,
    <PreprocessStage key="p" split={meta.split} />,
    <TrainingStage key="t" />,
    <ValidationStage key="v" loc={loc} />,
  ][stage];

  return (
    <Card>
      <View style={s.top}>
        <View style={s.flex}>
          <Text style={ui.muted}>Training run · {loc.label}</Text>
          <Text style={ui.h3}>{stage + 1}. {STAGES[stage]}</Text>
        </View>
        <LinkButton label="Back to globe" onPress={onClose} />
      </View>

      <View style={s.progress} accessibilityLabel="Pipeline stages">
        {STAGES.map((name, i) => (
          <Pressable key={name} style={s.step} onPress={() => setStage(i)} accessibilityRole="button"
            accessibilityState={{ selected: i === stage }}>
            <View style={s.stepTrack}>
              <View style={[s.stepFill, { width: i < stage ? "100%" : i === stage ? `${progress * 100}%` : "0%" }]} />
            </View>
            <Text style={[s.stepLabel, i === stage && s.stepCurrent]} numberOfLines={1}>{name}</Text>
          </Pressable>
        ))}
      </View>
      <Text style={ui.note} accessibilityLiveRegion="polite">
        {progress < 1 ? `Replaying the ${STAGES[stage].toLowerCase()} step of a precomputed run…` : `${STAGES[stage]} complete`}
      </Text>

      <View style={s.body}>
        <Fragment>{body}</Fragment>
      </View>

      <View style={s.nav}>
        {stage > 0 ? <LinkButton label="Back" onPress={() => setStage((v) => v - 1)} /> : <View />}
        {stage < last ? (
          <Button label={`Next: ${STAGES[stage + 1]}`} onPress={() => setStage((v) => v + 1)} />
        ) : (
          <Button label="Done" onPress={onClose} />
        )}
      </View>
    </Card>
  );
}

const s = StyleSheet.create({
  flex: { flex: 1 },
  strong: { fontFamily: fonts.semi },
  center: { textAlign: "center" },
  top: { flexDirection: "row", alignItems: "flex-start", gap: 8 },
  progress: { flexDirection: "row", gap: 4, marginTop: 14, marginBottom: 6 },
  step: { flex: 1, gap: 4 },
  stepTrack: { height: 4, borderRadius: 2, backgroundColor: "rgba(255,255,255,0.18)", overflow: "hidden" },
  stepFill: { height: 4, backgroundColor: GREEN },
  stepLabel: { color: "rgba(225,228,255,0.6)", fontFamily: fonts.medium, fontSize: 10 },
  stepCurrent: { color: "#fff" },
  body: { gap: 12, marginTop: 12 },
  nav: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginTop: 16 },
  facts: { gap: 6 },
  fact: { flexDirection: "row", gap: 10 },
  factKey: { width: 130 },
  factVal: { flex: 1, fontSize: 14 },
  counter: { color: "rgba(225,228,255,0.8)", fontFamily: fonts.regular, fontSize: 15 },
  counterBig: { color: "#fff", fontFamily: fonts.light, fontSize: 40 },
  split: { flexDirection: "row", borderRadius: 8, overflow: "hidden" },
  splitPart: { color: "#fff", fontFamily: fonts.semi, fontSize: 11, paddingVertical: 6, paddingHorizontal: 6 },
  chips: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  chip: {
    color: "#fff", fontFamily: fonts.regular, fontSize: 12, paddingVertical: 4, paddingHorizontal: 9,
    borderRadius: 999, backgroundColor: "rgba(255,255,255,0.12)",
  },
  arch: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  branch: { flexBasis: "47%", flexGrow: 1, padding: 10, borderRadius: 12, backgroundColor: "rgba(255,255,255,0.1)", gap: 2 },
  head: { padding: 10, borderRadius: 12, backgroundColor: "rgba(16,185,129,0.22)" },
  result: { gap: 8 },
  resultHead: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 6 },
  badge: {
    color: "#04241a", backgroundColor: "#6ee7b7", fontFamily: fonts.semi, fontSize: 12,
    paddingVertical: 3, paddingHorizontal: 8, borderRadius: 999, overflow: "hidden",
  },
  barRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  barLabel: { width: 72 },
  barTrack: { flex: 1, height: 10, borderRadius: 5, backgroundColor: "rgba(255,255,255,0.12)", overflow: "hidden" },
  bar: { height: 10, borderRadius: 5 },
  barNum: { width: 86, textAlign: "right" },
  table: { borderRadius: 10, overflow: "hidden", borderWidth: 1, borderColor: "rgba(255,255,255,0.14)" },
  tr: { flexDirection: "row", borderTopWidth: 1, borderTopColor: "rgba(255,255,255,0.1)" },
  thRow: { borderTopWidth: 0, backgroundColor: "rgba(255,255,255,0.06)" },
  td: { flex: 1, paddingVertical: 6, paddingHorizontal: 8, fontSize: 13 },
  tdFirst: { flex: 1.6 },
});
