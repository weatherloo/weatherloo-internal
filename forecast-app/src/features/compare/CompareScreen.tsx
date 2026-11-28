import { Fragment, useEffect, useState } from "react";
import { Platform, Pressable, ScrollView, StyleSheet, Text, View, type GestureResponderEvent } from "react-native";
import { Link } from "expo-router";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import Svg, { Circle, Line, Path, Text as SvgText } from "react-native-svg";
import type { ForecastRecord } from "@/data/contract";
import {
  type BenchmarkScore,
  type ModelBenchmark,
  BENCHMARK_LEAD_HOURS,
  BENCHMARK_YEAR,
  BENCHMARK_STATION_ID,
  getBenchmark,
} from "@/data/benchmark";
import { useForecast } from "@/data/ForecastProvider";
import { buildReport, type Report } from "@/data/report";
import { forecastSource } from "@/data/source";
import { MODELS, type ModelInfo } from "@/data/registry";
import { Card } from "@/components/Card";
import { SkyBackground } from "@/components/SkyBackground";
import { glassPill } from "@/theme/glass";
import { pageStyle, useWide } from "@/theme/layout";
import { fonts, nightTheme, ThemeContext } from "@/theme/tokens";

type ModelView = { model: ModelInfo; report: Report };
type Variable = "t2m" | "wind_speed" | "pop";
type HourPoint = Report["allHours"][number];
interface VariableSpec {
  label: string; chartTitle: string; short: string; unit: string;
  value: (hour: HourPoint) => number;
  format: (value: number) => string;
  /** Fixed chart range; otherwise the data's range. */
  range?: [number, number];
  /** Historical benchmark score, when the site has one for this variable. */
  score: (benchmark?: ModelBenchmark) => BenchmarkScore | null | undefined;
}
const VARIABLES: Record<Variable, VariableSpec> = {
  t2m: {
    label: "Temperature", chartTitle: "Temperature", short: "Temp", unit: "°C",
    value: (h) => h.temp, format: (v) => `${Math.round(v)}°C`, score: (b) => b?.t2m,
  },
  wind_speed: {
    label: "Wind", chartTitle: "Wind speed", short: "Wind", unit: "km/h",
    value: (h) => h.windKmh, format: (v) => `${Math.round(v)} km/h`, score: (b) => b?.wind_speed,
  },
  pop: {
    label: "Precipitation", chartTitle: "Chance of precipitation", short: "Precip", unit: "%",
    value: (h) => h.pop * 100, format: (v) => `${Math.round(v)}%`, range: [0, 100], score: () => undefined,
  },
};
const VARIABLE_OPTIONS = (Object.keys(VARIABLES) as Variable[]).map((id) => ({ id, label: VARIABLES[id].label }));
const MODEL_COLORS = ["#ffd43b", "#6ec6ff", "#ff8f7a", "#a9e89b", "#d0b4ff"];

export function CompareScreen() {
  const { station } = useForecast();
  const insets = useSafeAreaInsets();
  const wide = useWide();
  const [models, setModels] = useState<ModelView[]>();
  const [variable, setVariable] = useState<Variable>("t2m");
  const [error, setError] = useState<string>();

  useEffect(() => {
    let live = true;
    Promise.all(
      MODELS.map(async (model) => {
        const records: ForecastRecord[] = await forecastSource.load(model.id, station.id);
        if (!records.length) throw new Error(`No forecast fixture is available for ${model.label}`);
        return { model, report: buildReport(records, station, forecastSource.now()) };
      }),
    )
      .then((results) => live && setModels(results))
      .catch((cause) => live && setError(String(cause)));
    return () => {
      live = false;
    };
  }, [station]);

  return (
    <ThemeContext.Provider value={nightTheme}>
      <View style={s.root}>
        <SkyBackground />
        <ScrollView contentContainerStyle={[s.content, { paddingTop: insets.top + (wide ? 32 : 20), paddingBottom: insets.bottom + 40 }]}>
          <View style={pageStyle(wide)}>
            <View style={s.topBar}>
              <Link href="/" replace asChild>
                <Text style={s.backLink}>‹  Forecast</Text>
              </Link>
              <Text style={s.station}>{station.place}</Text>
            </View>
            <View style={wide ? s.headerWide : s.header}>
              <View style={s.headerText}>
                <Text style={s.title}>Compare models</Text>
                <Text style={s.subtitle}>A side-by-side look at this forecast demo.</Text>
              </View>
              <View style={[s.variablePicker, glassPill, wide && s.variablePickerWide]}>
                {VARIABLE_OPTIONS.map((option) => {
                  const selected = variable === option.id;
                  return (
                    <Pressable
                      key={option.id}
                      onPress={() => setVariable(option.id)}
                      accessibilityRole="button"
                      accessibilityState={{ selected }}
                      style={[s.variableOption, selected ? s.variableOptionSelected : null]}
                    >
                      <Text style={s.variableOptionText}>{option.label}</Text>
                    </Pressable>
                  );
                })}
              </View>
            </View>
            <Card style={s.noteCard}>
              <Text style={s.noteTitle}>Illustrative forecasts</Text>
              <Text style={s.noteText}>
                Every forecast line and f024 value is a synthetic demo fixture, not live model output. Historical RMSE
                measures the typical forecast error (lower is better). Scores below are from 2025 at Eric D. Soulis, lead
                24 h; they are not scores for these demo forecasts. HRRR CNN has no exact f024 score here, so no
                head-to-head improvement is shown.
              </Text>
            </Card>
            {error ? (
              <Card><Text style={s.error}>Couldn’t load model comparison. {error}</Text></Card>
            ) : models ? (
              <>
                <ForecastComparisonChart models={models} variable={variable} tall={wide} />
                <View style={wide ? s.grid : s.stack}>
                  <PrimaryPairCard models={models} variable={variable} stationId={station.id} stationName={station.name} wide={wide} />
                  {models.filter(({ model }) => !["hrrr_interpolated", "hrrr_cnn"].includes(model.id)).map(({ model, report }) => (
                    <ModelCard
                      key={model.id}
                      model={model}
                      report={report}
                      variable={variable}
                      stationId={station.id}
                      stationName={station.name}
                      wide={wide}
                    />
                  ))}
              </View>
              </>
            ) : (
              <Card><Text style={s.loading}>Loading model fixtures…</Text></Card>
            )}
          </View>
        </ScrollView>
      </View>
    </ThemeContext.Provider>
  );
}

function PrimaryPairCard({
  models,
  variable,
  stationId,
  stationName,
  wide,
}: {
  models: ModelView[];
  variable: Variable;
  stationId: string;
  stationName: string;
  wide?: boolean;
}) {
  const hrrr = models.find(({ model }) => model.id === "hrrr_interpolated");
  const cnn = models.find(({ model }) => model.id === "hrrr_cnn");
  if (!hrrr || !cnn) return null;

  const benchmark = stationId === BENCHMARK_STATION_ID ? getBenchmark(hrrr.model.benchmarkMethodId) : undefined;
  const aggregateScore = VARIABLES[variable].score(benchmark);
  const scoreUnit = VARIABLES[variable].unit;
  const score = aggregateScore?.rmse;
  return (
    <Card style={[s.primaryCard, wide && s.gridCell]}>
      <View style={s.primaryHeading}>
        <View>
          <Text style={s.modelName}>HRRR vs HRRR CNN</Text>
          <Text style={s.family}>Primary comparison · f024</Text>
        </View>
        <View style={s.fixtureBadge}><Text style={s.fixtureText}>Demo</Text></View>
      </View>
      {[hrrr, cnn].map(({ model }) => (
        <Text key={model.id} style={s.about}><Text style={s.aboutName}>{model.label}. </Text>{model.description}</Text>
      ))}
      <View style={s.rule} />
      <Text style={s.sectionLabel}>Historical RMSE · {BENCHMARK_YEAR} · f024 · {scoreUnit}</Text>
      <View style={s.forecastRow}>
        <Metric label={score === undefined ? "HRRR RMSE" : `HRRR · ${aggregateScore?.n_samples?.toLocaleString()} pairs`} value={score === undefined ? "Unavailable" : `${score.toFixed(2)} ${scoreUnit}`} />
        <Metric label="HRRR CNN RMSE" value="Unscored" />
      </View>
      <Text style={s.scoreContext}>
        {score === undefined
          ? variable === "pop"
            ? "The historical benchmark doesn't score precipitation yet."
            : "No exact f024 benchmark is available for this station and variable."
          : `${stationName} · no exact HRRR CNN f024 score is available. Percent improvement is not calculated.`}
      </Text>
      <View style={s.rule} />
      <Text style={s.sectionLabel}>Synthetic demo values · f024</Text>
      <View style={s.forecastRow}>
        <Metric label="HRRR" value={f024Value(hrrr.report, variable)} />
        <Metric label="HRRR CNN" value={f024Value(cnn.report, variable)} />
      </View>
    </Card>
  );
}

function ModelCard({
  model,
  report,
  variable,
  stationId,
  stationName,
  wide,
}: ModelView & { variable: Variable; stationId: string; stationName: string; wide?: boolean }) {
  const benchmark = stationId === BENCHMARK_STATION_ID ? getBenchmark(model.benchmarkMethodId) : undefined;
  const aggregateScore = VARIABLES[variable].score(benchmark);
  const score = aggregateScore?.rmse;
  const scoreUnit = VARIABLES[variable].unit;
  const forecastValue = f024Value(report, variable);
  return (
    <Card style={wide && s.gridCell}>
      <View style={s.modelHeader}>
        <View style={s.modelTitles}>
          <Text style={s.modelName}>{model.label}</Text>
          <Text style={s.family}>{model.family}</Text>
        </View>
        <View style={s.fixtureBadge}><Text style={s.fixtureText}>Demo</Text></View>
      </View>
      {model.description ? <Text style={s.about}>{model.description}</Text> : null}

      <View style={s.benchmarkHeading}>
        <Text style={s.sectionLabel}>f{String(BENCHMARK_LEAD_HOURS).padStart(3, "0")} forecast · synthetic</Text>
        <Text style={s.leadNote}>{VARIABLES[variable].label}</Text>
      </View>
      <View style={s.forecastRow}><Metric label="Synthetic f024" value={forecastValue} /></View>

      <View style={s.rule} />
      <View style={s.benchmarkHeading}>
        <Text style={s.sectionLabel}>Historical RMSE · {BENCHMARK_YEAR}</Text>
        <Text style={s.leadNote}>{BENCHMARK_LEAD_HOURS} h · {scoreUnit}</Text>
      </View>
      {score !== undefined ? (
        <View style={s.forecastRow}>
          <Metric label={`${VARIABLES[variable].short} RMSE`} value={`${score.toFixed(2)} ${scoreUnit}`} />
          {aggregateScore?.n_samples ? <Metric label="Valid pairs" value={aggregateScore.n_samples.toLocaleString()} /> : null}
        </View>
      ) : (
        <Text style={s.noScore}>
          {stationId !== BENCHMARK_STATION_ID
            ? "Historical benchmark data is not available for this station."
            : variable === "pop"
              ? "The historical benchmark doesn't score precipitation yet."
              : `No exact ${BENCHMARK_LEAD_HOURS}-hour historical benchmark is available for ${model.label}.`}
        </Text>
      )}
      {score !== undefined ? (
        <Text style={s.scoreContext}>
          {stationName} · {aggregateScore?.n_samples?.toLocaleString()} valid pairs. Historical result, separate from the demo forecast.
        </Text>
      ) : null}
    </Card>
  );
}

function f024Value(report: Report, variable: Variable, lead = BENCHMARK_LEAD_HOURS) {
  const point = report.allHours.find(
    (hour) => hour.time.getTime() === report.initTime.getTime() + lead * 3600e3,
  );
  if (!point) return "—";
  return VARIABLES[variable].format(VARIABLES[variable].value(point));
}

function ForecastComparisonChart({ models, variable, tall }: { models: ModelView[]; variable: Variable; tall?: boolean }) {
  const hourCount = Math.min(...models.map(({ report }) => report.allHours.length));
  const last = Math.max(0, hourCount - 1);
  const [selectedLeadIndex, setSelectedLeadIndex] = useState(Math.min(23, last));
  // Drawn in real pixels at the measured width so it fills phones and desktop alike.
  const [width, setWidth] = useState(360);
  const height = tall ? 240 : 148;
  const values = models.flatMap(({ report }) =>
    report.allHours.slice(0, hourCount).map(VARIABLES[variable].value),
  );
  const [min, max] = VARIABLES[variable].range ?? [Math.floor(Math.min(...values)), Math.ceil(Math.max(...values))];
  const spread = Math.max(1, max - min);
  const chart = { left: 36, right: width - 10, top: 12, bottom: height - 32 };
  const x = (index: number) => chart.left + (index / Math.max(1, last)) * (chart.right - chart.left);
  const y = (value: number) => chart.bottom - ((value - min) / spread) * (chart.bottom - chart.top);
  const unit = VARIABLES[variable].unit;
  const selectedLead = selectedLeadIndex + 1;
  const selectLead = (event: GestureResponderEvent) => {
    const px = pressX(event);
    if (px === undefined) return;
    const fraction = (px - chart.left) / (chart.right - chart.left);
    setSelectedLeadIndex(Math.max(0, Math.min(last, Math.round(fraction * last))));
  };
  const step = (dir: 1 | -1) => setSelectedLeadIndex((i) => Math.max(0, Math.min(last, i + dir)));

  return (
    <Card>
      <View style={s.chartHeading}>
        <Text style={s.chartTitle}>{VARIABLES[variable].chartTitle}</Text>
        <Text style={s.leadNote}>{`${hourCount}-hour outlook · ${Platform.OS === "web" ? "click" : "tap"} to inspect`}</Text>
      </View>
      <Pressable
        onLayout={(event) => setWidth(Math.max(240, Math.round(event.nativeEvent.layout.width)))}
        onPress={selectLead}
        // Arrow keys step through leads once the chart has focus (web, hardware keyboards).
        onKeyDown={(event) => {
          const key = (event.nativeEvent as unknown as { key?: string }).key;
          if (key !== "ArrowLeft" && key !== "ArrowRight") return;
          event.preventDefault();
          step(key === "ArrowLeft" ? -1 : 1);
        }}
        accessibilityRole="button"
        accessibilityLabel={`Forecast comparison chart, selected lead ${selectedLead} hours`}
        accessibilityHint="Tap a point on the chart, or use the arrow keys, to inspect demo values for every model at that lead."
      >
        <Svg width={width} height={height}>
          {[0, 0.5, 1].map((fraction) => {
            const value = min + spread * fraction;
            const yy = y(value);
            return (
              <Fragment key={fraction}>
                <Line x1={chart.left} y1={yy} x2={chart.right} y2={yy} stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
                <SvgText x={2} y={yy + 4} fill="rgba(225,228,255,0.6)" fontSize={10} fontFamily={fonts.medium}>{`${Math.round(value)}${unit}`}</SvgText>
              </Fragment>
            );
          })}
          {[{ index: 0, label: "1h" }, { index: Math.min(23, last), label: "24h" }, { index: last, label: `${last + 1}h` }].map(({ index, label }) => (
            <SvgText key={label} x={x(index)} y={height - 10} fill="rgba(225,228,255,0.55)" fontSize={10} fontFamily={fonts.medium} textAnchor="middle">
              {label}
            </SvgText>
          ))}
          {models.map(({ model, report }, modelIndex) => {
            const series = report.allHours.slice(0, hourCount).map(VARIABLES[variable].value);
            const path = series.map((value, index) => `${index === 0 ? "M" : "L"}${x(index)},${y(value)}`).join(" ");
            return <Path key={model.id} d={path} fill="none" stroke={MODEL_COLORS[modelIndex % MODEL_COLORS.length]} strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" />;
          })}
          <Line x1={x(selectedLeadIndex)} y1={chart.top} x2={x(selectedLeadIndex)} y2={chart.bottom} stroke="rgba(255,255,255,0.65)" strokeWidth={1} strokeDasharray="3 3" />
          {models.map(({ model, report }, modelIndex) => {
            const point = report.allHours[selectedLeadIndex];
            const value = VARIABLES[variable].value(point);
            return (
              <Circle
                key={`${model.id}-selected`}
                cx={x(selectedLeadIndex)}
                cy={y(value)}
                r={4}
                fill={MODEL_COLORS[modelIndex % MODEL_COLORS.length]}
                stroke="#fff"
                strokeWidth={1}
              />
            );
          })}
        </Svg>
      </Pressable>
      <View style={s.tooltip}>
        <Text style={s.tooltipHeading}>Synthetic demo values · f{String(selectedLead).padStart(3, "0")}</Text>
        <View style={s.tooltipValues}>
          {models.map(({ model, report }, index) => (
            <View key={model.id} style={[s.tooltipItem, tall && s.tooltipItemWide]}>
              <View style={[s.legendDot, { backgroundColor: MODEL_COLORS[index % MODEL_COLORS.length] }]} />
              <Text style={s.tooltipLabel}>{model.label}</Text>
              <Text style={s.tooltipValue}>{f024Value(report, variable, selectedLead)}</Text>
            </View>
          ))}
        </View>
      </View>
      <View style={s.legend}>
        {models.map(({ model }, index) => (
          <View key={model.id} style={s.legendItem}>
            <View style={[s.legendDot, { backgroundColor: MODEL_COLORS[index % MODEL_COLORS.length] }]} />
            <Text style={s.legendLabel}>{model.label}</Text>
          </View>
        ))}
      </View>
    </Card>
  );
}

/** Horizontal press position inside the pressed element. On web, onPress gets a DOM mouse event with no locationX. */
function pressX(event: GestureResponderEvent): number | undefined {
  const native = event.nativeEvent as unknown as { locationX?: number; clientX?: number };
  if (Platform.OS === "web" && typeof native.clientX === "number") {
    const rect = (event.currentTarget as unknown as HTMLElement | null)?.getBoundingClientRect?.();
    if (rect) return native.clientX - rect.left;
  }
  return Number.isFinite(native.locationX) ? native.locationX : undefined;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <View style={s.metric}>
      <Text style={s.metricLabel}>{label}</Text>
      <Text style={s.metricValue}>{value}</Text>
    </View>
  );
}

const s = StyleSheet.create({
  root: { flex: 1 },
  content: { alignItems: "center" },
  header: { gap: 12 },
  headerWide: { flexDirection: "row", alignItems: "flex-end", justifyContent: "space-between", gap: 24 },
  headerText: { gap: 8, flexShrink: 1 },
  stack: { gap: 12 },
  grid: { flexDirection: "row", flexWrap: "wrap", gap: 16 },
  gridCell: { flexBasis: "40%", flexGrow: 1 },
  topBar: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  backLink: { color: "#fff", fontFamily: fonts.medium, fontSize: 14, paddingVertical: 6 },
  station: { color: "rgba(225,228,255,0.7)", fontFamily: fonts.medium, fontSize: 13 },
  title: { color: "#fff", fontFamily: fonts.semi, fontSize: 34, letterSpacing: -0.5, marginTop: 4 },
  subtitle: { color: "rgba(225,228,255,0.7)", fontFamily: fonts.regular, fontSize: 15, marginBottom: 4 },
  noteCard: { backgroundColor: "rgba(255,255,255,0.12)", borderWidth: 1, borderColor: "rgba(255,255,255,0.17)" },
  noteTitle: { color: "#fff", fontFamily: fonts.semi, fontSize: 14, marginBottom: 4 },
  noteText: { color: "rgba(245,247,255,0.78)", fontFamily: fonts.regular, fontSize: 13, lineHeight: 19 },
  scoreContext: { color: "rgba(225,228,255,0.64)", fontFamily: fonts.regular, fontSize: 11, lineHeight: 16, marginTop: 4 },
  modelHeader: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 10 },
  about: { color: "rgba(245,247,255,0.78)", fontFamily: fonts.regular, fontSize: 13, lineHeight: 19, marginBottom: 14 },
  aboutName: { color: "#fff", fontFamily: fonts.semi },
  modelTitles: { gap: 3 },
  primaryCard: { borderWidth: 1, borderColor: "rgba(255,255,255,0.22)", backgroundColor: "rgba(76,92,155,0.52)" },
  primaryHeading: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 10 },
  modelName: { color: "#fff", fontFamily: fonts.semi, fontSize: 19 },
  family: { color: "rgba(225,228,255,0.58)", fontFamily: fonts.regular, fontSize: 12 },
  fixtureBadge: { borderRadius: 12, borderWidth: 1, borderColor: "rgba(255,255,255,0.2)", paddingHorizontal: 9, paddingVertical: 5 },
  fixtureText: { color: "rgba(245,247,255,0.72)", fontFamily: fonts.medium, fontSize: 11 },
  sectionLabel: { color: "rgba(225,228,255,0.62)", fontFamily: fonts.medium, fontSize: 12 },
  forecastRow: { flexDirection: "row", justifyContent: "space-between", gap: 6, marginTop: 11 },
  metric: { flex: 1, gap: 4 },
  metricLabel: { color: "rgba(225,228,255,0.58)", fontFamily: fonts.regular, fontSize: 10 },
  metricValue: { color: "#fff", fontFamily: fonts.medium, fontSize: 18 },
  rule: { height: 1, backgroundColor: "rgba(255,255,255,0.14)", marginVertical: 17 },
  benchmarkHeading: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  leadNote: { color: "rgba(225,228,255,0.48)", fontFamily: fonts.regular, fontSize: 10 },
  noScore: { color: "rgba(225,228,255,0.56)", fontFamily: fonts.regular, fontSize: 12, marginTop: 11 },
  variablePicker: { flexDirection: "row", borderRadius: 16, padding: 4, backgroundColor: "rgba(70,78,140,0.45)" },
  variablePickerWide: { width: 300, marginBottom: 4 },
  variableOption: { flex: 1, alignItems: "center", borderRadius: 12, paddingVertical: 9 },
  variableOptionSelected: { backgroundColor: "rgba(255,255,255,0.18)" },
  variableOptionText: { color: "#fff", fontFamily: fonts.medium, fontSize: 13 },
  chartHeading: { flexDirection: "row", alignItems: "baseline", justifyContent: "space-between", marginBottom: 8 },
  chartTitle: { color: "#fff", fontFamily: fonts.semi, fontSize: 17 },
  tooltip: { borderRadius: 12, backgroundColor: "rgba(255,255,255,0.08)", padding: 10, marginTop: 2 },
  tooltipHeading: { color: "rgba(225,228,255,0.7)", fontFamily: fonts.medium, fontSize: 11, marginBottom: 8 },
  tooltipValues: { flexDirection: "row", flexWrap: "wrap", columnGap: 12, rowGap: 8 },
  tooltipItem: { flexDirection: "row", alignItems: "center", gap: 5, minWidth: "45%" },
  tooltipItemWide: { minWidth: "18%" },
  tooltipLabel: { color: "rgba(245,247,255,0.72)", fontFamily: fonts.regular, fontSize: 10, flexShrink: 1 },
  tooltipValue: { color: "#fff", fontFamily: fonts.medium, fontSize: 11 },
  legend: { flexDirection: "row", flexWrap: "wrap", columnGap: 12, rowGap: 8, paddingTop: 2 },
  legendItem: { flexDirection: "row", alignItems: "center", gap: 5 },
  legendDot: { width: 7, height: 7, borderRadius: 4 },
  legendLabel: { color: "rgba(245,247,255,0.78)", fontFamily: fonts.regular, fontSize: 10 },
  loading: { color: nightTheme.text, fontFamily: fonts.regular, fontSize: 15 },
  error: { color: "#fff", fontFamily: fonts.regular, fontSize: 14, lineHeight: 21 },
});
