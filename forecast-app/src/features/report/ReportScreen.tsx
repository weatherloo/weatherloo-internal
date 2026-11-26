import { ScrollView, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useForecast } from "@/data/ForecastProvider";
import { buildNarrative, deg } from "@/data/narrative";
import { clock, localParts } from "@/data/time";
import { Icon } from "@/components/Icon";
import { Meta, Runs, SectionTitle } from "@/components/Type";
import { Pill } from "@/components/Pill";
import { HourlyChart } from "@/components/HourlyChart";
import { DayRow } from "@/components/DayRow";
import { FinePrint } from "./FinePrint";
import { colors, fonts, MAX_WIDTH, space } from "@/theme/tokens";

/** The morning report: one scrollable page, top to bottom. */
export function ReportScreen() {
  const { report, status, error } = useForecast();
  const insets = useSafeAreaInsets();
  if (status === "error") return <Centered text={`couldn't load the forecast. ${error ?? ""}`} />;
  if (!report) return <Centered text="loading the forecast…" />;

  const tz = report.station.timeZone;
  const p = localParts(report.now, tz);
  const { headline, lede, chips } = buildNarrative(report);
  const c = report.current;
  const min = Math.min(...report.days.map((d) => d.low)), max = Math.max(...report.days.map((d) => d.high));

  return (
    <ScrollView style={s.scroll} contentContainerStyle={[s.content, { paddingTop: insets.top + 28, paddingBottom: insets.bottom + 64 }]}>
      <View style={s.page}>
        <View style={s.metaRow}>
          <Icon name={c.isDay ? "sun" : "moon"} size={22} />
          <Meta style={{ flex: 1 }}>{`${p.weekday}, ${p.monthShort} ${p.day}  ·  ${report.station.place}`}</Meta>
        </View>
        <Meta style={{ marginTop: 4 }}>{`sunrise ${clock(report.sun.sunrise, tz)} · sunset ${clock(report.sun.sunset, tz)}`}</Meta>

        <View style={s.hero}>
          <Text style={s.big}>{deg(c.temp)}</Text>
          <Text style={s.unit}>°C</Text>
        </View>

        <View style={s.condRow}>
          <Icon name={c.sky === "rain" ? "rain" : c.sky === "snow" ? "snow" : c.isDay ? "sun" : "moon"} size={72} stroke={1.4} />
          <View style={{ flex: 1 }}>
            <Text style={s.cond}>{c.sky === "dry" ? (c.windKmh < 8 ? "dry & calm" : "dry") : c.sky}</Text>
            <Meta style={{ letterSpacing: 0.6, fontSize: 18 }}>
              {`feels ${deg(c.feels)}° · high ${deg(report.today.high)}° · low ${deg(report.today.low)}°`}
            </Meta>
          </View>
        </View>

        <Runs runs={headline} style={s.headline} />
        <Runs runs={lede} style={s.lede} />

        <View style={s.chips}>{chips.map((x) => <Pill key={x.label} {...x} />)}</View>

        <View style={s.section}>
          <SectionTitle aside={`${report.modelLabel}, ${report.station.name.toLowerCase()}, issued ${clock(report.issuedAt, tz)}`}>
            Hour by hour
          </SectionTitle>
          <HourlyChart hours={report.hours} />
          <Meta style={s.foot}>blue figures are the chance of precipitation each hour. scroll sideways for more.</Meta>
        </View>

        <View style={s.section}>
          <View style={s.titleRow}>
            <SectionTitle>The days ahead</SectionTitle>
            <Meta style={{ marginTop: 18 }}>{`${report.modelLabel}, 48 h`}</Meta>
          </View>
          <View style={s.rule} />
          {report.days.map((d) => <DayRow key={d.key} day={d} min={min} max={max} />)}
        </View>

        <View style={s.section}>
          <SectionTitle aside={`${report.station.name.toLowerCase()}, forecast run ${clock(report.initTime, tz)}`}>
            The fine print
          </SectionTitle>
          <FinePrint report={report} />
        </View>
      </View>
    </ScrollView>
  );
}

function Centered({ text }: { text: string }) {
  return <View style={[s.scroll, { justifyContent: "center", padding: 32 }]}><Meta>{text}</Meta></View>;
}

const s = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: colors.bg },
  content: { alignItems: "center" },
  page: { width: "100%", maxWidth: MAX_WIDTH, paddingHorizontal: space.page },
  metaRow: { flexDirection: "row", alignItems: "center", gap: 14 },
  hero: { flexDirection: "row", alignItems: "flex-start", marginTop: 22 },
  big: { fontFamily: fonts.serifLight, fontSize: 168, lineHeight: 176, color: colors.text, letterSpacing: -4 },
  unit: { fontFamily: fonts.serifLight, fontSize: 44, color: colors.muted, marginTop: 22, marginLeft: 6 },
  condRow: { flexDirection: "row", alignItems: "center", gap: 24, marginTop: 28 },
  cond: { fontFamily: fonts.serif, fontSize: 32, color: colors.text, marginBottom: 6 },
  headline: { fontFamily: fonts.serif, fontSize: 40, lineHeight: 46, color: colors.text, marginTop: 48, letterSpacing: -0.4 },
  lede: { fontFamily: fonts.serif, fontSize: 22, lineHeight: 38, color: colors.body, marginTop: 28 },
  chips: { gap: 14, marginTop: 36 },
  section: { marginTop: space.section + 16 },
  titleRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start" },
  rule: { height: 1.5, backgroundColor: colors.strongRule, marginTop: 20 },
  foot: { marginTop: 20, paddingTop: 20, borderTopWidth: 1, borderTopColor: colors.rule },
});
