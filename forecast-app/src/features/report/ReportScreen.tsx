import { ScrollView, StyleSheet, Text, View } from "react-native";
import { Link } from "expo-router";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useForecast } from "@/data/ForecastProvider";
import { getModel } from "@/data/registry";
import { conditionLabel, deg, summaryText } from "@/data/narrative";
import { Card } from "@/components/Card";
import { DayRow } from "@/components/DayRow";
import { HourlyStrip } from "@/components/HourlyStrip";
import { Icon } from "@/components/Icon";
import { SkyBackground } from "@/components/SkyBackground";
import { ModelAbout } from "./ModelAbout";
import { ModelPicker } from "./ModelPicker";
import { SpatialMap } from "./SpatialMap";
import { Precip, Sunrise, Tiles, Wind } from "./Tiles";
import { pageStyle, useRoomy, useWide } from "@/theme/layout";
import { dayTheme, fonts, nightTheme, ThemeContext } from "@/theme/tokens";

/**
 * The forecast page: hero, summary + hourly, day forecast, detail tiles.
 * Phones get one column; desktop puts the hero beside the hourly card and lays the tiles out in rows.
 */
export function ReportScreen() {
  const { report, status, error, activeModelId, setActiveModelId } = useForecast();
  const insets = useSafeAreaInsets();
  const wide = useWide();
  const roomy = useRoomy();
  if (status === "error") return <Centered text={`Couldn't load the forecast. ${error ?? ""}`} />;
  if (!report) return <Centered text="Loading…" />;

  const theme = report.current.isDay ? dayTheme : nightTheme;
  const c = report.current;
  const min = Math.min(...report.days.map((d) => d.low)), max = Math.max(...report.days.map((d) => d.high));

  const model = getModel(activeModelId);
  const picker = <ModelPicker activeModelId={activeModelId} onSelect={setActiveModelId} />;
  const hero = (
    <View style={[s.hero, wide && s.heroWide]}>
      <View style={s.place}>
        <Icon name="nav" size={16} color={theme.text} />
        <Text style={s.placeText}>{report.station.place}</Text>
      </View>
      <Text style={s.station}>{report.station.name}</Text>
      <Text style={s.big}>{`${deg(c.temp)}°`}</Text>
      <Text style={[s.cond, { color: theme.muted }]}>{conditionLabel(report)}</Text>
      <Text style={s.hl}>{`H:${deg(report.today.high)}°  L:${deg(report.today.low)}°`}</Text>
    </View>
  );
  const hourly = (
    <Card>
      <Text style={s.summary}>{summaryText(report)}</Text>
      <View style={[s.rule, { backgroundColor: theme.hairline }]} />
      <HourlyStrip hours={report.hours} arrows={wide} />
    </Card>
  );
  const days = (
    <Card title={`${report.days.length}-day forecast`} icon="calendar" style={[{ paddingBottom: 4 }, wide && s.fill]}>
      {report.days.map((d, i) => <DayRow key={d.key} day={d} min={min} max={max} first={i === 0} />)}
    </Card>
  );

  return (
    <ThemeContext.Provider value={theme}>
      <View style={s.root}>
        <SkyBackground />
        <ScrollView contentContainerStyle={[s.content, { paddingTop: insets.top + (wide ? 32 : 20), paddingBottom: insets.bottom + 48 }]}>
          <View style={pageStyle(wide)}>
            <View style={[s.topBar, wide && s.topBarWide]}>
              {wide ? picker : null}
              <View style={s.links}>
                <Link href="/demo" replace asChild>
                  <Text style={s.navLink}>Training demo  ›</Text>
                </Link>
                <Link href="/compare" replace asChild>
                  <Text style={s.navLink}>Compare models  ›</Text>
                </Link>
              </View>
            </View>
            {wide ? null : picker}

            {wide ? (
              <>
                <View style={s.row}>
                  <View style={[s.third, s.column]}>
                    {hero}
                    {days}
                  </View>
                  <View style={[s.twoThirds, s.column]}>
                    {hourly}
                    {roomy ? (
                      <View style={s.row}>
                        <View style={[s.cell, { flex: 1.6 }]}><Wind report={report} style={s.fill} compact /></View>
                        <View style={s.cell}><Precip report={report} style={s.fill} /></View>
                        <View style={s.cell}><Sunrise report={report} style={s.fill} /></View>
                      </View>
                    ) : (
                      <>
                        <View style={s.row}>
                          <View style={s.cell}><Precip report={report} style={s.fill} /></View>
                          <View style={s.cell}><Sunrise report={report} style={s.fill} /></View>
                        </View>
                        <Wind report={report} compact />
                      </>
                    )}
                  </View>
                </View>
              </>
            ) : (
              <>
                {hero}
                {hourly}
                {days}
                <Tiles report={report} />
              </>
            )}
            <SpatialMap wide={wide} timeZone={report.station.timeZone} now={report.now} stationName={report.station.name} />
            {model ? <ModelAbout model={model} wide={wide} /> : null}
          </View>
        </ScrollView>
      </View>
    </ThemeContext.Provider>
  );
}

function Centered({ text }: { text: string }) {
  return (
    <View style={[s.root, { backgroundColor: nightTheme.sky[0], justifyContent: "center", alignItems: "center" }]}>
      <Text style={{ color: nightTheme.muted, fontFamily: fonts.regular, fontSize: 17 }}>{text}</Text>
    </View>
  );
}

const s = StyleSheet.create({
  root: { flex: 1 },
  content: { alignItems: "center" },
  // zIndex keeps the open model dropdown above the rows below.
  topBar: { flexDirection: "row", justifyContent: "flex-end", alignItems: "center", zIndex: 20 },
  topBarWide: { justifyContent: "space-between", marginBottom: 8 },
  links: { flexDirection: "row", gap: 18 },
  navLink: { color: "#fff", fontFamily: fonts.medium, fontSize: 14, paddingVertical: 6 },
  row: { flexDirection: "row", gap: 16 },
  third: { flex: 1, minWidth: 0 },
  twoThirds: { flex: 2, minWidth: 0 },
  column: { gap: 16 },
  fill: { flex: 1 },
  cell: { flex: 1, minWidth: 0 },
  hero: { alignItems: "center", paddingBottom: 56 },
  heroWide: { paddingTop: 4, paddingBottom: 12 },
  place: { flexDirection: "row", alignItems: "center", gap: 6 },
  placeText: { fontFamily: fonts.medium, fontSize: 15, color: "rgba(255,255,255,0.8)" },
  station: { fontFamily: fonts.regular, fontSize: 30, lineHeight: 36, color: "#fff", marginTop: 6, textAlign: "center" },
  big: { fontFamily: fonts.thin, fontSize: 120, lineHeight: 132, color: "#fff", marginTop: 2, letterSpacing: -2 },
  cond: { fontFamily: fonts.medium, fontSize: 22, marginTop: 2 },
  hl: { fontFamily: fonts.medium, fontSize: 21, color: "#fff", marginTop: 4 },
  summary: { fontFamily: fonts.regular, fontSize: 17, lineHeight: 24, color: "#fff" },
  rule: { height: 1, marginVertical: 14 },
});
