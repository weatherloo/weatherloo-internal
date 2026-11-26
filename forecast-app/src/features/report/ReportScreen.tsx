import { ScrollView, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { useForecast } from "@/data/ForecastProvider";
import { conditionLabel, deg, summaryText } from "@/data/narrative";
import { Card } from "@/components/Card";
import { DayRow } from "@/components/DayRow";
import { HourlyStrip } from "@/components/HourlyStrip";
import { Icon } from "@/components/Icon";
import { SkyBackground } from "@/components/SkyBackground";
import { Tiles } from "./Tiles";
import { dayTheme, fonts, MAX_WIDTH, nightTheme, ThemeContext } from "@/theme/tokens";

/** The forecast page: hero, summary + hourly, day forecast, detail tiles. */
export function ReportScreen() {
  const { report, status, error } = useForecast();
  const insets = useSafeAreaInsets();
  if (status === "error") return <Centered text={`Couldn't load the forecast. ${error ?? ""}`} />;
  if (!report) return <Centered text="Loading…" />;

  const theme = report.current.isDay ? dayTheme : nightTheme;
  const c = report.current;
  const min = Math.min(...report.days.map((d) => d.low)), max = Math.max(...report.days.map((d) => d.high));

  return (
    <ThemeContext.Provider value={theme}>
      <View style={s.root}>
        <SkyBackground />
        <ScrollView contentContainerStyle={[s.content, { paddingTop: insets.top + 56, paddingBottom: insets.bottom + 48 }]}>
          <View style={s.page}>
            <View style={s.hero}>
              <View style={s.place}>
                <Icon name="nav" size={16} color={theme.text} />
                <Text style={s.placeText}>{report.station.place.toUpperCase()}</Text>
              </View>
              <Text style={s.station}>{report.station.name}</Text>
              <Text style={s.big}>{`${deg(c.temp)}°`}</Text>
              <Text style={[s.cond, { color: theme.muted }]}>{conditionLabel(report)}</Text>
              <Text style={s.hl}>{`H:${deg(report.today.high)}°  L:${deg(report.today.low)}°`}</Text>
            </View>

            <Card>
              <Text style={s.summary}>{summaryText(report)}</Text>
              <View style={[s.rule, { backgroundColor: theme.hairline }]} />
              <HourlyStrip hours={report.hours} />
            </Card>

            <Card title={`${report.days.length}-day forecast`} icon="calendar" style={{ paddingBottom: 4 }}>
              {report.days.map((d, i) => <DayRow key={d.key} day={d} min={min} max={max} first={i === 0} />)}
            </Card>

            <Tiles report={report} />
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
  page: { width: "100%", maxWidth: MAX_WIDTH, paddingHorizontal: 16, gap: 12 },
  hero: { alignItems: "center", paddingBottom: 56 },
  place: { flexDirection: "row", alignItems: "center", gap: 6 },
  placeText: { fontFamily: fonts.semi, fontSize: 17, color: "#fff", letterSpacing: 0.4 },
  station: { fontFamily: fonts.regular, fontSize: 28, color: "#fff", marginTop: 4 },
  big: { fontFamily: fonts.thin, fontSize: 120, lineHeight: 132, color: "#fff", marginTop: 2, letterSpacing: -2 },
  cond: { fontFamily: fonts.medium, fontSize: 22, marginTop: 2 },
  hl: { fontFamily: fonts.medium, fontSize: 21, color: "#fff", marginTop: 4 },
  summary: { fontFamily: fonts.regular, fontSize: 17, lineHeight: 24, color: "#fff" },
  rule: { height: 1, marginVertical: 14 },
});
