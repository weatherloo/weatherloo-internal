import { useEffect, useMemo, useRef, useState, type ComponentRef } from "react";
import { Image, PanResponder, Platform, Pressable, StyleSheet, Text, View, type ViewStyle } from "react-native";
import Svg, { Circle, G, Line, LinearGradient, Path, Rect, Stop as SvgStop, Text as SvgText } from "react-native-svg";
import { Card } from "@/components/Card";
import { pngDataUri } from "@/components/png";
import { useSvgId } from "@/components/svgId";
import { hrrrMap, type HrrrMap, type MapField } from "@/data/hrrrMap";
import { capitalize, hourWord, localParts } from "@/data/time";
import { colorAt, PRECIP_MMH, TEMPERATURE, toHex, WIND_KMH, type Stop } from "@/theme/colormaps";
import { glassPill } from "@/theme/glass";
import { fonts, useTheme } from "@/theme/tokens";

const FIELDS: Record<MapField, { label: string; unit: string; stops: Stop[]; format: (v: number) => string }> = {
  t2m: { label: "Temperature", unit: "°C", stops: TEMPERATURE, format: (v) => `${Math.round(v)}` },
  wind_speed: { label: "Wind", unit: "km/h", stops: WIND_KMH, format: (v) => `${Math.round(v)}` },
  precip_rate: { label: "Precipitation", unit: "mm/h", stops: PRECIP_MMH, format: (v) => (v < 0.05 ? "0" : v < 1 ? v.toFixed(1) : `${Math.round(v)}`) },
};
const ORDER: MapField[] = ["t2m", "wind_speed", "precip_rate"];
const PLAY_MS = 800;

function render(map: HrrrMap, field: MapField, frame: number) {
  const n = map.cells, rgba = new Uint8Array(n * n * 4), stops = FIELDS[field].stops;
  for (let i = 0; i < n * n; i++) {
    const [r, g, b] = colorAt(stops, map.value(field, frame, i));
    rgba[i * 4] = r; rgba[i * 4 + 1] = g; rgba[i * 4 + 2] = b; rgba[i * 4 + 3] = 255;
  }
  return pngDataUri(n, n, rgba);
}

/**
 * Real NOAA HRRR fields around the station: temperature, wind and precipitation on the native 3 km grid,
 * every 3 hours of one run, with a time scrubber. Raw HRRR; our correction applies only at stations.
 */
export function SpatialMap({ wide, timeZone, now, stationName }: { wide: boolean; timeZone: string; now: Date; stationName: string }) {
  const t = useTheme();
  const map = useMemo(hrrrMap, []);
  const [field, setField] = useState<MapField>("t2m");
  const startFrame = Math.max(0, map.leads.findIndex((l) => map.initTime.getTime() + l * 3600e3 >= now.getTime()));
  const [frame, setFrame] = useState(startFrame);
  const [playing, setPlaying] = useState(false);
  const [box, setBox] = useState(0);
  const images = useRef(new Map<string, string>());

  const uri = useMemo(() => {
    const key = `${field}:${frame}`;
    if (!images.current.has(key)) images.current.set(key, render(map, field, frame));
    return images.current.get(key)!;
  }, [map, field, frame]);

  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => setFrame((f) => (f + 1) % map.leads.length), PLAY_MS);
    return () => clearInterval(id);
  }, [playing, map.leads.length]);

  const lead = map.leads[frame];
  const valid = new Date(map.initTime.getTime() + lead * 3600e3);
  const local = localParts(valid, timeZone);
  const when = `${capitalize(local.weekday.slice(0, 3))} ${hourWord(local.hour)}`;
  const run = localParts(map.initTime, "UTC");
  const spec = FIELDS[field];
  const atSite = map.sample(field, frame, map.site.col, map.site.row);
  const size = wide ? Math.min(box, 520) : box;

  const tabs = (
    <View style={[s.tabs, glassPill]}>
      {ORDER.map((k) => (
        <Pressable key={k} onPress={() => setField(k)} accessibilityRole="button" accessibilityState={{ selected: field === k }}
          style={[s.tab, field === k && s.tabOn]}>
          <Text style={[s.tabText, { color: t.text }]}>{FIELDS[k].label}</Text>
        </Pressable>
      ))}
    </View>
  );

  const readout = (
    <View style={s.readout}>
      <Text style={[s.when, { color: t.text }]}>{when}</Text>
      <Text style={[s.sub, { color: t.muted }]}>+{lead} h · {stationName}: {spec.format(atSite)} {spec.unit}</Text>
    </View>
  );

  const controls = (
    <View style={s.controls}>
      <Legend field={field} range={map.range(field)} />
      <View style={s.scrubRow}>
        <Pressable onPress={() => setPlaying((p) => !p)} accessibilityRole="button"
          accessibilityLabel={playing ? "Pause" : "Play through the forecast"} style={[s.play, glassPill]}>
          <Text style={[s.playText, { color: t.text }]}>{playing ? "❚❚" : "▶"}</Text>
        </Pressable>
        <TimeScrubber
          count={map.leads.length}
          frame={frame}
          label={`+${lead} hours, ${when}`}
          onChange={(i) => { setPlaying(false); setFrame(i); }}
        />
      </View>
      <Text style={[s.note, { color: t.muted }]}>
        Real NOAA HRRR output (run {String(run.hour).padStart(2, "0")} UTC, {run.day} {run.monthShort}), 3 km grid. Raw HRRR:
        our correction only applies at stations, and the station forecast above is a demo fixture.
      </Text>
    </View>
  );

  const mapView = (
    <View style={[s.mapWrap, wide && { flex: 1.15 }]} onLayout={(e) => setBox(Math.floor(e.nativeEvent.layout.width))}>
      {size ? (
        <View style={[s.map, { width: size, height: size }]}>
          <Image source={{ uri }} style={{ width: size, height: size }} resizeMode="stretch"
            accessibilityLabel={`${spec.label} map, ${when}`} />
          <Overlay map={map} field={field} frame={frame} size={size} />
        </View>
      ) : null}
    </View>
  );

  return (
    <Card title="HRRR forecast map" icon="nav">
      {wide ? (
        <View style={s.wideRow}>
          {mapView}
          <View style={s.side}>
            {tabs}
            {readout}
            {controls}
          </View>
        </View>
      ) : (
        <View style={s.stack}>
          {tabs}
          {mapView}
          {readout}
          {controls}
        </View>
      )}
    </Card>
  );
}

/** Wind arrows, place names, the station, a north arrow and a scale bar, in cell units. */
function Overlay({ map, field, frame, size }: { map: HrrrMap; field: MapField; frame: number; size: number }) {
  const n = map.cells, every = map.windEvery, coarse = Math.floor(n / every), start = Math.floor(every / 2);
  const font = (12.5 * n) / size; // ~12.5 px on screen at any map size
  const arrows = field === "wind_speed" ? Array.from({ length: coarse * coarse }, (_, k) => {
    const r = Math.floor(k / coarse), c = k % coarse;
    const col = c * every + start + 0.5, row = r * every + start + 0.5;
    const deg = map.windToward(frame, r, c) * (Math.PI / 180);
    const speed = map.sample("wind_speed", frame, col - 0.5, row - 0.5);
    const len = 0.9 + Math.min(speed / 45, 1) * 1.6;
    const dx = Math.sin(deg) * len, dy = -Math.cos(deg) * len;
    const hx = col + dx, hy = row + dy, back = Math.atan2(-dy, -dx);
    const head = `M${hx},${hy} L${hx + Math.cos(back + 0.5) * 0.8},${hy + Math.sin(back + 0.5) * 0.8} M${hx},${hy} L${hx + Math.cos(back - 0.5) * 0.8},${hy + Math.sin(back - 0.5) * 0.8}`;
    return (
      <G key={k} opacity={0.75}>
        <Line x1={col - dx} y1={row - dy} x2={hx} y2={hy} stroke="#fff" strokeWidth={0.28} strokeLinecap="round" />
        <Path d={head} stroke="#fff" strokeWidth={0.28} strokeLinecap="round" fill="none" />
      </G>
    );
  }) : null;
  const north = map.northDeg * (Math.PI / 180), nx = n - 5, ny = 5;
  const scaleCells = 50 / map.spacingKm;

  return (
    <Svg width={size} height={size} viewBox={`0 0 ${n} ${n}`} style={StyleSheet.absoluteFill} pointerEvents="none">
      {arrows}
      {map.places.map((p) => {
        const x = p.col + 0.5, y = p.row + 0.5;
        if (p.water) {
          return <SvgText key={p.name} x={x} y={y} fontSize={font * 0.9} fontFamily={fonts.medium} fill="rgba(220,235,255,0.75)"
            fontStyle="italic" textAnchor="middle">{p.name}</SvgText>;
        }
        const ly = y < 6 ? y + font + 0.4 : y - 0.8; // names near the top edge go under the dot
        const right = x > n - 16; // and names near the right edge go to the left of it
        const lx = right ? x - 1 : x + 1, anchor = right ? "end" : "start";
        const label = p.name === "Waterloo" ? null : (
          <>
            <SvgText x={lx} y={ly} textAnchor={anchor} fontSize={font} fontFamily={fonts.semi} stroke="rgba(8,12,40,0.7)" strokeWidth={0.5}
              strokeLinejoin="round">{p.name}</SvgText>
            <SvgText x={lx} y={ly} textAnchor={anchor} fontSize={font} fontFamily={fonts.semi} fill="#fff">{p.name}</SvgText>
          </>
        );
        return (
          <G key={p.name}>
            <Circle cx={x} cy={y} r={0.45} fill="#fff" stroke="rgba(8,12,40,0.7)" strokeWidth={0.2} />
            {label}
          </G>
        );
      })}
      <Circle cx={map.site.col + 0.5} cy={map.site.row + 0.5} r={1.4} fill="none" stroke="#fff" strokeWidth={0.45} />
      <Circle cx={map.site.col + 0.5} cy={map.site.row + 0.5} r={0.55} fill="#fff" />
      <G>
        <Circle cx={nx} cy={ny} r={2.6} fill="rgba(8,12,40,0.45)" />
        <Line x1={nx - Math.sin(north) * 1.6} y1={ny + Math.cos(north) * 1.6} x2={nx + Math.sin(north) * 1.6} y2={ny - Math.cos(north) * 1.6}
          stroke="#fff" strokeWidth={0.35} strokeLinecap="round" />
        <SvgText x={nx + Math.sin(north) * 2.1} y={ny - Math.cos(north) * 2.1 - 0.9} fontSize={font * 0.8} fontFamily={fonts.semi}
          fill="#fff" textAnchor="middle">N</SvgText>
      </G>
      <G>
        <Rect x={3} y={n - 4.2} width={scaleCells} height={0.35} fill="#fff" />
        <SvgText x={3} y={n - 5} fontSize={font * 0.8} fontFamily={fonts.medium} fill="#fff">50 km</SvgText>
      </G>
    </Svg>
  );
}

/**
 * Draggable time bar: press anywhere to jump, drag to scrub, arrow keys or screen-reader
 * increment/decrement to step. Positions snap to the nearest frame.
 */
function TimeScrubber({ count, frame, label, onChange }: {
  count: number; frame: number; label: string; onChange: (frame: number) => void;
}) {
  const track = useRef<ComponentRef<typeof View>>(null);
  const box = useRef({ left: 0, width: 1 });
  const change = useRef(onChange);
  change.current = onChange;
  const [dragging, setDragging] = useState(false);

  const at = (pageX: number) => {
    const f = Math.max(0, Math.min(1, (pageX - box.current.left) / box.current.width));
    change.current(Math.round(f * (count - 1)));
  };
  const responder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: () => true,
    onPanResponderTerminationRequest: () => false,
    onPanResponderGrant: (e) => {
      const pageX = e.nativeEvent.pageX; // read now: the event is gone by the time measure answers
      setDragging(true);
      track.current?.measure((_x, _y, width, _h, left) => {
        box.current = { left, width: width || 1 };
        at(pageX);
      });
    },
    onPanResponderMove: (e) => at(e.nativeEvent.pageX),
    onPanResponderRelease: () => setDragging(false),
    onPanResponderTerminate: () => setDragging(false),
  }), [count]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = (d: number) => onChange(Math.max(0, Math.min(count - 1, frame + d)));
  const pct = count > 1 ? (frame / (count - 1)) * 100 : 0;
  return (
    <View
      ref={track}
      {...responder.panHandlers}
      focusable
      accessible
      accessibilityRole="adjustable"
      accessibilityLabel="Forecast hour"
      accessibilityValue={{ min: 0, max: count - 1, now: frame, text: label }}
      // react-native-web reads the aria-* props, not accessibilityValue.
      aria-valuemin={0}
      aria-valuemax={count - 1}
      aria-valuenow={frame}
      aria-valuetext={label}
      accessibilityActions={[{ name: "increment" }, { name: "decrement" }]}
      onAccessibilityAction={(e) => step(e.nativeEvent.actionName === "increment" ? 1 : -1)}
      onKeyDown={(e) => {
        const key = (e.nativeEvent as unknown as { key?: string }).key;
        if (key === "ArrowRight" || key === "ArrowLeft") { e.preventDefault(); step(key === "ArrowRight" ? 1 : -1); }
      }}
      style={[s.scrub, WEB_DRAG]}
    >
      <View style={s.rail}>
        <View style={[s.railFill, { width: `${pct}%` }]} />
        {Array.from({ length: count }, (_, i) => (
          <View key={i} style={[s.notch, { left: `${count > 1 ? (i / (count - 1)) * 100 : 0}%` }]} />
        ))}
      </View>
      <View style={[s.thumb, dragging && s.thumbActive, { left: `${pct}%` }]} />
    </View>
  );
}

// Web: no page scroll or text selection while dragging the bar, and a grab cursor.
const WEB_DRAG = (Platform.OS === "web" ? { touchAction: "none", userSelect: "none", cursor: "grab" } : {}) as ViewStyle;

function Legend({ field, range }: { field: MapField; range: [number, number] }) {
  const t = useTheme();
  const id = useSvgId("legend");
  const spec = FIELDS[field];
  const [lo, hi] = range;
  // Sample the scale through the data range; precipitation uses sqrt spacing so light rain gets room.
  const pos = (f: number) => (field === "precip_rate" ? lo + (hi - lo) * f * f : lo + (hi - lo) * f);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => ({ f, v: pos(f) }));
  return (
    <View style={s.legend}>
      <View style={s.legendHead}>
        <Text style={[s.legendTitle, { color: t.text }]}>{spec.label}</Text>
        <Text style={[s.legendUnit, { color: t.muted }]}>{spec.unit}</Text>
      </View>
      <Svg width="100%" height={12} viewBox="0 0 100 12" preserveAspectRatio="none">
        <LinearGradient id={id} x1="0" y1="0" x2="1" y2="0">
          {Array.from({ length: 11 }, (_, i) => (
            <SvgStop key={i} offset={i / 10} stopColor={toHex(colorAt(spec.stops, pos(i / 10)))} />
          ))}
        </LinearGradient>
        <Rect width={100} height={12} rx={4} fill={`url(#${id})`} />
      </Svg>
      <View style={s.legendTicks}>
        {ticks.map(({ f, v }) => <Text key={f} style={[s.legendTick, { color: t.muted }]}>{spec.format(v)}</Text>)}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  wideRow: { flexDirection: "row", gap: 24, alignItems: "flex-start" },
  side: { flex: 1, gap: 18, minWidth: 0 },
  stack: { gap: 14 },
  mapWrap: { width: "100%" },
  map: { borderRadius: 16, overflow: "hidden", borderWidth: 1, borderColor: "rgba(255,255,255,0.14)" },
  tabs: { flexDirection: "row", borderRadius: 16, padding: 4, backgroundColor: "rgba(70,78,140,0.45)" },
  tab: { flex: 1, alignItems: "center", borderRadius: 12, paddingVertical: 9 },
  tabOn: { backgroundColor: "rgba(255,255,255,0.18)" },
  tabText: { fontFamily: fonts.medium, fontSize: 13 },
  readout: { gap: 2 },
  when: { fontFamily: fonts.light, fontSize: 34 },
  sub: { fontFamily: fonts.regular, fontSize: 14 },
  controls: { gap: 14 },
  legend: { gap: 6 },
  legendHead: { flexDirection: "row", justifyContent: "space-between" },
  legendTitle: { fontFamily: fonts.medium, fontSize: 13 },
  legendUnit: { fontFamily: fonts.regular, fontSize: 12 },
  legendTicks: { flexDirection: "row", justifyContent: "space-between" },
  legendTick: { fontFamily: fonts.regular, fontSize: 11, fontVariant: ["tabular-nums"] },
  scrubRow: { flexDirection: "row", alignItems: "center", gap: 12 },
  play: { width: 36, height: 36, borderRadius: 18, alignItems: "center", justifyContent: "center", backgroundColor: "rgba(255,255,255,0.1)" },
  playText: { fontFamily: fonts.medium, fontSize: 12 },
  // Tall hit area around a thin rail; the thumb is centred on the current frame.
  scrub: { flex: 1, height: 32, justifyContent: "center", marginHorizontal: 9 },
  rail: { height: 6, borderRadius: 3, backgroundColor: "rgba(255,255,255,0.16)" },
  railFill: { position: "absolute", left: 0, top: 0, bottom: 0, borderRadius: 3, backgroundColor: "rgba(255,255,255,0.45)" },
  notch: { position: "absolute", top: 1, width: 2, height: 4, marginLeft: -1, borderRadius: 1, backgroundColor: "rgba(8,12,40,0.35)" },
  thumb: {
    position: "absolute", top: 7, width: 18, height: 18, marginLeft: -9, borderRadius: 9, backgroundColor: "#fff",
    boxShadow: "0 2px 8px rgba(3,6,25,0.45)",
  },
  thumbActive: { transform: [{ scale: 1.2 }] },
  note: { fontFamily: fonts.regular, fontSize: 12, lineHeight: 17 },
});
