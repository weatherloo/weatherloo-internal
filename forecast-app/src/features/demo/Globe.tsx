import { useEffect, useMemo, useRef, useState } from "react";
import { PanResponder, Pressable, StyleSheet, Text, View, type LayoutChangeEvent } from "react-native";
import Svg, { Circle, Defs, G, Path, RadialGradient, Stop, Text as SvgText } from "react-native-svg";
import { geoCircle, geoDistance, geoGraticule10, geoOrthographic, geoPath } from "d3-geo";
import { feature, mesh } from "topojson-client";
import type { GeometryCollection, Topology } from "topojson-specification";
import world from "world-atlas/countries-110m.json";
import { fonts } from "@/theme/tokens";
import type { LatLon } from "./checks";

const SIZE = 600; // SVG units; the view scales to its width
const topo = world as unknown as Topology<{ land: GeometryCollection; countries: GeometryCollection }>;
const LAND = feature(topo, topo.objects.land);
const BORDERS = mesh(topo, topo.objects.countries, (a, b) => a !== b);
const GRATICULE = geoGraticule10();
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

const C = {
  cyan: "#67e8f9", ok: "#10b981", bad: "#f87171", ink: "#f8fbff", dark: "#081224",
};

export interface Pin extends LatLon { id: string; label: string; left?: boolean }

interface Props {
  domain: GeoJSON.Polygon;
  pins: Pin[];
  tap: LatLon | null;
  station: LatLon | null;
  radiusKm: number;
  tapOk: boolean;
  focus: LatLon | null;
  onTap: (p: LatLon) => void;
  onPin: (id: string) => void;
}

/** Orthographic globe: drag to rotate, buttons to zoom, tap to pick a spot or a pin. */
export function Globe({ domain, pins, tap, station, radiusKm, tapOk, focus, onTap, onPin }: Props) {
  const [rotation, setRotation] = useState<[number, number, number]>([97.5, -40, 0]);
  const [zoom, setZoom] = useState(1.6);
  const [width, setWidth] = useState(SIZE);

  useEffect(() => {
    if (focus) setRotation([-focus.lon, -focus.lat, 0]);
  }, [focus]);

  const projection = useMemo(
    () => geoOrthographic().scale((SIZE / 2 - 8) * zoom).translate([SIZE / 2, SIZE / 2]).rotate(rotation).clipAngle(90),
    [rotation, zoom],
  );
  const path = geoPath(projection);
  const centre: [number, number] = [-rotation[0], -rotation[1]];
  const project = (p: LatLon) =>
    geoDistance([p.lon, p.lat], centre) < Math.PI / 2 - 0.01 ? projection([p.lon, p.lat]) : null;

  // The responder is created once, so it reads the latest values through a ref.
  const live = useRef({ rotation, zoom, width, projection, pins, project, onTap, onPin });
  live.current = { rotation, zoom, width, projection, pins, project, onTap, onPin };
  const start = useRef<[number, number, number]>(rotation);

  const responder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: () => true,
        onPanResponderTerminationRequest: () => false,
        onPanResponderGrant: () => {
          start.current = live.current.rotation;
        },
        onPanResponderMove: (_e, g) => {
          if (Math.abs(g.dx) + Math.abs(g.dy) <= 4) return;
          const k = 0.3 / live.current.zoom;
          const [r0, r1] = start.current;
          setRotation([r0 + g.dx * k, clamp(r1 - g.dy * k, -85, 85), 0]);
        },
        onPanResponderRelease: (e, g) => {
          if (Math.abs(g.dx) + Math.abs(g.dy) > 4) return;
          const L = live.current;
          const scale = SIZE / L.width;
          const x = e.nativeEvent.locationX * scale;
          const y = e.nativeEvent.locationY * scale;
          const hit = L.pins.find((p) => {
            const xy = L.project(p);
            return xy && Math.hypot(xy[0] - x, xy[1] - y) < 16;
          });
          if (hit) return L.onPin(hit.id);
          if (Math.hypot(x - SIZE / 2, y - SIZE / 2) > L.projection.scale()) return;
          const ll = L.projection.invert?.([x, y]);
          if (ll) L.onTap({ lat: ll[1], lon: ll[0] });
        },
      }),
    [],
  );

  const circle = tap ? geoCircle().center([tap.lon, tap.lat]).radius(radiusKm / 111.195)() : null;
  const tapXY = tap && project(tap);
  const d = (g: Parameters<typeof path>[0]) => path(g) ?? undefined;

  return (
    <View style={s.wrap} onLayout={(e: LayoutChangeEvent) => setWidth(e.nativeEvent.layout.width || SIZE)}>
      <View
        {...responder.panHandlers}
        style={{ width: "100%", aspectRatio: 1 }}
        accessible
        accessibilityLabel="Globe. Drag to rotate, use the buttons to zoom, tap a spot to check it."
      >
        <Svg width="100%" height="100%" viewBox={`0 0 ${SIZE} ${SIZE}`} pointerEvents="none">
          <Defs>
            <RadialGradient id="ocean" cx="40%" cy="35%" r="75%">
              <Stop offset="0" stopColor="#173158" />
              <Stop offset="1" stopColor={C.dark} />
            </RadialGradient>
          </Defs>
          <Path d={d({ type: "Sphere" })} fill="url(#ocean)" stroke="rgba(103,232,249,0.35)" />
          <Path d={d(GRATICULE)} fill="none" stroke="rgba(148,163,184,0.14)" />
          <Path d={d(LAND)} fill="#22395f" stroke="#2f4c78" strokeWidth={0.6} />
          <Path d={d(BORDERS)} fill="none" stroke="rgba(203,213,225,0.35)" strokeWidth={0.6} />
          <Path d={d(domain)} fill="rgba(103,232,249,0.14)" stroke={C.cyan} strokeWidth={1.2} />
          {circle ? (
            <Path
              d={d(circle)}
              fill={tapOk ? "rgba(16,185,129,0.16)" : "rgba(248,113,113,0.14)"}
              stroke={tapOk ? C.ok : C.bad}
              strokeDasharray="4 3"
            />
          ) : null}
          {tap && station ? (
            <Path
              d={d({ type: "LineString", coordinates: [[tap.lon, tap.lat], [station.lon, station.lat]] })}
              fill="none" stroke={C.ink} strokeWidth={1.2} strokeDasharray="3 3"
            />
          ) : null}
          {pins.map((p) => {
            const xy = project(p);
            return xy ? (
              <G key={p.id} x={xy[0]} y={xy[1]}>
                <Circle r={6} fill={C.ok} stroke={C.ink} strokeWidth={1.5} />
                {/* Outline first, then the label: react-native-svg has no paint-order. */}
                {[{ stroke: "rgba(8,18,36,0.85)", strokeWidth: 3 }, {}].map((extra, k) => (
                  <SvgText
                    key={k} x={p.left ? -10 : 10} y={-9} textAnchor={p.left ? "end" : "start"}
                    fill={C.ink} fontSize={13} fontWeight="600" {...extra}
                  >
                    {p.label}
                  </SvgText>
                ))}
              </G>
            ) : null;
          })}
          {tapXY ? <Circle cx={tapXY[0]} cy={tapXY[1]} r={4.5} fill={C.ink} stroke={C.dark} strokeWidth={1.5} /> : null}
        </Svg>
      </View>
      <View style={s.zoom}>
        {([["+", 1.25, "Zoom in"], ["−", 1 / 1.25, "Zoom out"]] as const).map(([label, k, a11y]) => (
          <Pressable key={label} accessibilityRole="button" accessibilityLabel={a11y} style={s.zoomBtn}
            onPress={() => setZoom((z) => clamp(z * k, 1, 10))}>
            <Text style={s.zoomText}>{label}</Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

const s = StyleSheet.create({
  wrap: { width: "100%", position: "relative" },
  zoom: { position: "absolute", right: 8, bottom: 8, gap: 6 },
  zoomBtn: {
    width: 36, height: 36, borderRadius: 18, alignItems: "center", justifyContent: "center",
    backgroundColor: "rgba(8,18,36,0.75)", borderWidth: 1, borderColor: "rgba(255,255,255,0.2)",
  },
  zoomText: { color: "#fff", fontFamily: fonts.semi, fontSize: 18, lineHeight: 20 },
});
