import { useEffect, useRef } from "react";
import OlMap from "ol/Map";
import View from "ol/View";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import XYZ from "ol/source/XYZ";
import VectorSource from "ol/source/Vector";
import Feature from "ol/Feature";
import Point from "ol/geom/Point";
import { fromLonLat, transformExtent } from "ol/proj";
import Style from "ol/style/Style";
import CircleStyle from "ol/style/Circle";
import Fill from "ol/style/Fill";
import Stroke from "ol/style/Stroke";
import "ol/ol.css";
import {
  SOUTH_ONTARIO_EXTENT_4326,
  STATIONS,
} from "../constants.js";

const SOUTH_ONTARIO_EXTENT = transformExtent(
  SOUTH_ONTARIO_EXTENT_4326,
  "EPSG:4326",
  "EPSG:3857",
);

function stationStyle(selectedId) {
  return (feature) => {
    const selected = feature.get("id") === selectedId;
    return new Style({
      image: new CircleStyle({
        radius: selected ? 9 : 7,
        fill: new Fill({ color: selected ? "#f0f6fc" : "#58a6ff" }),
        stroke: new Stroke({ color: "#0d1117", width: 2 }),
      }),
    });
  };
}

export default function StationMap({ selectedLocationId, onSelectLocation }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const stationsLayerRef = useRef(null);

  useEffect(() => {
    const stationFeatures = STATIONS.map(
      (station) =>
        new Feature({
          geometry: new Point(fromLonLat([station.lon, station.lat])),
          id: station.id,
          label: station.label,
        }),
    );

    const stationsSource = new VectorSource({ features: stationFeatures });
    const stationsLayer = new VectorLayer({
      source: stationsSource,
      style: stationStyle(selectedLocationId),
    });
    stationsLayerRef.current = stationsLayer;

    const darkBasemap = new TileLayer({
      source: new XYZ({
        url: "https://{a-d}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        attributions:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
        maxZoom: 19,
      }),
    });

    const map = new OlMap({
      target: containerRef.current,
      layers: [darkBasemap, stationsLayer],
      view: new View({
        extent: SOUTH_ONTARIO_EXTENT,
        constrainOnlyCenter: false,
        minZoom: 7,
        maxZoom: 13,
      }),
      controls: [],
    });

    map.getView().fit(SOUTH_ONTARIO_EXTENT, {
      padding: [24, 24, 24, 24],
      maxZoom: 9,
    });

    map.on("click", (evt) => {
      const feature = map.forEachFeatureAtPixel(evt.pixel, (f) => f, {
        layerFilter: (layer) => layer === stationsLayer,
      });
      if (feature) onSelectLocation(feature.get("id"));
    });

    map.on("pointermove", (evt) => {
      const hit = map.hasFeatureAtPixel(evt.pixel, {
        layerFilter: (layer) => layer === stationsLayer,
      });
      map.getTargetElement().style.cursor = hit ? "pointer" : "";
    });

    mapRef.current = map;
    const resizeTimer = setTimeout(() => map.updateSize(), 0);

    return () => {
      clearTimeout(resizeTimer);
      map.setTarget(undefined);
      mapRef.current = null;
      stationsLayerRef.current = null;
    };
  }, [onSelectLocation]);

  useEffect(() => {
    stationsLayerRef.current?.setStyle(stationStyle(selectedLocationId));
  }, [selectedLocationId]);

  useEffect(() => {
    mapRef.current?.updateSize();
  }, [selectedLocationId]);

  return (
    <div
      id="map"
      ref={containerRef}
      aria-label="Southern Ontario map with weather stations"
    />
  );
}
