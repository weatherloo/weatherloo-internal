import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { ForecastRecord } from "./contract";
import { DEFAULT_MODEL_ID } from "./registry";
import { DEFAULT_STATION_ID, getStation, type Station } from "./stations";
import { forecastSource } from "./source";
import { buildReport, type Report } from "./report";

interface ForecastState {
  /** Active model id. A model switcher only has to call setActiveModelId. */
  activeModelId: string;
  setActiveModelId: (id: string) => void;
  station: Station;
  setStationId: (id: string) => void;
  records: ForecastRecord[];
  report: Report | null;
  status: "loading" | "ready" | "error";
  error?: string;
}

const Ctx = createContext<ForecastState | null>(null);

export function ForecastProvider({ children }: { children: ReactNode }) {
  const [activeModelId, setActiveModelId] = useState(DEFAULT_MODEL_ID);
  const [stationId, setStationId] = useState(DEFAULT_STATION_ID);
  const [records, setRecords] = useState<ForecastRecord[]>([]);
  const [status, setStatus] = useState<ForecastState["status"]>("loading");
  const [error, setError] = useState<string>();

  useEffect(() => {
    let live = true;
    setStatus("loading");
    forecastSource
      .load(activeModelId, stationId)
      .then((r) => live && (setRecords(r), setStatus("ready")))
      .catch((e) => live && (setError(String(e)), setStatus("error")));
    return () => {
      live = false;
    };
  }, [activeModelId, stationId]);

  const station = getStation(stationId);
  const report = useMemo(
    () => (records.length ? buildReport(records, station, forecastSource.now()) : null),
    [records, station],
  );

  const value: ForecastState = { activeModelId, setActiveModelId, station, setStationId, records, report, status, error };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useForecast(): ForecastState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useForecast must be used inside <ForecastProvider>");
  return v;
}
