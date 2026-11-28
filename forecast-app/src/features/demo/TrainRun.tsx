import { useEffect, useState } from "react";
import { Pressable, StyleSheet, Switch, Text, View } from "react-native";
import { fonts } from "@/theme/tokens";
import type { Station } from "./checks";
import { ui } from "./ui";

// Plan -> go-ahead -> train for a location without a prepared run. Talks to the
// training API in benchmarking-site/server/train_api.py; nothing trains until the
// user approves. Set EXPO_PUBLIC_TRAIN_API_URL when the API is not on localhost.
const API = (process.env.EXPO_PUBLIC_TRAIN_API_URL ?? "http://localhost:5174").replace(/\/$/, "");

interface PlanStep { stage: string; target?: string; months?: number; compute?: unknown }
interface RunState {
  status?: string;
  exit_code?: number | null;
  plan?: { steps?: PlanStep[]; problems?: string[]; note?: string };
  log_tail?: string;
}

const regionFor = (station: Station) => `demo_${station.id.toLowerCase()}`;

// Same box as train_api.HALF_LON / HALF_LAT: the networks whose stations can land in the table.
function networksNear(station: Station, stations: Station[]) {
  const near = stations.filter(
    (s) => s.network && Math.abs(s.lon - station.lon) <= 3 && Math.abs(s.lat - station.lat) <= 2.75,
  );
  return [...new Set(near.map((s) => s.network as string))].sort();
}

async function call<T>(path: string, body?: object): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, body
      ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : undefined);
  } catch {
    throw new Error(`Can't reach the training API at ${API}. Start benchmarking-site/server/npz_api.py.`);
  }
  const data = await res.json().catch(() => ({ error: `The training API answered ${res.status}` }));
  if (!res.ok) throw new Error(data.error ?? `Request failed (${res.status})`);
  return data as T;
}

const describe = (c: unknown) => (c == null ? "" : typeof c === "string" ? c : JSON.stringify(c));

export function TrainRun({ station, stations }: { station: Station; stations: Station[] }) {
  const region = regionFor(station);
  const [run, setRun] = useState<RunState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [approved, setApproved] = useState(false);

  const status = run?.status ?? "none";
  const active = status.startsWith("running");

  useEffect(() => {
    setRun(null);
    setError(null);
    setApproved(false);
    call<RunState>(`/api/train/status?region=${region}`).then(setRun).catch(() => {});
  }, [region]);

  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(() => {
      call<RunState>(`/api/train/status?region=${region}`).then(setRun).catch(() => {});
    }, 15000);
    return () => clearInterval(id);
  }, [active, region]);

  const act = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const plan = () => act(async () => {
    const res = await call<{ state: RunState }>("/api/train/plan", {
      station_id: station.id, name: station.name, lat: station.lat, lon: station.lon,
      network: station.network, networks: networksNear(station, stations),
    });
    setRun(res.state);
    setApproved(false);
  });

  const start = () => act(async () => {
    const res = await call<{ state: RunState }>("/api/train/start", { region, approve: region });
    setRun(res.state);
  });

  const steps = run?.plan?.steps ?? [];
  const problems = run?.plan?.problems ?? [];

  return (
    <View style={s.root}>
      {status === "none" ? (
        <>
          <Text style={ui.body}>Valid location, no run yet. Plan one to see what training {station.id} would take.</Text>
          <Button label={busy ? "Planning… (up to a few minutes)" : "Plan a training run"} disabled={busy} onPress={plan} />
          <Text style={ui.note}>Planning only checks the setup on Modal. It does not download or train anything.</Text>
        </>
      ) : null}

      {status === "planned" ? (
        <>
          <Text style={ui.h4}>Plan for {region}</Text>
          {problems.map((p) => <Text key={p} style={[ui.body, s.bad]}>• {p}</Text>)}
          {steps.map((st, i) => (
            <Text key={i} style={ui.body}>
              {i + 1}. <Text style={s.strong}>{st.stage}</Text>
              {st.target ? ` · ${st.target}` : ""}
              {st.months ? ` · ${st.months} months` : ""}
              {st.compute ? ` · ${describe(st.compute)}` : ""}
            </Text>
          ))}
          {run?.plan?.note ? <Text style={ui.note}>{run.plan.note}</Text> : null}
          {problems.length ? (
            <Text style={[ui.body, s.bad]}>Fix these before training. Try a different spot.</Text>
          ) : (
            <View style={s.approve}>
              <Text style={ui.body}>
                Training downloads about five years of HRRR and trains on GPUs. It takes hours and runs on the
                shared Modal account.
              </Text>
              <View style={s.row}>
                <Switch value={approved} onValueChange={setApproved} accessibilityLabel={`Go-ahead for ${region}`} />
                <Text style={[ui.body, s.flex]}>I have the go-ahead to spend Modal compute on {region}</Text>
              </View>
              <Button label="Start training" disabled={!approved || busy} onPress={start} />
            </View>
          )}
          <LinkButton label="Re-plan" disabled={busy} onPress={plan} />
        </>
      ) : null}

      {active || status === "finished" || status === "failed" ? (
        <>
          <Text style={ui.h4}>
            {region}: {active ? "training…" : status === "finished" ? "finished" : `failed (exit ${run?.exit_code})`}
          </Text>
          <Text style={ui.note}>
            {active
              ? "Refreshes every 15 seconds. Completed stages are skipped if the run is restarted."
              : `Results are on the Modal data volume under /processed/regions/${region}/.`}
          </Text>
          {run?.log_tail ? <Text style={s.log}>{run.log_tail}</Text> : null}
          {status === "failed" ? <LinkButton label="Plan again" disabled={busy} onPress={plan} /> : null}
        </>
      ) : null}

      {error ? <Text style={[ui.body, s.bad]}>{error}</Text> : null}
    </View>
  );
}

export function Button({ label, onPress, disabled }: { label: string; onPress: () => void; disabled?: boolean }) {
  return (
    <Pressable accessibilityRole="button" disabled={disabled} onPress={onPress}
      style={[s.button, disabled && s.disabled]}>
      <Text style={s.buttonText}>{label}</Text>
    </Pressable>
  );
}

export function LinkButton({ label, onPress, disabled }: { label: string; onPress: () => void; disabled?: boolean }) {
  return (
    <Pressable accessibilityRole="button" disabled={disabled} onPress={onPress} style={disabled && s.disabled}>
      <Text style={s.link}>{label}</Text>
    </Pressable>
  );
}

const s = StyleSheet.create({
  root: { gap: 10, marginTop: 8 },
  strong: { fontFamily: fonts.semi },
  bad: { color: "#ffb4ab" },
  approve: { gap: 10, padding: 12, borderRadius: 14, borderWidth: 1, borderColor: "rgba(255,255,255,0.25)" },
  row: { flexDirection: "row", alignItems: "center", gap: 10 },
  flex: { flex: 1 },
  log: {
    color: "#e6e9ff", fontFamily: "monospace", fontSize: 11, lineHeight: 15, padding: 10,
    borderRadius: 10, backgroundColor: "rgba(0,0,0,0.3)", maxHeight: 260, overflow: "hidden",
  },
  button: { alignSelf: "flex-start", paddingVertical: 10, paddingHorizontal: 16, borderRadius: 999, backgroundColor: "#10b981" },
  buttonText: { color: "#04241a", fontFamily: fonts.semi, fontSize: 15 },
  disabled: { opacity: 0.45 },
  link: { color: "#fff", fontFamily: fonts.medium, fontSize: 14, textDecorationLine: "underline", paddingVertical: 4 },
});
