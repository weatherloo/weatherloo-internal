"""Training-demo endpoints: plan and launch an hrrr_cnn region run on Modal.

Nothing spends compute without a go-ahead:

  POST /api/train/plan    {station_id, name, lat, lon, network}
      Writes models/hrrr_cnn/regions/demo_<id>.json and runs the pipeline's
      plan (`modal run ... --region <name>`, no --execute). Returns the plan.
  POST /api/train/start   {region, approve}
      Starts `modal run ... --region <name> --execute` only when `approve`
      repeats the region name exactly. Output goes to logs/hrrr_cnn_runs/.
  GET  /api/train/status?region=<name>
      State (planned / running / finished / failed) and the log tail.

Needs the Modal CLI on PATH (or MODAL_BIN) and `modal setup` done once.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
REGION_DIR = REPO / "models" / "hrrr_cnn" / "regions"
PIPELINE = "models/hrrr_cnn/modal_region.py"
RUN_DIR = REPO / "logs" / "hrrr_cnn_runs"

STATION_ID_RE = re.compile(r"^[A-Z0-9]{3,5}$")
NETWORK_RE = re.compile(r"^(?:[A-Z]{2}|CA_[A-Z]{2})$")
REGION_RE = re.compile(r"^demo_[a-z0-9]{3,5}$")

# Station-table box around the target, in degrees. San Francisco used about
# 6 x 5.5; the anchor's native HRRR chunk (~800 x 900 km) is the real limit
# and the plan reports any target outside it before compute.
HALF_LON, HALF_LAT = 3.0, 2.75

_lock = threading.Lock()
_running: dict[str, subprocess.Popen] = {}


class TrainError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def modal_bin() -> str:
    found = os.environ.get("MODAL_BIN") or shutil.which("modal")
    if not found:
        raise TrainError(503, "Modal CLI not found. Run `pip install modal` and `modal setup`, then restart the API.")
    return found


def state_path(region: str) -> Path:
    return RUN_DIR / f"{region}.json"


def log_path(region: str) -> Path:
    return RUN_DIR / f"{region}.log"


def read_state(region: str) -> dict[str, Any]:
    path = state_path(region)
    return json.loads(path.read_text()) if path.is_file() else {}


def write_state(region: str, **changes: Any) -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    state = {**read_state(region), **changes, "updated": now()}
    state_path(region).write_text(json.dumps(state, indent=2) + "\n")
    return state


def region_config(station_id: str, name: str, lat: float, lon: float, network: str, networks: list[str]) -> dict:
    target = f"asos_{station_id}"
    return {
        "name": f"demo_{station_id.lower()}",
        "description": f"Training demo run: {name} (IEM ASOS {station_id}). Created from the demo globe.",
        "anchor": target,
        "method": "hrrr_cnn",
        "station_root": f"/data/processed/regions/demo_{station_id.lower()}/stations_v1",
        "stations": {
            "bbox": [round(lon - HALF_LON, 2), round(lat - HALF_LAT, 2),
                     round(lon + HALF_LON, 2), round(lat + HALF_LAT, 2)],
            "asos_networks": networks,
            "eccc": network.startswith("CA_") or lat + HALF_LAT > 42.0,
        },
        "k_neighbors": 30,
        "patch_half": 8,
        "version": "v1",
        # Temperature and wind only: precipitation needs a checked gauge record.
        "targets": [{"id": target, "cnn_options": {"trajectory": True}}],
    }


def parse_plan(stdout: str) -> dict[str, Any] | None:
    """The plan entrypoint prints one indented JSON object; pull it out of the CLI output."""
    start, end = stdout.find("{"), stdout.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(stdout[start:end + 1])
    except json.JSONDecodeError:
        return None


def plan(body: dict[str, Any]) -> dict[str, Any]:
    station_id = str(body.get("station_id", "")).upper()
    network = str(body.get("network", "")).upper()
    networks = sorted({str(n).upper() for n in body.get("networks", []) if NETWORK_RE.fullmatch(str(n).upper())})
    if not STATION_ID_RE.fullmatch(station_id) or not NETWORK_RE.fullmatch(network):
        raise TrainError(400, "station_id and network (IEM ASOS) are required")
    try:
        lat, lon = float(body["lat"]), float(body["lon"])
    except (KeyError, TypeError, ValueError):
        raise TrainError(400, "lat and lon are required") from None
    config = region_config(station_id, str(body.get("name", station_id))[:80], lat, lon, network,
                           sorted(set(networks) | {network}))
    region = config["name"]
    if region in _running and _running[region].poll() is None:
        raise TrainError(409, f"{region} is already training")

    modal = modal_bin()
    REGION_DIR.mkdir(parents=True, exist_ok=True)
    (REGION_DIR / f"{region}.json").write_text(json.dumps(config, indent=2) + "\n")
    try:
        result = subprocess.run([modal, "run", PIPELINE, "--region", region], cwd=REPO,
                                capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise TrainError(504, "Planning took longer than 10 minutes") from None
    output = (result.stdout or "") + (result.stderr or "")
    parsed = parse_plan(result.stdout or "")
    if result.returncode != 0 or parsed is None:
        raise TrainError(502, f"Plan failed (exit {result.returncode}): {output[-1500:]}")
    state = write_state(region, region=region, station_id=station_id, status="planned", plan=parsed,
                        planned_at=now(), approved_at=None, exit_code=None)
    return {"region": region, "config": config, "plan": parsed, "state": state}


def _wait(region: str, proc: subprocess.Popen, log) -> None:
    code = proc.wait()
    log.close()
    with _lock:
        _running.pop(region, None)
    write_state(region, status="finished" if code == 0 else "failed", exit_code=code, finished_at=now())


def start(body: dict[str, Any]) -> dict[str, Any]:
    region = str(body.get("region", ""))
    if not REGION_RE.fullmatch(region) or not (REGION_DIR / f"{region}.json").is_file():
        raise TrainError(404, "Unknown region; plan it first")
    if body.get("approve") != region:
        raise TrainError(403, f"Go-ahead required: send approve = {region!r}")
    state = read_state(region)
    if state.get("status") != "planned" or state.get("plan", {}).get("problems"):
        raise TrainError(409, "Region has no clean plan; plan it again first")
    modal = modal_bin()
    with _lock:
        if region in _running and _running[region].poll() is None:
            raise TrainError(409, f"{region} is already training")
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        log = log_path(region).open("w", encoding="utf-8")
        proc = subprocess.Popen([modal, "run", PIPELINE, "--region", region, "--execute"], cwd=REPO,
                                stdout=log, stderr=subprocess.STDOUT, text=True)
        _running[region] = proc
    threading.Thread(target=_wait, args=(region, proc, log), daemon=True).start()
    return {"region": region, "state": write_state(region, status="running", approved_at=now(), pid=proc.pid)}


def status(region: str) -> dict[str, Any]:
    if not REGION_RE.fullmatch(region):
        raise TrainError(400, "Bad region name")
    state = read_state(region)
    if not state:
        return {"region": region, "status": "none"}
    if state.get("status") == "running" and region not in _running:
        # The API restarted while Modal was running; the log keeps growing if the CLI is still alive.
        state["status"] = "running (detached)"
    tail = ""
    if log_path(region).is_file():
        tail = "\n".join(log_path(region).read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
    return {**state, "log_tail": tail}
