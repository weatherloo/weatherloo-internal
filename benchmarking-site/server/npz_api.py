#!/usr/bin/env python3
"""Thin HTTP API: aggregate benchmark metrics from consolidated NPZ files.

Endpoints:
  GET /api/benchmark/{method_id}
      -> { method_id, npz_available, n_inits, year, npz_file }
  GET /api/benchmark/{method_id}/aggregate?location=&variable=
      [&init_from=ISO][&init_to=ISO][&cycles=00,06,12,18]
      -> { lead_times_hours, rmse, mae, bias, acc, n_inits, n_samples }

Run: python3 server/npz_api.py  (default port 5174)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

SITE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = SITE_ROOT / "data"
DEFAULT_YEAR = 2025
DEFAULT_PORT = 5174

METRICS = ("rmse", "mae", "bias", "acc")
METHOD_ID_RE = re.compile(r"^[a-z0-9_]+$")

_npz_cache: dict[str, tuple[str, np.lib.npyio.NpzFile]] = {}


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def npz_path_for(method_id: str, year: int = DEFAULT_YEAR) -> Path | None:
    method_dir = DATA_ROOT / method_id
    if not method_dir.is_dir():
        return None

    filename = f"{method_id}_{year}.npz"
    meta_path = method_dir / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        filename = meta.get("npz_file", filename)

    path = method_dir / filename
    return path if path.is_file() else None


def load_npz(method_id: str, year: int = DEFAULT_YEAR) -> np.lib.npyio.NpzFile | None:
    path = npz_path_for(method_id, year)
    if path is None:
        return None

    cache_key = str(path)
    cached = _npz_cache.get(method_id)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    if cached is not None:
        cached[1].close()

    data = np.load(path, allow_pickle=False)
    _npz_cache[method_id] = (cache_key, data)
    return data


def json_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in values.tolist():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            out.append(None)
        else:
            out.append(float(value))
    return out


def build_init_mask(
    initializations: np.ndarray,
    init_from: str | None,
    init_to: str | None,
    cycles: str | None,
) -> np.ndarray:
    mask = np.ones(len(initializations), dtype=bool)
    from_dt = parse_utc(init_from) if init_from else None
    to_dt = parse_utc(init_to) if init_to else None
    cycle_hours: set[int] | None = None
    if cycles:
        cycle_hours = set()
        for part in cycles.split(","):
            part = part.strip().upper().removesuffix("Z")
            if part:
                cycle_hours.add(int(part))

    for idx, init in enumerate(initializations):
        dt = parse_utc(str(init))
        if from_dt is not None and dt < from_dt:
            mask[idx] = False
        if to_dt is not None and dt > to_dt:
            mask[idx] = False
        if cycle_hours is not None and dt.hour not in cycle_hours:
            mask[idx] = False

    return mask


def aggregate_from_npz(
    data: np.lib.npyio.NpzFile,
    location: str,
    variable: str,
    init_from: str | None = None,
    init_to: str | None = None,
    cycles: str | None = None,
) -> dict[str, Any]:
    station_ids = [str(x) for x in data["station_ids"]]
    variables = [str(x) for x in data["variables"]]
    if location not in station_ids:
        raise KeyError(f"Unknown location {location!r}")
    if variable not in variables:
        raise KeyError(f"Unknown variable {variable!r}")

    station_idx = station_ids.index(location)
    variable_idx = variables.index(variable)
    lead_times = [int(x) for x in data["lead_times_hours"].tolist()]
    inits = data["initializations"]
    mask = build_init_mask(inits, init_from, init_to, cycles)

    result: dict[str, Any] = {
        "method_id": str(data["method_id"].item()),
        "location": location,
        "variable": variable,
        "lead_times_hours": lead_times,
        "n_inits": int(mask.sum()),
    }

    n_samples = np.zeros(len(lead_times), dtype=np.int64)
    for metric in METRICS:
        slab = data[metric][mask, station_idx, variable_idx, :]
        if slab.size == 0:
            result[metric] = [None] * len(lead_times)
            continue
        with np.errstate(all="ignore"):
            if metric == "rmse":
                # Each per-init entry is a single forecast/obs pair, so its
                # stored rmse is |error|. Averaging those yields mae; pooling a
                # real RMSE means squaring first and taking the root last.
                agg = np.sqrt(np.nanmean(np.square(slab), axis=0))
            elif metric == "acc":
                # Single-sample ACC degenerates to +/-1; an average of that is
                # not a correlation. Report nothing rather than something wrong.
                agg = np.full(len(lead_times), np.nan)
            else:
                agg = np.nanmean(slab, axis=0)
        result[metric] = json_list(agg)
        n_samples = np.maximum(
            n_samples, np.sum(~np.isnan(slab), axis=0).astype(np.int64)
        )

    result["n_samples"] = n_samples.tolist()
    return result


def method_status(method_id: str, year: int = DEFAULT_YEAR) -> dict[str, Any]:
    path = npz_path_for(method_id, year)
    status: dict[str, Any] = {
        "method_id": method_id,
        "year": year,
        "npz_available": path is not None,
        "npz_file": path.name if path else None,
        "n_inits": 0,
    }
    if path is None:
        return status

    data = load_npz(method_id, year)
    assert data is not None
    status["n_inits"] = int(len(data["initializations"]))
    return status


class Handler(BaseHTTPRequestHandler):
    server_version = "BenchmarkNPZAPI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[npz_api] {self.address_string()} - {fmt % args}")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_route(self) -> tuple[str | None, str | None, dict[str, list[str]]]:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        parts = [p for p in parsed.path.split("/") if p]
        # /api/benchmark/{method_id}[/aggregate]
        if len(parts) < 3 or parts[0] != "api" or parts[1] != "benchmark":
            return None, None, query
        method_id = parts[2]
        action = parts[3] if len(parts) > 3 else None
        return method_id, action, query

    def do_GET(self) -> None:
        method_id, action, query = self._read_route()
        if method_id is None or not METHOD_ID_RE.fullmatch(method_id):
            self._send_json(404, {"error": "Not found"})
            return

        year = int(query.get("year", [str(DEFAULT_YEAR)])[0])

        if action is None:
            self._send_json(200, method_status(method_id, year))
            return

        if action == "aggregate":
            location = query.get("location", [None])[0]
            variable = query.get("variable", [None])[0]
            if not location or not variable:
                self._send_json(
                    400,
                    {"error": "Query params 'location' and 'variable' are required"},
                )
                return

            data = load_npz(method_id, year)
            if data is None:
                self._send_json(
                    404,
                    {"error": f"No NPZ file for method {method_id!r}", "method_id": method_id},
                )
                return

            try:
                payload = aggregate_from_npz(
                    data,
                    location,
                    variable,
                    init_from=query.get("init_from", [None])[0],
                    init_to=query.get("init_to", [None])[0],
                    cycles=query.get("cycles", [None])[0],
                )
            except KeyError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            self._send_json(200, payload)
            return

        self._send_json(404, {"error": "Not found"})


def main() -> None:
    port = int(os.environ.get("BENCHMARK_API_PORT", DEFAULT_PORT))
    host = os.environ.get("BENCHMARK_API_HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Benchmark NPZ API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        for _path, cached in _npz_cache.values():
            cached.close()
        server.server_close()


if __name__ == "__main__":
    main()
