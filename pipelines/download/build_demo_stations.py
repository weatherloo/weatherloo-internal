"""Station list for the demo globe's tap-time checks.

Writes forecast-app/src/data/demo/stations.json: every IEM ASOS station in a box
around the HRRR domain that is still reporting, with the date its archive starts.
The app's training demo uses it to find the nearest station to a tap and to check that the
station's records start early enough for the hrrr_cnn training split.
The network (IEM state code, e.g. CA or CA_ON) feeds a new region's station table.

ponytail: archive start date is a proxy for history. The pipeline's `stations`
stage still applies the real rule (>= 50% hourly coverage) before any training.

Usage (from the repo root; ~70 small requests to IEM, stdlib only):
    python pipelines/download/build_demo_stations.py
"""
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "forecast-app" / "src" / "data" / "demo" / "stations.json"
IEM = "https://mesonet.agron.iastate.edu/geojson/network/{}.geojson"

US_STATES = (
    "AL AZ AR CA CO CT DE DC FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY "
    "NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"
).split()
CA_PROVINCES = "BC AB SK MB ON QC NB NS PE NF".split()
NETWORKS = [f"{s}_ASOS" for s in US_STATES] + [f"CA_{p}_ASOS" for p in CA_PROVINCES]

# Loose lat/lon box around the HRRR domain; the browser does the exact grid test.
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 20.0, 56.0, -140.0, -55.0


def fetch(network):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(IEM.format(network), timeout=60) as r:
                return json.load(r)["features"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            if attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
        time.sleep(2 ** (attempt + 1))
    return []


def main():
    rows, seen = [], set()
    for network in NETWORKS:
        features = fetch(network)
        for f in features:
            p, (lon, lat) = f["properties"], f["geometry"]["coordinates"][:2]
            if not p.get("online") or p.get("archive_end") or not p.get("archive_begin"):
                continue
            if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX) or p["sid"] in seen:
                continue
            seen.add(p["sid"])
            rows.append([p["sid"], p["sname"].title(), round(lat, 4), round(lon, 4), p["archive_begin"],
                         network.removesuffix("_ASOS")])
        print(f"{network:>12}: {len(features):4d} listed, {len(rows):5d} kept so far", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "IEM ASOS network metadata (mesonet.agron.iastate.edu)",
        "columns": ["id", "name", "lat", "lon", "archive_begin", "network"],
        "stations": sorted(rows),
    }, separators=(",", ":")))
    print(f"wrote {len(rows)} stations to {OUT}")


if __name__ == "__main__":
    main()
