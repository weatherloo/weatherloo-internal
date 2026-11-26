"""Multi-station hourly labels for the paper-faithful HRRR experiment.

Stations: every ECCC climate-hourly station and every IEM ASOS station inside the
model domain with records spanning 2020-12 .. 2025-12, plus Soulis (v3 QC
overlay). Labels are hourly on the UTC hour: wind speed (m/s), 2 m temperature
(degC), station pressure (hPa), 1-hour precipitation (mm). Missing values stay
NaN; nothing is imputed.

ASOS routine METARs are issued a few minutes before the hour (typically :51-:56)
and are assigned to the following top of the hour; precipitation is the METAR
hourly accumulation; station pressure is derived from the altimeter setting and
station elevation (standard-atmosphere reduction).
"""

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import numpy as np

UTC = timezone.utc
BBOX = (-88.0, 40.0, -72.125, 47.875)  # lon/lat of the 32x64 0.25-degree domain (cell edges)
START, END = datetime(2020, 12, 1, tzinfo=UTC), datetime(2026, 1, 3, tzinfo=UTC)
UA = {"User-Agent": "weatherloo-internal/1.0"}
ECCC_STATIONS = "https://api.weather.gc.ca/collections/climate-stations/items"
ECCC_HOURLY = "https://api.weather.gc.ca/collections/climate-hourly/items"
ASOS_NETWORKS = ("MI", "OH", "IN", "IL", "WI", "NY", "PA", "NJ", "VT", "NH", "MA", "CT")
SOULIS = {"id": "eric_d_soulis", "source": "uw_soulis", "lat": 43.4668, "lon": -80.5164, "elevation": 337.0,
          "wind_height_m": 4.4}
# Measurement QC bounds (same as the ERA5 experiment): wind m/s, t2m degC, sp hPa, tp mm.
BOUNDS = ((0, 100), (-80, 60), (500, 1100), (0, 500))


def retry(action, attempts=4):
    for attempt in range(attempts):
        try:
            return action()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 2))


def get(url, as_json=True):
    def read():
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as response:
            data = response.read()
        return json.loads(data) if as_json else data.decode()
    return retry(read)


def eccc_degrees(value):
    """ECCC encodes coordinates as signed DDMMSSsss (e.g. 455700000 -> 45 deg 57' 00")."""
    sign, value = (-1 if value < 0 else 1), abs(int(value))
    return sign * (value // 10 ** 7 + (value // 10 ** 5 % 100) / 60 + value % 10 ** 5 / 1000 / 3600)


def inside(lat, lon, bbox=BBOX):
    return bbox[1] <= lat < bbox[3] and bbox[0] <= lon < bbox[2]


def inventory(bbox=BBOX, asos_networks=ASOS_NETWORKS, eccc=True, soulis=True):
    """Stations inside bbox (lon_min, lat_min, lon_max, lat_max) with hourly records spanning 2020-12..2025-12.

    Defaults reproduce the Waterloo table (Soulis first, ECCC and the listed ASOS states).
    """
    inside_box = lambda lat, lon: inside(lat, lon, bbox)
    stations = [dict(SOULIS)] if soulis else []
    query = urllib.parse.urlencode({"bbox": ",".join(map(str, bbox)), "limit": 10000, "f": "json"})
    for feature in (get(f"{ECCC_STATIONS}?{query}")["features"] if eccc else []):
        p = feature["properties"]
        if p.get("HAS_HOURLY_DATA") not in ("Y", True):
            continue
        first, last = p.get("HLY_FIRST_DATE") or "9999", p.get("HLY_LAST_DATE") or "0000"
        if first > "2020-12-01" or last < "2025-12-31":
            continue
        lat, lon = eccc_degrees(p["LATITUDE"]), eccc_degrees(p["LONGITUDE"])
        if inside_box(lat, lon):
            stations.append({"id": f"eccc_{p['STN_ID']}", "source": "eccc", "eccc": int(p["STN_ID"]),
                             "name": p.get("STATION_NAME"), "lat": lat, "lon": lon,
                             "elevation": float(p["ELEVATION"]) if p.get("ELEVATION") else None})
    for network in asos_networks:
        for feature in get(f"https://mesonet.agron.iastate.edu/geojson/network/{network}_ASOS.geojson")["features"]:
            p = feature["properties"]
            lon, lat = feature["geometry"]["coordinates"][:2]
            if not inside_box(lat, lon):
                continue
            if (p.get("archive_begin") or "9999") > "2020-12-01":
                continue
            if p.get("archive_end") not in (None, "") and p["archive_end"] < "2025-12-31":
                continue
            stations.append({"id": f"asos_{p['sid']}", "source": "asos", "asos": p["sid"], "name": p.get("sname"),
                             "state": network, "lat": lat, "lon": lon, "elevation": p.get("elevation")})
    return stations


def hour_index(stamp):
    return int((stamp - START).total_seconds() // 3600)


def fetch_eccc(station, n_hours):
    labels = np.full((n_hours, 4), np.nan, dtype=np.float32)
    rows_seen = 0
    for year in range(START.year, END.year + 1):
        offset = 0
        while True:
            query = urllib.parse.urlencode({"filter": f"properties.STN_ID={station['eccc']} AND properties.UTC_YEAR={year}",
                                            "limit": 10000, "offset": offset, "f": "json"})
            batch = get(f"{ECCC_HOURLY}?{query}").get("features", [])
            for feature in batch:
                p = feature["properties"]
                stamp = datetime.fromisoformat(p["UTC_DATE"].replace("Z", "+00:00")).astimezone(UTC)
                if stamp.minute or stamp.second:
                    continue
                index = hour_index(stamp)
                if not 0 <= index < n_hours:
                    continue
                raw = [p.get(k) for k in ("WIND_SPEED", "TEMP", "STATION_PRESSURE", "PRECIP_AMOUNT")]
                values = [float(v) if v not in (None, "") and float(v) > -9000 else np.nan for v in raw]
                labels[index] = (values[0] / 3.6, values[1], values[2] * 10, values[3])
                rows_seen += 1
            if len(batch) < 10000:
                break
            offset += 10000
    return labels, rows_seen


def station_pressure(altimeter_inhg, elevation_m):
    """Altimeter setting -> station pressure (hPa), standard-atmosphere reduction."""
    return altimeter_inhg * 33.8639 * ((288.0 - 0.0065 * elevation_m) / 288.0) ** 5.2561


def fetch_asos(station, n_hours):
    labels = np.full((n_hours, 4), np.nan, dtype=np.float32)
    begin, end = START - timedelta(hours=1), END
    query = urllib.parse.urlencode([("station", station["asos"]), ("data", "tmpc"), ("data", "sknt"), ("data", "alti"),
                                    ("data", "p01m"), ("year1", begin.year), ("month1", begin.month), ("day1", begin.day),
                                    ("year2", end.year), ("month2", end.month), ("day2", end.day), ("tz", "Etc/UTC"),
                                    ("format", "onlycomma"), ("latlon", "no"), ("missing", "null"), ("trace", "0.0001"),
                                    ("report_type", "3"), ("direct", "no")])
    text = get(f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?{query}", as_json=False)
    rows_seen = 0
    best_offset = np.full(n_hours, 99.)
    elevation = float(station["elevation"] or 0)
    for row in csv.DictReader(io.StringIO(text)):
        stamp = datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        target = (stamp + timedelta(minutes=15)).replace(minute=0)  # :45-:59 -> next hour, :00-:14 -> this hour
        offset = abs((stamp - target).total_seconds()) / 60
        if offset > 15:
            continue
        index = hour_index(target)
        if not 0 <= index < n_hours or offset >= best_offset[index]:
            continue
        number = lambda key: float(row[key]) if row.get(key) not in (None, "", "null", "M") else np.nan
        labels[index] = (number("sknt") * 0.514444, number("tmpc"),
                         station_pressure(number("alti"), elevation), number("p01m"))
        best_offset[index] = offset
        rows_seen += 1
    return labels, rows_seen


def build(destination, soulis_overlay):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "labels.npz"
    if output.is_file():
        return json.loads((destination / "stations.json").read_text())
    stations = inventory()
    n_hours = hour_index(END)
    times = np.array([int((START + timedelta(hours=h)).timestamp()) for h in range(n_hours)], dtype=np.int64)
    labels = np.full((n_hours, len(stations), 4), np.nan, dtype=np.float32)
    with np.load(soulis_overlay) as overlay:
        index = np.searchsorted(overlay["times"], times)
        ok = (index < len(overlay["times"])) & (overlay["times"][np.minimum(index, len(overlay["times"]) - 1)] == times)
        labels[ok, 0, :2] = overlay["values"][index[ok]]

    fetch_all(stations, labels, start=1)
    apply_bounds(labels)
    # Keep stations with at least half of all hours carrying temperature and wind.
    coverage = np.isfinite(labels[:, :, :2]).all(axis=2).mean(axis=0)
    for station, fraction in zip(stations, coverage):
        station["coverage_t2m_wind"] = float(fraction)
    keep = [i for i, station in enumerate(stations) if i == 0 or (coverage[i] >= .5 and "error" not in station)]
    labels, kept = labels[:, keep], [stations[i] for i in keep]
    np.savez_compressed(output, times=times, labels=labels)
    manifest = {"stations": kept, "dropped": [s for i, s in enumerate(stations) if i not in keep],
                "variables": ["wind_speed_m/s", "t2m_degC", "station_pressure_hPa", "precip_1h_mm"],
                "start": START.isoformat(), "end": END.isoformat(), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "notes": __doc__}
    (destination / "stations.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"station_labels_saved": len(kept), "dropped": len(manifest["dropped"])}), flush=True)
    return manifest


def fetch_all(stations, labels, start=0):
    """Fill labels[:, i] for stations[start:] from ECCC or IEM ASOS; failures are recorded on the station."""
    n_hours = labels.shape[0]

    def fetch(item):
        i, station = item
        try:
            values, rows = (fetch_eccc if station["source"] == "eccc" else fetch_asos)(station, n_hours)
            return i, values, rows, None
        except Exception as error:
            return i, None, 0, f"{type(error).__name__}: {error}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        for done, (i, values, rows, error) in enumerate(pool.map(fetch, list(enumerate(stations))[start:]), start=1):
            station = stations[i]
            if error:
                station["error"] = error
            else:
                labels[:, i] = values
                station["rows"] = rows
            if done % 20 == 0:
                print(json.dumps({"stations_fetched": done, "total": len(stations) - start}), flush=True)


def apply_bounds(labels):
    for variable, (low, high) in enumerate(BOUNDS):
        slab = labels[:, :, variable]
        slab[(slab < low) | (slab > high)] = np.nan


def build_region(destination, bbox, asos_networks=(), eccc=False, keep=(), min_coverage=0.5, dedup_km=2.0):
    """Station table for any region: hourly labels for ASOS (and optionally ECCC) stations inside bbox.

    Stations need min_coverage of hours with both temperature and wind; ids in `keep` (the targets) are
    kept regardless. Stations within dedup_km of a better-covered one are dropped (targets win ties).
    Writes stations.json and labels.npz in the same layout as build().
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "labels.npz"
    if output.is_file():
        return json.loads((destination / "stations.json").read_text())
    stations = inventory(tuple(bbox), tuple(asos_networks), eccc=eccc, soulis=False)
    missing = sorted(set(keep) - {s["id"] for s in stations})
    if missing:
        raise ValueError(f"targets not in the inventory: {missing}")
    n_hours = hour_index(END)
    times = np.array([int((START + timedelta(hours=h)).timestamp()) for h in range(n_hours)], dtype=np.int64)
    labels = np.full((n_hours, len(stations), 4), np.nan, dtype=np.float32)
    fetch_all(stations, labels)
    apply_bounds(labels)
    coverage = np.isfinite(labels[:, :, :2]).all(axis=2).mean(axis=0)
    for station, fraction in zip(stations, coverage):
        station["coverage_t2m_wind"] = float(fraction)
    usable = [i for i, s in enumerate(stations)
              if s["id"] in keep or (coverage[i] >= min_coverage and "error" not in s)]
    order = sorted(usable, key=lambda i: (stations[i]["id"] not in keep, -coverage[i]))
    kept, duplicates = [], {}
    for i in order:
        near = next((k for k in kept if distance_km(stations[i], stations[k]) < dedup_km), None)
        if near is None:
            kept.append(i)
        else:
            duplicates[stations[i]["id"]] = stations[near]["id"]
    kept.sort()
    np.savez_compressed(output, times=times, labels=labels[:, kept])
    manifest = {"stations": [stations[i] for i in kept],
                "dropped": [s for i, s in enumerate(stations) if i not in usable],
                "duplicates_removed": duplicates, "bbox": list(bbox), "asos_networks": list(asos_networks),
                "eccc": eccc, "variables": ["wind_speed_m/s", "t2m_degC", "station_pressure_hPa", "precip_1h_mm"],
                "start": START.isoformat(), "end": END.isoformat(),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "notes": __doc__}
    (destination / "stations.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"station_labels_saved": len(kept), "dropped": len(manifest["dropped"]),
                      "duplicates": len(duplicates)}), flush=True)
    return manifest


def split_stations(stations, seed=0, val_fraction=200 / 1866, test_fraction=400 / 1866,
                   forced_test=("eric_d_soulis", "asos_YYZ", "eccc_51459")):
    """Random station split with the paper's proportions; Waterloo/Toronto forced into test."""
    ids = [s["id"] for s in stations]
    forced = [i for i, sid in enumerate(ids) if sid in forced_test]
    rest = [i for i in range(len(ids)) if i not in forced]
    rng = np.random.default_rng(seed)
    rng.shuffle(rest)
    n_test = max(round(test_fraction * len(ids)) - len(forced), 0)
    n_val = round(val_fraction * len(ids))
    test = sorted(forced + rest[:n_test])
    val = sorted(rest[n_test:n_test + n_val])
    train = sorted(rest[n_test + n_val:])
    return {"train": train, "val": val, "test": test}


def distance_km(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def deduplicate(stations, radius_km=2.0):
    """Indices to keep: one station per cluster within radius_km (best t2m/wind coverage; Soulis always kept).

    ECCC lists several IDs for one site (e.g. "Kingston A" twice plus "Kingston Climate"); left in place they
    would put the same location in two splits.
    """
    order = sorted(range(len(stations)), key=lambda i: (stations[i]["id"] != "eric_d_soulis",
                                                         -stations[i].get("coverage_t2m_wind", 0)))
    kept, duplicates = [], {}
    for i in order:
        near = next((k for k in kept if distance_km(stations[i], stations[k]) < radius_km), None)
        if near is None:
            kept.append(i)
        else:
            duplicates[stations[i]["id"]] = stations[near]["id"]
    return sorted(kept), duplicates


def derive_deduplicated(source, destination, radius_km=2.0):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / "stations.json").read_text())
    keep, duplicates = deduplicate(meta["stations"], radius_km)
    with np.load(source / "labels.npz") as saved:
        times, labels = saved["times"], saved["labels"][:, keep]
    np.savez_compressed(destination / "labels.npz", times=times, labels=labels)
    stations = [meta["stations"][i] for i in keep]
    manifest = {**meta, "stations": stations, "deduplicated_from": str(source), "dedup_radius_km": radius_km,
                "duplicates_removed": duplicates,
                "sha256": hashlib.sha256((destination / "labels.npz").read_bytes()).hexdigest()}
    (destination / "stations.json").write_text(json.dumps(manifest, indent=2) + "\n")
    split = split_stations(stations)
    named = {name: [stations[i]["id"] for i in indices] for name, indices in split.items()}
    (destination / "split.json").write_text(json.dumps(named, indent=2) + "\n")
    return manifest, named
