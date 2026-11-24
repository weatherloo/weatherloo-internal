"""Issued NWP forecast fields (HRRR or AIFS) aligned to the ERA5/GOES/station shards.

Each monthly NWP shard stores, for every accepted initialization of the matching
ERA5 shard, the forecast trajectory regridded to the same 0.25 degree 32x64 grid
(``fields.npy``) and bilinearly interpolated at the benchmark stations on the
native grid (``station.npy``). ERA5 remains a valid-time label only.

Canonical variable order matches the rest of HyperDS: u10, v10, t2m (degC),
surface pressure (hPa), precipitation (mm/h, mean rate over the preceding output
interval). Lead-0 precipitation is undefined in both catalogs and is set to 0.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

# Domain and stations are frozen by the ERA5 shards; duplicated here so the
# builder image does not need the ERA5/zarr stack.
LAT = np.arange(40, 48, 0.25, dtype=np.float64)
LON = np.arange(-88, -72, 0.25, dtype=np.float64)
UTC = timezone.utc
CATALOG_VARIABLES = ("wind_u_10m", "wind_v_10m", "temperature_2m", "pressure_surface", "precipitation_surface")
STATIONS = (
    {"id": "eric_d_soulis", "lat": 43.4668, "lon": -80.5164},
    {"id": "kitchener_waterloo", "lat": 43.4608333333, "lon": -80.3786111111},
    {"id": "elora_rcs", "lat": 43.65, "lon": -80.4166666667},
    {"id": "guelph_turfgrass", "lat": 43.5418725, "lon": -80.2220811111},
    {"id": "cyyz", "lat": 43.6777, "lon": -79.6248},
)
SOURCES = {
    # HRRRv3 (archive start 2018-07-13 to 2020-12-02) issued only 36 h forecasts at
    # 00/06/12/18Z; HRRRv4 extended them to 48 h. Training uses v4 only, matching
    # the version evaluated in 2024/2025.
    "hrrr": {"catalog": "noaa-hrrr-forecast-48-hour", "leads": list(range(49)),
             "first_init": datetime(2020, 12, 3, tzinfo=UTC),
             "splits": {"train": (datetime(2020, 12, 3, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)),
                        "val": (datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)),
                        "test": (datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))}},
    # AIFS Single starts 2024-04-01 in the archive; 2025 stays the frozen test year,
    # so the short 2024 record is split chronologically into train/validation.
    "aifs": {"catalog": "ecmwf-aifs-single-forecast", "leads": list(range(0, 49, 6)),
             "first_init": datetime(2024, 4, 1, tzinfo=UTC),
             "splits": {"train": (datetime(2024, 4, 1, tzinfo=UTC), datetime(2024, 11, 1, tzinfo=UTC)),
                        "val": (datetime(2024, 11, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)),
                        "test": (datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))}},
}


def split_for(source, init):
    """Split by initialization; every target hour must precede the next split."""
    for split, (start, end) in SOURCES[source]["splits"].items():
        if start <= init < end:
            if split != "test" and init + timedelta(hours=48) >= end:
                return None
            return split
    return None


def months(source):
    first = SOURCES[source]["first_init"]
    keys, year, month = [], first.year, first.month
    while (year, month) <= (2025, 12):
        keys.append(f"{year}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return keys


def retry(action):
    for attempt in range(4):
        try:
            return action()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def bracket(axis, value):
    """Index pair and weight for linear interpolation on an ascending or descending axis."""
    axis = np.asarray(axis, dtype=np.float64)
    descending = axis[0] > axis[-1]
    ordered = axis[::-1] if descending else axis
    i = int(np.searchsorted(ordered, value)) - 1
    if not 0 <= i < len(ordered) - 1:
        raise ValueError("station outside forecast grid")
    w = (value - ordered[i]) / (ordered[i + 1] - ordered[i])
    if descending:
        n = len(axis) - 1
        return n - i, n - i - 1, float(w)
    return i, i + 1, float(w)


def station_weights(xs, ys, points):
    """Bilinear (y, x) indices/weights per station for 2-D arrays indexed [y, x]."""
    result = []
    for px, py in points:
        x0, x1, wx = bracket(xs, px)
        y0, y1, wy = bracket(ys, py)
        result.append(((y0, y1, x0, x1), np.array([[(1 - wy) * (1 - wx), (1 - wy) * wx], [wy * (1 - wx), wy * wx]])))
    return result


def interpolate_stations(values, weights):
    """values [..., y, x] -> [..., station]."""
    output = []
    for (y0, y1, x0, x1), w in weights:
        corner = np.stack((np.stack((values[..., y0, x0], values[..., y0, x1]), -1),
                           np.stack((values[..., y1, x0], values[..., y1, x1]), -1)), -2)
        output.append((corner * w).sum(axis=(-2, -1)))
    return np.stack(output, -1)


class Reader:
    """Opens one catalog and reads one initialization as grid + station arrays."""

    def __init__(self, source):
        import dynamical_catalog

        self.source = source
        self.leads = SOURCES[source]["leads"]
        ds = dynamical_catalog.open(SOURCES[source]["catalog"], chunks=None)
        for name in CATALOG_VARIABLES:
            units = ds[name].attrs.get("units")
            expected = {"temperature_2m": "degree_Celsius", "pressure_surface": "Pa",
                        "precipitation_surface": "kg m-2 s-1"}.get(name, "m s-1")
            if units != expected:
                raise ValueError(f"{source} {name} units {units!r}, expected {expected!r}")
        lead_values = (ds.lead_time.values / np.timedelta64(1, "h")).astype(int)
        self.lead_index = [int(np.flatnonzero(lead_values == lead)[0]) for lead in self.leads]
        if source == "hrrr":
            self._init_hrrr(ds)
        else:
            self._init_latlon(ds)
        self.provenance = {"catalog": SOURCES[source]["catalog"], "leads": self.leads,
                           "variables": list(CATALOG_VARIABLES), "regrid": self.regrid_note}

    def _init_hrrr(self, ds):
        from pyproj import CRS, Transformer
        from scipy import sparse

        lat, lon = ds.latitude.values, ds.longitude.values
        lon = np.where(lon > 180, lon - 360, lon)
        inside = (lat >= LAT[0] - .125) & (lat < LAT[-1] + .125) & (lon >= LON[0] - .125) & (lon < LON[-1] + .125)
        ys, xs = np.nonzero(inside)
        y0, y1, x0, x1 = max(ys.min() - 2, 0), ys.max() + 3, max(xs.min() - 2, 0), xs.max() + 3
        self.crop = {"y": slice(int(y0), int(y1)), "x": slice(int(x0), int(x1))}
        lat, lon = lat[self.crop["y"], self.crop["x"]], lon[self.crop["y"], self.crop["x"]]
        rows = np.floor((lat - (LAT[0] - .125)) / .25).astype(int)
        cols = np.floor((lon - (LON[0] - .125)) / .25).astype(int)
        valid = (rows >= 0) & (rows < len(LAT)) & (cols >= 0) & (cols < len(LON))
        cell = (rows * len(LON) + cols)[valid]
        counts = np.bincount(cell, minlength=len(LAT) * len(LON))
        if (counts == 0).any():
            raise ValueError("HRRR does not cover every 0.25 degree cell")
        columns = np.flatnonzero(valid.ravel())
        self.mean = sparse.csr_matrix((1 / counts[cell], (cell, columns)), shape=(len(LAT) * len(LON), lat.size))
        crs = CRS.from_cf(ds.spatial_ref.attrs)
        transform = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        points = [transform.transform(s["lon"], s["lat"]) for s in STATIONS]
        self.weights = station_weights(ds.x.values[self.crop["x"]], ds.y.values[self.crop["y"]], points)
        self.subset = ds[list(CATALOG_VARIABLES)].isel(self.crop)
        self.regrid_note = ("native 3 km points averaged within each 0.25 degree cell "
                            f"(min {int(counts.min())}, max {int(counts.max())} points); stations bilinear in native x/y")

    def _init_latlon(self, ds):
        lat, lon = ds.latitude.values, ds.longitude.values
        rows = [int(np.flatnonzero(np.isclose(lat, value))[0]) for value in LAT]
        cols = [int(np.flatnonzero(np.isclose(lon, value))[0]) for value in LON]
        if np.any(np.diff(cols) != 1) or np.any(np.abs(np.diff(rows)) != 1):
            raise ValueError("AIFS grid does not contain the 0.25 degree domain contiguously")
        lat_slice = slice(min(rows) - 2, max(rows) + 3)
        lon_slice = slice(min(cols) - 2, max(cols) + 3)
        self.subset = ds[list(CATALOG_VARIABLES)].isel(latitude=lat_slice, longitude=lon_slice)
        sub_lat, sub_lon = lat[lat_slice], lon[lon_slice]
        self.rows = np.array([int(np.flatnonzero(np.isclose(sub_lat, v))[0]) for v in LAT])
        self.cols = np.array([int(np.flatnonzero(np.isclose(sub_lon, v))[0]) for v in LON])
        self.weights = station_weights(sub_lon, sub_lat, [(s["lon"], s["lat"]) for s in STATIONS])
        self.mean = None
        self.regrid_note = "native 0.25 degree points identical to the ERA5 grid; stations bilinear in lat/lon"

    def read(self, init):
        values = retry(lambda: self.subset.sel(init_time=np.datetime64(init.replace(tzinfo=None))).isel(
            lead_time=self.lead_index).to_array().values.astype(np.float32))  # [var, lead, y, x]
        values[4, 0] = 0  # lead-0 precipitation rate is undefined
        if not np.isfinite(values).all():
            bad = [CATALOG_VARIABLES[v] for v in range(5) if not np.isfinite(values[v]).all()]
            raise ValueError(f"non-finite forecast values in {bad}")
        values[3] /= 100  # Pa -> hPa
        values[4] = np.maximum(values[4] * 3600, 0)  # kg m-2 s-1 -> mm/h
        values = values.transpose(1, 0, 2, 3)  # [lead, var, y, x]
        station = interpolate_stations(values, self.weights).transpose(0, 2, 1)  # [lead, station, var]
        if self.mean is not None:
            flat = values.reshape(-1, values.shape[-2] * values.shape[-1])
            grid = (self.mean @ flat.T).T.reshape(len(self.leads), 5, len(LAT), len(LON))
        else:
            grid = values[:, :, self.rows][:, :, :, self.cols]
        grid, station = grid.astype(np.float32), station.astype(np.float32)
        if not (-80 < grid[:, 2].min() and grid[:, 2].max() < 60 and 500 < grid[:, 3].min() and grid[:, 3].max() < 1100):
            raise ValueError("forecast temperature/pressure values out of range")
        return grid, station


def era5_rows(era5_shard):
    """Initializations of an immutable ERA5 shard, in stored row order."""
    detail = json.loads((Path(era5_shard) / "manifest.json").read_text())
    stamps = [datetime.fromisoformat(r["initialization"]).astimezone(UTC) for r in detail["records"]]
    return detail["split"], stamps, hashlib.sha256((Path(era5_shard) / "manifest.json").read_bytes()).hexdigest()


def moments(values, axes):
    return {"sum": values.sum(axis=axes, dtype=np.float64).tolist(),
            "sum_sq": np.square(values, dtype=np.float64).sum(axis=axes, dtype=np.float64).tolist(),
            "count": int(np.prod([values.shape[a] for a in axes]))}


def build_month(source, key, era5_shard, destination, reader=None):
    destination = Path(destination)
    if (destination / "manifest.json").is_file():
        return json.loads((destination / "manifest.json").read_text())
    era5_split, stamps, era5_sha = era5_rows(era5_shard)
    planned = [(row, init, split_for(source, init)) for row, init in enumerate(stamps)]
    planned = [item for item in planned if item[2] is not None]
    reader = reader or Reader(source)

    def fetch(item):
        row, init, split = item
        try:
            return item, reader.read(init), None
        except Exception as error:
            return item, None, {"initialization": init.isoformat(), "error": f"{type(error).__name__}: {error}"}

    accepted, failures = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (item, result, failure) in enumerate(pool.map(fetch, planned)):
            if failure:
                failures.append(failure)
                print(json.dumps({"rejected_initialization": failure, "source": source, "month": key}), flush=True)
            else:
                accepted.append((item, result))
            if index % 20 == 0:
                print(json.dumps({"source": source, "month": key, "read": index + 1, "planned": len(planned)}), flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"source": source, "month": key, "era5_split": era5_split, "era5_manifest_sha256": era5_sha,
                "planned": len(planned), "samples": len(accepted), "failures": failures,
                "leads": SOURCES[source]["leads"], "stations": [s["id"] for s in STATIONS], **reader.provenance}
    if accepted:
        fields = np.stack([result[0] for _, result in accepted])
        station = np.stack([result[1] for _, result in accepted])
        arrays = {"fields": fields, "station": station,
                  "init_time": np.array([int(item[1].timestamp()) for item, _ in accepted], dtype=np.int64),
                  "era5_row": np.array([item[0] for item, _ in accepted], dtype=np.int64),
                  "split": np.array([item[2] for item, _ in accepted])}
        for name, value in arrays.items():
            np.save(destination / f"{name}.npy", value)
        train = arrays["split"] == "train"
        if train.any():
            manifest["field_moments"] = moments(fields[train], (0, 1, 3, 4))
        manifest["sha256"] = {name: hashlib.sha256(np.load(destination / f"{name}.npy").tobytes()).hexdigest()
                              for name in arrays}
        manifest["split_counts"] = {s: int((arrays["split"] == s).sum()) for s in ("train", "val", "test")}
    manifest["verified"] = True
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"nwp_month_completed": key, "source": source, "samples": len(accepted),
                      "rejected": len(failures)}), flush=True)
    return manifest


def finalize(source, era5_root, root):
    """Root manifest with chronology, coverage and checksum checks."""
    root, era5_root = Path(root), Path(era5_root)
    entries, stamps = [], {"train": [], "val": [], "test": []}
    for key in months(source):
        detail_path = root / "shards" / key / "manifest.json"
        detail = json.loads(detail_path.read_text())
        if not detail["verified"] or (detail["planned"] and detail["samples"] < .8 * detail["planned"]):
            raise ValueError(f"{source} {key}: fewer than 80% of planned forecasts verified")
        era5_split, _, era5_sha = era5_rows(era5_root / "shards" / key)
        if era5_sha != detail["era5_manifest_sha256"]:
            raise ValueError(f"{key}: ERA5 shard changed after NWP build")
        if detail["samples"]:
            init_time = np.load(root / "shards" / key / "init_time.npy")
            split = np.load(root / "shards" / key / "split.npy")
            for name, digest in detail["sha256"].items():
                if hashlib.sha256(np.load(root / "shards" / key / f"{name}.npy").tobytes()).hexdigest() != digest:
                    raise ValueError(f"{key}: {name} checksum mismatch")
            for s in stamps:
                stamps[s].extend(init_time[split == s].tolist())
        entries.append({"month": key, "path": f"shards/{key}", "era5_shard": f"shards/{key}", "era5_split": era5_split,
                        "samples": detail["samples"], "planned": detail["planned"], "rejected": len(detail["failures"]),
                        "manifest_sha256": hashlib.sha256(detail_path.read_bytes()).hexdigest()})
    for split, following in (("train", "val"), ("val", "test")):
        if stamps[split] and max(stamps[split]) + 48 * 3600 >= min(stamps[following]):
            raise ValueError("target hours cross a split boundary")
    manifest = {"schema": "hyperds_nwp_v1", "source": source, "catalog": SOURCES[source]["catalog"],
                "leads": SOURCES[source]["leads"], "stations": [s["id"] for s in STATIONS],
                "split_ranges": {s: [a.isoformat(), b.isoformat()] for s, (a, b) in SOURCES[source]["splits"].items()},
                "splits": {s: len(v) for s, v in stamps.items()}, "months": entries,
                "input_contract": ("Issued forecast trajectory from one initialization, regridded to the ERA5 0.25 degree grid; "
                                   "station observations at the initialization time (lead 0) may enter as issue-time "
                                   "inputs; ERA5 and later station values are labels only."),
                "units": ["m/s", "m/s", "degC", "hPa", "mm/h"]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
