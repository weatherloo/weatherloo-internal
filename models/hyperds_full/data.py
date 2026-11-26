"""Public ERA5, GOES-16 and station ingestion; bulk IO runs on Modal only."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import xarray as xr


ERA5_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VARIABLES = ("10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature",
             "surface_pressure", "total_precipitation")
# Pooling follows the paper's 4x average-pooling resolution ratio.
LAT = np.arange(40, 48, 0.25, dtype=np.float64)
LON = np.arange(-88, -72, 0.25, dtype=np.float64)
SAT_SHAPE = (128, 256)
STATIONS = (
    {"id": "eric_d_soulis", "lat": 43.4737778, "lon": -80.5576389, "eccc": None, "split": "train"},
    {"id": "kitchener_waterloo", "lat": 43.4608333333, "lon": -80.3786111111, "eccc": 48569, "split": "train"},
    {"id": "elora_rcs", "lat": 43.65, "lon": -80.4166666667, "eccc": 41983, "split": "val"},
    {"id": "guelph_turfgrass", "lat": 43.5418725, "lon": -80.2220811111, "eccc": 55299, "split": "test"},
)


def initializations():
    for split, year, days in (("train", 2021, range(2, 16)), ("val", 2024, range(2, 7))):
        for day in days:
            for hour in (0, 12):
                yield split, datetime(year, 1, day, hour, tzinfo=timezone.utc)


def utc_stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def goes_time(key, kind):
    value = re.search(rf"_{kind}(\d{{14}})", key).group(1)
    return datetime.strptime(value[:13], "%Y%j%H%M%S").replace(tzinfo=timezone.utc)


@lru_cache(maxsize=64)
def list_goes_hour(hour, bucket):
    # List all bands together; 2018/early-2019 scans use M3 rather than M6.
    prefix = f"ABI-L1b-RadC/{hour:%Y/%j/%H}/"
    params = urllib.parse.urlencode({"list-type": "2", "prefix": prefix})
    with urllib.request.urlopen(f"https://{bucket}.s3.amazonaws.com/?{params}", timeout=90) as response:
        doc = ET.fromstring(response.read())
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    if doc.findtext("s:IsTruncated", namespaces=ns) == "true":
        raise RuntimeError("unexpected truncated GOES hourly band listing")
    return [{"bucket":bucket, "key": item.findtext("s:Key", namespaces=ns),
             "modified": item.findtext("s:LastModified", namespaces=ns),
             "etag": item.findtext("s:ETag", namespaces=ns)}
            for item in doc.findall("s:Contents", ns)]


def list_goes(hour, band, bucket="noaa-goes16"):
    return [row for row in list_goes_hour(hour,bucket)
            if re.search(rf"-M\dC{band:02d}_",row["key"])]


def select_goes(issue_time, bucket="noaa-goes16", production_cutoff=False):
    bands = (2, 5, 8, 15)
    cutoff = issue_time-timedelta(minutes=15) if production_cutoff else issue_time
    listings = {}
    for band in bands:
        entries = []
        for lag in (1, 0):
            entries += list_goes(issue_time - timedelta(hours=lag), band, bucket)
        listings[band] = [row for row in entries
                          if (production_cutoff or utc_stamp(row["modified"]) <= issue_time)
                          and goes_time(row["key"], "e") <= cutoff
                          and goes_time(row["key"], "c") <= cutoff]
    selected = []
    previous_start = None
    for candidate in sorted(listings[2], key=lambda row: row["key"], reverse=True):
        start = goes_time(candidate["key"], "s")
        if previous_start and previous_start - start < timedelta(minutes=9):
            continue
        frame = []
        for band in bands:
            matches = [row for row in listings[band]
                       if abs((goes_time(row["key"], "s") - start).total_seconds()) < 5]
            if not matches:
                break
            frame.append(max(matches, key=lambda row: row["key"]))
        if len(frame) == 4:
            selected.append(frame)
            previous_start = start
            if len(selected) == 2:
                return list(reversed(selected))
    raise RuntimeError(f"no two complete pre-issue GOES frames at {issue_time.isoformat()}")


def fill_satellite_gaps(values):
    """Fill small invalid-pixel holes using only valid pixels in this scan."""
    from scipy.ndimage import distance_transform_edt

    invalid = ~np.isfinite(values)
    fraction = float(invalid.mean())
    if fraction>.01:
        raise RuntimeError(f"GOES invalid pixel fraction {fraction:.5f} exceeds 1%")
    if invalid.any():
        nearest = distance_transform_edt(invalid,return_distances=False,return_indices=True)
        values = values.copy()
        values[invalid] = values[tuple(nearest[:,invalid])]
    return values,fraction


def read_goes_band(row, band, fs, fill_small_gaps=False):
    from pyproj import CRS, Transformer
    from scipy.interpolate import RegularGridInterpolator

    with fs.open(row.get("bucket","noaa-goes16") + "/" + row["key"], "rb", block_size=2**20) as stream:
        with xr.open_dataset(stream, engine="h5netcdf", chunks=None) as ds:
            projection = ds.goes_imager_projection.attrs
            height = projection["perspective_point_height"]
            crs = CRS.from_proj4(
                f"+proj=geos +h={height} +lon_0={projection['longitude_of_projection_origin']} "
                f"+sweep={projection['sweep_angle_axis']} +a={projection['semi_major_axis']} "
                f"+b={projection['semi_minor_axis']} +units=m"
            )
            lon, lat = np.meshgrid(np.linspace(LON[0], LON[-1], SAT_SHAPE[1]),
                                   np.linspace(LAT[0], LAT[-1], SAT_SHAPE[0]))
            px, py = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
            px, py = px / height, py / height
            xs, ys = ds.x.values, ds.y.values
            xi = np.flatnonzero((xs >= px.min()) & (xs <= px.max()))
            yi = np.flatnonzero((ys >= py.min()) & (ys <= py.max()))
            region = ds.isel(x=slice(max(0, xi[0]-2), xi[-1]+3),
                             y=slice(max(0, yi[0]-2), yi[-1]+3))
            values = region.Rad.values.astype(np.float32)
            quality = region.DQF.values
            values[(quality > 1) | ~np.isfinite(values)] = np.nan
            if band in (2, 5):
                values *= float(ds.kappa0.values)
            else:
                fk1, fk2, bc1, bc2 = [float(ds[name].values) for name in
                                     ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2")]
                values = (fk2 / np.log1p(fk1 / np.maximum(values, 1e-6)) - bc1) / bc2
            if fill_small_gaps:
                values,fraction = fill_satellite_gaps(values)
                row["quality_control"] = {"invalid_fraction":fraction,"fill":"nearest_valid_same_scan","maximum_fraction":.01}
            interp = RegularGridInterpolator((region.y.values[::-1], region.x.values),
                                             values[::-1], bounds_error=True)
            output = interp(np.stack((py, px), axis=-1)).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError(f"missing GOES pixels in {row['key']}")
    return output


def fetch_satellite(init, bucket="noaa-goes16", production_cutoff=False, fill_small_gaps=False):
    import s3fs

    selected = select_goes(init,bucket,production_cutoff)
    fs = s3fs.S3FileSystem(anon=True)
    frames = [np.stack([read_goes_band(row, band, fs,fill_small_gaps) for row, band in zip(frame, (2, 5, 8, 15))])
              for frame in selected]
    return np.stack(frames), selected


def fetch_station_data(scratch):
    sys.path.insert(0, "/workspace/pipelines/download")
    sys.path.insert(0, "/workspace/src")
    from fetch_station_observations import fetch_eccc_hourly
    from hrrr_bias_correction.make_data import SoulisObsIndex

    sources = []
    for year in (2021, 2024):
        url = f"https://www.civil.uwaterloo.ca/weather/download/Hobo_15minutedata_{year}.csv"
        request = urllib.request.Request(url, headers={"User-Agent": "weatherloo-internal/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            blob = response.read()
        (scratch / f"uw_hobo_15min_{year}.csv").write_bytes(blob)
        sources.append({"url": url, "sha256": hashlib.sha256(blob).hexdigest()})
    observations = {"eric_d_soulis": SoulisObsIndex(scratch, (2021, 2024))}
    for station in STATIONS[1:]:
        hourly = {}
        for year in (2021, 2024):
            rows = fetch_eccc_hourly(station["eccc"], year)
            sources.append({"eccc_stn_id": station["eccc"], "year": year,
                            "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()})
            for row in rows:
                stamp = utc_stamp(row["UTC_DATE"] + "Z")
                raw = [row.get(name) for name in ("WIND_SPEED", "TEMP", "STATION_PRESSURE", "PRECIP_AMOUNT")]
                values = np.array([float(value) if value is not None and float(value) > -9000 else np.nan
                                   for value in raw], dtype=np.float32)
                values[0] /= 3.6  # km/h -> m/s
                values[2] *= 10  # kPa -> hPa; use station pressure, never sea-level pressure
                hourly[stamp] = values
        observations[station["id"]] = hourly
    return observations, sources


def station_labels(init, observations):
    labels = np.full((49, len(STATIONS), 4), np.nan, dtype=np.float32)
    for hour in range(49):
        valid = init + timedelta(hours=hour)
        for index, station in enumerate(STATIONS):
            source = observations[station["id"]]
            if index == 0:
                value = source.lookup(valid)
                if value is not None:
                    labels[hour, index, :2] = (value[1]/3.6, value[0])
            elif valid in source:
                labels[hour, index] = source[valid]
    return labels


def ingest(destination):
    import tempfile

    destination = Path(destination)
    records = list(initializations())
    # Every target hour is read once even when overlapping initializations reuse it.
    times = sorted({(init + timedelta(hours=hour)).replace(tzinfo=None)
                    for _, init in records for hour in range(49)})
    print(f"ERA5: {len(times)} unique target hours, five variables", flush=True)
    ds = xr.open_zarr(ERA5_STORE, chunks=None, storage_options={"token": "anon"}, consolidated=True)
    subset = ds[list(VARIABLES)].sel(latitude=LAT, longitude=LON % 360)

    def fetch_hour(valid):
        values = subset.sel(time=np.datetime64(valid)).to_array().transpose("variable", "latitude", "longitude").values
        values = values.astype(np.float32)
        values[2] -= 273.15
        values[3] /= 100  # Pa -> hPa
        values[4] *= 1000  # ERA5 hourly accumulation, m -> mm (not differenced)
        if not np.isfinite(values).all() or values[4].min() < -1e-4:
            raise RuntimeError(f"invalid ERA5 fields at {valid}")
        values[4] = np.maximum(values[4], 0)
        return values

    with ThreadPoolExecutor(max_workers=8) as pool:
        fields = {}
        for index, (valid, values) in enumerate(zip(times, pool.map(fetch_hour, times))):
            fields[valid] = values
            if index % 24 == 0:
                print(f"ERA5: {index+1}/{len(times)} hours", flush=True)
    with tempfile.TemporaryDirectory(prefix="hyperds-observations-") as temp:
        observations, sources = fetch_station_data(Path(temp))
    destination.mkdir(parents=True, exist_ok=False)
    provenance = []
    arrays = {split: [] for split in ("train", "val")}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for (split, init), (satellite, satellite_sources) in zip(records, pool.map(fetch_satellite, [r[1] for r in records])):
            grid = np.stack([fields[(init+timedelta(hours=hour)).replace(tzinfo=None)] for hour in range(49)])
            labels = station_labels(init, observations)
            arrays[split].append((int(init.timestamp()), grid, satellite, labels))
            provenance.append({"initialization": init.isoformat(), "satellite": satellite_sources,
                               "grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest()})
            print(f"GOES + station labels: {init.isoformat()}", flush=True)
    query = np.array([[2*(station['lon']-(LON[0]-.125))/(len(LON)*.25)-1,
                       2*(station['lat']-(LAT[0]-.125))/(len(LAT)*.25)-1] for station in STATIONS], dtype=np.float32)
    stats = {}
    for split, samples in arrays.items():
        grid = np.stack([row[1] for row in samples])
        sat = np.stack([row[2] for row in samples])
        labels = np.stack([row[3] for row in samples])
        if split == "train":
            # Unique hours avoid overweighting overlap in normalization statistics.
            train_times = {init.replace(tzinfo=None)+timedelta(hours=h) for s,init in records if s=='train' for h in range(49)}
            train_fields = np.stack([fields[t] for t in sorted(train_times)])
            stats = {"field_mean": train_fields.mean(axis=(0,2,3)).tolist(),
                     "field_std": np.maximum(train_fields.std(axis=(0,2,3)), 1e-3).tolist(),
                     "satellite_mean": sat.mean(axis=(0,1,3,4)).tolist(),
                     "satellite_std": np.maximum(sat.std(axis=(0,1,3,4)), 1e-3).tolist()}
        group = xr.Dataset({
            "grid": (("sample", "lead", "variable", "latitude", "longitude"), grid),
            "satellite": (("sample", "frame", "band", "sat_y", "sat_x"), sat),
            "station": (("sample", "lead", "station_id", "target"), labels),
            "query_xy": (("station_id", "xy"), query),
            "init_time": ("sample", np.array([row[0] for row in samples], dtype=np.int64)),
        })
        group.to_zarr(destination, group=split, mode="a", consolidated=False,
                      encoding={"grid": {"chunks": (1,49,5,len(LAT),len(LON))},
                                "satellite": {"chunks": (1,2,4,*SAT_SHAPE)}})
    manifest = {"schema": "hyperds_full_v1", "variables": ["u10", "v10", "t2m", "sp", "tp1h"],
                "units": ["m/s", "m/s", "degC", "hPa", "mm"], "station_targets": ["wind_speed", "t2m", "sp", "tp1h"],
                "era5_store": ERA5_STORE, "latitude": LAT.tolist(), "longitude": LON.tolist(),
                "stations": STATIONS, "normalization": stats, "records": provenance, "station_sources": sources,
                "splits": {split: {"samples": len(samples), "valid_station_labels": np.isfinite(np.stack([r[3] for r in samples])).sum(axis=(0,1)).tolist()}
                           for split,samples in arrays.items()},
                "input_contract": "ERA5 at initialization only, 4x avgpool; two GOES frames published by initialization; future ERA5/stations are labels only",
                "experiment": "Waterloo regional adaptation, retrospective initialization; not a real-time availability claim"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"dataset_verified":manifest["splits"]}), flush=True)
    return manifest


def validate_store(destination):
    """Verify stored geometry, units, provenance, temporal isolation and missingness."""
    import zarr

    destination = Path(destination)
    manifest = json.loads((destination/"manifest.json").read_text())
    store = zarr.open_group(str(destination),mode="r")
    provenance = {int(utc_stamp(record["initialization"]).timestamp()):record for record in manifest["records"]}
    seen = set()
    report = {}
    for split,expected in (("train",28),("val",10)):
        group = store[split]
        if group["grid"].shape!=(expected,49,5,32,64) or group["satellite"].shape!=(expected,2,4,128,256):
            raise ValueError(f"incorrect {split} field/satellite geometry")
        query = np.asarray(group["query_xy"])
        if query.shape!=(4,2) or not np.isfinite(query).all() or np.abs(query).max()>1:
            raise ValueError("station geometry is invalid")
        counts = np.zeros((4,4),dtype=np.int64)
        for index in range(expected):
            init = int(group["init_time"][index])
            if init in seen:
                raise ValueError("duplicate initialization across splits")
            seen.add(init)
            record = provenance[init]
            field,satellite = np.asarray(group["grid"][index]),np.asarray(group["satellite"][index])
            if not np.isfinite(field).all() or not np.isfinite(satellite).all():
                raise ValueError("missing required model input")
            if hashlib.sha256(field.tobytes()).hexdigest()!=record["grid_sha256"]:
                raise ValueError("stored field checksum mismatch")
            cutoff = datetime.fromtimestamp(init,tz=timezone.utc)
            for frame in record["satellite"]:
                for band in frame:
                    if any(value>cutoff for value in (utc_stamp(band["modified"]),goes_time(band["key"],"e"),goes_time(band["key"],"c"))):
                        raise ValueError("satellite file was published after initialization")
            if not (-100<field[:,2].min() and field[:,2].max()<70 and 500<field[:,3].min()<1100 and field[:,3].max()<1100):
                raise ValueError("temperature/pressure units or values are invalid")
            if field[:,4].min()<0:
                raise ValueError("negative hourly precipitation")
            counts += np.isfinite(np.asarray(group["station"][index])).sum(axis=0)
        for station in ([0,1] if split=="train" else [2,3]):
            if (counts[station,:2]<24).any():
                raise ValueError(f"insufficient {split} station temperature/wind labels")
        report[split] = {"samples":expected,"valid_station_labels":counts.tolist()}
    if max(int(v) for v in store['train']['init_time'][:])+48*3600 >= min(int(v) for v in store['val']['init_time'][:]):
        raise ValueError("training valid times overlap validation initialization times")
    return {"passed":True,"checks":["geometry","finite_inputs","grid_sha256","satellite_publication_cutoff",
            "station_units","chronological_isolation","station_label_coverage"],"splits":report}
