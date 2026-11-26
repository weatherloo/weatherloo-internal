"""Region pipeline on Modal: HRRR points/patches -> patch CNN per site -> site export.

    modal run models/hrrr_cnn/modal_region.py --region waterloo              # plan only (default)
    modal run models/hrrr_cnn/modal_region.py --region waterloo --execute    # run missing stages
    modal run models/hrrr_cnn/modal_region.py --region waterloo --execute --force cnn

Every stage is idempotent: extraction months with a manifest are skipped, and a
CNN stage is skipped when its output exists unless --force cnn;
re-extracting data needs a new ``version`` in the region config. The plan
lists what would run and its compute class before anything is spent.
"""

import json
from pathlib import Path
import sys

import modal

if not modal.is_local():
    sys.path.insert(0, "/app")

app = modal.App("weatherloo-region-pipeline")
HERE = Path(__file__).resolve().parent
data_volume = modal.Volume.from_name("weatherloo-temporal-data")

SHARED_FILES = (("pipelines/download/station_labels.py", "/app/station_labels.py"),
                ("pipelines/benchmarks/site_scoring.py", "/app/site_scoring.py"))


def with_shared(image):
    """Add this directory and the shared repository modules to an image at /app."""
    image = image.add_local_dir(str(HERE), "/app")
    for source, target in SHARED_FILES:
        image = image.add_local_file(str(HERE.parents[1] / source), target)
    return image
cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "dynamical-catalog>=0.8.0,<1", "pyproj==3.7.0", "scipy", "numpy", "lightgbm==4.5.0", "scikit-learn==1.5.2")
gpu_image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "torch==2.5.1", "scikit-learn==1.5.2")
precip_image = gpu_image.pip_install("lightgbm==4.5.0")
if modal.is_local():
    cpu_image, gpu_image, precip_image = (with_shared(i) for i in (cpu_image, gpu_image, precip_image))
    site = HERE.parents[1] / "benchmarking-site/data"
    for source in sorted(site.glob("*/*_2025.npz")) + sorted(site.glob("observations/*/observations_6h_2025.json")):
        cpu_image = cpu_image.add_local_file(str(source), "/benchmark/" + str(source.relative_to(site)))
    # Soulis HOBO 15-min files (tipping-bucket precipitation truth), HRRRv4 years only.
    for year in range(2021, 2026):
        source = site / f"observations/eric_d_soulis/raw/uw_hobo_15min_{year}.csv"
        precip_image = precip_image.add_local_file(str(source), f"/soulis_raw/{source.name}")
STAGES = ("stations", "points", "patches", "cnn", "final", "export")
COMPUTE = {"stations": "2 cpu, IEM/ECCC downloads", "points": "cpu x months", "patches": "cpu x months (all fields + chunk view)", "cnn": "L4/A10 gpu",
           "final": "2 cpu", "export": "2 cpu"}


def region_of(name):
    from regions import load
    return load(name)


def stations_of(region):
    return json.loads((Path(region.station_root) / "stations.json").read_text())["stations"]


def done_months(root):
    return {p.parent.name for p in Path(root).glob("*/manifest.json")}


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=30 * 60, volumes={"/data": data_volume})
def plan(name: str, force: str = ""):
    """Validate the config against the station table and native chunk; list pending work."""
    from points import PointReader, months
    from regions import check

    data_volume.reload()
    region = region_of(name)
    if not (Path(region.station_root) / "stations.json").is_file():
        if not region.station_build:
            return {"region": region.name, "problems": [f"no station table at {region.station_root} and no stations spec"],
                    "steps": [], "months": [], "method": ""}
        return {"region": region.name, "anchor": region.anchor, "problems": [], "months": months(),
                "method": region.extra.get("method", ""), "note": "remaining stages are planned once the table exists",
                "steps": [{"stage": "stations", "spec": region.station_build, "compute": COMPUTE["stations"]}]}
    stations = stations_of(region)
    reader = PointReader(stations, region.anchor)
    chunk = [s["id"] for s in reader.stations]
    problems = check(region, stations, chunk)
    forced = set(filter(None, force.split(",")))
    all_months = months()
    steps = []
    missing = sorted(set(all_months) - done_months(region.point_root))
    if missing:
        steps.append({"stage": "points", "months": len(missing), "compute": COMPUTE["points"]})
    for t in region.targets:
        missing = sorted(set(all_months) - done_months(region.patch_root(t.id)))
        if missing:
            steps.append({"stage": "patches", "target": t.id, "months": len(missing), "compute": COMPUTE["patches"]})
        if "cnn" in forced or not Path(region.outputs(t.id)["cnn"]).is_file():
            steps.append({"stage": "cnn", "target": t.id, "compute": COMPUTE["cnn"]})
        steps.append({"stage": "final", "target": t.id, "compute": COMPUTE["final"]})
    steps.append({"stage": "export", "compute": COMPUTE["export"]})
    return {"region": region.name, "anchor": region.anchor, "chunk_stations": len(chunk),
            "native_window": reader.window, "problems": problems, "steps": steps, "months": all_months,
            "method": region.extra.get("method", "")}


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=4 * 60 * 60, volumes={"/data": data_volume})
def build_stations(name: str):
    """Station table for a region with a stations spec (IEM ASOS, optionally ECCC); targets are always kept."""
    import numpy as np
    from station_labels import build_region

    data_volume.reload()
    region = region_of(name)
    spec = region.station_build
    manifest = build_region(region.station_root, spec["bbox"], spec.get("asos_networks", ()), spec.get("eccc", False),
                            keep=[t.id for t in region.targets])
    data_volume.commit()
    with np.load(Path(region.station_root) / "labels.npz") as saved:
        labels = saved["labels"]
    ids = [s["id"] for s in manifest["stations"]]
    coverage = {t.id: {v: float(np.isfinite(labels[:, ids.index(t.id), j]).mean())
                       for j, v in enumerate(("wind", "t2m", "pressure", "precip"))} for t in region.targets}
    return {"stations": len(ids), "dropped": len(manifest["dropped"]), "target_coverage": coverage}


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=3 * 60 * 60, max_containers=20, retries=1,
              volumes={"/data": data_volume})
def point_month(name: str, key: str):
    from points import build_month

    data_volume.reload()
    region = region_of(name)
    manifest = build_month(key, stations_of(region), region.anchor, Path(region.point_root) / key)
    data_volume.commit()
    return {"month": key, "samples": manifest["samples"], "planned": manifest["planned"]}


@app.function(image=gpu_image, gpu=["L4", "A10"], cpu=8, memory=65536, timeout=4 * 60 * 60,
              volumes={"/data": data_volume})
def cnn(name: str, target: str):
    from cnn_correction import run_cnn

    data_volume.reload()
    region = region_of(name)
    out = Path(region.outputs(target)["cnn"])
    out.parent.mkdir(parents=True, exist_ok=True)
    from points import PATCH_VARIABLES

    options = {"variables": list(PATCH_VARIABLES), **region.target(target).cnn_options}
    report = run_cnn(region.point_root, region.patch_root(target), region.station_root, target,
                     out=out.with_suffix(".json"), **options)
    data_volume.commit()
    return json.loads(json.dumps({"val_rmse": report["val_rmse"], "test_rmse": {
        m: {v: r[v]["rmse"] for v in r} for m, r in report["methods"].items()}}, default=float))


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=30 * 60, volumes={"/data": data_volume})
def finalize(name: str, method: str = ""):
    """Score each target's CNN, write the site export and the matched comparison with every site method."""
    from site_scoring import matched_comparison
    from final import final_target, site_export

    data_volume.reload()
    region = region_of(name)
    finals, summary = {}, {}
    for t in region.targets:
        report, finals[t.id] = final_target(region, t.id)
        summary[t.id] = {**{v: report["variables"][v]["rmse"] for v in ("t2m", "wind")},
                         **{f"{v}_raw": report["variables"][v]["raw_rmse"] for v in ("t2m", "wind")}}
    if not any(t.site_id for t in region.targets):
        data_volume.commit()
        return json.loads(json.dumps({"targets": summary, "site_export": None}, default=float))
    method = method or region.extra.get("method", f"hrrr_correction_{region.name}")
    out = Path(region.output_root) / "site_export"
    exported = site_export(region, finals, "/benchmark/observations", out, method)
    comparison = {p.parent.name: matched_comparison(exported, p) for p in sorted(Path("/benchmark").glob("*/*_2025.npz"))}
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    data_volume.commit()
    pooled = {base: {s: {v: {k: e["pooled"][k]["rmse"] for k in ("hyperds", "baseline")} if e["pooled"] else None
                         for v, e in by_var.items()} for s, by_var in by_station.items()}
              for base, by_station in comparison.items()}
    # JSON round trip: the local entrypoint has no numpy to unpickle numpy scalars.
    return json.loads(json.dumps({"targets": summary, "site_export": str(exported), "pooled_vs_site_methods": pooled},
                                 default=float))


@app.local_entrypoint()
def main(region: str = "waterloo", execute: bool = False, force: str = ""):
    """Print the plan; with --execute, run the pending stages in order."""
    pending = plan.remote(region, force)
    if execute and any(s["stage"] == "stations" for s in pending["steps"]):
        print(json.dumps({"stations": build_stations.remote(region)}, indent=2))
        pending = plan.remote(region, force)
    months = pending.pop("months")
    print(json.dumps(pending, indent=2))
    if pending["problems"]:
        raise SystemExit("config problems; fix before executing")
    if not execute:
        print("plan only; pass --execute to run")
        return
    steps = pending["steps"]
    if any(s["stage"] == "points" for s in steps):
        results = list(point_month.starmap([(region, k) for k in months]))
        print(json.dumps({"points": sum(r["samples"] for r in results), "planned": sum(r["planned"] for r in results)}))
    patch_jobs = [(region, s["target"], k) for s in steps if s["stage"] == "patches" for k in months]
    if patch_jobs:
        results = list(context_month.starmap(patch_jobs))
        print(json.dumps({"patches": sum(r["samples"] for r in results), "planned": sum(r["planned"] for r in results)}))
    # Sites train concurrently; the final step waits for all of them.
    calls = [(s, cnn.spawn(region, s["target"])) for s in steps if s["stage"] == "cnn"]
    for step, call in calls:
        print(json.dumps({step["stage"]: step["target"], "result": call.get()}))
    result = finalize.remote(region, pending["method"])
    print(json.dumps(result, indent=2))


# CNN input ablations: all catalog fields + whole-chunk view, time context, site trajectory.

def context_root(region, target):
    return region.patch_root(target)


@app.function(image=cpu_image, cpu=8, memory=32768, timeout=3 * 60 * 60, max_containers=20, retries=1,
              volumes={"/data": data_volume})
def context_month(name: str, target: str, key: str):
    from points import build_context_month

    data_volume.reload()
    region = region_of(name)
    site = next(s for s in stations_of(region) if s["id"] == target)
    manifest = build_context_month(key, site, Path(context_root(region, target)) / key, region.patch_half)
    data_volume.commit()
    return {"month": key, "samples": manifest["samples"], "planned": manifest["planned"],
            "failures": len(manifest["failures"])}


@app.function(image=gpu_image, gpu=["L4", "A10"], cpu=8, memory=98304, timeout=6 * 60 * 60,
              volumes={"/data": data_volume})
def cnn_ablation(name: str, target: str, tag: str, options: str):
    """One CNN variant; options is JSON: context, trajectory, wide, variables, source ("v1" or "context")."""
    from cnn_correction import run_cnn

    data_volume.reload()
    region = region_of(name)
    opts = json.loads(options)
    source = opts.pop("source", "v1")
    legacy = f"/data/processed/patches/hrrr_{target}_{2 * region.patch_half + 1}_v1"  # Pa/rate float16 bug
    patch_root = region.patch_root(target) if source == "context" else legacy
    out = Path(region.target_root(target)) / "cnn_ablation" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = run_cnn(region.point_root, patch_root, region.station_root, target,
                     out=out, **opts)
    data_volume.commit()
    return json.loads(json.dumps({"tag": tag, "options": report["options"] | {"source": source},
                                  "val_rmse": report["val_rmse"],
                                  "test_rmse": {m: {v: r[v]["rmse"] for v in r} for m, r in report["methods"].items()}},
                                 default=float))


# All variants read the context extraction (float16-safe units). The control uses the original 15
# fields, so a3 isolates the extra fields and a1/a2/a4 isolate one input each.
ORIGINAL = ["temperature_2m", "dew_point_temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m",
            "wind_u_80m", "wind_v_80m", "wind_gust_surface", "pressure_surface", "total_cloud_cover_atmosphere",
            "downward_short_wave_radiation_flux_surface", "downward_long_wave_radiation_flux_surface",
            "snow_area_fraction_surface", "precipitable_water_atmosphere", "precipitation_surface"]  # = PATCH_VARIABLES
ABLATIONS = {
    "a0_control": {"source": "context", "variables": ORIGINAL},
    "a1_time_context": {"source": "context", "variables": ORIGINAL, "context": [-3, 0, 3]},
    "a2_trajectory": {"source": "context", "variables": ORIGINAL, "trajectory": True},
    "a3_all_fields": {"source": "context"},
    "a4_wide_view": {"source": "context", "variables": ORIGINAL, "wide": True},
    # Combinations of the inputs that improved 2024 validation (a1, a2; a4 for wind only).
    "a5_traj_context": {"source": "context", "variables": ORIGINAL, "trajectory": True, "context": [-3, 0, 3]},
    "a6_traj_context_wide": {"source": "context", "variables": ORIGINAL, "trajectory": True, "context": [-3, 0, 3],
                             "wide": True},
}


@app.local_entrypoint()
def ablate(region: str = "waterloo", target: str = "eric_d_soulis", tags: str = "", extract: bool = False,
           options: str = ""):
    """--extract builds the context patches; --tags runs named ablations (or --options JSON with one tag)."""
    if extract:
        months = plan.remote(region)["months"]
        results = list(context_month.starmap([(region, target, k) for k in months]))
        print(json.dumps({"context_samples": sum(r["samples"] for r in results),
                          "planned": sum(r["planned"] for r in results),
                          "failures": sum(r["failures"] for r in results)}))
    names = [t for t in tags.split(",") if t]
    jobs = [(t, json.dumps(json.loads(options) if options else ABLATIONS[t])) for t in names]
    calls = [(t, cnn_ablation.spawn(region, target, t, o)) for t, o in jobs]
    for tag, call in calls:
        print(json.dumps(call.get()), flush=True)


# Precipitation (targets with "precip": true).

@app.function(image=precip_image, cpu=4, memory=32768, timeout=60 * 60, volumes={"/data": data_volume})
def precip_check(name: str, target: str = "eric_d_soulis"):
    """Gauge coverage and timing: wet/dry correlation of the Soulis hourly series with HRRR (leads 1-6 at the
    site) and with the nearest gauges, at lags -3..3 h, training period only."""
    import numpy as np
    import mos
    from precip import soulis_hourly, lag_check, WET

    data_volume.reload()
    region = region_of(name)
    meta = stations_of(region)
    with np.load(Path(region.station_root) / "labels.npz") as saved:
        label_times, labels = saved["times"], saved["labels"]
    ids = [s["id"] for s in meta]
    times, mm = soulis_hourly("/soulis_raw")
    index = np.searchsorted(times, label_times)
    hit = (index < len(times)) & (times[np.minimum(index, len(times) - 1)] == label_times)
    soulis = np.full(len(label_times), np.nan)
    soulis[hit] = mm[index[hit]]
    years = {}
    for year in range(2021, 2026):
        start, end = (int(__import__("datetime").datetime(y, 1, 1, tzinfo=__import__("datetime").timezone.utc).timestamp())
                      for y in (year, year + 1))
        sel = (label_times >= start) & (label_times < end)
        years[year] = {"coverage": float(np.isfinite(soulis[sel]).mean()), "total_mm": float(np.nansum(soulis[sel])),
                       "wet_hour_rate": float(np.nanmean(soulis[sel] >= WET))}
    values, inits, point_ids, variables = mos.load_points(region.point_root)
    hrrr = np.full(len(label_times), np.nan)
    pr = values[:, 1:7, point_ids.index(target), variables.index("precipitation_surface")] * 3600
    for lead in range(6):
        stamps = inits + (lead + 1) * 3600
        i = np.searchsorted(label_times, stamps)
        ok = (i < len(label_times)) & (label_times[np.minimum(i, len(label_times) - 1)] == stamps)
        hrrr[i[ok]] = pr[ok, lead]
    train = label_times < mos.TRAIN_END
    coverage = np.isfinite(labels[train, :, 3]).mean(axis=0)
    gauges = [i for i in ids if i != target and coverage[ids.index(i)] >= 0.3]
    nearest = mos.nearest_stations(meta, gauges + [target], target, 3)[1:]
    out = {"years": years, "lag_vs_hrrr": lag_check(label_times, soulis, hrrr, before=mos.TRAIN_END),
           "note": "lag L correlates Soulis at t with the reference at t + L; the peak should be at 0"}
    for station in nearest:
        out[f"lag_vs_{station}"] = lag_check(label_times, soulis, labels[:, ids.index(station), 3].astype(float),
                                             before=mos.TRAIN_END)
    return json.loads(json.dumps(out, default=float))


@app.function(image=precip_image, gpu=["L4", "A10"], cpu=16, memory=98304, timeout=6 * 60 * 60,
              volumes={"/data": data_volume})
def precip(name: str, target: str = "eric_d_soulis"):
    from precip import run

    data_volume.reload()
    region = region_of(name)
    out = Path(region.outputs(target)["precip"])
    out.parent.mkdir(parents=True, exist_ok=True)
    report = run(region.point_root, region.station_root, target, k=region.k_neighbors,
                 soulis_raw="/soulis_raw" if target == "eric_d_soulis" else None,
                 patch_root=region.patch_root(target), out=out)
    data_volume.commit()
    summary = {"model": "cnn", "val_score": {p: v["val_score"] for p, v in report["selection"].items()}}
    for product, by_season in report["scores"].items():
        summary[product] = {season: {k: (v["brier"] if "brier" in v else v["rmse"]) if isinstance(v, dict) else v
                                     for k, v in entry.items()} for season, entry in by_season.items()}
    return json.loads(json.dumps(summary, default=float))


@app.local_entrypoint()
def rain(region: str = "waterloo", target: str = "eric_d_soulis", check: bool = False):
    """--check reports gauge coverage and timing only; otherwise trains and scores the precipitation models."""
    if check:
        print(json.dumps(precip_check.remote(region, target), indent=1))
    else:
        print(json.dumps(precip.remote(region, target), indent=1))
