"""Region pipeline on Modal: HRRR points/patches -> gradient boosting + CNN -> validation blend -> site export.

    modal run models/hyperds_full/modal_region.py --region waterloo              # plan only (default)
    modal run models/hyperds_full/modal_region.py --region waterloo --execute    # run missing stages
    modal run models/hyperds_full/modal_region.py --region waterloo --execute --force gbm,cnn

Every stage is idempotent: extraction months with a manifest are skipped, and a
model stage (gbm, cnn) is skipped when its output exists unless named in --force;
re-extracting data needs a new ``version`` in the region config. The plan
lists what would run and its compute class before anything is spent.
"""

import json
from pathlib import Path
import sys

import modal

if not modal.is_local():
    sys.path.insert(0, "/app")
from modal_app import data_volume

app = modal.App("weatherloo-region-pipeline")
HERE = Path(__file__).resolve().parent
cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "dynamical-catalog>=0.8.0,<1", "pyproj==3.7.0", "scipy", "numpy", "lightgbm==4.5.0", "scikit-learn==1.5.2")
gpu_image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "torch==2.5.1", "scikit-learn==1.5.2")
if modal.is_local():
    cpu_image = cpu_image.add_local_dir(str(HERE), "/app")
    gpu_image = gpu_image.add_local_dir(str(HERE), "/app")
    site = HERE.parents[1] / "benchmarking-site/data"
    for source in sorted(site.glob("*/*_2025.npz")) + sorted(site.glob("observations/*/observations_6h_2025.json")):
        cpu_image = cpu_image.add_local_file(str(source), "/benchmark/" + str(source.relative_to(site)))
STAGES = ("points", "patches", "gbm", "cnn", "blend", "export")
COMPUTE = {"points": "cpu x months", "patches": "cpu x months", "gbm": "16 cpu", "cnn": "L4/A10 gpu",
           "blend": "2 cpu", "export": "2 cpu"}


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
        outputs = region.outputs(t.id)
        if "cnn" in t.models:
            missing = sorted(set(all_months) - done_months(region.patch_root(t.id)))
            if missing:
                steps.append({"stage": "patches", "target": t.id, "months": len(missing), "compute": COMPUTE["patches"]})
        for stage in ("gbm", "cnn"):
            if stage in t.models and (stage in forced or not Path(outputs[stage]).is_file()):
                steps.append({"stage": stage, "target": t.id, "compute": COMPUTE[stage]})
        steps.append({"stage": "blend", "target": t.id, "compute": COMPUTE["blend"]})
    steps.append({"stage": "export", "compute": COMPUTE["export"]})
    return {"region": region.name, "anchor": region.anchor, "chunk_stations": len(chunk),
            "native_window": reader.window, "problems": problems, "steps": steps, "months": all_months,
            "method": region.extra.get("method", "")}


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=3 * 60 * 60, max_containers=20, retries=1,
              volumes={"/data": data_volume})
def point_month(name: str, key: str):
    from points import build_month

    data_volume.reload()
    region = region_of(name)
    manifest = build_month(key, stations_of(region), region.anchor, Path(region.point_root) / key)
    data_volume.commit()
    return {"month": key, "samples": manifest["samples"], "planned": manifest["planned"]}


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=3 * 60 * 60, max_containers=20, retries=1,
              volumes={"/data": data_volume})
def patch_month(name: str, target: str, key: str):
    from points import build_patch_month

    data_volume.reload()
    region = region_of(name)
    site = next(s for s in stations_of(region) if s["id"] == target)
    manifest = build_patch_month(key, site, Path(region.patch_root(target)) / key, region.patch_half)
    data_volume.commit()
    return {"month": key, "samples": manifest["samples"], "planned": manifest["planned"]}


@app.function(image=cpu_image, cpu=16, memory=65536, timeout=4 * 60 * 60, volumes={"/data": data_volume})
def gbm(name: str, target: str):
    from mos import run_study_v2

    data_volume.reload()
    region = region_of(name)
    out = Path(region.outputs(target)["gbm"])
    out.parent.mkdir(parents=True, exist_ok=True)
    report = run_study_v2(region.point_root, region.station_root, target, k=region.k_neighbors,
                          out=out.with_suffix(".json"))
    data_volume.commit()
    return {v: {"chosen": e["chosen_by_validation"], **{m: e["methods"][m]["test_rmse"]
                                                        for m in ("raw_hrrr", "final_seed_ensemble")}}
            for v, e in report["targets"].items()}


@app.function(image=gpu_image, gpu=["L4", "A10"], cpu=8, memory=65536, timeout=4 * 60 * 60,
              volumes={"/data": data_volume})
def cnn(name: str, target: str):
    from cnn_correction import run_cnn

    data_volume.reload()
    region = region_of(name)
    out = Path(region.outputs(target)["cnn"])
    out.parent.mkdir(parents=True, exist_ok=True)
    report = run_cnn(region.point_root, region.patch_root(target), region.station_root, target,
                     out=out.with_suffix(".json"))
    data_volume.commit()
    return {"val_rmse": report["val_rmse"], "test_rmse": {m: {v: r[v]["rmse"] for v in r}
                                                         for m, r in report["methods"].items()}}


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=30 * 60, volumes={"/data": data_volume})
def finalize(name: str, method: str = ""):
    """Blend every target, write the site export and the matched comparison with every site method."""
    from benchmark import matched_comparison
    from blend import blend_target, site_export

    data_volume.reload()
    region = region_of(name)
    finals, summary = {}, {}
    for t in region.targets:
        report, finals[t.id] = blend_target(region, t.id)
        summary[t.id] = {"weights_on_gbm": report["weights_on_gbm"],
                         **{v: report["variables"][v]["test_final"]["rmse"] for v in ("t2m", "wind")},
                         **{f"{v}_raw": report["variables"][v]["test_final"]["raw_rmse"] for v in ("t2m", "wind")}}
    method = method or region.extra.get("method", f"hrrr_correction_{region.name}")
    out = Path(region.output_root) / "site_export"
    exported = site_export(region, finals, "/benchmark/observations", out, method)
    comparison = {p.parent.name: matched_comparison(exported, p) for p in sorted(Path("/benchmark").glob("*/*_2025.npz"))}
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    data_volume.commit()
    pooled = {base: {s: {v: {k: e["pooled"][k]["rmse"] for k in ("hyperds", "baseline")} if e["pooled"] else None
                         for v, e in by_var.items()} for s, by_var in by_station.items()}
              for base, by_station in comparison.items()}
    return {"targets": summary, "site_export": str(exported), "pooled_vs_site_methods": pooled}


@app.local_entrypoint()
def main(region: str = "waterloo", execute: bool = False, force: str = ""):
    """Print the plan; with --execute, run the pending stages in order."""
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
        results = list(patch_month.starmap(patch_jobs))
        print(json.dumps({"patches": sum(r["samples"] for r in results), "planned": sum(r["planned"] for r in results)}))
    # Independent model fits run concurrently; blending waits for all of them.
    calls = [(s, (gbm if s["stage"] == "gbm" else cnn).spawn(region, s["target"]))
             for s in steps if s["stage"] in ("gbm", "cnn")]
    for step, call in calls:
        print(json.dumps({step["stage"]: step["target"], "result": call.get()}))
    result = finalize.remote(region, pending["method"])
    print(json.dumps(result, indent=2))
