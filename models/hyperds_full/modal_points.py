"""Remote native-HRRR point extraction and station-level forecast-correction (MOS) study.

Experiment record for the Soulis rounds 1-2 and the first CNN run. The packaged, region-configurable
version is modal_region.py (see PIPELINE.md); both share the same point and patch caches.

    modal deploy models/hyperds_full/modal_points.py
"""

import json
from pathlib import Path
import sys

import modal

if not modal.is_local():
    sys.path.insert(0, "/app")
from modal_app import data_volume, checkpoint_volume

app = modal.App("weatherloo-points")
STATION_ROOT = "/data/processed/hyperds_nwp/stations_v2"
TARGET = "eric_d_soulis"
POINT_ROOT = f"/data/processed/points/hrrr_{TARGET}_v1"
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "dynamical-catalog>=0.8.0,<1", "pyproj==3.7.0", "scipy", "numpy", "lightgbm==4.5.0", "scikit-learn==1.5.2")
if modal.is_local():
    image = image.add_local_dir(str(Path(__file__).resolve().parent), "/app")


def load_stations():
    return json.loads((Path(STATION_ROOT) / "stations.json").read_text())["stations"]


@app.function(image=image, cpu=2, memory=8192, timeout=20 * 60, volumes={"/data": data_volume})
def probe():
    """One initialization: which stations share the target's native chunk, and value sanity."""
    from datetime import datetime, timezone
    import numpy as np
    from points import PointReader, VARIABLES

    reader = PointReader(load_stations(), TARGET)
    values = reader.read(datetime(2024, 7, 1, 12, tzinfo=timezone.utc))
    soulis = [s["id"] for s in reader.stations].index(TARGET)
    report = {"stations": [s["id"] for s in reader.stations], "window": reader.window, "units": reader.units,
            "soulis_lead6": dict(zip(VARIABLES, np.round(values[6, soulis], 3).tolist())),
            "finite_fraction_by_variable": dict(zip(VARIABLES, np.isfinite(values).mean(axis=(0, 1)).round(3).tolist()))}
    print(json.dumps(report, indent=1), flush=True)
    return report


@app.function(image=image, cpu=8, memory=16384, timeout=3 * 60 * 60, max_containers=20, retries=1,
              volumes={"/data": data_volume})
def month(key: str):
    from points import build_month

    data_volume.reload()
    manifest = build_month(key, load_stations(), TARGET, Path(POINT_ROOT) / key)
    data_volume.commit()
    return {key: manifest["samples"], "planned": manifest["planned"]}


@app.function(image=image, cpu=0.25, memory=1024, timeout=24 * 60 * 60)
def extract():
    from points import months
    results = list(month.map(months()))
    return {"months": len(results), "samples": sum(next(iter(r.values())) for r in results),
            "planned": sum(r["planned"] for r in results)}


@app.function(image=image, cpu=16, memory=65536, timeout=3 * 60 * 60,
              volumes={"/data": data_volume, "/checkpoints": checkpoint_volume.with_mount_options(read_only=True)})
def study(k: int = 30, tag: str = "v1"):
    """Station-level correction study at the target: bias table, ridge, gradient boosting, r5, bootstrap."""
    from mos import run_study

    data_volume.reload()
    out = Path(POINT_ROOT) / f"study_{tag}_k{k}.json"
    report = run_study(POINT_ROOT, STATION_ROOT, TARGET, k=k,
                       r5_predictions="/checkpoints/hyperds_nwp/hrrr_v1_r5/nwp/predictions.npz", out=out)
    data_volume.commit()
    return report


bench_image = image
if modal.is_local():
    site = Path(__file__).resolve().parents[2] / "benchmarking-site/data"
    for source in sorted(site.glob("*/*_2025.npz")):
        bench_image = bench_image.add_local_file(str(source), "/benchmark/" + str(source.relative_to(site)))
    for station in ("cyyz", "eric_d_soulis"):
        source = site / "observations" / station / "observations_6h_2025.json"
        bench_image = bench_image.add_local_file(str(source), f"/benchmark/observations/{station}/{source.name}")


@app.function(image=image, cpu=16, memory=65536, timeout=4 * 60 * 60,
              volumes={"/data": data_volume, "/checkpoints": checkpoint_volume.with_mount_options(read_only=True)})
def study_v2(target: str = TARGET, k: int = 30):
    from mos import run_study_v2

    data_volume.reload()
    out = Path(POINT_ROOT) / f"study_v2_{target}.json"
    report = run_study_v2(POINT_ROOT, STATION_ROOT, target, k=k,
                          r5_predictions="/checkpoints/hyperds_nwp/hrrr_v1_r5/nwp/predictions.npz", out=out)
    data_volume.commit()
    return report


@app.function(image=bench_image, cpu=2, memory=8192, timeout=30 * 60, volumes={"/data": data_volume})
def site_benchmark(method: str = "gbm_hrrr_points"):
    """Score the round-2 exports with the benchmarking site's truth and matched comparisons."""
    import numpy as np
    from benchmark import site_export, matched_comparison
    from mos import site_predictions

    data_volume.reload()
    exports = {}
    for site_id, target in (("eric_d_soulis", "eric_d_soulis"), ("cyyz", "eccc_51459")):
        with np.load(Path(POINT_ROOT) / f"study_v2_{target}.npz") as saved:
            exports[site_id] = {k: saved[k] for k in saved.files}
    predictions = site_predictions(exports, Path(POINT_ROOT) / "site_predictions.npz")
    out = Path(POINT_ROOT) / "site_export"
    exported = site_export(predictions, "/benchmark/observations", out, method)
    comparison = {path.parent.name: matched_comparison(exported, path) for path in sorted(Path("/benchmark").glob("*/*_2025.npz"))}
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    data_volume.commit()
    return comparison


PATCH_ROOT = f"/data/processed/patches/hrrr_{TARGET}_17_v1"


@app.function(image=image, cpu=8, memory=16384, timeout=3 * 60 * 60, max_containers=20, retries=1,
              volumes={"/data": data_volume})
def patch_month(key: str):
    from points import build_patch_month

    data_volume.reload()
    site = next(s for s in load_stations() if s["id"] == TARGET)
    manifest = build_patch_month(key, site, Path(PATCH_ROOT) / key)
    data_volume.commit()
    return {key: manifest["samples"], "planned": manifest["planned"]}


@app.function(image=image, cpu=0.25, memory=1024, timeout=24 * 60 * 60)
def extract_patches():
    from points import months
    results = list(patch_month.map(months()))
    return {"months": len(results), "samples": sum(next(iter(r.values())) for r in results),
            "planned": sum(r["planned"] for r in results)}


cnn_image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "torch==2.5.1", "scikit-learn==1.5.2")
if modal.is_local():
    cnn_image = cnn_image.add_local_dir(str(Path(__file__).resolve().parent), "/app")


@app.function(image=cnn_image, gpu=["L4", "A10"], cpu=8, memory=65536, timeout=4 * 60 * 60,
              volumes={"/data": data_volume, "/checkpoints": checkpoint_volume.with_mount_options(read_only=True)})
def cnn(target: str = TARGET):
    from cnn_correction import run_cnn

    data_volume.reload()
    out = Path(POINT_ROOT) / f"cnn_v1_{target}.json"
    report = run_cnn(POINT_ROOT, PATCH_ROOT, STATION_ROOT, target,
                     gbm_export=str(Path(POINT_ROOT) / f"study_v2_{target}.npz"),
                     r5_predictions="/checkpoints/hyperds_nwp/hrrr_v1_r5/nwp/predictions.npz", out=out)
    data_volume.commit()
    return {k: v for k, v in report.items() if k != "history"}
