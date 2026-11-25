"""Region configuration for the HRRR point-correction pipeline.

A region names an anchor station (whose native HRRR storage chunk is read once
per initialization) and the target stations to correct. Every target must lie
inside the anchor's chunk and in the labelled station table. Every target is
corrected by the same model type, the patch CNN (cnn_correction.run_cnn), with
optional per-target input options. Configs live in ``regions/<name>.json``.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

REGION_DIR = Path(__file__).resolve().parent / "regions"


@dataclass(frozen=True)
class Target:
    id: str                      # station id in the station table
    site_id: str | None = None   # benchmarking-site station id, when the site scores it
    cnn_options: dict = field(default_factory=dict)  # cnn_correction.run_cnn options (context, trajectory, ...)
    precip: bool = False  # also forecast precipitation (needs a gauge record for the target)


@dataclass(frozen=True)
class Region:
    name: str
    anchor: str
    targets: tuple
    station_root: str = "/data/processed/hyperds_nwp/stations_v2"
    k_neighbors: int = 30
    patch_half: int = 8
    version: str = "v1"
    description: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def point_root(self):
        return f"/data/processed/points/hrrr_{self.anchor}_{self.version}"

    def patch_root(self, target):
        """All catalog fields on the patch plus the whole-chunk view, in float16-safe units (points.ContextReader)."""
        return f"/data/processed/patches/hrrr_{target}_context_{2 * self.patch_half + 1}_b8_{self.version}"

    @property
    def output_root(self):
        return f"/data/processed/regions/{self.name}"

    def target_root(self, target):
        return f"{self.output_root}/{target}"

    def target(self, target_id):
        return next(t for t in self.targets if t.id == target_id)

    def outputs(self, target_id):
        root = self.target_root(target_id)
        return {"cnn": f"{root}/cnn.npz", "final": f"{root}/final.json", "precip": f"{root}/precip.json"}


def parse(document):
    targets = []
    for entry in document["targets"]:
        if "models" in entry and list(entry["models"]) != ["cnn"]:
            raise ValueError(f"{entry['id']}: the pipeline uses one model type, the CNN; drop 'models'")
        options = dict(entry.get("cnn_options", {}))
        unknown = set(options) - {"context", "trajectory", "wide", "variables"}
        if unknown:
            raise ValueError(f"{entry['id']}: unknown cnn_options {sorted(unknown)}")
        targets.append(Target(entry["id"], entry.get("site_id"), options, bool(entry.get("precip", False))))
    if len({t.id for t in targets}) != len(targets):
        raise ValueError("duplicate target")
    known = {"name", "anchor", "targets", "station_root", "k_neighbors", "patch_half", "version", "description"}
    region = Region(document["name"], document["anchor"], tuple(targets),
                    **{k: document[k] for k in known - {"name", "anchor", "targets"} if k in document},
                    extra={k: v for k, v in document.items() if k not in known})
    if region.anchor not in {t.id for t in targets} and not document.get("anchor_is_not_target"):
        raise ValueError("anchor should be a target (set anchor_is_not_target to override)")
    return region


def load(name):
    path = Path(name)
    if not path.suffix:
        path = REGION_DIR / f"{name}.json"
    return parse(json.loads(path.read_text()))


def check(region, stations, chunk_station_ids=None):
    """Problems that would make a stage fail; empty list when the config is usable.

    Correction models are site-specific with a temporal split (train to 2023, validate 2024, test
    2025), so the station table's spatial split does not constrain targets.
    """
    problems = []
    ids = {s["id"] for s in stations}
    for t in region.targets:
        if t.id not in ids:
            problems.append(f"{t.id}: not in station table {region.station_root}")
        if chunk_station_ids is not None and t.id not in chunk_station_ids:
            problems.append(f"{t.id}: outside the native chunk of anchor {region.anchor}")
    return problems
