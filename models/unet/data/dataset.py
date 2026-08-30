#!/usr/bin/env python3
"""PyTorch Dataset for the residual-learning U-Net.

Each sample pairs a GFS forecast region grid (the model **input**) with the
``GFS - ERA5`` residual over the same grid (the **target**). The network learns
to predict that error so that ``corrected = GFS - predicted_error`` (see
``models/unet/README.md``).

Design decisions
----------------
* **Units.** ``gfs_region_grid`` returns t2m in degC, but ``era5_region``
  returns t2m in **Kelvin** (it only renames). We convert ERA5 t2m to degC here
  so the residual is a physically meaningful degC error, and u10/v10 are already
  m/s on both sides.
* **Channels** are ``[t2m, u10, v10]`` in that fixed order, shape ``(C, H, W)``
  with ``H``=latitude (ascending), ``W``=longitude. For the region box this is
  ``(3, 21, 41)``.
* **Normalization.** Per-channel z-score. Stats (mean/std) are computed **once**
  over a sample of the *training* split only (no val leakage) and cached to
  ``stats.json`` so runs are reproducible. Both the GFS input and the residual
  target get their own stats; the corrected forecast is reconstructed downstream
  via :func:`denormalize_residual`.
* **Caching.** Un-normalized region grids are cached as ``.npz`` under
  ``.cache/unet_training/`` keyed by ``date+cycle+fxx`` (which fixes the valid
  time and therefore every variable), so epochs after the first do no network IO.
* **Split.** Chronological 80/20 by valid time — **not** random. This is a time
  series, so a random split would leak future information into training (same
  reasoning as the DLSD paper).

Run standalone for a quick self-test over one week::

    .venv/bin/python models/unet/data/dataset.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # when run as a script from this directory
    from fetch_era5 import load_config, open_era5, era5_region, _np_time
    from fetch_gfs import gfs_region_grid
except ImportError:  # when imported as models.unet.data.dataset
    from .fetch_era5 import load_config, open_era5, era5_region, _np_time
    from .fetch_gfs import gfs_region_grid

REPO_ROOT = Path(__file__).resolve().parents[3]
# Cache root is overridable via UNET_DATA_DIR (set by run_pipeline.py) so the
# whole cache can live on an arbitrary data volume, e.g. on WATcloud. An env
# var (not a function arg) so DataLoader worker subprocesses inherit it too.
CACHE_ROOT = Path(os.environ.get("UNET_DATA_DIR", str(REPO_ROOT / ".cache")))
CACHE_DIR = CACHE_ROOT / "unet_training"
STATS_PATH = Path(__file__).resolve().parent / "stats.json"

CHANNELS = ("t2m", "u10", "v10")

# Forecast lead is fed to the network as an extra, spatially-constant input
# channel. The residual the model has to predict grows with lead, and the grid
# alone does not say whether it is an f006 or an f048 field — without this one
# network trained across leads can only learn a single blended correction,
# over-correcting short leads and under-correcting long ones.
#
# Scaled by the longest configured lead so the channel lands in ~(0, 1], the
# same order as the z-scored physical channels.
LEAD_SCALE_HOURS = 48.0


def lead_plane(fxx: float, height: int, width: int,
               dtype=np.float32) -> np.ndarray:
    """Constant ``(1, H, W)`` plane encoding forecast lead in hours."""
    return np.full((1, height, width), float(fxx) / LEAD_SCALE_HOURS, dtype=dtype)


def build_model_input(gfs_norm: np.ndarray, fxx: float) -> np.ndarray:
    """Normalized GFS grid ``(C,H,W)`` + lead -> ``(C+1,H,W)`` network input.

    The single definition of the input layout: training, station evaluation and
    the dashboard benchmark all build their tensors through here so the lead
    encoding cannot drift between them.
    """
    h, w = gfs_norm.shape[-2:]
    return np.concatenate(
        [gfs_norm, lead_plane(fxx, h, w, gfs_norm.dtype)], axis=0)

# Usable overlap of the two archives (see README): GFS 0.25° AWS start ..
# the old 6-hourly ERA5 store's end. Kept as defaults for backwards compat;
# with the full_37-1h store the real end is read from the store's time axis.
DEFAULT_START = "2021-03-23"
DEFAULT_END = "2021-12-31"
# Fallback last analysis instant (the old store's end) when no ERA5 dataset is
# available to ask; a valid time past the store's end has no ground truth.
_ERA5_LAST_VALID_FALLBACK = datetime(2021, 12, 31, 18, tzinfo=timezone.utc)


def era5_last_valid_time(era5_ds) -> datetime:
    """Last *real* analysis instant in the opened ERA5 store (UTC).

    The ARCO "ar" stores pad the time coordinate far past the ingested data
    (full_37-1h runs 1900..2050 with empty chunks beyond the real boundary),
    so the raw axis end is useless. The true boundary is the
    ``valid_time_stop`` attribute — final ERA5, deliberately NOT the
    preliminary-ERA5T ``valid_time_stop_era5t`` (we don't train ground truth
    on preliminary data). Falls back to the raw time axis for stores without
    the attribute (e.g. the old 6-hourly store, whose axis is not padded).
    """
    stop = era5_ds.attrs.get("valid_time_stop")
    if stop:
        d = datetime.strptime(stop, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return d.replace(hour=23)  # hourly store: data through 23Z of stop date
    last = era5_ds.time.values[-1].astype("datetime64[s]").astype(object)
    return last.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Sample enumeration
# ---------------------------------------------------------------------------
def build_samples(cfg: dict, start: str, end: str,
                  last_valid: datetime | None = None) -> list[dict]:
    """All ``{date, cycle, fxx}`` triples in ``[start, end]``, sorted by valid time.

    Enumerates every configured init cycle and forecast hour whose init date is
    in range and whose valid time still has ERA5 ground truth. ``last_valid``
    should be :func:`era5_last_valid_time` of the opened store; when ``None``
    the old 6-hourly store's end (2021-12-31 18Z) is assumed.
    """
    gfs = cfg["data"]["gfs"]
    cycles = gfs["init_hours_utc"]
    leads = gfs["forecast_hours"]
    if last_valid is None:
        last_valid = _ERA5_LAST_VALID_FALLBACK

    start_d = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_d = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    samples: list[dict] = []
    day = start_d
    while day <= end_d:
        for cycle in cycles:
            init = day.replace(hour=cycle)
            if init < start_d:
                continue
            for fxx in leads:
                valid = init + timedelta(hours=fxx)
                if valid > last_valid:
                    continue
                samples.append({"date": day.strftime("%Y-%m-%d"), "cycle": int(cycle),
                                "fxx": int(fxx)})
        day += timedelta(days=1)

    samples.sort(key=lambda s: _valid_time(s))
    return samples


def _valid_time(sample: dict) -> datetime:
    d = datetime.strptime(sample["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return d.replace(hour=sample["cycle"]) + timedelta(hours=sample["fxx"])


def chronological_split(samples: list[dict], val_fraction: float = 0.2
                        ) -> tuple[list[dict], list[dict]]:
    """First ``1-val_fraction`` (earliest) → train, last ``val_fraction`` → val."""
    n_train = int(round(len(samples) * (1.0 - val_fraction)))
    return samples[:n_train], samples[n_train:]


def interleaved_month_split(samples: list[dict], train_days: int = 24
                            ) -> tuple[list[dict], list[dict]]:
    """Per-month split: valid-time day-of-month ≤ ``train_days`` → train, else val.

    Keeps ~80/20 (24/~30) while ensuring **every month is represented in both**
    sets, fixing the chronological split's "val is an unseen season" artifact.
    Val days are the *contiguous* tail of each month, so train/val only touch at
    two month-boundary days — minimizing the temporal-autocorrelation leakage that
    a random per-sample split would incur.
    """
    train, val = [], []
    for s in samples:
        (train if _valid_time(s).day <= train_days else val).append(s)
    return train, val


def split_samples(samples: list[dict], split_mode: str = "chronological",
                  val_fraction: float = 0.2, train_days: int = 24
                  ) -> tuple[list[dict], list[dict]]:
    if split_mode == "chronological":
        return chronological_split(samples, val_fraction)
    if split_mode == "interleaved_month":
        return interleaved_month_split(samples, train_days)
    raise ValueError(f"unknown split_mode {split_mode!r}")


# ---------------------------------------------------------------------------
# Grid loading (cached, un-normalized)
# ---------------------------------------------------------------------------
def _cache_path(sample: dict) -> Path:
    return CACHE_DIR / f"{sample['date']}_{sample['cycle']:02d}z_f{sample['fxx']:03d}.npz"


def load_sample_grids(cfg: dict, sample: dict, era5_ds=None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(gfs, era5)`` arrays, each ``(3, H, W)`` float32, t2m in degC.

    Cached per sample under ``.cache/unet_training/``. u10/v10 are m/s.
    """
    path = _cache_path(sample)
    if path.exists():
        with np.load(path) as z:
            return z["gfs"].astype(np.float32), z["era5"].astype(np.float32)

    # GFS input grid (t2m already degC, u10/v10 m/s), latitude ascending.
    gfs_grid = gfs_region_grid(cfg, sample["date"].replace("-", ""),
                               sample["cycle"], sample["fxx"])
    gfs = np.stack([gfs_grid[c].values for c in CHANNELS]).astype(np.float32)

    # ERA5 truth at the valid time; t2m is Kelvin here -> convert to degC.
    valid = _valid_time(sample)
    era5_grid = era5_region(cfg, valid, era5_ds)
    era5 = np.stack([era5_grid[c].values for c in CHANNELS]).astype(np.float32)
    era5[0] -= 273.15  # t2m K -> degC to match GFS

    if gfs.shape != era5.shape:
        raise ValueError(f"grid shape mismatch for {sample}: "
                         f"GFS {gfs.shape} vs ERA5 {era5.shape}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, gfs=gfs, era5=era5)
    return gfs, era5


# ---------------------------------------------------------------------------
# Normalization stats (computed once over the train split)
# ---------------------------------------------------------------------------
def compute_stats(cfg: dict, train_samples: list[dict], era5_ds=None,
                  max_samples: int = 80) -> dict:
    """Per-channel mean/std of GFS input and residual target over the train split.

    Uses up to ``max_samples`` evenly-spaced train samples to bound network IO.
    """
    if not train_samples:
        raise ValueError("cannot compute stats from an empty train split")

    if len(train_samples) > max_samples:
        idx = np.linspace(0, len(train_samples) - 1, max_samples).round().astype(int)
        chosen = [train_samples[i] for i in np.unique(idx)]
    else:
        chosen = train_samples

    gfs_stack, res_stack = [], []
    for s in chosen:
        gfs, era5 = load_sample_grids(cfg, s, era5_ds)
        gfs_stack.append(gfs)
        res_stack.append(gfs - era5)
    # (N, C, H, W) -> per-channel over N*H*W
    gfs_all = np.stack(gfs_stack)
    res_all = np.stack(res_stack)

    def _ms(a: np.ndarray) -> dict:
        mean = a.mean(axis=(0, 2, 3))
        std = a.std(axis=(0, 2, 3))
        std = np.where(std < 1e-6, 1.0, std)  # guard constant channels
        return {"mean": mean.tolist(), "std": std.tolist()}

    return {
        "channels": list(CHANNELS),
        "n_samples": len(chosen),
        "gfs": _ms(gfs_all),
        "residual": _ms(res_all),
    }


def sample_space(cfg: dict) -> dict:
    """The (cycles, leads) the stats were computed over — see load_or_compute_stats."""
    gfs = cfg["data"]["gfs"]
    return {"cycles": sorted(int(c) for c in gfs["init_hours_utc"]),
            "leads": sorted(int(f) for f in gfs["forecast_hours"])}


def load_or_compute_stats(cfg: dict, train_samples: list[dict], era5_ds=None,
                          stats_path: Path = STATS_PATH, recompute: bool = False) -> dict:
    want = sample_space(cfg)
    if stats_path.exists() and not recompute:
        with open(stats_path) as f:
            cached = json.load(f)
        # Residual magnitude scales with forecast lead, so stats computed over a
        # narrower cycle/lead set mis-scale a wider one — and the arrays stay
        # (3,) either way, so nothing would fail loudly. Recompute instead of
        # silently normalizing f048 residuals with f006-f024 statistics.
        if cached.get("sample_space") == want:
            return cached
        print(f"[stats] {stats_path.name}: sample space changed "
              f"{cached.get('sample_space')} -> {want}; recomputing")
    stats = compute_stats(cfg, train_samples, era5_ds)
    stats["sample_space"] = want
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def denormalize_residual(norm_residual: np.ndarray | torch.Tensor, stats: dict):
    """Invert z-score on a residual prediction: ``pred * std + mean`` (per channel)."""
    mean = np.asarray(stats["residual"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(stats["residual"]["std"], dtype=np.float32).reshape(-1, 1, 1)
    if isinstance(norm_residual, torch.Tensor):
        m = torch.as_tensor(mean, device=norm_residual.device)
        s = torch.as_tensor(std, device=norm_residual.device)
        return norm_residual * s + m
    return norm_residual * std + mean


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GFSResidualDataset(Dataset):
    """GFS region grid (input) + normalized ``GFS-ERA5`` residual (target).

    Parameters
    ----------
    split : "train" | "val"
        Chronological split selector (earliest ``1-val_fraction`` = train).
    start, end : str (YYYY-MM-DD)
        Init-date range. Defaults to the usable 2021 overlap window.
    cached_only : bool
        Restrict samples to those already cached as ``.npz`` (i.e. what a prior
        ``run_pipeline.py fetch`` produced) so training does zero network IO and
        never trips over permanent archive gaps (e.g. missing GFS cycles).
    """

    def __init__(self, split: Literal["train", "val"] = "train",
                 start: str = DEFAULT_START, end: str = DEFAULT_END,
                 val_fraction: float = 0.2, cfg: dict | None = None,
                 stats: dict | None = None, stats_path: Path = STATS_PATH,
                 recompute_stats: bool = False,
                 split_mode: str = "chronological", train_days: int = 24,
                 cached_only: bool = False):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.cfg = cfg if cfg is not None else load_config()
        self.split = split
        self.split_mode = split_mode
        data = self.cfg["data"]["era5"]
        self._era5_ds = open_era5(data["store"], data["token"])

        all_samples = build_samples(self.cfg, start, end,
                                    last_valid=era5_last_valid_time(self._era5_ds))
        if cached_only:
            all_samples = [s for s in all_samples if _cache_path(s).exists()]
            if not all_samples:
                raise RuntimeError(
                    f"cached_only=True but no cached samples in {CACHE_DIR} for "
                    f"{start} .. {end} — run `run_pipeline.py fetch` first.")
        train_samples, val_samples = split_samples(
            all_samples, split_mode, val_fraction, train_days)
        self.samples = train_samples if split == "train" else val_samples

        # Stats are always derived from the train split (no val leakage).
        self.stats = stats if stats is not None else load_or_compute_stats(
            self.cfg, train_samples, self._era5_ds, stats_path, recompute_stats)

        self._gfs_mean = np.asarray(self.stats["gfs"]["mean"], np.float32).reshape(-1, 1, 1)
        self._gfs_std = np.asarray(self.stats["gfs"]["std"], np.float32).reshape(-1, 1, 1)
        self._res_mean = np.asarray(self.stats["residual"]["mean"], np.float32).reshape(-1, 1, 1)
        self._res_std = np.asarray(self.stats["residual"]["std"], np.float32).reshape(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        gfs, era5 = load_sample_grids(self.cfg, sample, self._era5_ds)
        residual = gfs - era5

        gfs_norm = ((gfs - self._gfs_mean) / self._gfs_std).astype(np.float32)
        x = build_model_input(gfs_norm, sample["fxx"])
        res_norm = (residual - self._res_mean) / self._res_std
        return (torch.from_numpy(x),
                torch.from_numpy(res_norm.astype(np.float32)))


# ---------------------------------------------------------------------------
# Standalone self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:
    from torch.utils.data import DataLoader

    start, end = "2021-06-01", "2021-06-07"
    print(f"=== GFSResidualDataset self-test ({start} .. {end}) ===")

    cfg = load_config()
    all_samples = build_samples(cfg, start, end)
    train_s, val_s = chronological_split(all_samples, 0.2)
    print(f"  total samples: {len(all_samples)}  ->  train {len(train_s)} / val {len(val_s)}")
    print(f"  train span: {_valid_time(train_s[0]).isoformat()} .. "
          f"{_valid_time(train_s[-1]).isoformat()}")
    print(f"  val   span: {_valid_time(val_s[0]).isoformat()} .. "
          f"{_valid_time(val_s[-1]).isoformat()}")

    # Fresh stats for the test window (don't clobber any real stats.json).
    test_stats_path = CACHE_DIR / "stats_selftest.json"
    if test_stats_path.exists():
        test_stats_path.unlink()

    train_ds = GFSResidualDataset("train", start, end, cfg=cfg,
                                  stats_path=test_stats_path, recompute_stats=True)
    val_ds = GFSResidualDataset("val", start, end, cfg=cfg,
                                stats=train_ds.stats)

    x, y = train_ds[0]
    print("\n=== One sample ===")
    print(f"  input  (GFS)      tensor: shape={tuple(x.shape)} dtype={x.dtype}")
    print(f"  target (residual) tensor: shape={tuple(y.shape)} dtype={y.dtype}")

    st = train_ds.stats
    print("\n=== Normalization stats (per channel, from train split) ===")
    print(f"  channels: {st['channels']}  (computed over {st['n_samples']} samples)")
    for grp in ("gfs", "residual"):
        mean = ", ".join(f"{m:+.3f}" for m in st[grp]["mean"])
        std = ", ".join(f"{s:.3f}" for s in st[grp]["std"])
        print(f"  {grp:<9} mean=[{mean}]  std=[{std}]")

    print("\n=== NaN/Inf check over a full batch ===")
    loader = DataLoader(train_ds, batch_size=min(8, len(train_ds)), shuffle=False)
    xb, yb = next(iter(loader))
    print(f"  batch input  shape: {tuple(xb.shape)}")
    print(f"  batch target shape: {tuple(yb.shape)}")
    for name, t in (("input", xb), ("target", yb)):
        n_nan = int(torch.isnan(t).sum())
        n_inf = int(torch.isinf(t).sum())
        print(f"  {name:<7} NaN={n_nan}  Inf={n_inf}  "
              f"min={t.min():+.3f} max={t.max():+.3f} mean={t.mean():+.3f}")
        assert n_nan == 0 and n_inf == 0, f"found NaN/Inf in {name}!"

    print(f"\n  val dataset length: {len(val_ds)} (uses train stats)")
    print("\nOK: dataset builds, shapes correct, no NaN/Inf.")


if __name__ == "__main__":
    _selftest()
