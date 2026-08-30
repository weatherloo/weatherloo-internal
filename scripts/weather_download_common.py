#!/usr/bin/env python3
"""Shared config + helpers for the raw HRRR/ERA5 training-data downloaders.

Both ``download_hrrr.py`` and ``download_era5.py`` subset the same
Kitchener-Waterloo bounding box (from ``lit-review/weather-forecasting.md``) and
write to the same configurable data root. This module centralises those constants
plus the retry/backoff HTTP helper (lifted from
``benchmarking-site/data/hrrr_interpolated/compute_benchmark.py``).
"""

from __future__ import annotations

import http.client
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

# scripts/ sits at the repo root, so parents[1] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Region of interest (verbatim bounding box from weather-forecasting.md) ---
LAT_MIN = 42.1175
LAT_MAX = 44.8159
LON_MIN = -82.3481
LON_MAX = -78.6853
# Padding so native grid cells straddling the boundary are retained.
PAD_DEG = 0.5


def resolve_data_root(cli_value: str | None) -> Path:
    """Where downloads land. Precedence: ``--data-root`` -> env -> repo ``data/``.

    The env fallback (``WEATHERLOO_DATA_ROOT``) lets the same script target
    WATcloud bulk storage (``/mnt/wato-drive*``) on the cluster without editing
    code, while defaulting to the gitignored repo-root ``data/`` locally.
    """
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("WEATHERLOO_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "data"


def resolve_hrrr_cache_dir(data_root: Path, cli_value: str | None = None) -> Path:
    """GRIB byte-range cache base. Prefer data_root (bulk disk), not repo ``.cache/``.

    Precedence: ``--cache-dir`` -> ``$WEATHERLOO_HRRR_CACHE`` ->
    ``{data_root}/.cache/hrrr_grib``. Home/SSD fills and crashes around ~20 GB
    with many tiny GRIB blobs, so the default stays next to NetCDF outputs.

    Callers should normally wrap this with :func:`private_hrrr_cache_dir` so
    parallel Slurm jobs do not share one directory (eccodes segfaults when
    another worker deletes/prunes a GRIB still being read).
    """
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("WEATHERLOO_HRRR_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return data_root / ".cache" / "hrrr_grib"


def private_hrrr_cache_dir(base: Path) -> Path:
    """Per-process cache under ``base`` so Slurm workers do not race on NFS.

    Uses ``job-{SLURM_JOB_ID}`` when present, else ``pid-{pid}``. Downloaders
    remove this subdirectory when they finish.
    """
    job = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    leaf = f"job-{job}" if job else f"pid-{os.getpid()}"
    return base / leaf


def unlink_quiet(path: Path) -> None:
    """Best-effort delete; ignore races from parallel workers."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def unlink_grib_and_sidecars(grib_path: Path) -> None:
    """Remove a GRIB blob plus cfgrib's ``*.grib2.*.idx`` sidecars."""
    parent = grib_path.parent
    name = grib_path.name
    for path in parent.glob(f"{name}*"):
        unlink_quiet(path)


def cycle_cache_glob(day, cycle: int) -> str:
    """Filename prefix for one init-cycle's cached GRIB/idx files."""
    return f"{day:%Y%m%d}_t{cycle:02d}z"


def cleanup_cycle_cache(cache_dir: Path, day, cycle: int) -> int:
    """Delete all cache files for one init-cycle. Returns bytes freed."""
    if not cache_dir.is_dir():
        return 0
    prefix = cycle_cache_glob(day, cycle)
    freed = 0
    for path in cache_dir.glob(f"{prefix}*"):
        try:
            freed += path.stat().st_size
        except OSError:
            continue
        unlink_quiet(path)
    return freed


def cache_dir_nbytes(cache_dir: Path) -> int:
    if not cache_dir.is_dir():
        return 0
    total = 0
    for path in cache_dir.iterdir():
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def enforce_cache_budget(cache_dir: Path, max_bytes: int) -> int:
    """Delete oldest files until cache is under ``max_bytes``. Returns bytes freed.

    Safety valve for a *private* cache dir (see :func:`private_hrrr_cache_dir`).
    Prefer per-cycle cleanup after each NetCDF write; this catches leftovers so
    we never approach the ~20 GB home crash threshold.
    """
    if max_bytes <= 0 or not cache_dir.is_dir():
        return 0
    entries: list[tuple[float, int, Path]] = []
    total = 0
    for path in cache_dir.iterdir():
        try:
            if not path.is_file():
                continue
            st = path.stat()
        except OSError:
            continue
        entries.append((st.st_mtime, st.st_size, path))
        total += st.st_size
    if total <= max_bytes:
        return 0
    entries.sort(key=lambda t: t[0])  # oldest first
    freed = 0
    for _mtime, size, path in entries:
        if total - freed <= max_bytes:
            break
        unlink_quiet(path)
        freed += size
    return freed


def download_bytes(
    url: str,
    start: int | None = None,
    end: int | None = None,
    retries: int = 6,
) -> bytes:
    """HTTP GET (optionally a byte range) with exponential-backoff retries.

    Copied from the hrrr_interpolated benchmark; ``end`` is exclusive, matching
    the byte offsets parsed out of a GRIB ``.idx`` file.
    """
    headers = {"User-Agent": "weatherloo-internal/1.0"}
    if start is not None and end is not None:
        # end == -1 => last GRIB message: open-ended range to EOF.
        headers["Range"] = f"bytes={start}-" if end < 0 else f"bytes={start}-{end - 1}"

    last_err: BaseException | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            # http.client.HTTPException covers IncompleteRead (truncated S3 response).
            last_err = exc
            if attempt == retries - 1:
                raise

        delay = min(60.0, (2**attempt) + random.uniform(0, 1))
        print(f"  download retry {attempt + 1}/{retries - 1} in {delay:.1f}s: {url[:80]}…")
        time.sleep(delay)

    if last_err is not None:
        raise last_err
    raise RuntimeError(f"download failed: {url}")


def parse_idx_ranges(
    idx_text: str, needles: dict[str, str]
) -> dict[str, tuple[int, int]]:
    """Map each key in ``needles`` to its (start, end) byte range in the GRIB2.

    ``needles`` is ``{field_key: substring}`` where the substring uniquely
    identifies a GRIB message line in the ``.idx`` inventory (e.g.
    ``":TMP:2 m above ground:"``). ``end`` is the start offset of the next
    message (exclusive); the last message runs to EOF, encoded here as ``-1``
    handled by the caller.
    """
    lines = idx_text.strip().splitlines()
    ranges: dict[str, tuple[int, int]] = {}
    for i, line in enumerate(lines):
        for key, needle in needles.items():
            if key in ranges:
                continue
            if needle in line:
                start = int(line.split(":")[1])
                if i + 1 < len(lines):
                    end = int(lines[i + 1].split(":")[1])
                else:
                    end = -1  # last message: read to end of object
                ranges[key] = (start, end)
    return ranges
