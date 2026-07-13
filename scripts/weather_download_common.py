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
