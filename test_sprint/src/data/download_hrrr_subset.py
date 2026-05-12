from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from botocore.exceptions import ConnectionClosedError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError
from zarr.errors import GroupNotFoundError

from src.data.hrrr import HRRRPatchLoader, WaterlooPatchConfig


DEFAULT_START_INIT_UTC = "2025-12-31 06:00:00+00:00"
DEFAULT_END_INIT_UTC = "2026-02-28 22:00:00+00:00"
DEFAULT_LEAD_HOURS = tuple(range(1, 19))
DEFAULT_ARCHIVE_RETRY_ATTEMPTS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a local Waterloo HRRR subset from the public archive.")
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/hrrr/waterloo_janfeb_2026_subset.zarr",
        help="Local Zarr output path.",
    )
    parser.add_argument(
        "--start-init-utc",
        type=str,
        default=DEFAULT_START_INIT_UTC,
        help="First HRRR init time to include.",
    )
    parser.add_argument(
        "--end-init-utc",
        type=str,
        default=DEFAULT_END_INIT_UTC,
        help="Last HRRR init time to include.",
    )
    parser.add_argument(
        "--lead-hours",
        type=str,
        default=",".join(str(hour) for hour in DEFAULT_LEAD_HOURS),
        help="Comma-separated forecast lead hours to keep.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing output store.",
    )
    return parser.parse_args()


def _parse_lead_hours(raw: str) -> list[int]:
    lead_hours = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not lead_hours:
        msg = "At least one lead hour is required."
        raise ValueError(msg)
    return lead_hours


def _to_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _completed_runs_log_path(output_path: Path) -> Path:
    return output_path.with_suffix(".init_times.txt")


def _missing_runs_log_path(output_path: Path) -> Path:
    return output_path.with_suffix(".missing_runs.txt")


def _load_logged_init_times(log_path: Path) -> list[pd.Timestamp]:
    if not log_path.exists():
        return []
    values: list[pd.Timestamp] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            values.append(_to_utc_timestamp(line))
    return values


def _write_logged_init_times(log_path: Path, timestamps: list[pd.Timestamp]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{timestamp.isoformat()}\n" for timestamp in timestamps)
    log_path.write_text(text, encoding="utf-8")


def _append_logged_init_time(log_path: Path, timestamp: pd.Timestamp) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp.isoformat()}\n")


def _close_archive_loader(loader: HRRRPatchLoader) -> None:
    archive_s3 = getattr(loader, "_archive_s3", None)
    if archive_s3 is None:
        return

    archive_loop = getattr(archive_s3, "loop", None)
    http_session = getattr(getattr(archive_s3, "_s3", None), "_endpoint", None)
    http_session = getattr(http_session, "http_session", None)
    session_map = getattr(http_session, "_sessions", {})
    if archive_loop is not None and session_map:
        from fsspec.asyn import sync

        for session in session_map.values():
            if not session.closed:
                sync(archive_loop, session.close, timeout=5)
    connector = getattr(http_session, "_connector", None)
    if connector is not None and hasattr(connector, "_close"):
        connector._close()


def _is_retryable_archive_error(exc: BaseException) -> bool:
    retryable_error_types = (
        ConnectionClosedError,
        ConnectTimeoutError,
        EndpointConnectionError,
        ReadTimeoutError,
        TimeoutError,
    )
    if isinstance(exc, retryable_error_types):
        return True
    cause = exc.__cause__
    if cause is not None and cause is not exc:
        return _is_retryable_archive_error(cause)
    context = exc.__context__
    if context is not None and context is not exc:
        return _is_retryable_archive_error(context)
    return False


def _build_run_dataset(
    *,
    loader: HRRRPatchLoader,
    init_time_utc: pd.Timestamp,
    lead_hours: list[int],
) -> xr.Dataset:
    window = loader._get_archive_window()
    run_data = loader._get_archive_run_data(init_time_utc)
    lead_indices = [lead_hour - 1 for lead_hour in lead_hours]
    max_required_index = max(lead_indices)

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for variable_name in loader.config.dynamic_variables:
        variable_block = run_data[variable_name]
        if variable_block.shape[0] <= max_required_index:
            msg = (
                f"Run {init_time_utc.isoformat()} only has {variable_block.shape[0]} forecast steps; "
                f"cannot satisfy requested lead hour {max(lead_hours)}."
            )
            raise ValueError(msg)
        data_vars[variable_name] = (
            ("time", "step", "y", "x"),
            variable_block[lead_indices][None, ...],
        )

    lat_grid = window["lat_grid"]
    lon_grid = window["lon_grid"]
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.asarray([init_time_utc.tz_convert(None).to_datetime64()]),
            "step": np.asarray(lead_hours, dtype=np.int16),
            "y": np.arange(lat_grid.shape[0], dtype=np.int32),
            "x": np.arange(lat_grid.shape[1], dtype=np.int32),
            "lat": (("y", "x"), lat_grid),
            "lon": (("y", "x"), lon_grid),
        },
        attrs={
            "source": "HRRR archive subset from s3://hrrrzarr",
            "region": "Waterloo",
            "start_init_utc": init_time_utc.isoformat(),
        },
    )
    return ds


def download_archive_subset_to_zarr(
    *,
    output_path: Path,
    start_init_utc: pd.Timestamp,
    end_init_utc: pd.Timestamp,
    lead_hours: list[int],
    overwrite: bool,
) -> None:
    completed_log_path = _completed_runs_log_path(output_path)
    missing_log_path = _missing_runs_log_path(output_path)
    if output_path.exists():
        if not overwrite:
            pass
        if output_path.is_dir():
            import shutil

            if overwrite:
                shutil.rmtree(output_path)
        else:
            if overwrite:
                output_path.unlink()
    if overwrite:
        for log_path in (completed_log_path, missing_log_path):
            if log_path.exists():
                log_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    loader = HRRRPatchLoader("s3://hrrrzarr", config=WaterlooPatchConfig())
    try:
        init_times = pd.date_range(start=start_init_utc, end=end_init_utc, freq="1h", tz="UTC")
        total_runs = len(init_times)
        if total_runs == 0:
            msg = "No init times selected for download."
            raise ValueError(msg)

        completed_init_times = _load_logged_init_times(completed_log_path)
        if output_path.exists() and not completed_init_times:
            existing_ds = xr.open_zarr(output_path, consolidated=False, decode_times=False)
            try:
                existing_count = int(existing_ds.sizes.get("time", 0))
            finally:
                existing_ds.close()
            completed_init_times = [
                _to_utc_timestamp(init_times[index])
                for index in range(min(existing_count, total_runs))
            ]
            _write_logged_init_times(completed_log_path, completed_init_times)
            print(
                f"Recovered {len(completed_init_times)} completed runs from existing subset metadata",
                flush=True,
            )

        missing_init_times = _load_logged_init_times(missing_log_path)
        completed_init_time_set = {timestamp.isoformat(): timestamp for timestamp in completed_init_times}
        missing_init_time_set = {timestamp.isoformat(): timestamp for timestamp in missing_init_times}
        remaining_init_times = [
            _to_utc_timestamp(init_time)
            for init_time in init_times
            if _to_utc_timestamp(init_time).isoformat() not in completed_init_time_set
            and _to_utc_timestamp(init_time).isoformat() not in missing_init_time_set
        ]

        print(
            (
                f"Downloading {len(remaining_init_times)} remaining HRRR runs to {output_path} "
                f"for lead_hours={lead_hours[0]}-{lead_hours[-1]} "
                f"(completed={len(completed_init_times)} skipped={len(missing_init_times)} total={total_runs})"
            ),
            flush=True,
        )

        encoding = None
        wrote_new_store = output_path.exists()
        downloaded_this_run = 0
        skipped_this_run = 0
        for index, init_time_utc in enumerate(remaining_init_times, start=1):
            try:
                for attempt in range(1, DEFAULT_ARCHIVE_RETRY_ATTEMPTS + 1):
                    try:
                        ds = _build_run_dataset(
                            loader=loader,
                            init_time_utc=init_time_utc,
                            lead_hours=lead_hours,
                        )
                        break
                    except Exception as exc:
                        if not _is_retryable_archive_error(exc) or attempt == DEFAULT_ARCHIVE_RETRY_ATTEMPTS:
                            raise
                        sleep_seconds = min(60, 5 * (2 ** (attempt - 1)))
                        print(
                            (
                                f"Retrying HRRR run {init_time_utc.isoformat()} "
                                f"attempt={attempt}/{DEFAULT_ARCHIVE_RETRY_ATTEMPTS} "
                                f"after {type(exc).__name__}: {exc} "
                                f"sleep_s={sleep_seconds}"
                            ),
                            flush=True,
                        )
                        _close_archive_loader(loader)
                        loader = HRRRPatchLoader("s3://hrrrzarr", config=WaterlooPatchConfig())
                        time.sleep(sleep_seconds)
            except GroupNotFoundError:
                skipped_this_run += 1
                if init_time_utc.isoformat() not in missing_init_time_set:
                    missing_init_times.append(init_time_utc)
                    missing_init_time_set[init_time_utc.isoformat()] = init_time_utc
                    _append_logged_init_time(missing_log_path, init_time_utc)
                print(
                    (
                        f"Skipping missing HRRR run {init_time_utc.isoformat()} "
                        f"remaining_index={index}/{len(remaining_init_times)}"
                    ),
                    flush=True,
                )
                continue

            if encoding is None:
                time_chunk = 1
                step_chunk = len(lead_hours)
                y_chunk = ds.sizes["y"]
                x_chunk = ds.sizes["x"]
                encoding = {
                    variable_name: {
                        "chunks": (time_chunk, step_chunk, y_chunk, x_chunk),
                    }
                    for variable_name in ds.data_vars
                }
                encoding["time"] = {
                    "units": "hours since 1970-01-01 00:00:00",
                    "dtype": "int64",
                }
            if not wrote_new_store:
                ds.to_zarr(
                    output_path,
                    mode="w",
                    consolidated=False,
                    encoding=encoding,
                    zarr_format=2,
                )
                wrote_new_store = True
            else:
                ds.to_zarr(output_path, mode="a", append_dim="time")

            downloaded_this_run += 1
            completed_init_times.append(init_time_utc)
            completed_init_time_set[init_time_utc.isoformat()] = init_time_utc
            _append_logged_init_time(completed_log_path, init_time_utc)

            if downloaded_this_run == 1 or downloaded_this_run % 24 == 0 or index == len(remaining_init_times):
                print(
                    (
                        f"Downloaded run {downloaded_this_run}/{len(remaining_init_times)} "
                        f"init={init_time_utc.isoformat()} "
                        f"window_shape={ds.sizes['y']}x{ds.sizes['x']} "
                        f"skipped_this_run={skipped_this_run}"
                    ),
                    flush=True,
                )

        if output_path.exists():
            zarr.consolidate_metadata(str(output_path))
        print(
            (
                f"Finished writing {output_path} "
                f"(completed={len(completed_init_times)} skipped={len(missing_init_times)} total={total_runs})"
            ),
            flush=True,
        )
    finally:
        _close_archive_loader(loader)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    lead_hours = _parse_lead_hours(args.lead_hours)
    download_archive_subset_to_zarr(
        output_path=output_path,
        start_init_utc=_to_utc_timestamp(args.start_init_utc),
        end_init_utc=_to_utc_timestamp(args.end_init_utc),
        lead_hours=lead_hours,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
