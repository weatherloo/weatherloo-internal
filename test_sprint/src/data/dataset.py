from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.hrrr import HRRRPatchLoader


UTC = "UTC"


DEFAULT_V1_LEAD_HOURS: tuple[int, ...] = tuple(range(1, 19))
DEFAULT_V1_DYNAMIC_VARIABLES: tuple[str, ...] = ("t2m", "d2m", "u10", "v10", "sp")


@dataclass(frozen=True)
class V1SplitConfig:
    train_start_utc: str = "2026-01-01 00:00:00+00:00"
    train_end_utc: str = "2026-01-26 23:00:00+00:00"
    validation_start_utc: str = "2026-01-27 00:00:00+00:00"
    validation_end_utc: str = "2026-01-31 23:00:00+00:00"
    test_start_utc: str = "2026-02-01 00:00:00+00:00"
    test_end_utc: str = "2026-02-28 23:00:00+00:00"


@dataclass(frozen=True)
class SampleIndexConfig:
    lead_hours: tuple[int, ...] = DEFAULT_V1_LEAD_HOURS
    min_station_count: int = 1


def _to_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def build_v1_time_splits() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    cfg = V1SplitConfig()
    return {
        "train": (
            pd.Timestamp(cfg.train_start_utc, tz=UTC),
            pd.Timestamp(cfg.train_end_utc, tz=UTC),
        ),
        "validation": (
            pd.Timestamp(cfg.validation_start_utc, tz=UTC),
            pd.Timestamp(cfg.validation_end_utc, tz=UTC),
        ),
        "test": (
            pd.Timestamp(cfg.test_start_utc, tz=UTC),
            pd.Timestamp(cfg.test_end_utc, tz=UTC),
        ),
    }


def build_sample_index(
    station_panel: pd.DataFrame,
    *,
    split_start_utc: str | pd.Timestamp,
    split_end_utc: str | pd.Timestamp,
    lead_hours: Iterable[int] = DEFAULT_V1_LEAD_HOURS,
    station_ids: Iterable[str] | None = None,
    min_station_count: int = 1,
) -> pd.DataFrame:
    df = station_panel.copy()
    start_utc = _to_utc_timestamp(split_start_utc)
    end_utc = _to_utc_timestamp(split_end_utc)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df[(df["timestamp_utc"] >= start_utc) & (df["timestamp_utc"] <= end_utc)]
    if station_ids is not None:
        df = df[df["station_id"].isin(list(station_ids))]

    station_counts = (
        df.groupby("timestamp_utc")["station_id"]
        .nunique()
        .rename("available_station_count")
        .reset_index()
    )
    station_counts = station_counts[station_counts["available_station_count"] >= min_station_count]

    rows: list[dict[str, object]] = []
    for valid_time in station_counts["timestamp_utc"]:
        for lead_hour in lead_hours:
            rows.append(
                {
                    "valid_time_utc": _to_utc_timestamp(valid_time),
                    "lead_hour": int(lead_hour),
                    "init_time_utc": _to_utc_timestamp(valid_time) - pd.Timedelta(hours=int(lead_hour)),
                }
            )
    return pd.DataFrame(rows).sort_values(["valid_time_utc", "lead_hour"]).reset_index(drop=True)


class HRRRTemperatureDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        hrrr_loader: HRRRPatchLoader,
        station_panel: pd.DataFrame,
        sample_index: pd.DataFrame,
        station_ids: Iterable[str] | None = None,
    ) -> None:
        self.hrrr_loader = hrrr_loader
        self.station_panel = station_panel.copy()
        self.station_panel["timestamp_utc"] = pd.to_datetime(self.station_panel["timestamp_utc"], utc=True)
        self.sample_index = sample_index.reset_index(drop=True).copy()

        if station_ids is None:
            station_ids = sorted(self.station_panel["station_id"].unique().tolist())
        self.station_ids = list(station_ids)

        metadata = (
            self.station_panel.sort_values("station_id")
            .drop_duplicates(subset=["station_id"])
            .set_index("station_id")
            .loc[self.station_ids]
        )
        self.station_latitudes = metadata["latitude"].to_numpy(dtype=np.float32)
        self.station_longitudes = metadata["longitude"].to_numpy(dtype=np.float32)
        self.station_lookup = {
            timestamp: group.set_index("station_id")["temperature_c"].to_dict()
            for timestamp, group in self.station_panel.groupby("timestamp_utc", sort=False)
        }

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.sample_index.iloc[index]
        patch = self.hrrr_loader.load_patch(
            init_time_utc=row["init_time_utc"],
            lead_hour=int(row["lead_hour"]),
        )
        station_values, station_mask = self._build_station_targets(_to_utc_timestamp(row["valid_time_utc"]))
        station_xy = self.hrrr_loader.station_coords_to_normalized_grid(
            station_latitudes=self.station_latitudes,
            station_longitudes=self.station_longitudes,
            patch_lat_grid=patch["lat_grid"],
            patch_lon_grid=patch["lon_grid"],
        )

        return {
            "dynamic": torch.from_numpy(patch["dynamic"]),
            "static": torch.from_numpy(patch["static"]),
            "baseline_t2m_c": torch.from_numpy(patch["baseline_t2m_c"][None, ...]),
            "lead_hour": torch.tensor(int(row["lead_hour"]), dtype=torch.long),
            "station_xy": torch.from_numpy(station_xy),
            "station_values_c": torch.from_numpy(station_values),
            "station_mask": torch.from_numpy(station_mask),
            "valid_time_utc": str(_to_utc_timestamp(row["valid_time_utc"])),
            "init_time_utc": str(_to_utc_timestamp(row["init_time_utc"])),
        }

    def _build_station_targets(self, valid_time_utc: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros(len(self.station_ids), dtype=np.float32)
        mask = np.zeros(len(self.station_ids), dtype=np.bool_)
        station_map = self.station_lookup.get(valid_time_utc, {})
        for idx, station_id in enumerate(self.station_ids):
            if station_id in station_map and pd.notna(station_map[station_id]):
                values[idx] = float(station_map[station_id])
                mask[idx] = True
        return values, mask


def build_v1_datasets(
    *,
    hrrr_loader: HRRRPatchLoader,
    station_panel: pd.DataFrame,
    station_ids: Iterable[str] | None = None,
    lead_hours: Iterable[int] = DEFAULT_V1_LEAD_HOURS,
    min_station_count: int = 1,
) -> dict[str, HRRRTemperatureDataset]:
    datasets: dict[str, HRRRTemperatureDataset] = {}
    for split_name, (start_utc, end_utc) in build_v1_time_splits().items():
        sample_index = build_sample_index(
            station_panel,
            split_start_utc=start_utc,
            split_end_utc=end_utc,
            lead_hours=lead_hours,
            station_ids=station_ids,
            min_station_count=min_station_count,
        )
        datasets[split_name] = HRRRTemperatureDataset(
            hrrr_loader=hrrr_loader,
            station_panel=station_panel,
            sample_index=sample_index,
            station_ids=station_ids,
        )
    return datasets

