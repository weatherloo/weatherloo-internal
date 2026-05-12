from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


LOCAL_TIMEZONE = "America/Toronto"
UTC = "UTC"
TEMPERATURE_MIN_C = -60.0
TEMPERATURE_MAX_C = 50.0


@dataclass(frozen=True)
class StationConfig:
    station_id: str
    station_name: str
    source: str
    latitude: float
    longitude: float
    elevation_m: float
    timezone: str = LOCAL_TIMEZONE
    csv_url: str | None = None
    weatherstats_host: str | None = None


DEFAULT_V1_STATIONS: tuple[StationConfig, ...] = (
    StationConfig(
        station_id="uwaterloo_eds",
        station_name="University of Waterloo Eric D. Soulis Station",
        source="uwaterloo_15min",
        latitude=43.4725,
        longitude=-80.5449,
        elevation_m=321.0,
        csv_url=(
            "https://www.civil.uwaterloo.ca/weather/download/"
            "Hobo%5F15minutedata%5F2026.csv"
        ),
    ),
    StationConfig(
        station_id="weatherstats_kw",
        station_name="Kitchener-Waterloo Weatherstats",
        source="weatherstats_hourly",
        latitude=43.59,
        longitude=-80.48,
        elevation_m=336.0,
        weatherstats_host="kitchenerwaterloo.weatherstats.ca",
    ),
    StationConfig(
        station_id="weatherstats_guelph",
        station_name="Guelph Weatherstats",
        source="weatherstats_hourly",
        latitude=43.55,
        longitude=-80.25,
        elevation_m=334.0,
        weatherstats_host="guelph.weatherstats.ca",
    ),
)


def _fetch_text(url: str, data: bytes | None = None) -> str:
    request = Request(url, data=data)
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", "ignore")


def _clip_temperature_range(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["temperature_c"].between(TEMPERATURE_MIN_C, TEMPERATURE_MAX_C)
    return df.loc[mask].copy()


def _finalize_station_frame(
    df: pd.DataFrame,
    *,
    station: StationConfig,
    method_column: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["station_id"] = station.station_id
    df["station_name"] = station.station_name
    df["source"] = station.source
    df["latitude"] = station.latitude
    df["longitude"] = station.longitude
    df["elevation_m"] = station.elevation_m
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.drop_duplicates(subset=["timestamp_utc"], keep="last")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    columns = [
        "timestamp_utc",
        "temperature_c",
        method_column,
        "station_id",
        "station_name",
        "source",
        "latitude",
        "longitude",
        "elevation_m",
    ]
    return df.loc[:, columns].rename(columns={method_column: "qc_method"})


def load_uw_station(csv_url: str) -> pd.DataFrame:
    """
    Load the UW 2026 15-minute CSV and collapse it to hourly targets.

    Hourly collapse rules:
    1. Use an exact top-of-hour finite temperature if present.
    2. Otherwise use the mean if at least 2 finite quarter-hour values exist.
    3. Otherwise drop the hour.
    """

    csv_text = _fetch_text(csv_url)
    reader = csv.reader(StringIO(csv_text), skipinitialspace=True)
    try:
        header = next(reader)
    except StopIteration:
        return pd.DataFrame(columns=["timestamp_utc", "temperature_c", "hourly_method"])

    cleaned_header = [column.strip() for column in header]
    normalized_rows: list[list[str]] = []
    for row in reader:
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) < len(cleaned_header):
            row = row + [""] * (len(cleaned_header) - len(row))
        elif len(row) > len(cleaned_header):
            # The UW export currently appends unlabeled trailing cells we do not use.
            row = row[: len(cleaned_header)]
        normalized_rows.append(row)

    raw = pd.DataFrame(normalized_rows, columns=cleaned_header)
    raw["temperature_c"] = pd.to_numeric(raw["Temperature"], errors="coerce")
    raw.loc[raw["temperature_c"] <= -9990.0, "temperature_c"] = pd.NA

    local_timestamp = pd.to_datetime(
        dict(
            year=raw["year"],
            month=raw["month"],
            day=raw["day"],
            hour=raw["hour"],
            minute=raw["minute"],
        ),
        errors="coerce",
    )
    raw["timestamp_local"] = (
        local_timestamp.dt.tz_localize(
            LOCAL_TIMEZONE,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    )
    raw["timestamp_utc"] = raw["timestamp_local"].dt.tz_convert(UTC)
    raw = raw.dropna(subset=["timestamp_utc", "temperature_c"]).copy()
    raw = _clip_temperature_range(raw)
    raw["hour_bucket_utc"] = raw["timestamp_utc"].dt.floor("h")

    collapsed_rows: list[dict[str, object]] = []
    for hour_bucket, group in raw.groupby("hour_bucket_utc", sort=True):
        group = group.sort_values("timestamp_utc")
        exact = group[group["timestamp_utc"].dt.minute == 0]
        if not exact.empty:
            record = exact.iloc[-1]
            collapsed_rows.append(
                {
                    "timestamp_utc": hour_bucket,
                    "temperature_c": float(record["temperature_c"]),
                    "hourly_method": "top_of_hour",
                }
            )
            continue

        finite = group["temperature_c"].dropna()
        if len(finite) >= 2:
            collapsed_rows.append(
                {
                    "timestamp_utc": hour_bucket,
                    "temperature_c": float(finite.mean()),
                    "hourly_method": "mean_15min_fallback",
                }
            )

    return pd.DataFrame(collapsed_rows)


def _build_weatherstats_download_url(host: str) -> str:
    return f"https://{host}/download.html"


def load_weatherstats_station(
    host: str,
    *,
    limit_rows: int = 3500,
) -> pd.DataFrame:
    payload = urlencode(
        {
            "formdata": "ok",
            "type": "hourly",
            "limit": str(limit_rows),
            "submit": "Download",
        }
    ).encode()
    csv_text = _fetch_text(_build_weatherstats_download_url(host), data=payload)
    frame = pd.read_csv(StringIO(csv_text))
    frame["temperature_c"] = pd.to_numeric(frame["temperature"], errors="coerce")
    frame["timestamp_local"] = pd.to_datetime(
        frame["date_time_local"].str.replace(r" (EST|EDT)$", "", regex=True),
        errors="coerce",
    )
    frame["timestamp_local"] = frame["timestamp_local"].dt.tz_localize(
        LOCAL_TIMEZONE,
        ambiguous="infer",
        nonexistent="shift_forward",
    )
    frame["timestamp_utc"] = frame["timestamp_local"].dt.tz_convert(UTC)
    frame = frame.dropna(subset=["timestamp_utc", "temperature_c"]).copy()
    frame = _clip_temperature_range(frame)
    frame["hourly_method"] = "native_hourly"
    return frame[["timestamp_utc", "temperature_c", "hourly_method"]].copy()


def load_station_dataframe(
    station: StationConfig,
    *,
    weatherstats_limit_rows: int = 3500,
) -> pd.DataFrame:
    if station.source == "uwaterloo_15min":
        if not station.csv_url:
            msg = f"Station {station.station_id} is missing a CSV URL."
            raise ValueError(msg)
        df = load_uw_station(station.csv_url)
    elif station.source == "weatherstats_hourly":
        if not station.weatherstats_host:
            msg = f"Station {station.station_id} is missing a Weatherstats host."
            raise ValueError(msg)
        df = load_weatherstats_station(
            station.weatherstats_host,
            limit_rows=weatherstats_limit_rows,
        )
    else:
        msg = f"Unsupported station source: {station.source}"
        raise ValueError(msg)

    return _finalize_station_frame(df, station=station, method_column="hourly_method")


def load_station_panel(
    stations: Iterable[StationConfig] = DEFAULT_V1_STATIONS,
    *,
    weatherstats_limit_rows: int = 3500,
) -> pd.DataFrame:
    frames = [
        load_station_dataframe(
            station,
            weatherstats_limit_rows=weatherstats_limit_rows,
        )
        for station in stations
    ]
    if not frames:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "temperature_c",
                "qc_method",
                "station_id",
                "station_name",
                "source",
                "latitude",
                "longitude",
                "elevation_m",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def subset_station_panel(
    station_panel: pd.DataFrame,
    *,
    start_utc: str | pd.Timestamp | None = None,
    end_utc: str | pd.Timestamp | None = None,
    station_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = station_panel.copy()
    if start_utc is not None:
        df = df[df["timestamp_utc"] >= pd.Timestamp(start_utc, tz=UTC)]
    if end_utc is not None:
        df = df[df["timestamp_utc"] <= pd.Timestamp(end_utc, tz=UTC)]
    if station_ids is not None:
        df = df[df["station_id"].isin(list(station_ids))]
    return df.sort_values(["timestamp_utc", "station_id"]).reset_index(drop=True)

