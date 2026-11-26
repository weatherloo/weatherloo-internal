"""Frozen chronology and benchmark station contract; no data dependencies."""

from datetime import datetime, timedelta, timezone


DATASET = "waterloo_2018_2025_v2"
TRAIN_YEARS = tuple(range(2018,2024))
STATIONS = (
    {"id":"eric_d_soulis","lat":43.4668,"lon":-80.5164,"eccc":None},
    {"id":"kitchener_waterloo","lat":43.4608333333,"lon":-80.3786111111,"eccc":48569},
    {"id":"elora_rcs","lat":43.65,"lon":-80.4166666667,"eccc":41983},
    {"id":"guelph_turfgrass","lat":43.5418725,"lon":-80.2220811111,"eccc":55299},
    {"id":"cyyz","lat":43.6777,"lon":-79.6248,"eccc":51459},
)
UTC = timezone.utc
GOES19_START = datetime(2025,4,7,15,10,tzinfo=UTC)


def split_for_year(year):
    return "train" if year in TRAIN_YEARS else {2024:"val",2025:"test"}[year]


def month_records(key):
    year,month = map(int,key.split("-"))
    split = split_for_year(year)
    current = datetime(year,month,1,tzinfo=UTC)
    end = datetime(year+int(month==12),month%12+1,1,tzinfo=UTC)
    boundary = {"train":datetime(2024,1,1,tzinfo=UTC),"val":datetime(2025,1,1,tzinfo=UTC)}.get(split)
    records = []
    while current<end:
        if boundary is None or current+timedelta(hours=48)<boundary:
            records.append((split,current))
        current += timedelta(hours=6)
    return records


def plan():
    months = [f"{year}-{month:02d}" for year in range(2018,2026) for month in range(1,13)]
    counts = {split:sum(len(month_records(key)) for key in months if split_for_year(int(key[:4]))==split)
              for split in ("train","val","test")}
    return {"dataset":DATASET,"months":months,"planned_initializations":counts,
            "train_years":list(TRAIN_YEARS),"validation_year":2024,"test_year":2025,
            "init_hours_utc":[0,6,12,18],"lead_hours":list(range(49)),
            "primary_station":"eric_d_soulis","secondary_station":"cyyz",
            "stations":STATIONS,"training_station_indices":list(range(len(STATIONS))),
            "selection_station_indices":[0],"excluded_validation_station_indices":[],
            "benchmark_leads_hours":list(range(6,73,6)),"maximum_forecast_lead":48,
            "split_guard":"training/validation targets must precede the next split initialization",
            "input_contract":"ERA5 at initialization only, 4x avgpool; two GOES-East frames scanned/produced at least 15 minutes before initialization; future fields/stations are labels only",
            "satellite_availability":"Historical S3 replicas were backfilled after observation time. Filename production time plus a 15-minute margin is an availability proxy; original dissemination latency is not verified. S3 LastModified is retained as copy provenance.",
            "initialization_regime":"retrospective ERA5 initialization; operational NWP initialization is a separate comparison"}
