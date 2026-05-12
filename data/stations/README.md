# Station Sources And QC

This document records the station choices and hourly alignment rules for the
Waterloo HRRR post-processing v1 experiment.

## V1 stations

Anchor station:

- `uwaterloo_eds`
  - Source: University of Waterloo Eric D. Soulis weather station
  - Native cadence: 15 minutes
  - URL pattern confirmed for 2026:
    - `https://www.civil.uwaterloo.ca/weather/download/Hobo%5F15minutedata%5F2026.csv`

Nearby hourly auxiliary stations:

- `weatherstats_kw`
  - Source: `kitchenerwaterloo.weatherstats.ca`
  - Native cadence: hourly
- `weatherstats_guelph`
  - Source: `guelph.weatherstats.ca`
  - Native cadence: hourly

Audit notes:

- The nearby-stations list shared earlier clearly surfaced **Elora** as an
  active nearby hourly climate station.
- For implementation, the programmatic hourly exports that were confirmed to
  work cleanly for Jan-Feb 2026 were:
  - `kitchenerwaterloo.weatherstats.ca`
  - `guelph.weatherstats.ca`
- Historical or daily-only nearby stations from the Waterloo nearby list are
  not used in v1 because the task is hourly HRRR post-processing.

## Time standard

All station timestamps are converted to **UTC** before they are matched with
HRRR valid times.

Local source timezone:

- UW station: `America/Toronto`
- Weatherstats stations: `America/Toronto`

## Temperature QC rules

Apply the following source-agnostic checks before building hourly targets:

- Parse numeric temperature values into degrees Celsius.
- Treat sentinel missing values such as `-9999.99` as missing.
- Drop temperatures outside `[-60, 50]` C.
- Drop duplicated timestamps after UTC conversion, keeping the last value.

## UW 15-minute to hourly collapse

The UW source is 15-minute data, but v1 trains against hourly HRRR valid times.

Hourly collapse rule for each UTC hour:

1. If an exact top-of-hour observation exists and passes QC, use it.
2. Otherwise, if at least 2 finite 15-minute observations exist within that
   hour, use their mean temperature.
3. Otherwise, drop that hour for UW supervision.

This keeps the default target as close as possible to an instantaneous valid
 time while still recovering hours where the exact top-of-hour record is absent.

## Weatherstats hourly handling

Weatherstats exports are already hourly. For these sources:

- parse `date_time_local`
- localize to `America/Toronto`
- convert to UTC
- keep the hourly record as-is after basic QC

## HRRR alignment

For each training/evaluation sample:

- `valid_time_utc` is the station observation time
- `lead_hour` is one of the selected short HRRR leads
- `init_time_utc = valid_time_utc - lead_hour`

Only keep a sample if:

- at least one selected station has a valid temperature at `valid_time_utc`
- the corresponding HRRR initialization/lead pair can be loaded

## Planned v1 split

- train window: `2026-01-01 00:00 UTC` through late January 2026
- validation window: final days of January 2026
- test window: February 2026

