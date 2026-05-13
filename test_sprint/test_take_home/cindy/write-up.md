> Meta notes:
>
> - Completed this in 20 min 44 seconds half-procrastinating so time estimate should be reasonable
> - This is not an example of what a "good" submission should look like - this is very rough and unpolished



# Task 1

I chose mean sea level pressure.

- An ELI5-style explanation of what it represents physically

The pressure/how heavy a column of air feels, normalized at sea level.

- Whether it's a single-level variable (e.g. surface temperature) or one associated with pressure levels (e.g. wind at different altitudes in the atmosphere)

- It's a single-level variable

- Common abbreviation(s) for the variable

- MSLP, MSL

# Task 2

Implementation and how to run: see **`test_take_home/cindy/README.md`** (script `mslp_globe.py` writes `mslp_globe_may1_2021.html`).

Design decisions:
- Interactive globe (Plotly orthographic `Scattergeo`) so you can rotate and use a time slider for the four 6-hourly steps on 2021-05-01 UTC.
- ARCO Zarr field **`mean_sea_level_pressure`** (ERA5 short name **msl**), shown in **hPa**.

# Task 3
1. 6 hours
2. UTC
3. Grid size
4. File format, it's chunked and compressed