# MSLP interactive globe (take-home)

This folder builds an **interactive 3D-style globe** (Plotly orthographic projection) of **mean sea level pressure** from [ARCO ERA5 on GCS](https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr).

- **Variable in Zarr:** `mean_sea_level_pressure` (ERA5 short name **msl**; same field as “MSLP”).
- **Window:** UTC calendar day **2021-05-01** (00, 06, 12, 18 UTC) — **24 hours** at the dataset’s native **6-hourly** cadence (four snapshots).

## How to run

```bash
cd test_take_home/cindy
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python mslp_globe.py
```

Open the generated **`mslp_globe_may1_2021.html`** in a browser (the file is gitignored at ~10 MB; regenerate with the command above). Use the **time slider** and **Play / Pause** to step through the day; drag the globe to rotate.

Optional: sharper / larger file — `python mslp_globe.py --stride 2`; coarser / smaller — `--stride 6`.

**Note:** Reads use **anonymous** access to the public bucket (`token='anon'` via `gcsfs`); no Google Cloud login is required.
