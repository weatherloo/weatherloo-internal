# Weatherloo forecast app

iOS-Weather-style forecast app for Waterloo (Eric D. Soulis station). Expo / React Native + TypeScript; the same code ships as a static web build via react-native-web. The default model is the HRRR station-correction CNN (`models/hrrr_cnn/`).

> **Data note:** forecast values are **synthetic demo fixtures**, not live forecasts. Model selection and comparison are interactive, but the forecast trajectories are illustrative only. Historical RMSE values on the comparison screen are drawn from the benchmarking site's 2025 aggregates and are not the demo trajectories' scores.

## Run

```bash
cd forecast-app
npm install
npm run web        # dev server, web (http://localhost:8081)
npm start          # Expo dev server; press i / a for iOS / Android
npm run typecheck
npm run test:data  # validate fixture contracts and benchmark provenance
npm run test:demo  # training-demo location checks (HRRR grid, tiles, station history)
```

## Build for web

```bash
npm run build:web  # static SPA in dist/
npm run serve:web  # preview dist/ at http://localhost:8081
```

## Deploy (Vercel)

Create a **separate** Vercel project for this app (the benchmarking site has its own and is untouched):

```bash
cd forecast-app
vercel          # first run: link/create a project; root directory = forecast-app
vercel --prod
```

`vercel.json` sets the build (`npx expo export -p web`), output (`dist`) and a rewrite to `/` for the single-page build. If you connect the GitHub repo instead of using the CLI, set **Root Directory** to `forecast-app`.

## Layout

```
src/app/              routes (Expo Router): _layout.tsx (providers, fonts), index.tsx (report),
                      compare.tsx, demo.tsx (training demo)
src/features/report/  the forecast screen (ReportScreen.tsx) and detail tiles (Tiles.tsx)
src/components/       Card, Icon, SkyBackground, HourlyStrip, DayRow
src/data/             contract.ts, registry.ts, stations.ts, source.ts, ForecastProvider.tsx,
                      report.ts (view model), benchmark-summary.json, fixtures/
src/features/compare/  side-by-side synthetic forecast and historical skill comparison
src/features/demo/    training demo: globe, location checks, walkthrough, plan/approve/train panel
src/data/demo/        prepared hrrr_cnn runs (locations.json) and the IEM ASOS station list (stations.json)
src/theme/tokens.ts   night/day themes, fonts
scripts/              generate-fixtures.mjs
```

## What the fixtures contain

Station `eric_d_soulis`, init `2026-11-27T06:00:00Z` (demo issue time, leads out to 48 h). HRRR CNN, ECMWF AIFS, GraphCast, HRRR and persistence forecast records are synthetic variations for the interactive demo; their model IDs identify the scenario being shown, not the source of the values.

| variable | unit | leads |
|---|---|---|
| `t2m` | degC | hourly 1-48 |
| `wind_speed` | m/s | hourly 1-48 |
| `pop1` (chance of >= 0.2 mm in the hour) | fraction | hourly 1-48 |
| `pop6` (same, 6 h block) | fraction | 6, 12, ... 48 |
| `amt6` (expected amount, 6 h block) | mm | 6, 12, ... 48 |

The report's "now" is fixed to `2026-11-27T12:00:00Z` for fixtures (`FIXTURE_NOW`). Display is in `America/Toronto`, °C and km/h; the wire format stays UTC/SI. Feels-like is Environment Canada wind chill, derived in the app. Sunrise/sunset and moon phase are computed from the station's coordinates and date. Humidity, pressure, UV, visibility and wind direction are omitted because the CNN doesn't forecast them. The sky background switches between night and day from the report's sunrise/sunset.

The comparison screen plots synthetic temperature and wind trajectories for every app model and shows the synthetic value at f024. These are demo fixtures, not live model output. Historical RMSE is shown separately for exact benchmark methods at Eric D. Soulis in 2025 and lead 24 h. The site aggregate contains one signed forecast error per pair; the app calculates `sqrt(mean(error²))` across valid pairs and shows the count because observations can be missing. RMSE is the typical size of forecast errors in the variable's units; lower is better. These scores are historical results, not scores for the demo fixtures.

HRRR CNN remains demo-only and unscored: no local f024 CNN prediction/checkpoint is available, so the app does not show pooled CNN scores or claim an improvement over HRRR. Regenerate the compact summary from the existing benchmark aggregates with `npm run benchmark-summary`.

**Contract gap:** the v1 schema's `unit` enum only has `degC` and `m/s`. Precipitation needs `fraction` and `mm`, so `src/data/contract.ts` extends the type. The schema in `weatherloo-forecast/contract/` needs a matching bump before live precipitation can be published.

## Training demo (`/demo`)

Opened from "Training demo ›" on the forecast screen. Tap a spot on the globe; the app checks it the way the hrrr_cnn pipeline would (`src/features/demo/checks.ts`: inside HRRR, an ASOS station within 125 km with records back to Dec 2020, and the 17x17 patch inside one stored HRRR tile).

- **Prepared runs** (Waterloo, Pearson, SFO) replay the pipeline with real 2025 scores from `models/hrrr_cnn/README.md`, stored in `src/data/demo/locations.json`.
- **Any other valid spot** can be planned and trained through the training API (`benchmarking-site/server/train_api.py`, served by `npz_api.py` on port 5174). Planning only checks the setup on Modal; training starts only after the user switches on the go-ahead and presses Start, and the API also requires the region name as approval. Set `EXPO_PUBLIC_TRAIN_API_URL` when the API is not at `http://localhost:5174`, and add the app's origin to `BENCHMARK_API_CORS` on the API.
- `src/data/demo/stations.json` is rebuilt with `python pipelines/download/build_demo_stations.py` (IEM metadata, about 70 small requests).

## HRRR forecast map

The map card on the forecast screen shows **real** NOAA HRRR output, unlike the station fixtures: 2 m temperature, 10 m wind and precipitation rate on HRRR's native 3 km grid, in an 80 × 80-cell (240 km) square around Eric D. Soulis, every 3 hours to 48 h. It is raw HRRR; the CNN correction only applies at stations. The data is `src/data/hrrr-map.json`, quantized to uint8 (about 410 KB). Regenerate it from the repository root:

```bash
python3 -m venv .venv && .venv/bin/pip install "dynamical-catalog>=0.8.0,<1" pyproj==3.7.0
.venv/bin/python pipelines/download/export_hrrr_map.py --init 2026-11-27T06:00 --out forecast-app/src/data/hrrr-map.json
```

It reads one to four storage chunks per field from the dynamical.org catalog (tens of MB) and takes a few seconds.

## Swapping fixtures for live data

Only `src/data/source.ts` knows where data comes from. Implement `ForecastSource` against the read API (`GET /forecast`, see `weatherloo-forecast/docs/DESIGN.md` §7), return `ForecastRecord[]` for a `(modelId, stationId)`, make `now()` return `new Date()`, and point `forecastSource` at it. The UI reads only from `useForecast()`.

## Where to add things

- **A model:** add an entry to `MODELS` in `src/data/registry.ts` and make its records available (fixture JSON listed in `src/data/fixtures/index.ts`, or the live source). The active model is `activeModelId` / `setActiveModelId` from `useForecast()`. To show a historical score, set `benchmarkMethodId` only when the benchmark method directly matches the forecast model, then regenerate the summary using `npm run benchmark-summary`.
- **A screen** (coolcucumber, AutoML globe): add a file in `src/app/` (e.g. `globe.tsx`) with its code under `src/features/globe/`, and link to it with `expo-router`'s `Link`. Providers in `_layout.tsx` are shared. Native-only libraries need a development build; keep web fallbacks in mind.
- **A variable:** add a record for it, read it with `seriesOf` in `src/data/report.ts`, and surface it as a tile in `src/features/report/Tiles.tsx`.

## Credits

Background photo: [University of Waterloo William G. Davis Computer Research Center Bridge](https://commons.wikimedia.org/wiki/File:University_of_Waterloo_William_G._Davis_Computer_Research_Center_Bridge.jpg) by Victor Vucicevich, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0). Resized and toned to a dark-blue duotone for `assets/images/campus-davis-centre.jpg`.
