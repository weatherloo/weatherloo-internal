# Weatherloo forecast app

Morning-report forecast app for Waterloo (Eric D. Soulis station). Expo / React Native + TypeScript; the same code ships as a static web build via react-native-web. The default model is the HRRR station-correction CNN (`models/hrrr_cnn/`).

> **Data note:** v1 runs on **placeholder fixtures**, not live forecasts. They are shaped like the CNN's outputs (see below) and follow the forecast record contract, but the numbers are synthetic. The UI does not say so; this README is where it is recorded.

## Run

```bash
cd forecast-app
npm install
npm run web        # dev server, web (http://localhost:8081)
npm start          # Expo dev server; press i / a for iOS / Android
npm run typecheck
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
src/app/              routes (Expo Router): _layout.tsx (providers, fonts), index.tsx (report)
src/features/report/  the morning report screen
src/components/       Icon, Type, Pill, HourlyChart, DayRow, SunArc
src/data/             contract.ts, registry.ts, stations.ts, source.ts, ForecastProvider.tsx,
                      report.ts (view model), narrative.ts (copy), sun.ts, time.ts, fixtures/
src/theme/tokens.ts   colors, fonts, spacing
scripts/              generate-fixtures.mjs
```

## What the fixtures contain

Station `eric_d_soulis`, init `2026-11-27T06:00:00Z` (HRRR 06 UTC run, leads out to 48 h), model `hrrr_cnn`:

| variable | unit | leads |
|---|---|---|
| `t2m` | degC | hourly 1-48 |
| `wind_speed` | m/s | hourly 1-48 |
| `pop1` (chance of >= 0.2 mm in the hour) | fraction | hourly 1-48 |
| `pop6` (same, 6 h block) | fraction | 6, 12, ... 48 |
| `amt6` (expected amount, 6 h block) | mm | 6, 12, ... 48 |

The report's "now" is fixed to `2026-11-27T12:00:00Z` for fixtures (`FIXTURE_NOW`). Display is in `America/Toronto`, °C and km/h; the wire format stays UTC/SI. Feels-like is Environment Canada wind chill, derived in the app. Sunrise/sunset are computed from the station's coordinates. Humidity, pressure and dew point are omitted because the CNN doesn't forecast them.

**Contract gap:** the v1 schema's `unit` enum only has `degC` and `m/s`. Precipitation needs `fraction` and `mm`, so `src/data/contract.ts` extends the type. The schema in `weatherloo-forecast/contract/` needs a matching bump before live precipitation can be published.

## Swapping fixtures for live data

Only `src/data/source.ts` knows where data comes from. Implement `ForecastSource` against the read API (`GET /forecast`, see `weatherloo-forecast/docs/DESIGN.md` §7), return `ForecastRecord[]` for a `(modelId, stationId)`, make `now()` return `new Date()`, and point `forecastSource` at it. The UI reads only from `useForecast()`.

## Where to add things

- **A model** (riza): add an entry to `MODELS` in `src/data/registry.ts` and make its records available (fixture JSON listed in `src/data/fixtures/index.ts`, or the live source). The active model is `activeModelId` / `setActiveModelId` from `useForecast()`; a switcher only calls the setter. A compare view can call `forecastSource.load(id, stationId)` for several models and reuse `buildReport`. The benchmarking data is in `benchmarking-site/data/aggregates/`.
- **A screen** (coolcucumber, AutoML globe): add a file in `src/app/` (e.g. `globe.tsx`) with its code under `src/features/globe/`, and link to it with `expo-router`'s `Link`. Providers in `_layout.tsx` are shared. Native-only libraries need a development build; keep web fallbacks in mind.
- **A variable:** add a record for it, read it with `seriesOf` in `src/data/report.ts`, and surface it in `FinePrint.tsx` or the narrative.
