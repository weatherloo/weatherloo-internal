# Deploying the benchmarking site (Vercel, static)

**There is no backend server and no NPZ API in production.** The site is a
fully static Vite build: the dashboard reads one precomputed
`data/<method_id>/aggregate.json` per method. `server/npz_api.py` is a legacy
dev tool only and is never deployed (excluded via `.vercelignore`).

## Build

From `benchmarking-site/`:

```bash
npm install
npm run build
```

`npm run build` runs two steps:

1. **`npm run build:data`** — `scripts/build_site_aggregates.py` (repo root)
   (via a wrapper that picks the repo `.venv` Python or any `python3` with
   numpy) reads each method's consolidated `<method_id>_<year>.npz` and
   writes `data/<method_id>/aggregate.json`: the mean across all inits per
   station/variable/metric/lead, plus per-(month × cycle) partial sums and
   counts. The partials let the dashboard reproduce every cycle/quarter/month
   filter exactly, client-side, with no API.
2. **`vite build`** — bundles the app into `dist/` and copies the data the
   dashboard needs: only `aggregate.json` for methods that have one; all JSON
   (index/sample/per-init) for methods without an NPZ (fallback path).
   `data/observations/`, `*.npz`, and per-init JSON of aggregated methods are
   **not** shipped. Total `dist/` is ~2.6 MB.

Commit the regenerated `aggregate.json` files — Vercel's build image has no
numpy, so its build reuses the committed files (the wrapper script detects
this and skips the aggregation step with a warning).

## Deploy

```bash
cd benchmarking-site
vercel --prod
```

`vercel.json` sets `buildCommand: npm run build` and `outputDirectory: dist`.
First deploy: accept the CLI prompts to create/link the project (root
directory = `benchmarking-site`).

## Refreshing after a new compute_benchmark.py run

After a method pipeline updates its per-init JSON + NPZ (e.g.
`.venv/bin/python data/gfs_interpolated/compute_benchmark.py …`):

```bash
cd benchmarking-site
npm run build          # regenerates data/<method_id>/aggregate.json + dist/
git add data/aggregates/ && git commit -m "chore: refresh static aggregates"
vercel --prod
```

Only rebuild one method's aggregate if you prefer:

```bash
python3 ../scripts/build_site_aggregates.py
```

## What works where

| Dashboard feature | Production (static) | Local dev (`npm run dev`) |
|---|---|---|
| Default view (all inits) | `aggregate.json` | same |
| Cycle filter (00/06/12/18Z) | recombined from `aggregate.json` partials — exact | same |
| Quarter / month presets | recombined from partials — exact | same |
| Custom **month-aligned** range | recombined from partials — exact | same |
| Custom non-month-aligned range | **no data** for aggregated methods (per-init JSON not shipped) | works (dev serves per-init JSON from `data/`) |
| Methods without NPZ (e.g. `hrdps_analysis`) | per-init/sample JSON fallback | same |

If free-form custom ranges ever need to work in production, ship the per-init
JSON by removing the `aggregate.json` short-circuit in
`vite.config.js:dataCopyTargets()` for that method (~12 MB + 1460 files per
method, and the fallback fetches them sequentially — slow).
