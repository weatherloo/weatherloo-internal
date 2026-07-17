import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

const root = path.dirname(fileURLToPath(import.meta.url));
const dataRoot = path.join(root, "data");

// Production data bundle: methods with a static aggregate.json (built by
// scripts/build_static_aggregates.py) ship only that file — the dashboard
// never needs their per-init JSON or NPZ. Methods without one ship all their
// JSON (index/sample/per-init) for the client-side fallback. observations/
// is ground truth for compute scripts only and is never fetched by the app.
function dataCopyTargets() {
  return fs
    .readdirSync(dataRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name !== "observations")
    .map((entry) =>
      fs.existsSync(path.join(dataRoot, entry.name, "aggregate.json"))
        ? { src: `data/${entry.name}/aggregate.json`, dest: `data/${entry.name}` }
        : { src: `data/${entry.name}/*.json`, dest: `data/${entry.name}` },
    );
}

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: dataCopyTargets(),
    }),
    {
      name: "serve-benchmark-data",
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (!req.url?.startsWith("/data/")) {
            next();
            return;
          }
          const rel = decodeURIComponent(
            req.url.slice("/data/".length).split("?")[0],
          );
          const filePath = path.normalize(path.join(dataRoot, rel));
          if (!filePath.startsWith(dataRoot)) {
            res.statusCode = 403;
            res.end();
            return;
          }
          fs.readFile(filePath, (err, body) => {
            if (err) {
              next();
              return;
            }
            if (filePath.endsWith(".json")) {
              res.setHeader("Content-Type", "application/json");
            }
            res.end(body);
          });
        });
      },
    },
  ],
});
