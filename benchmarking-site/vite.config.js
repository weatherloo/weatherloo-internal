import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

const root = path.dirname(fileURLToPath(import.meta.url));
const dataRoot = path.join(root, "data");
const apiTarget = process.env.BENCHMARK_API_URL ?? "http://127.0.0.1:5174";

export default defineConfig({
  server: {
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
    },
  },
  preview: {
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
    },
  },
  plugins: [
    react(),
    viteStaticCopy({
      // Ship only what the browser fetches. Copying all of data/ put ~440 MB in
      // dist -- 271 MB of it raw CSVs and 13 MB of npz that no client code can
      // read -- which blows past static-host deployment limits.
      targets: [
        { src: "data/aggregates", dest: "data" },
        { src: "data/observations/stations.json", dest: "data/observations" },
        {
          src: "data/observations/cyyz/observations_6h_*.json",
          dest: "data/observations/cyyz",
        },
        {
          src: "data/observations/eric_d_soulis/observations_6h_*.json",
          dest: "data/observations/eric_d_soulis",
        },
      ],
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
