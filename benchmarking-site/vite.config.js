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
      targets: [{ src: "data", dest: "." }],
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
