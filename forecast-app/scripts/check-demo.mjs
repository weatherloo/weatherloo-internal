// Self-check for the demo location checks. Run from forecast-app/: npm run test:demo
import assert from "node:assert/strict";
import { geoArea } from "d3-geo";
import { gridPosition, hrrrDomain, insideHrrr, patchInsideTile, runChecks } from "../src/features/demo/checks.ts";

// Grid corners from the NCEP HRRR definition land on (0, 0) and (1798, 1058).
const [i0, j0] = gridPosition(21.138123, -122.719528);
const [i1, j1] = gridPosition(47.842195, -60.917193);
assert.ok(Math.abs(i0) < 0.01 && Math.abs(j0) < 0.01, `lower-left ${i0}, ${j0}`);
assert.ok(Math.abs(i1 - 1798) < 1 && Math.abs(j1 - 1058) < 1, `upper-right ${i1}, ${j1}`);

assert.ok(insideHrrr(43.4668, -80.5164), "Soulis inside HRRR");
assert.ok(!insideHrrr(51.5, -0.12), "London outside HRRR");
assert.ok(!insideHrrr(60, -100), "northern Canada outside HRRR");

// These three sites have run through the real pipeline, so their patches fit a tile.
for (const [name, lat, lon] of [["Soulis", 43.4668, -80.5164], ["CYYZ", 43.6777, -79.6248], ["SFO", 37.619, -122.3749]]) {
  assert.ok(patchInsideTile(lat, lon), `${name} patch inside one tile`);
}

// d3-geo fills the small side only if the ring is clockwise: ~0.4 sr, not ~12.
const area = geoArea(hrrrDomain());
assert.ok(area > 0.2 && area < 1, `domain area ${area}`);

const stations = [{ id: "AAA", name: "Near", lat: 43.5, lon: -80.4, begin: "1990-01-01" }];
assert.equal(runChecks({ lat: 43.46, lon: -80.5 }, stations).ok, true);
const far = runChecks({ lat: 30, lon: -100 }, stations);
assert.deepEqual(far.steps.map((s) => s.status), ["pass", "fail", "skip", "skip"]);
const young = runChecks({ lat: 43.46, lon: -80.5 }, [{ ...stations[0], begin: "2025-06-01" }]);
assert.equal(young.steps[2].status, "fail");
// The pipeline needs records from the start of training (2020-12), not just 2025.
const late = runChecks({ lat: 43.46, lon: -80.5 }, [{ ...stations[0], begin: "2021-03-01" }]);
assert.equal(late.steps[2].status, "fail");
assert.equal(runChecks({ lat: 51.5, lon: -0.12 }, stations).steps[0].status, "fail");

console.log("demoChecks: all checks pass");
