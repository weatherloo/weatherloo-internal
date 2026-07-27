"""Verify make_sequences respects the lead-time observability cutoff."""
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import make_sequences  # noqa: E402


def mk_inits(n, start="2026-01-01T00:00:00", step_h=6):
    t0 = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return np.array(
        [(t0 + timedelta(hours=i * step_h)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n)]
    )


def feats(n):
    # feature 0 = bias; make it an identity marker so we can trace provenance
    f = np.zeros((n, 5), dtype=np.float32)
    f[:, 0] = np.arange(n)
    return f


fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fail += 1


# --- 1. lead=6h reproduces the naive next-step window (the one case that was right)
n, seq_len = 40, 4
inits, f = mk_inits(n), feats(n)
X, y, idx = make_sequences(f, seq_len, inits, lead_time=6)
old_X = np.array([f[i : i + seq_len] for i in range(n - seq_len)])
old_y = np.array([f[i + seq_len, 0] for i in range(n - seq_len)])
check("lead=6h matches previous behaviour", np.array_equal(X, old_X) and np.array_equal(y, old_y))

# --- 2. lead=48h puts 8 six-hourly steps between window end and target
X, y, idx = make_sequences(f, seq_len, inits, lead_time=48)
k = idx[0]
window_last_bias = X[0, -1, 0]
gap_steps = k - window_last_bias
check("lead=48h window ends 8 steps before target", gap_steps == 8, f"(gap={gap_steps})")

# --- 3. no window value may come from at/after the target
ok = all(X[r, -1, 0] <= idx[r] - 8 for r in range(len(X)))
check("no window sample at/after target-lead", ok)

# --- 4. every window sample's bias is observable by target init time
#         init[j] + lead <= init[k]
t = np.array(
    [datetime.strptime(s.rstrip("Z"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
     / 3600.0 for s in inits]
)
ok = all(t[int(X[r, -1, 0])] + 48 <= t[idx[r]] + 1e-9 for r in range(len(X)))
check("observability holds against real timestamps", ok)

# --- 5. uneven spacing (NaN-dropped series) uses time, not index offset
#        drop indices 10..17 so an index-based offset would silently reach too far
keep = np.array([i for i in range(n) if not (10 <= i < 18)])
inits_g, f_g = inits[keep], feats(n)[keep]
X, y, idx = make_sequences(f_g, seq_len, inits_g, lead_time=48)
tg = np.array(
    [datetime.strptime(s.rstrip("Z"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
     / 3600.0 for s in inits_g]
)
# X stores original bias markers; map back to position within the gapped series
pos = {v: p for p, v in enumerate(f_g[:, 0])}
ok = all(tg[pos[X[r, -1, 0]]] + 48 <= tg[idx[r]] + 1e-9 for r in range(len(X)))
check("gapped series still respects cutoff", ok, f"({len(X)} sequences)")

# --- 6. targets too early to have a full window are dropped, not silently truncated
X, y, idx = make_sequences(feats(12), 4, mk_inits(12), lead_time=72)
ok = all(len(x) == 4 for x in X)
check("all emitted windows are full length", ok, f"({len(X)} sequences from 12 samples)")

print()
print("ALL PASS" if fail == 0 else f"{fail} FAILURE(S)")
sys.exit(1 if fail else 0)
