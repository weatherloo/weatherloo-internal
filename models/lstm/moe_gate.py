"""
Mixture-of-experts gate over the LSTM-corrected forecasts (MoWE-style, scaled
down to point stations). A small MLP outputs a softmax weight per expert
(method) plus a bias; final forecast = sum_i w_i * corrected_i + b.

Input is retrain_output/<station>_<variable>/combined_predictions.csv (from
combine_predictions.py). Its predicted/actual columns are biases, so each
expert's corrected-forecast error is e_i = actual_i - predicted_i, and with
weights summing to 1 the blended error is sum_i w_i * e_i + b. The gate can be
trained on that without the raw forecasts.

Gate features use only what is known at issue time: lead time, init hour, and
each expert's RMSE over its last 8 *verified* inits (init + lead <= now).

Usage:
    python models/lstm/moe_gate.py --station eric_d_soulis --variable t2m
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LSTM_DIR = os.path.dirname(os.path.abspath(__file__))
HIST = 8  # verified inits used for each expert's recent RMSE


def build_table(csv_path):
    d = pd.read_csv(csv_path)
    d["t"] = pd.to_datetime(d.timestamp, utc=True)
    d["e"] = d.actual - d.predicted
    methods = sorted(d.method.unique())

    # Recent RMSE per expert, lagged so only errors verified by issue time are used.
    recent = []
    for (m, lead), g in d.groupby(["method", "lead_time"]):
        s = g.set_index("t").e.sort_index()
        r = (s**2).rolling(HIST, min_periods=3).mean() ** 0.5
        recent.append(pd.DataFrame({
            "method": m, "lead_time": lead, "t": s.index,
            "recent": r.asof(s.index - pd.Timedelta(hours=int(lead))).values,
        }))
    d = d.merge(pd.concat(recent), on=["method", "lead_time", "t"])

    E = d.pivot_table(index=["lead_time", "t"], columns="method", values="e").reindex(columns=methods)
    R = d.pivot_table(index=["lead_time", "t"], columns="method", values="recent").reindex(index=E.index, columns=methods)
    return E, R, methods


def features(E, R, med):
    lead = E.index.get_level_values(0).values / 72.0
    hour = E.index.get_level_values(1).hour.values * 2 * np.pi / 24
    return np.column_stack([
        lead, np.sin(hour), np.cos(hour),
        R.fillna(med).values, R.isna().values, E.isna().values,
    ]).astype(np.float32)


class Gate(nn.Module):
    def __init__(self, n_in, n_exp, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_exp + 1)
        )

    def forward(self, x, err, avail):
        out = self.net(x)
        w = torch.softmax(out[:, :-1].masked_fill(~avail, -1e9), dim=1)
        return (w * err).sum(1) + out[:, -1], w


def train_gate(Xtr, Etr, Atr, Xva, Eva, Ava, seed, epochs=2000, patience=100):
    torch.manual_seed(seed)
    g = Gate(Xtr.shape[1], Etr.shape[1])
    opt = torch.optim.AdamW(g.parameters(), lr=3e-3, weight_decay=1e-2)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        g.train()
        opt.zero_grad()
        loss = (g(Xtr, Etr, Atr)[0] ** 2).mean()
        loss.backward()
        opt.step()
        g.eval()
        with torch.no_grad():
            v = (g(Xva, Eva, Ava)[0] ** 2).mean().item()
        if v < best:
            best, best_state, bad = v, {k: x.clone() for k, x in g.state_dict().items()}, 0
        elif (bad := bad + 1) >= patience:
            break
    g.load_state_dict(best_state)
    return g.eval()


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--station", default="eric_d_soulis")
    p.add_argument("--variable", default="t2m")
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()

    csv_path = os.path.join(LSTM_DIR, "retrain_output", f"{args.station}_{args.variable}", "combined_predictions.csv")
    E, R, methods = build_table(csv_path)

    # Chronological 60/20/20 split on init time (shared across leads -> no look-ahead).
    times = np.sort(E.index.get_level_values(1).unique())
    c1, c2 = times[int(0.6 * len(times))], times[int(0.8 * len(times))]
    t = E.index.get_level_values(1)
    tr, va, te = t < c1, (t >= c1) & (t < c2), t >= c2

    med = R[tr].median()
    X = torch.tensor(features(E, R, med))
    A = torch.tensor(E.notna().values)
    Ev = torch.tensor(E.fillna(0).values, dtype=torch.float32)

    gates = [train_gate(X[tr], Ev[tr], A[tr], X[va], Ev[va], A[va], s) for s in range(args.seeds)]
    with torch.no_grad():
        moe = torch.stack([g(X, Ev, A)[0] for g in gates]).mean(0).numpy()
        W = torch.stack([g(X, Ev, A)[1] for g in gates]).mean(0).numpy()

    # Baselines picked on train+val, all scored on test rows where every expert of that lead exists.
    fit = tr | va
    print(f"{args.station} {args.variable}: train<{c1}  val<{c2}  test>={c2}")
    print(f"{'lead':>4} {'n':>4} {'best-1':>16} {'top-K avg':>12} {'MoE gate':>9}")
    tot = {"best": [], "topk": [], "moe": []}
    for lead in sorted(E.index.get_level_values(0).unique()):
        L = E.index.get_level_values(0) == lead
        cols = E.columns[E.loc[L].notna().any()]
        rank = E.loc[L & fit, cols].pow(2).mean().pow(0.5).sort_values()
        k = min(range(1, len(rank) + 1), key=lambda k: rmse(E.loc[L & fit, rank.index[:k]].mean(axis=1).dropna()))
        rows = L & te & E[cols].notna().all(axis=1).values
        b, tk, mo = E.loc[rows, rank.index[0]], E.loc[rows, rank.index[:k]].mean(axis=1), moe[rows]
        tot["best"].append(b.values); tot["topk"].append(tk.values); tot["moe"].append(mo)
        print(f"{lead:>4} {rows.sum():>4} {rmse(b):>6.3f} {rank.index[0][:9]:>9} {rmse(tk):>6.3f} (K={k}) {rmse(mo):>9.3f}")
    print(f" all {sum(map(len, tot['moe'])):>4} {rmse(np.concatenate(tot['best'])):>6.3f} {'':>9} "
          f"{rmse(np.concatenate(tot['topk'])):>6.3f} {'':>5} {rmse(np.concatenate(tot['moe'])):>9.3f}")

    print("\nmean gate weight on test:")
    print(pd.Series(W[te].mean(0), index=methods).sort_values(ascending=False).round(3).to_string())


def _check():
    # Equal weights + zero bias must reproduce the plain mean of available expert errors.
    g = Gate(3, 2)
    for prm in g.parameters():
        nn.init.zeros_(prm)
    err = torch.tensor([[1.0, 3.0], [2.0, 0.0]])
    avail = torch.tensor([[True, True], [True, False]])
    out, w = g(torch.zeros(2, 3), err, avail)
    assert torch.allclose(out, torch.tensor([2.0, 2.0])), out
    assert torch.allclose(w.sum(1), torch.ones(2))


if __name__ == "__main__":
    _check()
    main()
