"""CNN correction of HRRR at a site from a native-grid patch plus issue-time scalar inputs.

Sample = (initialization, lead 1..48). Inputs: the 17x17 native HRRR patch (15
fields) valid at that lead, and the target-station scalar features from
mos.build (time, lagged runs, issue-time observations and errors; all known at
issue). Target: observation minus the HRRR point forecast for t2m and wind speed.
Same split as mos (train to 2023, validate 2024, test 2025); epochs and the
checkpoint are chosen on 2024 validation loss only.
"""

import json
from pathlib import Path

import numpy as np

import mos

STATION_CONSTANT = ("station_lat", "station_lon", "station_elev", "station_index")


def load_patches(root):
    root = Path(root)
    patches, inits, meta = [], [], None
    for path in sorted(root.glob("*/manifest.json")):
        detail = json.loads(path.read_text())
        meta = meta or detail
        if detail["samples"]:
            patches.append(np.load(path.parent / "patches.npy", mmap_mode="r"))
            inits.append(np.load(path.parent / "init_time.npy"))
    inits = np.concatenate(inits)
    order = np.argsort(inits)
    return np.concatenate(patches)[order], inits[order], meta["variables"]


def assemble(point_root, patch_root, station_root, target):
    values, inits, point_ids, variables = mos.load_points(point_root)
    station_root = Path(station_root)
    meta = json.loads((station_root / "stations.json").read_text())["stations"]
    with np.load(station_root / "labels.npz") as saved:
        label_times, labels = saved["times"], saved["labels"]
    X, names, y, base, split = mos.build(values, inits, point_ids, variables, meta, label_times, labels,
                                         [s["id"] for s in meta], [target])
    patches, patch_inits, patch_vars = load_patches(patch_root)
    common, a, b = np.intersect1d(inits, patch_inits, return_indices=True)
    keep = [i for i, n in enumerate(names) if n not in STATION_CONSTANT]
    scalars = X[a][:, :, 0][:, :, keep]  # [init, lead(1..48), feature]
    residual = np.stack((y["t2m"][a][:, :, 0], y["wind"][a][:, :, 0]), -1)  # [init, lead, 2]
    hrrr = np.stack((base["t2m"][a][:, :, 0], base["wind"][a][:, :, 0]), -1)
    return {"inits": common, "split": split[a], "patches": patches[b][:, 1:], "scalars": scalars,
            "scalar_names": [names[i] for i in keep], "residual": residual, "hrrr": hrrr, "patch_vars": patch_vars}


def build_model(n_channels, n_scalars, width=32):
    import torch
    from torch import nn

    class PatchNet(nn.Module):
        def __init__(self):
            super().__init__()
            act = nn.GELU
            self.conv = nn.Sequential(
                nn.Conv2d(n_channels, width, 3, padding=1), act(), nn.Conv2d(width, width, 3, padding=1), act(),
                nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), act(),            # 17 -> 9
                nn.Conv2d(2 * width, 2 * width, 3, padding=1), act(),
                nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1), act(),        # 9 -> 5
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.centre = nn.Sequential(nn.Linear(n_channels * 9, 2 * width), act())  # 3x3 around the site
            self.scalar = nn.Sequential(nn.Linear(2 * n_scalars, 128), act(), nn.Linear(128, 128), act())
            self.head = nn.Sequential(nn.Linear(4 * width + 2 * width + 128, 256), act(), nn.Dropout(0.1),
                                      nn.Linear(256, 2))

        def forward(self, patch, scalars):
            mid = patch.shape[-1] // 2
            centre = patch[:, :, mid - 1:mid + 2, mid - 1:mid + 2].flatten(1)
            return self.head(torch.cat((self.conv(patch), self.centre(centre), self.scalar(scalars)), 1))

    return PatchNet()


def run_cnn(point_root, patch_root, station_root, target="eric_d_soulis", gbm_export=None, r5_predictions=None,
            out=None, seeds=3, max_epochs=40, patience=5, batch_size=512):
    import torch
    from torch.nn import functional as F

    data = assemble(point_root, patch_root, station_root, target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = data["split"]
    train_i, val_i, test_i = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
    # Normalization from the training split only.
    sample = data["patches"][train_i[::7]].astype(np.float32)
    p_mean = np.nanmean(sample, axis=(0, 1, 3, 4))
    p_std = np.nanstd(sample, axis=(0, 1, 3, 4)) + 1e-3
    s_train = data["scalars"][train_i].reshape(-1, data["scalars"].shape[-1])
    s_mean, s_std = np.nanmean(s_train, axis=0), np.nanstd(s_train, axis=0) + 1e-3
    r_train = data["residual"][train_i].reshape(-1, 2)
    r_std = np.nanstd(r_train, axis=0)

    def tensors(index):
        patch = (data["patches"][index].astype(np.float32) - p_mean[None, None, :, None, None]) / p_std[None, None, :, None, None]
        patch = np.nan_to_num(patch)
        s = (data["scalars"][index] - s_mean) / s_std
        scalars = np.concatenate((np.nan_to_num(s), np.isfinite(s).astype(np.float32)), -1)
        res = data["residual"][index] / r_std
        lead = np.broadcast_to(np.arange(48), res.shape[:2])
        flat = lambda a: a.reshape(-1, *a.shape[2:])
        return (torch.tensor(flat(patch), dtype=torch.float16, device=device),
                torch.tensor(flat(scalars), dtype=torch.float32, device=device),
                torch.tensor(flat(res), dtype=torch.float32, device=device), flat(lead))

    train, val, test = tensors(train_i), tensors(val_i), tensors(test_i)

    def loss_on(model, part):
        model.eval()
        total, count = 0., 0.
        with torch.no_grad():
            for start in range(0, len(part[0]), 4096):
                pred = model(part[0][start:start + 4096].float(), part[1][start:start + 4096])
                target_ = part[2][start:start + 4096]
                mask = torch.isfinite(target_)
                total += ((pred - torch.nan_to_num(target_)) ** 2 * mask).sum().item()
                count += mask.sum().item()
        return total / max(count, 1)

    def predict(model, part):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(part[0][s:s + 4096].float(), part[1][s:s + 4096])
                              for s in range(0, len(part[0]), 4096)]).cpu().numpy()

    members, history = [], []
    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(train[0].shape[1], data["scalars"].shape[-1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        best, best_state, stale = np.inf, None, 0
        for epoch in range(max_epochs):
            model.train()
            order = torch.randperm(len(train[0]), device=device)
            for start in range(0, len(order), batch_size):
                pick = order[start:start + batch_size]
                pred = model(train[0][pick].float(), train[1][pick])
                target_ = train[2][pick]
                mask = torch.isfinite(target_)
                loss = ((pred - torch.nan_to_num(target_)) ** 2 * mask).sum() / mask.sum().clamp_min(1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            score = loss_on(model, val)
            history.append({"seed": seed, "epoch": epoch + 1, "val_loss": score})
            if score < best:
                best, stale = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        members.append({"val_loss": best, "test": predict(model, test), "val": predict(model, val)})
    pred_test = np.mean([m["test"] for m in members], axis=0) * r_std  # residual, physical units
    pred_val = np.mean([m["val"] for m in members], axis=0) * r_std
    n_test = len(test_i)
    hrrr = data["hrrr"][test_i].reshape(-1, 2)
    obs = hrrr + data["residual"][test_i].reshape(-1, 2)
    cnn = hrrr + pred_test
    lead = test[3]
    units = np.array([1.0, 3.6])
    report = {"target": target, "train_inits": int(len(train_i)), "val_inits": int(len(val_i)),
              "test_inits": int(n_test), "seed_val_loss": [m["val_loss"] for m in members], "history": history,
              "patch_variables": data["patch_vars"], "scalar_features": data["scalar_names"], "methods": {}}
    val_obs = data["hrrr"][val_i].reshape(-1, 2) + data["residual"][val_i].reshape(-1, 2)
    report["val_rmse"] = {v: mos.rmse(((data["hrrr"][val_i].reshape(-1, 2) + pred_val) - val_obs)[:, j] * units[j])
                          for j, v in enumerate(("t2m", "wind"))}
    candidates = {"raw_hrrr": hrrr, "cnn": cnn}
    if gbm_export:
        with np.load(gbm_export) as g:
            position = {int(t): i for i, t in enumerate(g["init"])}
            rows = [position.get(int(t), -1) for t in data["inits"][test_i]]
            gbm = np.full((n_test, 48, 2), np.nan)
            for k, r in enumerate(rows):
                if r >= 0:
                    gbm[k, :, 0], gbm[k, :, 1] = g["t2m"][r], g["wind"][r]
            gbm = gbm.reshape(-1, 2)
        candidates["gbm_round2"] = gbm
        candidates["cnn_gbm_mean"] = (cnn + gbm) / 2
    if r5_predictions:
        with np.load(r5_predictions) as r5:
            s = list(r5["station_ids"]).index({"eccc_51459": "cyyz"}.get(target, target))
            position = {int(t): i for i, t in enumerate(r5["initializations"])}
            leads = list(r5["lead_times_hours"])
            r5_pred = np.full((n_test, 48, 2), np.nan)
            for k, t in enumerate(data["inits"][test_i]):
                if int(t) in position:
                    block = r5["forecast"][position[int(t)], [leads.index(h) for h in range(1, 49)], s, :2]
                    r5_pred[k, :, 0], r5_pred[k, :, 1] = block[:, 1], block[:, 0]
        candidates["r5"] = r5_pred.reshape(-1, 2)
    mask = np.isfinite(obs)
    for values in candidates.values():
        mask &= np.isfinite(values)
    for name, values in candidates.items():
        report["methods"][name] = {}
        for j, variable in enumerate(("t2m", "wind")):
            m = mask[:, j]
            error = (values[:, j] - obs[:, j])[m] * units[j]
            report["methods"][name][variable] = {
                "rmse": mos.rmse(error), "count": int(m.sum()),
                "rmse_by_lead": [mos.rmse(error[lead[m] == h]) for h in range(48)]}
    if out:
        Path(out).write_text(json.dumps(report, indent=2) + "\n")
        n_val = len(val_i)
        np.savez_compressed(Path(out).with_suffix(".npz"), init=data["inits"][test_i],
                            cnn=cnn.reshape(n_test, 48, 2), obs=obs.reshape(n_test, 48, 2),
                            val_init=data["inits"][val_i],
                            val_cnn=(data["hrrr"][val_i].reshape(-1, 2) + pred_val).reshape(n_val, 48, 2),
                            val_obs=val_obs.reshape(n_val, 48, 2), variables=np.array(["t2m", "wind"]))
    return report
