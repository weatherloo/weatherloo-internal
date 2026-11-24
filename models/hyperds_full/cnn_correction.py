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


def load_patches(root, name="patches.npy"):
    root = Path(root)
    patches, inits, meta = [], [], None
    for path in sorted(root.glob("*/manifest.json")):
        detail = json.loads(path.read_text())
        meta = meta or detail
        if detail["samples"]:
            patches.append(np.load(path.parent / name, mmap_mode="r"))
            inits.append(np.load(path.parent / "init_time.npy"))
    inits = np.concatenate(inits)
    order = np.argsort(inits)
    return np.concatenate(patches)[order], inits[order], meta


def assemble(point_root, patch_root, station_root, target, wide=False):
    """Scalars, residuals and HRRR at leads 1..48; patches (and the wide view) at all leads 0..48."""
    values, inits, point_ids, variables = mos.load_points(point_root)
    station_root = Path(station_root)
    meta = json.loads((station_root / "stations.json").read_text())["stations"]
    with np.load(station_root / "labels.npz") as saved:
        label_times, labels = saved["times"], saved["labels"]
    X, names, y, base, split = mos.build(values, inits, point_ids, variables, meta, label_times, labels,
                                         [s["id"] for s in meta], [target])
    patches, patch_inits, patch_meta = load_patches(patch_root)
    common, a, b = np.intersect1d(inits, patch_inits, return_indices=True)
    keep = [i for i, n in enumerate(names) if n not in STATION_CONSTANT]
    out = {"inits": common, "split": split[a], "patches": patches[b], "scalars": X[a][:, :, 0][:, :, keep],
           "scalar_names": [names[i] for i in keep],
           "residual": np.stack((y["t2m"][a][:, :, 0], y["wind"][a][:, :, 0]), -1),  # [init, lead 1..48, 2]
           "hrrr": np.stack((base["t2m"][a][:, :, 0], base["wind"][a][:, :, 0]), -1),
           "patch_vars": patch_meta["variables"]}
    if wide:
        view, view_inits, view_meta = load_patches(patch_root, "wide.npy")
        if not np.array_equal(view_inits, patch_inits):
            raise ValueError("wide view and patches cover different initializations")
        out["wide"], out["wide_vars"] = view[b], view_meta["wide_variables"]
    return out


def build_model(n_channels, n_scalars, width=32, traj_channels=0, wide_channels=0):
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
            features = 4 * width + 2 * width + 128
            if traj_channels:
                # Site trajectory over leads 0..48 plus a one-hot of the lead being predicted.
                self.traj = nn.Sequential(nn.Conv1d(traj_channels + 1, 64, 5, padding=2), act(),
                                          nn.Conv1d(64, 64, 5, padding=2), act())
                features += 128
            if wide_channels:
                self.wide = nn.Sequential(nn.Conv2d(wide_channels, width, 3, padding=1), act(),
                                          nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), act(),
                                          nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1), act(),
                                          nn.AdaptiveAvgPool2d(1), nn.Flatten())
                features += 2 * width
            self.head = nn.Sequential(nn.Linear(features, 256), act(), nn.Dropout(0.1), nn.Linear(256, 2))

        def forward(self, patch, scalars, traj=None, lead=None, wide=None):
            mid = patch.shape[-1] // 2
            centre = patch[:, :, mid - 1:mid + 2, mid - 1:mid + 2].flatten(1)
            parts = [self.conv(patch), self.centre(centre), self.scalar(scalars)]
            if traj_channels:
                onehot = torch.zeros(traj.shape[0], 1, traj.shape[2], device=traj.device, dtype=traj.dtype)
                onehot[torch.arange(len(lead)), 0, lead] = 1
                hidden = self.traj(torch.cat((traj, onehot), 1))
                parts += [hidden[torch.arange(len(lead)), :, lead], hidden.mean(-1)]
            if wide_channels:
                parts.append(self.wide(wide))
            return self.head(torch.cat(parts, 1))

    return PatchNet()


def run_cnn(point_root, patch_root, station_root, target="eric_d_soulis", gbm_export=None, r5_predictions=None,
            out=None, seeds=3, max_epochs=40, patience=5, batch_size=512, context=(0,), trajectory=False,
            variables=None, wide=False):
    """Train the CNN ensemble; options: context = patch lead offsets, trajectory = site 0..48 h sequence,
    variables = patch fields to use (default all stored), wide = block-averaged whole-chunk view."""
    import torch

    data = assemble(point_root, patch_root, station_root, target, wide=wide)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = data["split"]
    train_i, val_i, test_i = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
    channels = [data["patch_vars"].index(v) for v in (variables or data["patch_vars"])]
    offsets = torch.tensor(sorted(set(context) | {0}), device=device)

    def to_gpu(array, index, sample):
        """Normalize with training statistics, NaN -> 0, float16 on the device, built in slices."""
        stats = array[sample][:, :, index].astype(np.float32)
        mean = np.nanmean(stats, axis=(0, 1, 3, 4))[None, None, :, None, None]
        std = (np.nanstd(stats, axis=(0, 1, 3, 4)) + 1e-3)[None, None, :, None, None]
        out_ = torch.empty((len(array),) + array.shape[1:2] + (len(index),) + array.shape[3:], dtype=torch.float16,
                           device=device)
        for s in range(0, len(array), 256):
            block = (array[s:s + 256][:, :, index].astype(np.float32) - mean) / std
            out_[s:s + 256] = torch.from_numpy(np.nan_to_num(block)).half()
        return out_

    patches = to_gpu(data["patches"], channels, train_i[::7])  # [init, lead 0..48, C, H, W]
    views = to_gpu(data["wide"], list(range(data["wide"].shape[2])), train_i[::7]) if wide else None
    mid = patches.shape[-1] // 2
    traj = patches[:, :, :, mid, mid].transpose(1, 2).contiguous() if trajectory else None  # [init, C, 49]
    s_train = data["scalars"][train_i].reshape(-1, data["scalars"].shape[-1])
    s_mean, s_std = np.nanmean(s_train, axis=0), np.nanstd(s_train, axis=0) + 1e-3
    s = (data["scalars"] - s_mean) / s_std
    scalars = torch.tensor(np.concatenate((np.nan_to_num(s), np.isfinite(s)), -1), dtype=torch.float32, device=device)
    r_std = np.nanstd(data["residual"][train_i].reshape(-1, 2), axis=0)
    residual = torch.tensor(data["residual"] / r_std, dtype=torch.float32, device=device)

    def batch(model, inits, leads):
        """inits: init rows; leads: 1..48."""
        idx = (leads[:, None] + offsets[None]).clamp(0, 48)
        patch = patches[inits[:, None], idx].flatten(1, 2).float()
        extra = {}
        if trajectory:
            extra.update(traj=traj[inits].float(), lead=leads)
        if wide:
            extra["wide"] = views[inits, leads].float()
        return model(patch, scalars[inits, leads - 1], **extra)

    def samples(rows):
        rows = torch.tensor(rows, device=device)
        return rows.repeat_interleave(48), torch.arange(1, 49, device=device).repeat(len(rows))

    def predict(model, rows):
        model.eval()
        inits, leads = samples(rows)
        with torch.no_grad():
            pred = torch.cat([batch(model, inits[k:k + 4096], leads[k:k + 4096]) for k in range(0, len(inits), 4096)])
        return pred.cpu().numpy().reshape(len(rows), 48, 2)

    def val_loss(model):
        pred = torch.tensor(predict(model, val_i), device=device)
        target_ = residual[torch.tensor(val_i, device=device)]
        mask = torch.isfinite(target_)
        return (((pred - torch.nan_to_num(target_)) ** 2) * mask).sum().item() / max(mask.sum().item(), 1)

    n_channels = patches.shape[2] * len(offsets)
    members, history = [], []
    train_inits, train_leads = samples(train_i)
    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(n_channels, data["scalars"].shape[-1], traj_channels=patches.shape[2] if trajectory else 0,
                            wide_channels=views.shape[2] if wide else 0).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        best, best_state, stale = np.inf, None, 0
        for epoch in range(max_epochs):
            model.train()
            order = torch.randperm(len(train_inits), device=device)
            for start in range(0, len(order), batch_size):
                pick = order[start:start + batch_size]
                pred = batch(model, train_inits[pick], train_leads[pick])
                target_ = residual[train_inits[pick], train_leads[pick] - 1]
                mask = torch.isfinite(target_)
                loss = ((pred - torch.nan_to_num(target_)) ** 2 * mask).sum() / mask.sum().clamp_min(1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            score = val_loss(model)
            history.append({"seed": seed, "epoch": epoch + 1, "val_loss": score})
            if score < best:
                best, stale = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        members.append({"val_loss": best, "test": predict(model, test_i), "val": predict(model, val_i)})
    pred_test = (np.mean([m["test"] for m in members], axis=0) * r_std).reshape(-1, 2)  # residual, physical units
    pred_val = (np.mean([m["val"] for m in members], axis=0) * r_std).reshape(-1, 2)
    n_test = len(test_i)
    hrrr = data["hrrr"][test_i].reshape(-1, 2)
    obs = hrrr + data["residual"][test_i].reshape(-1, 2)
    cnn = hrrr + pred_test
    lead = np.tile(np.arange(48), n_test)
    units = np.array([1.0, 3.6])
    report = {"target": target, "train_inits": int(len(train_i)), "val_inits": int(len(val_i)),
              "test_inits": int(n_test), "seed_val_loss": [m["val_loss"] for m in members], "history": history,
              "options": {"context": list(map(int, offsets.tolist())), "trajectory": trajectory, "wide": wide,
                          "variables": [data["patch_vars"][c] for c in channels]},
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
