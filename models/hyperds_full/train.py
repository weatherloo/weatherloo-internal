#!/usr/bin/env python3
"""Train the released full HyperDS architecture and its hourly extensions."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model import HyperDS, UPSTREAM_REVISION, grid_average, subgrid_queries


class WeatherDataset(Dataset):
    def __init__(self, store, split, manifest):
        import zarr

        if "shards" in manifest:
            self.groups = [zarr.open_group(str(Path(store)/s["path"]),mode="r")[split]
                           for s in manifest["shards"] if s["split"]==split]
        else:
            self.groups = [zarr.open_group(str(store),mode="r")[split]]
        self.rows = [(group,index) for group in self.groups for index in range(group["grid"].shape[0])]
        self.init_times = np.concatenate([np.asarray(group["init_time"]) for group in self.groups])
        self.coordinates = torch.from_numpy(np.asarray(self.groups[0]["query_xy"],dtype=np.float32))
        stats = manifest["normalization"]
        self.mean = np.array(stats["field_mean"],dtype=np.float32)
        self.std = np.array(stats["field_std"],dtype=np.float32)
        self.sat_mean = np.array(stats["satellite_mean"],dtype=np.float32)
        self.sat_std = np.array(stats["satellite_std"],dtype=np.float32)
        self.station_overlay = None
        self.label_end = None
        if "station_overlay" in manifest:
            with np.load(Path(store)/manifest["station_overlay"]["path"]) as overlay:
                self.station_overlay = (overlay["times"],overlay["values"])
            if split=="test" and manifest.get("test_year")==2025:
                self.label_end = 1767225600  # 2026-01-01 UTC

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        group,index = self.rows[index]
        grid = np.asarray(group["grid"][index],dtype=np.float32)
        satellite = np.asarray(group["satellite"][index],dtype=np.float32)
        station = np.asarray(group["station"][index],dtype=np.float32)
        if self.station_overlay is not None:
            times,values = self.station_overlay
            stamps = int(group["init_time"][index])+np.arange(station.shape[0])*3600
            indices = np.searchsorted(times,stamps)
            if (indices>=len(times)).any() or not np.array_equal(times[indices],stamps):
                raise ValueError("station overlay time mismatch")
            station[:,0,:2] = values[indices]
            if self.label_end is not None:
                station[stamps>=self.label_end] = np.nan
        if not np.isfinite(grid).all() or not np.isfinite(satellite).all():
            raise ValueError("incomplete grid or satellite input")
        grid = (grid-self.mean[None,:,None,None])/self.std[None,:,None,None]
        satellite = (satellite-self.sat_mean[None,:,None,None])/self.sat_std[None,:,None,None]
        return torch.from_numpy(grid),torch.from_numpy(satellite),self.coordinates,torch.from_numpy(station)


def physical_station(prediction, mean, std):
    physical = prediction*std+mean
    wind = (physical[...,0].square()+physical[...,1].square()+1e-8).sqrt()
    return torch.stack((wind,physical[...,2],physical[...,3],physical[...,4]),dim=-1)


def station_loss(prediction, target, mean, std, station_indices):
    prediction = physical_station(prediction,mean,std)[:,station_indices]
    target = target[:,station_indices]
    mask = torch.isfinite(target)
    scale = torch.stack(((std[0].square()+std[1].square()).sqrt(),std[2],std[3],std[4]))
    error = ((prediction-torch.nan_to_num(target))/scale).square()
    return (error*mask).sum()/mask.sum().clamp_min(1)


def predict_step(model, context, field, query, grid_query, hour, feedback_query=None):
    all_queries = torch.cat((query,grid_query) if feedback_query is None else (query,grid_query,feedback_query),dim=1)
    prediction = model.decode(context,field,all_queries,hour)
    station = prediction[:,:query.shape[1]]
    grid = prediction[:,query.shape[1]:query.shape[1]+grid_query.shape[1]]
    feedback = prediction[:,query.shape[1]+grid_query.shape[1]:]
    return station,grid,feedback


def train_epoch(model,loader,optimizer,device,mean,std,training_indices=(0,1)):
    model.train()
    total_grid,total_station,steps = 0.,0.,0
    for batch_index,(target,satellite,query,station) in enumerate(loader):
        target,satellite,query,station = (value.to(device) for value in (target,satellite,query,station))
        initial = F.avg_pool2d(target[:,0],4)
        batch,_,height,width = target[:,0].shape
        state = initial
        hours = 1 if model.mode=="paper" else target.shape[1]
        # Full predicted-state rollouts with gradients truncated every six hours.
        # All 49 target leads are trained; no observed future field enters state.
        for start in range(0,hours,6):
            optimizer.zero_grad()
            satellite_context = model.encode_satellite(satellite)
            direct_context = model.encode_field(initial,satellite_context) if model.mode!="autoregressive" else None
            losses = []
            for hour in range(start,min(start+6,hours)):
                field = state if model.mode=="autoregressive" else initial
                context = model.encode_field(field,satellite_context) if direct_context is None else direct_context
                all_grid = subgrid_queries(batch,height,width,device,stations=query[:,training_indices])
                # Uniform cell minibatches preserve ten subgrid samples per cell.
                cells = torch.randperm(height*width,device=device)[:128]
                sampled_grid = all_grid.reshape(batch,height*width,10,2)[:,cells].reshape(batch,-1,2)
                feedback_query = subgrid_queries(batch,*initial.shape[-2:],device) if model.mode=="autoregressive" else None
                point,grid,feedback = predict_step(model,context,field,query,sampled_grid,hour,feedback_query)
                grid_pred = grid.reshape(batch,len(cells),10,5).mean(dim=2)
                grid_target = target[:,hour].flatten(2).transpose(1,2)[:,cells]
                grid_loss = F.mse_loss(grid_pred,grid_target)
                stn_loss = station_loss(point,station[:,hour],mean,std,training_indices)
                loss = grid_loss+.05*stn_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at lead {hour}")
                losses.append(loss)
                total_grid += grid_loss.item()
                total_station += stn_loss.item()
                steps += 1
                if model.mode=="autoregressive":
                    state = grid_average(feedback,*initial.shape[-2:])
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if start==0 and batch_index%100==0:
                print(json.dumps({"optimizer_step_completed":model.mode,"window_hours":len(losses),
                                  "loss":torch.stack(losses).mean().item()}),flush=True)
            state = state.detach()
    return {"grid_mse":total_grid/steps,"station_mse":total_station/steps,"steps":steps}


@torch.no_grad()
def evaluate(model,loader,device,mean,std,include_test=False,selection_indices=(2,),
             excluded_indices=(3,),prediction_output=None,station_ids=None):
    model.eval()
    n_stations = loader.dataset.coordinates.shape[0]
    absolute = torch.zeros(n_stations,4,dtype=torch.float64)
    counts = torch.zeros_like(absolute)
    grid_losses,station_losses = [],[]
    lead_errors = {h:torch.zeros(n_stations,4,dtype=torch.float64) for h in (0,1,3,6,12,24,48)}
    lead_counts = {h:torch.zeros(n_stations,4,dtype=torch.float64) for h in lead_errors}
    predictions,observations = [],[]
    for target,satellite,query,station in loader:
        target,satellite,query,station = (value.to(device) for value in (target,satellite,query,station))
        initial = F.avg_pool2d(target[:,0],4)
        state = initial
        batch,_,height,width = target[:,0].shape
        sat = model.encode_satellite(satellite)
        direct_context = model.encode_field(initial,sat) if model.mode!="autoregressive" else None
        all_grid = subgrid_queries(batch,height,width,device,random=False)
        # Fixed spatial coverage; full-grid evaluation is reserved for final export.
        cells = torch.linspace(0,height*width-1,128,device=device).long()
        grid_query = all_grid.reshape(batch,height*width,10,2)[:,cells].reshape(batch,-1,2)
        feedback_query = subgrid_queries(batch,*initial.shape[-2:],device,random=False)
        hours = 1 if model.mode=="paper" else target.shape[1]
        batch_predictions = np.full((batch,hours,n_stations,4),np.nan,dtype=np.float32) if prediction_output else None
        for hour in range(hours):
            field = state if model.mode=="autoregressive" else initial
            context = model.encode_field(field,sat) if direct_context is None else direct_context
            point,grid,feedback = predict_step(model,context,field,query,grid_query,hour,
                                              feedback_query if model.mode=="autoregressive" else None)
            if model.mode=="autoregressive":
                state = grid_average(feedback,*initial.shape[-2:])
            physical = physical_station(point,mean,std)
            if prediction_output:
                batch_predictions[:,hour] = physical.cpu().numpy()
            if hour==0 and model.mode!="paper":
                continue
            observed = station[:,hour]
            mask = torch.isfinite(observed)
            if not include_test:
                mask[:,excluded_indices] = False
            error = (physical-torch.nan_to_num(observed)).abs()*mask
            absolute += error.sum(dim=0).cpu()
            counts += mask.sum(dim=0).cpu()
            if hour in lead_errors:
                lead_errors[hour] += error.sum(dim=0).cpu()
                lead_counts[hour] += mask.sum(dim=0).cpu()
            station_losses.append(station_loss(point,observed,mean,std,selection_indices).item())
            grid_pred = grid.reshape(batch,len(cells),10,5).mean(dim=2)
            grid_losses.append(F.mse_loss(grid_pred,target[:,hour].flatten(2).transpose(1,2)[:,cells]).item())
        if prediction_output:
            predictions.append(batch_predictions)
            observations.append(station[:,:hours].cpu().numpy())
    if prediction_output:
        np.savez_compressed(prediction_output,forecast=np.concatenate(predictions),observation=np.concatenate(observations),
                            initializations=loader.dataset.init_times,station_ids=np.array(station_ids),
                            variables=np.array(["wind_speed","t2m","sp","tp1h"]),
                            units=np.array(["m/s","degC","hPa","mm"]),lead_times_hours=np.arange(hours))
    return {"station_mae":(absolute/counts.clamp_min(1)).tolist(),"count":counts.int().tolist(),
            "grid_mse":float(np.mean(grid_losses)),"validation_station_mse":float(np.mean(station_losses)),
            "by_lead":{str(h):{"mae":(lead_errors[h]/lead_counts[h].clamp_min(1)).tolist(),
                               "count":lead_counts[h].int().tolist()} for h in lead_errors if lead_counts[h].sum()>0}}


def commit_checkpoints():
    import modal
    if not modal.is_local():
        modal.Volume.from_name("weatherloo-temporal-checkpoints").commit()


def smoke():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for mode in ("paper","direct","autoregressive"):
        model = HyperDS(mode,satellite_shape=(32,64))
        field = torch.randn(2,5,8,16)
        satellite = torch.randn(2,2,4,32,64)
        query = torch.zeros(2,4,2)
        prediction = model(field,satellite,query,hour=1)
        assert prediction.shape==(2,4,5)
        loss = station_loss(prediction,torch.randn(2,4,4),torch.zeros(5),torch.ones(5),[0,1])
        loss.backward()
        assert model.fusion.h8_conv3.weight.grad is not None
        assert model.fusion.pred_conv3.weight.grad is not None
        assert all(decoder.feat_fc1.weight.grad is not None for decoder in model.decoders)
        print(json.dumps({"smoke":mode,"shape":list(prediction.shape),"loss":loss.item(),
                          "satellite_and_field_gradients":True}),flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--store",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--mode",choices=("paper","direct","autoregressive"),default="paper")
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--stop-after-epoch",type=int)
    args = parser.parse_args(argv)
    if args.smoke:
        smoke()
        return
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda":
        raise RuntimeError("real training requires a GPU")
    manifest = json.loads((args.store/"manifest.json").read_text())
    if "shards" in manifest:
        from multiyear import validate as validate_store
    else:
        from data import validate_store
    print(json.dumps({"data_audit":validate_store(args.store)}),flush=True)
    training = WeatherDataset(args.store,"train",manifest)
    validation = WeatherDataset(args.store,"val",manifest)
    workers = 2 if "shards" in manifest else 0
    train_loader = DataLoader(training,batch_size=4,shuffle=True,num_workers=workers,persistent_workers=bool(workers))
    val_loader = DataLoader(validation,batch_size=2,num_workers=workers,persistent_workers=bool(workers))
    training_indices = manifest.get("training_station_indices",[0,1])
    evaluation_options = {"selection_indices":manifest.get("selection_station_indices",[2]),
                          "excluded_indices":manifest.get("excluded_validation_station_indices",[3])}
    stats = manifest["normalization"]
    mean = torch.tensor(stats["field_mean"],device=device)
    std = torch.tensor(stats["field_std"],device=device)
    model = HyperDS(args.mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5)
    args.out.mkdir(parents=True,exist_ok=True)
    first,best = 1,float("inf")
    source = {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
              for name in (("model.py","train.py","data.py","multiyear.py","multiyear_plan.py")
                           if "shards" in manifest else ("model.py","train.py","data.py"))}
    latest = args.out/"latest.pt"
    warm_start = args.out.parent/"paper/best.pt"
    if args.mode!="paper" and not latest.is_file() and warm_start.is_file():
        model.load_state_dict(torch.load(warm_start,map_location=device,weights_only=False)["state_dict"])
        print(json.dumps({"initialized_from":str(warm_start)}),flush=True)
    if latest.is_file():
        checkpoint = torch.load(latest,map_location=device,weights_only=False)
        if checkpoint["source_sha256"]!=source:
            raise RuntimeError("source changed; use a new run directory")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        first,best = checkpoint["epoch"]+1,checkpoint["best_validation_station_mse"]
        torch.set_rng_state(checkpoint["rng_state"].cpu())
        if "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
            np.random.set_state(checkpoint["numpy_rng_state"])
    print(json.dumps({"training_started":args.mode,"device":torch.cuda.get_device_name(),
                      "epochs":args.epochs,"train_initializations":len(training),"validation_initializations":len(validation),
                      "upstream_revision":UPSTREAM_REVISION,"source_sha256":source}),flush=True)
    stop = min(args.epochs,args.stop_after_epoch or args.epochs)
    for epoch in range(first,stop+1):
        started = time.monotonic()
        train_metrics = train_epoch(model,train_loader,optimizer,device,mean,std,training_indices)
        val_metrics = evaluate(model,val_loader,device,mean,std,**evaluation_options)
        score = val_metrics["validation_station_mse"]
        if not np.isfinite(score):
            raise RuntimeError("non-finite validation loss")
        scheduler.step()
        improved = score<best
        best = min(best,score)
        checkpoint = {"state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                      "epoch":epoch,"mode":args.mode,"best_validation_station_mse":best,"validation":val_metrics,
                      "rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all(),
                      "numpy_rng_state":np.random.get_state(),"source_sha256":source,"manifest":manifest,
                      "upstream_revision":UPSTREAM_REVISION,"config":{"epochs":args.epochs,"batch_size":4,"lr":1e-4,
                      "station_beta":.05,"subgrid_samples":10,"grid_cells_per_step":128,"bptt_hours":6,"seed":42,
                      "warm_start":str(warm_start) if args.mode!="paper" and warm_start.is_file() else None}}
        torch.save(checkpoint,latest)
        if improved:
            torch.save(checkpoint,args.out/"best.pt")
        record = {"epoch":epoch,"mode":args.mode,"training":train_metrics,"validation":val_metrics,
                  "seconds":time.monotonic()-started,"best":improved}
        print(json.dumps(record),flush=True)
        with (args.out/"metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record)+"\n")
        commit_checkpoints()
    if stop<args.epochs:
        print(json.dumps({"training_stage_saved":args.mode,"epoch":stop,"epochs":args.epochs}),flush=True)
        return
    model.load_state_dict(torch.load(args.out/"best.pt",map_location=device,weights_only=False)["state_dict"])
    if "shards" in manifest:
        test = WeatherDataset(args.store,"test",manifest)
        final_loader = DataLoader(test,batch_size=2,num_workers=2)
    else:
        final_loader = val_loader
    final = evaluate(model,final_loader,device,mean,std,include_test=True,**evaluation_options,
                     prediction_output=args.out/"predictions.npz",station_ids=[s["id"] for s in manifest["stations"]])
    (args.out/"evaluation.json").write_text(json.dumps(final,indent=2)+"\n")
    print(json.dumps({"completed":args.mode,"evaluation":final}),flush=True)
    commit_checkpoints()


if __name__=="__main__":
    main()
