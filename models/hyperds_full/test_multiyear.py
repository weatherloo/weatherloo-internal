"""Chronology, shard loading and Waterloo evaluation contract checks."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader
import zarr

from data import list_goes, select_goes, goes_time, fill_satellite_gaps
from multiyear import combine_moments, moments
from multiyear_plan import STATIONS, UTC, GOES19_START, month_records, plan
from train import WeatherDataset, evaluate
from benchmark import STATIONS as SITE_STATIONS, site_export, matched_comparison


class ConstantField(torch.nn.Module):
    mode = "direct"

    def encode_satellite(self,satellite):
        return None

    def encode_field(self,field,satellite):
        return None

    def decode(self,context,field,query,hour):
        return field[:,:,0,0][:,None].expand(-1,query.shape[1],-1)


class MultiyearTests(unittest.TestCase):
    def test_satellite_quality_holes_and_rejection(self):
        values = np.arange(10000,dtype=np.float32).reshape(100,100)
        values[50,50] = np.nan
        filled,fraction = fill_satellite_gaps(values)
        self.assertAlmostEqual(fraction,.0001)
        self.assertTrue(np.isfinite(filled).all())
        np.testing.assert_array_equal(filled[np.isfinite(values)],values[np.isfinite(values)])
        values[:2] = np.nan
        with self.assertRaises(RuntimeError):
            fill_satellite_gaps(values)

    def test_site_truth_wind_conversion_and_matched_rmse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = np.array([int(datetime(2025,1,1,h,tzinfo=UTC).timestamp()) for h in (0,6)])
            values = np.zeros((2,49,2,4),dtype=np.float32)
            values[...,0] = 5  # m/s -> 18 km/h, compared with 15 km/h below
            values[0,...,1] = 1
            values[1,...,1] = 3
            predictions = root/"predictions.npz"
            np.savez(predictions,initializations=times,station_ids=np.array(SITE_STATIONS),
                     variables=np.array(["wind_speed","t2m","sp","tp1h"]),
                     units=np.array(["m/s","degC","hPa","mm"]),lead_times_hours=np.arange(49),forecast=values)
            for station in SITE_STATIONS:
                path = root/"truth"/station
                path.mkdir(parents=True)
                (path/"observations_6h_2025.json").write_text(json.dumps({"observations":[
                    {"valid_time":f"2025-01-01T{h:02d}:00:00Z","t2m":0.,"wind_speed":15.} for h in (6,12)]}))
            exported = site_export(predictions,root/"truth",root/"out","hyperds_direct")
            full = matched_comparison(exported,exported)["eric_d_soulis"]["t2m"]["by_lead"]["6"]
            self.assertAlmostEqual(full["hyperds"]["rmse"],np.sqrt(5))
            self.assertEqual(full["hyperds"]["mae"],2.)
            with np.load(exported) as saved:
                self.assertEqual(saved["bias"].shape,(1460,2,2,12))
                self.assertEqual(saved["bias"][0,1,1,0],3.)
                self.assertTrue(np.isnan(saved["bias"][:,:,:,8:]).all())
                baseline = {key:saved[key] for key in saved.files}
                baseline["bias"] = saved["bias"].copy()
                baseline["bias"][0,1,0,0] = np.nan
                np.savez(root/"baseline.npz",**baseline)
            matched = matched_comparison(exported,root/"baseline.npz")["eric_d_soulis"]["t2m"]["by_lead"]["6"]
            self.assertEqual(matched["hyperds"]["n"],1)
            self.assertEqual(matched["hyperds"]["rmse"],3.)

    def test_calendar_split_guards_and_all_cycles(self):
        specification = plan()
        self.assertEqual(specification["planned_initializations"],{"train":8756,"val":1456,"test":1460})
        self.assertEqual(len(specification["months"]),96)
        records = [record for key in specification["months"] for record in month_records(key)]
        for split,following in (("train","val"),("val","test")):
            self.assertLess(max(t for s,t in records if s==split)+timedelta(hours=48),
                            min(t for s,t in records if s==following))
        self.assertEqual({t.hour for _,t in records},{0,6,12,18})
        self.assertEqual(len(month_records("2020-02")),29*4)
        self.assertEqual(specification["primary_station"],"eric_d_soulis")
        self.assertEqual(specification["selection_station_indices"],[0])
        self.assertEqual((STATIONS[0]["lat"],STATIONS[0]["lon"]),(43.4668,-80.5164))

    def test_training_moments_weight_counts(self):
        first = np.array([[1.,2.],[3.,6.]],dtype=np.float32)
        second = np.array([[9.,12.]],dtype=np.float32)
        mean,std = combine_moments([moments(first,(0,)),moments(second,(0,))])
        joined = np.concatenate((first,second))
        np.testing.assert_allclose(mean,joined.mean(axis=0))
        np.testing.assert_allclose(std,joined.std(axis=0),rtol=1e-6)

    def test_goes_scan_modes_and_operational_transition(self):
        entries = [{"key":"OR_ABI-L1b-RadC-M3C02_G16"},
                   {"key":"OR_ABI-L1b-RadC-M6C02_G19"},
                   {"key":"OR_ABI-L1b-RadC-M6C05_G16"}]
        with patch("data.list_goes_hour",return_value=entries):
            self.assertEqual(len(list_goes(datetime(2018,1,1,tzinfo=UTC),2)),2)
        self.assertEqual(GOES19_START,datetime(2025,4,7,15,10,tzinfo=UTC))

    def test_backfilled_copy_does_not_admit_future_production(self):
        issue = datetime(2018,1,2,tzinfo=UTC)
        def rows(hour,band,bucket):
            result = []
            for minute in (20,30,50):
                start = datetime(2018,1,1,23,minute,tzinfo=UTC)
                stamp = lambda value:value.strftime("%Y%j%H%M%S")+"0"
                key = f"OR_ABI-L1b-RadC-M3C{band:02d}_G16_s{stamp(start)}_e{stamp(start+timedelta(minutes=2))}_c{stamp(start+timedelta(minutes=3))}.nc"
                result.append({"key":key,"modified":"2018-07-24T00:00:00Z","bucket":bucket})
            return result
        with patch("data.list_goes",side_effect=rows):
            with self.assertRaises(RuntimeError):
                select_goes(issue)
            selected = select_goes(issue,production_cutoff=True)
        self.assertEqual(len(selected),2)
        for frame in selected:
            for band in frame:
                self.assertLessEqual(goes_time(band["key"],"c"),issue-timedelta(minutes=15))

    def test_shard_loading_and_waterloo_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"normalization":{"field_mean":[0.]*5,"field_std":[1.]*5,
                                           "satellite_mean":[0.]*4,"satellite_std":[1.]*4},"shards":[]}
            for name,split,n,stamp in (("one","train",2,100), ("two","train",1,200),
                                       ("validation","val",2,300),("final","test",1,400)):
                group = zarr.open_group(str(root/name),mode="w").require_group(split)
                grid = np.broadcast_to(np.array([3,4,7,1000,.1],dtype=np.float32)[None,None,:,None,None],(n,49,5,32,64))
                labels = np.zeros((n,49,5,4),dtype=np.float32)
                labels[:,:,0] = [5,7,1000,.1]
                for key,value in {"grid":grid,"satellite":np.zeros((n,2,4,128,256),dtype=np.float32),
                                  "station":labels,"query_xy":np.zeros((5,2),dtype=np.float32),
                                  "init_time":np.arange(stamp,stamp+n,dtype=np.int64)}.items():
                    group.create_dataset(key,data=value)
                manifest["shards"].append({"path":name,"split":split,"samples":n})
            training = WeatherDataset(root,"train",manifest)
            self.assertEqual(len(training),3)
            np.testing.assert_array_equal(training.init_times,[100,101,200])
            self.assertEqual(training[2][0].shape,(49,5,32,64))
            validation = WeatherDataset(root,"val",manifest)
            output = root/"predictions.npz"
            report = evaluate(ConstantField(),DataLoader(validation,batch_size=2),"cpu",torch.zeros(5),torch.ones(5),
                              selection_indices=[0],excluded_indices=[],prediction_output=output,
                              station_ids=[s["id"] for s in STATIONS])
            self.assertAlmostEqual(report["validation_station_mse"],0.,places=7)
            self.assertGreater(report["station_mae"][3][1],1.)
            self.assertEqual(report["count"][0],[96]*4)
            with np.load(output) as saved:
                self.assertEqual(saved["forecast"].shape,(2,49,5,4))
                self.assertEqual(saved["station_ids"][0],"eric_d_soulis")
                np.testing.assert_array_equal(saved["initializations"],[300,301])

    def test_station_overlay_preserves_sources_and_final_year_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init = int(datetime(2025,12,31,tzinfo=UTC).timestamp())
            group = zarr.open_group(str(root/"month"),mode="w").require_group("test")
            for name,array in {"grid":np.zeros((1,49,5,32,64),dtype=np.float32),
                               "satellite":np.zeros((1,2,4,128,256),dtype=np.float32),
                               "station":np.zeros((1,49,5,4),dtype=np.float32),
                               "query_xy":np.zeros((5,2),dtype=np.float32),"init_time":np.array([init])}.items():
                group.create_dataset(name,data=array)
            labels = np.full((49,2),7.,dtype=np.float32)
            labels[:,0] = np.nan
            np.savez(root/"overlay.npz",times=init+np.arange(49)*3600,values=labels)
            manifest = {"normalization":{"field_mean":[0.]*5,"field_std":[1.]*5,
                                           "satellite_mean":[0.]*4,"satellite_std":[1.]*4},
                        "shards":[{"path":"month","split":"test"}],"test_year":2025,
                        "station_overlay":{"path":"overlay.npz"}}
            corrected = WeatherDataset(root,"test",manifest)[0][3].numpy()
            self.assertTrue(np.isnan(corrected[:,0,0]).all())
            self.assertTrue((corrected[:24,0,1]==7).all())
            self.assertTrue(np.isnan(corrected[24:]).all())
            self.assertTrue((np.asarray(group["station"])==0).all())


if __name__=="__main__":
    torch.set_num_threads(2)
    unittest.main()
