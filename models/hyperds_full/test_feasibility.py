"""Check the small experiment's chronology and benchmark-year isolation."""

import repo_paths  # noqa: F401  (shared pipelines modules on sys.path)
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import zarr

from site_scoring import site_export
from feasibility import valid_rows,ViewDataset


class FeasibilityTests(unittest.TestCase):
    def test_guard_removes_target_overlap(self):
        stamp = lambda text:int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())
        times = np.array([stamp(value) for value in ("2018-12-29T18:00:00","2018-12-30T00:00:00","2019-01-01T00:00:00")])
        np.testing.assert_array_equal(valid_rows(times,"train"),[0])
        np.testing.assert_array_equal(valid_rows(times,"val"),[2])

    def test_rows_map_to_source_groups_without_copying(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = zarr.open_group(str(root/"2019-02"),mode="w").require_group("train")
            for name,array in {"grid":np.zeros((3,49,5,32,64),dtype=np.float32),
                               "satellite":np.zeros((3,2,4,128,256),dtype=np.float32),
                               "station":np.zeros((3,49,5,4),dtype=np.float32),
                               "query_xy":np.zeros((5,2),dtype=np.float32),"init_time":np.array([10,20,30])}.items():
                group.create_dataset(name,data=array)
            manifest = {"normalization":{"field_mean":[0.]*5,"field_std":[1.]*5,
                                           "satellite_mean":[0.]*4,"satellite_std":[1.]*4},
                        "shards":[{"path":"2019-02","source_split":"train","split":"val","indices":[0,2]}]}
            dataset = ViewDataset(root,manifest,"val")
            self.assertEqual(len(dataset),2)
            np.testing.assert_array_equal(dataset.init_times,[10,30])
            self.assertEqual(dataset[1][0].shape,(49,5,32,64))
            # The repaired view must apply labels, rather than silently reading
            # the source shard's original, quality-control-failed observations.
            times = np.unique(np.concatenate([t+np.arange(49)*3600 for t in (10,20,30)]))
            values = np.tile([np.nan,7.],(len(times),1)).astype(np.float32)
            np.savez(root/"overlay.npz",times=times,values=values)
            manifest["station_overlay"] = {"path":"overlay.npz"}
            corrected = ViewDataset(root,manifest,"val")[1][3].numpy()
            self.assertTrue(np.isnan(corrected[:,0,0]).all())
            self.assertTrue((corrected[:,0,1]==7).all())

    def test_2020_export_uses_leap_year_and_2020_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for station in ("cyyz","eric_d_soulis"):
                path = root/"truth"/station
                path.mkdir(parents=True)
                (path/"observations_6h_2020.json").write_text(json.dumps({"observations":[
                    {"valid_time":"2020-01-01T06:00:00Z","t2m":2.,"wind_speed":0.}]}))
            values = np.zeros((1,49,2,4),dtype=np.float32)
            values[...,1] = 3
            predictions = root/"predictions.npz"
            np.savez(predictions,forecast=values,initializations=np.array([1577836800]),
                     station_ids=np.array(["cyyz","eric_d_soulis"]),variables=np.array(["wind_speed","t2m","sp","tp1h"]),
                     units=np.array(["m/s","degC","hPa","mm"]),lead_times_hours=np.arange(49))
            exported = site_export(predictions,root/"truth",root/"out","feasibility",year=2020)
            with np.load(exported) as saved:
                self.assertEqual(saved["bias"].shape,(1464,2,2,12))
                self.assertEqual(saved["bias"][0,1,0,0],1.)
                self.assertTrue(str(saved["initializations"][-1]).startswith("2020-12-31"))


if __name__=="__main__":
    unittest.main()
