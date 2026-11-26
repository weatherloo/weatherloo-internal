"""Regression checks for both actual Soulis CSV layouts and sensor missingness."""

import csv
from datetime import datetime,timezone
import importlib.util
from pathlib import Path
import tempfile
import unittest


def parsers():
    root = Path("/workspace")
    if not root.is_dir():
        root = Path(__file__).resolve().parents[2]
    result = []
    for name,path in (("download","pipelines/download/fetch_station_observations.py"),
                      ("training","src/hrrr_bias_correction/make_data.py")):
        spec = importlib.util.spec_from_file_location(name,root/path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module.load_soulis_hobo_csv)
    return result


class SoulisTests(unittest.TestCase):
    def check_rows(self,header,row,expected):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"station.csv"
            with path.open("w",newline="") as stream:
                writer = csv.writer(stream)
                writer.writerows([header,row])
            for parse in parsers():
                with self.subTest(parser=parse.__module__):
                    values = parse(path)
                    self.assertEqual(len(values),1)
                    self.assertEqual(values[0][0],datetime(2020,1,2,5,tzinfo=timezone.utc))
                    self.assertEqual(values[0][1],expected)

    def test_legacy_columns(self):
        self.check_rows(["year","month","day","hour","minute"]+["sensor"]*7,
                        [2020,1,2,0,0,3.25,0,0,0,0,7.2,10],{"t2m":3.25,"wind_speed":7.2})

    def test_average_above_gust_masks_only_wind(self):
        self.check_rows(["year","month","day","hour","minute"]+["sensor"]*7,
                        [2020,1,2,0,0,3.25,0,0,0,0,185.4,46.73],{"t2m":3.25,"wind_speed":None})

    def test_temperature_range_masks_only_temperature(self):
        self.check_rows(["year","month","day","hour","minute"]+["sensor"]*7,
                        [2020,1,2,0,0,355,0,0,0,0,7.2,10],{"t2m":None,"wind_speed":7.2})

    def test_gross_wind_range_even_without_gust(self):
        self.check_rows(["year","month","day","hour","minute"]+["sensor"]*7,
                        [2020,1,2,0,0,3.25,0,0,0,0,1004.73,"NAN"],{"t2m":3.25,"wind_speed":None})

    def test_dated_temperature_with_missing_44m_wind(self):
        self.check_rows(["Date","Time","Temperature","2 m Wind Speed","4.4 m Wind Speed"]+["sensor"]*7,
                        ["2020-01-02","0:00:00",3.25,18,"NAN"]+[0]*7,
                        {"t2m":3.25,"wind_speed":None})

    def test_never_substitute_2m_wind(self):
        self.check_rows(["Date","Time","Temperature","2 m Wind Speed"]+["sensor"]*8,
                        ["2020-01-02","00:00:00",3.25,18]+[0]*8,
                        {"t2m":3.25,"wind_speed":None})


if __name__=="__main__":
    unittest.main()
