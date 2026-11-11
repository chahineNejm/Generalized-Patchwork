import json

import numpy as np
import pandas as pd
import pytest

from pwgp.data.loaders import prepare_flight
from pwgp.data.manager import DatasetManager


def flight_row(**changes):
    return {"Year": 2008, "Month": 1, "DayofMonth": 1, "DayOfWeek": 1, "DepTime": 1200,
            "ArrTime": 1300, "AirTime": 60, "ArrDelay": 7, "Distance": 100, "TailNum": "A",
            "Cancelled": 0, "Diverted": 0, **changes}


def write_raw(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    planes = directory / "plane-data.csv"
    flights = directory / "2008.csv.bz2"
    pd.DataFrame({"tailnum": ["A", "B"], "year": [2000, "bad"]}).to_csv(planes, index=False)
    pd.DataFrame(rows).to_csv(flights, index=False, compression="bz2")
    return flights, planes


def test_flight_removes_invalid_rows_without_filling_and_records_counts(tmp_path):
    rows = [flight_row(), flight_row(Cancelled=None), flight_row(Diverted="bad"),
            flight_row(Cancelled=2), flight_row(Cancelled=1), flight_row(Diverted=1),
            flight_row(Distance="bad"), flight_row(ArrDelay=None),
            flight_row(AirTime=float("inf")), flight_row(TailNum="B"),
            flight_row(Year=1990)]
    flights, planes = write_raw(tmp_path / "raw", rows)
    output = prepare_flight(flights, planes, tmp_path / "flight.npz")
    with np.load(output) as data:
        np.testing.assert_array_equal(data["X"], [[8, 100, 60, 1200, 1300, 1, 1, 1]])
        np.testing.assert_array_equal(data["y"], [7])
    record = json.loads(output.with_suffix(".json").read_text())
    assert record["imputation"] == "none"
    assert record["row_counts"] == {"input_rows": 11, "removed_invalid_flags": 3,
                                    "removed_cancelled_or_diverted": 2,
                                    "removed_invalid_features": 4, "removed_invalid_age": 1,
                                    "eligible_rows": 1, "sampled_rows": 1}


def test_flight_all_invalid_rows_raise_without_output(tmp_path):
    flights, planes = write_raw(tmp_path / "raw", [flight_row(Cancelled="bad")])
    output = tmp_path / "flight.npz"
    with pytest.raises(RuntimeError, match="No usable FLIGHT observations"):
        prepare_flight(flights, planes, output)
    assert not output.exists()


def test_manager_records_flight_cleaning_provenance(tmp_path):
    manager = DatasetManager(tmp_path)
    source = tmp_path / "airline.pickle"
    pd.DataFrame([{"plane_age": 8, "Distance": 100, "AirTime": 60, "DepTime": 1230,
                   "ArrTime": 1345, "DayOfWeek": 1, "DayofMonth": 1, "Month": 1,
                   "ArrDelay": 7}]).to_pickle(source)
    manager.import_file("flight", source)
    output = manager.prepare("flight")
    manifest = manager.manifest("flight")
    assert manifest["provenance"]["cleaning"]["row_counts"]["output_rows"] == 1
    with np.load(output) as data:
        np.testing.assert_array_equal(data["X"], [[8, 100, 60, 750, 825, 1, 1, 1]])
        np.testing.assert_array_equal(data["y"], [7])
    assert manager.load("flight").x.shape == (1, 8)
    assert any(entry["path"].endswith(output.with_suffix(".json").name)
               for entry in manifest["files"])


def test_author_download_rejects_html_without_using_raw_sources(tmp_path, monkeypatch):
    calls = []

    def download(url, destination):
        calls.append(url)
        destination.parent.mkdir(parents=True)
        destination.write_text("<html>shared_link_disabled</html>")
        return destination, url

    monkeypatch.setattr("pwgp.data.manager.download_file", download)
    with pytest.raises(ValueError, match="did not return a ZIP archive"):
        DatasetManager(tmp_path).download("flight")
    assert len(calls) == 1
    assert "dropbox.com" in calls[0]


def test_author_zip_is_the_only_download_and_extracts_expected_pickle(tmp_path, monkeypatch):
    import zipfile

    def download(url, destination):
        destination.parent.mkdir(parents=True)
        with zipfile.ZipFile(destination, "w") as bundle:
            bundle.writestr("airline.pickle", b"fixture")
        return destination, url

    monkeypatch.setattr("pwgp.data.manager.download_file", download)
    paths = DatasetManager(tmp_path).download("flight")
    assert paths[1].read_bytes() == b"fixture"
