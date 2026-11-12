import json
import zipfile

import numpy as np
import pytest

from pwgp.data import benchmarks
from pwgp.data.download import file_checksum


def test_cached_source_checksum_failure_never_downloads(tmp_path, monkeypatch):
    path = tmp_path / benchmarks.SOURCES["california"][1]
    path.write_bytes(b"wrong source")

    def forbidden(*args, **kwargs):
        raise AssertionError("Corrupt cache must not cause a replacement download")

    monkeypatch.setattr("requests.get", forbidden)
    with pytest.raises(ValueError, match="checksum"):
        benchmarks.source_file("california", tmp_path)
    assert path.read_bytes() == b"wrong source"


def test_exact_protein_archive_schema(tmp_path, monkeypatch):
    path = tmp_path / "protein.zip"
    with zipfile.ZipFile(path, "x") as archive:
        archive.writestr("unexpected.csv", "RMSD,F1\n1,2\n")
    monkeypatch.setattr(benchmarks, "source_file", lambda *args: path)
    with pytest.raises(ValueError, match="archive members"):
        benchmarks.load_real("protein", cache=tmp_path, input_path=None)


def test_icethick_removes_malformed_rows_never_fills(tmp_path, monkeypatch):
    path = tmp_path / "icethk.txt"
    path.write_text("Season/Flight\tYear\tDay\tLongitude W\tLatitude S\tThickness (m)\n"
                    "a\t1990\t1\t1\t2\t3\n"
                    "b\t1990\t2\tbad\t2\t9\n"
                    "c\t1990\t3\t4\t5\t6\n")
    monkeypatch.setattr(benchmarks, "ICETHICK_MD5", file_checksum(path, "md5"))
    result = benchmarks.load_real("icethick", cache=tmp_path, input_path=path)
    np.testing.assert_array_equal(result.x, [[1, 2], [4, 5]])
    assert result.metadata["removed_nonfinite"] == 1


def test_explicit_converted_flight_format_and_provenance(tmp_path, monkeypatch):
    path = tmp_path / "airline.npz"
    np.savez_compressed(path, X=np.arange(32, dtype=float).reshape(4, 8), y=np.arange(4, dtype=float))
    checksum = file_checksum(path)
    provenance = {"source_sha256": benchmarks.FLIGHT_SHA256, "output_sha256": checksum,
                  "row_counts": {"output_rows": 4}}
    path.with_suffix(".json").write_text(json.dumps(provenance))
    monkeypatch.setattr(benchmarks, "FLIGHT_NPZ_SHA256", checksum)
    monkeypatch.setattr(benchmarks, "FLIGHT_PROVENANCE_SHA256", file_checksum(path.with_suffix(".json")))
    result = benchmarks.load_real("flight", cache=tmp_path, input_path=path)
    assert result.x.shape == (4, 8)
    assert result.metadata["conversion_provenance"] == provenance
    path.with_suffix(".json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        benchmarks.verify_flight_npz(path)


def test_conversion_script_never_modifies_source_or_runs_in_wrong_environment(tmp_path):
    from experiments.convert_legacy_flight import convert

    source = tmp_path / "airline.pickle"
    source.write_bytes(b"never unpickle in modern runtime")
    with pytest.raises(RuntimeError, match="isolated"):
        convert(source, tmp_path / "airline.npz")
    assert source.read_bytes() == b"never unpickle in modern runtime"


def test_unapproved_synthetic_values_raise():
    options = {"dimension": 3, "training_points": 80, "test_points": 20, "seed": 31,
               "noise_variance": 1e-8, "michalewicz_m": 10, "g_coefficients": [0, .5, 1]}
    with pytest.raises(ValueError, match="coefficients"):
        benchmarks.synthetic_dataset("g_function", **{**options, "g_coefficients": [0]})
    with pytest.raises(ValueError, match="positive integer"):
        benchmarks.synthetic_dataset("michalewicz", **{**options, "michalewicz_m": .5})
