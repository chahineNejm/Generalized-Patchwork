import hashlib
import io

import pytest

from pwgp.data import ukhousing


@pytest.mark.parametrize("matching", [False, True])
def test_download_requires_exact_pin_and_preserves_mismatch(tmp_path, monkeypatch, matching):
    content = b"explicit test source\n"
    actual = hashlib.sha256(content).hexdigest()
    expected = actual if matching else "0" * 64
    calls = []

    def response(request, timeout):
        calls.append(request.full_url)
        stream = io.BytesIO(content)
        stream.headers = {"Content-Length": str(len(content))}
        return stream

    monkeypatch.setattr(ukhousing, "urlopen", response)
    path = tmp_path / "prices.csv"
    if matching:
        assert ukhousing.acquire("https://example.invalid/source", path, expected) == actual
        assert path.read_bytes() == content
        assert not path.with_suffix(".csv.part").exists()
    else:
        with pytest.raises(ValueError, match="Source checksum mismatch") as error:
            ukhousing.acquire("https://example.invalid/source", path, expected)
        assert actual in str(error.value) and expected in str(error.value)
        assert str(path.with_suffix(".csv.part")) in str(error.value)
        assert not path.exists()
        assert path.with_suffix(".csv.part").read_bytes() == content
    assert calls == ["https://example.invalid/source"]


def test_old_cache_is_not_replaced_or_retried(tmp_path, monkeypatch):
    path = tmp_path / "prices.csv"
    path.write_bytes(b"previous snapshot")

    def forbidden(*args, **kwargs):
        raise AssertionError("Checksum mismatch must not trigger a replacement download")

    monkeypatch.setattr(ukhousing, "urlopen", forbidden)
    with pytest.raises(ValueError, match="Cached source checksum mismatch") as error:
        ukhousing.acquire("https://example.invalid/source", path, "0" * 64)
    assert hashlib.sha256(b"previous snapshot").hexdigest() in str(error.value)
    assert path.read_bytes() == b"previous snapshot"
