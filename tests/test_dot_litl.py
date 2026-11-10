import struct
import warnings

import pytest
import zstd

from litl import DotLitl

BLOB = b"some compressed bytes " * 50
META = {"bits": 8, "min": -1.5, "max": 1.5, "nested": {"shape": [2, 3]}}


@pytest.fixture
def litl_file(tmp_path):
    path = str(tmp_path / "data.litl")
    DotLitl.save(path, BLOB, META, "my_compressor", "1.2.3")
    return path


def test_round_trip(litl_file):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a valid file must not trigger the checksum warning
        blob, compressor_name, compressor_version, meta, litl_version = DotLitl.read(litl_file)

    assert blob == BLOB
    assert compressor_name == "my_compressor"
    assert compressor_version == "1.2.3"
    assert meta == META
    assert isinstance(litl_version, str) and litl_version


def test_save_returns_file_size(tmp_path):
    path = tmp_path / "data.litl"
    total_size = DotLitl.save(str(path), BLOB, META, "my_compressor", "1.2.3")
    assert total_size == path.stat().st_size


def test_byte_layout(litl_file):
    with open(litl_file, "rb") as f:
        raw = f.read()

    # 4 bytes magic, 1 byte version, 4 bytes checksum, 4 bytes meta length, meta, blob
    assert raw[:4] == b"LITL"
    assert raw[4] == DotLitl.version == 1
    checksum, meta_length = struct.unpack("<II", raw[5:13])
    meta_bytes = raw[13:13 + meta_length]
    blob_bytes = raw[13 + meta_length:]

    assert checksum == DotLitl.checksum(blob_bytes, meta_bytes)
    assert zstd.decompress(blob_bytes) == BLOB
    assert b"my_compressor" in zstd.decompress(meta_bytes)


def test_empty_meta_and_blob(tmp_path):
    path = str(tmp_path / "empty.litl")
    DotLitl.save(path, b"", {}, "c", "0")
    blob, _, _, meta, _ = DotLitl.read(path)
    assert blob == b""
    assert meta == {}


def test_bad_magic(tmp_path):
    path = tmp_path / "not_litl.litl"
    path.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(ValueError, match="Invalid file format"):
        DotLitl.read(str(path))


def test_unsupported_version(litl_file):
    with open(litl_file, "r+b") as f:
        f.seek(4)
        f.write(struct.pack("<B", DotLitl.version + 1))
    with pytest.raises(ValueError, match="Unsupported version"):
        DotLitl.read(litl_file)


@pytest.mark.parametrize("keep", [4, 8, 20])
def test_truncated_file(litl_file, keep):
    with open(litl_file, "rb") as f:
        raw = f.read()
    with open(litl_file, "wb") as f:
        f.write(raw[:keep])
    with pytest.raises(ValueError, match="truncated"):
        DotLitl.read(litl_file)


def test_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Failed to read file"):
        DotLitl.read(str(tmp_path / "missing.litl"))


def test_checksum_mismatch_warns(litl_file):
    # flip one bit of the stored checksum: the payload itself is intact and must still be returned
    with open(litl_file, "r+b") as f:
        f.seek(5)
        byte = f.read(1)
        f.seek(5)
        f.write(bytes([byte[0] ^ 0x01]))

    with pytest.warns(UserWarning, match="Checksum mismatch"):
        blob, _, _, meta, _ = DotLitl.read(litl_file)
    assert blob == BLOB
    assert meta == META


def test_corrupted_payload_warns(litl_file):
    # flip the last byte of the compressed blob
    with open(litl_file, "r+b") as f:
        f.seek(-1, 2)
        byte = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([byte[0] ^ 0xFF]))

    with pytest.warns(UserWarning, match="Checksum mismatch"):
        try:
            blob, _, _, _, _ = DotLitl.read(litl_file)
        except zstd.Error:
            # the corruption may also make the payload undecodable, the warning comes first either way
            blob = None
    assert blob != BLOB
