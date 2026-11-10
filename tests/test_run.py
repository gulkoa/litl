import math
import os
import textwrap

import numpy as np
import pytest

import litl
from litl import DataWrapper, DotLitl
from litl.blobs import ArrayBlob
from litl.compressors import Compressor, CompressorAbout
from litl.run import get_compressor

SIZE_METRICS = {"original_size", "compressed_size", "compression_ratio"}
QUALITY_METRICS = {"mse", "psnr", "ssim", "l1"}


def write_script(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    return str(path)


# --- loading compressors ---

def test_get_compressor_from_a_file(quantize):
    compressor = get_compressor(quantize)

    assert issubclass(compressor, Compressor)
    assert compressor.blob_class() is ArrayBlob
    about = compressor.about()
    assert isinstance(about, CompressorAbout)
    assert about.name == "Quantize"
    assert about.author is None


def test_get_compressor_unknown_name():
    with pytest.raises(ValueError, match="not found in entry points"):
        get_compressor("definitely_not_installed")


def test_get_compressor_missing_file(tmp_path):
    with pytest.raises(ImportError, match="does not exist"):
        get_compressor(str(tmp_path / "missing.py") + ":Anything")


def test_get_compressor_missing_class(quantize):
    path = quantize.rsplit(":", 1)[0]
    with pytest.raises(ValueError, match="does not have class"):
        get_compressor(path + ":NotThere")


def test_get_compressor_not_a_compressor(tmp_path):
    path = write_script(tmp_path, "not_a_compressor.py", """
        class JustAClass:
            pass

        not_even_a_class = 42
    """)
    with pytest.raises(ValueError, match="not a subclass of Compressor"):
        get_compressor(path + ":JustAClass")
    with pytest.raises(ValueError, match="not a subclass of Compressor"):
        get_compressor(path + ":not_even_a_class")


def test_errors_in_the_compressor_script_are_not_hidden(tmp_path):
    path = write_script(tmp_path, "broken.py", """
        raise ValueError("boom inside the compressor script")
    """)
    with pytest.raises(ValueError, match="boom inside the compressor script"):
        get_compressor(path + ":Anything")


# --- compress / decompress / evaluate ---

def test_compress_to_a_litl_file(tmp_path, quantize, field, field_path):
    litl_path = str(tmp_path / "field.litl")
    blob, meta, metrics = litl.compress(quantize, field_path, config={"bits": 8}, litl_file_path=litl_path)

    assert isinstance(blob, ArrayBlob)
    assert blob.array.dtype == np.uint8
    assert blob.array.shape == field.shape
    assert meta == {"bits": 8, "min": float(field.min()), "max": float(field.max()), "dtype": "float32"}

    assert set(metrics) == SIZE_METRICS | {"compression_time", "io_time", "total_litl_size"}
    assert metrics["original_size"] == os.path.getsize(field_path)
    assert metrics["compressed_size"] == len(blob.to_bytes())
    assert metrics["compression_ratio"] == pytest.approx(metrics["original_size"] / metrics["compressed_size"])
    # float32 to uint8 is close to 4x, a bit less because of the .npy headers
    assert 3.5 < metrics["compression_ratio"] < 4.0
    assert metrics["total_litl_size"] == os.path.getsize(litl_path)

    # what is in the file is what compress returned
    blob_bytes, compressor_name, compressor_version, stored_meta, _ = DotLitl.read(litl_path)
    assert blob_bytes == blob.to_bytes()
    assert compressor_name == quantize
    assert compressor_version == "1.0.0"
    assert stored_meta == meta


def test_compress_without_a_file(quantize, field_path):
    blob, meta, metrics = litl.compress(quantize, field_path, config={"bits": 4})
    assert "total_litl_size" not in metrics
    assert blob.array.max() <= 15


def test_decompress_a_litl_file(tmp_path, quantize, field, field_path):
    litl_path = str(tmp_path / "field.litl")
    decompressed_path = str(tmp_path / "decompressed.npy")
    litl.compress(quantize, field_path, config={"bits": 8}, litl_file_path=litl_path)

    data, metrics = litl.decompress(litl_file_path=litl_path, decompressed_path=decompressed_path, original_data_path=field_path)

    assert isinstance(data, DataWrapper)
    assert data.numpy().dtype == field.dtype
    assert data.shape() == field.shape
    # uniform quantization is off by at most half a step
    step = (field.max() - field.min()) / 255
    assert np.abs(data.numpy() - field).max() <= step / 2 + 1e-6

    assert set(metrics) == QUALITY_METRICS | SIZE_METRICS | {"decompression_time"}
    assert metrics["psnr"] > 40
    assert metrics["ssim"] > 0.99

    np.testing.assert_array_equal(np.load(decompressed_path), data.numpy())


def test_decompress_without_the_original_has_no_quality_metrics(tmp_path, quantize, field_path):
    litl_path = str(tmp_path / "field.litl")
    litl.compress(quantize, field_path, config={"bits": 8}, litl_file_path=litl_path)

    data, metrics = litl.decompress(litl_file_path=litl_path)
    assert set(metrics) == {"decompression_time"}


def test_decompress_from_bytes(quantize, field, field_path):
    blob, meta, _ = litl.compress(quantize, field_path, config={"bits": 12})
    data, metrics = litl.decompress(quantize, compressed_bytes=blob.to_bytes(), meta=meta)

    step = (field.max() - field.min()) / (2 ** 12 - 1)
    assert np.abs(data.numpy() - field).max() <= step / 2 + 1e-6


def test_decompress_needs_some_input(quantize):
    with pytest.raises(ValueError, match="must be provided"):
        litl.decompress(quantize)


def test_decompress_checks_the_compressor_version(tmp_path, field_path):
    source = """
        import numpy as np
        from litl import DataWrapper
        from litl.blobs import ArrayBlob
        from litl.compressors import Compressor, CompressorAbout

        class Identity(Compressor):
            @classmethod
            def about(cls):
                return CompressorAbout(name="Identity", description="Stores the data as is", version="{version}")

            @classmethod
            def blob_class(cls):
                return ArrayBlob

            @classmethod
            def compress(cls, data, config):
                return ArrayBlob(data.numpy()), {{}}

            @classmethod
            def decompress(cls, blob, meta):
                return DataWrapper(blob.array)
    """
    script = write_script(tmp_path, "identity.py", source.format(version="1.0.0"))
    litl_path = str(tmp_path / "field.litl")
    litl.compress(script + ":Identity", field_path, config={}, litl_file_path=litl_path)

    data, _ = litl.decompress(litl_file_path=litl_path)
    np.testing.assert_array_equal(data.numpy(), np.load(field_path))

    # the same file can not be decompressed by another version of the compressor
    # (the new version string has a different length, so that python does not reuse the cached bytecode of the script)
    write_script(tmp_path, "identity.py", source.format(version="2.0.0-beta"))
    with pytest.raises(ValueError, match="version mismatch"):
        litl.decompress(litl_file_path=litl_path)


def test_evaluate(tmp_path, quantize, field, field_path):
    litl_path = str(tmp_path / "field.litl")
    decompressed_path = str(tmp_path / "decompressed.pt")
    data, metrics = litl.evaluate(quantize, field_path, config={"bits": 8}, decompressed_path=decompressed_path, litl_file_path=litl_path)

    assert data.shape() == field.shape
    assert set(metrics) == QUALITY_METRICS | SIZE_METRICS | {"compression_time", "io_time", "decompression_time", "total_litl_size"}
    assert all(isinstance(value, (int, float)) for value in metrics.values())
    assert os.path.exists(litl_path)
    np.testing.assert_array_equal(DataWrapper.from_file(decompressed_path).numpy(), data.numpy())


def test_evaluate_without_saving_anything(tmp_path, quantize, field_path):
    data, metrics = litl.evaluate(quantize, field_path, config={"bits": 8})
    assert "total_litl_size" not in metrics
    assert sorted(os.listdir(tmp_path)) == ["field.npy"]


def test_more_bits_give_better_quality(quantize, field_path):
    psnr = [litl.evaluate(quantize, field_path, config={"bits": bits})[1]["psnr"] for bits in (2, 4, 8, 16)]
    assert psnr == sorted(psnr)
    assert all(math.isfinite(value) for value in psnr)
    # each extra bit halves the quantization step, which is worth about 6 dB
    assert psnr[2] - psnr[1] == pytest.approx(4 * 6.02, abs=2.5)


def test_integer_data_round_trip(tmp_path, quantize):
    original = (np.arange(16 * 16 * 16) % 200).astype(np.int16).reshape(16, 16, 16)
    path = str(tmp_path / "ints.npy")
    np.save(path, original)

    data, metrics = litl.evaluate(quantize, path, config={"bits": 8})
    # 200 distinct values fit into 8 bits, so nothing is lost
    assert data.numpy().dtype == np.int16
    np.testing.assert_array_equal(data.numpy(), original)
    assert metrics["mse"] == 0


def test_example_compressor_rejects_bad_config(quantize, field_path):
    with pytest.raises(ValueError, match="bits must be between 1 and 16"):
        litl.compress(quantize, field_path, config={"bits": 32})
