import math

import numpy as np
import pytest

from litl import DataWrapper, Evaluator
from litl.blobs import BytesBlob


def test_identical_data_is_a_perfect_reconstruction(field):
    metrics = Evaluator.quality(DataWrapper(field), DataWrapper(field))

    assert set(metrics) == {"mse", "psnr", "ssim", "l1"}
    assert metrics["mse"] == 0
    assert metrics["l1"] == 0
    assert metrics["psnr"] == math.inf
    assert metrics["ssim"] == pytest.approx(1.0)


def test_known_error():
    # values between 0 and 1, reconstruction is off by exactly 0.1 everywhere
    original = np.linspace(0, 1, 16 * 16 * 16, dtype=np.float64).reshape(16, 16, 16)
    reconstructed = original + 0.1
    metrics = Evaluator.quality(DataWrapper(original), DataWrapper(reconstructed))

    assert metrics["mse"] == pytest.approx(0.01)
    assert metrics["l1"] == pytest.approx(0.1)
    # PSNR = 10 * log10(data_range^2 / MSE), with the maximum of the original data (1.0) as the data range
    assert metrics["psnr"] == pytest.approx(20.0, abs=1e-3)
    assert 0 < metrics["ssim"] < 1


def test_more_noise_is_worse(field):
    rng = np.random.default_rng(0)
    slightly_noisy = field + rng.normal(0, 0.01, field.shape).astype(np.float32)
    very_noisy = field + rng.normal(0, 0.2, field.shape).astype(np.float32)

    good = Evaluator.quality(DataWrapper(field), DataWrapper(slightly_noisy))
    bad = Evaluator.quality(DataWrapper(field), DataWrapper(very_noisy))

    assert good["mse"] < bad["mse"]
    assert good["l1"] < bad["l1"]
    assert good["psnr"] > bad["psnr"]
    assert good["ssim"] > bad["ssim"]


def test_integer_data():
    original = (np.arange(16 * 16 * 16) % 251).astype(np.uint8).reshape(16, 16, 16)
    reconstructed = original.copy()
    reconstructed[0, 0, 0] += 4  # one value out of 4096 is off by 4
    metrics = Evaluator.quality(DataWrapper(original), DataWrapper(reconstructed))

    assert metrics["mse"] == pytest.approx(16 / 4096)
    assert metrics["l1"] == pytest.approx(4 / 4096)
    assert metrics["psnr"] == pytest.approx(10 * math.log10(250 ** 2 / (16 / 4096)), abs=1e-3)


def test_mismatched_float_types(field):
    metrics = Evaluator.quality(DataWrapper(field.astype(np.float64)), DataWrapper(field))
    assert metrics["mse"] == 0


def test_data_without_positive_values(field):
    # the maximum is not a usable data range here, the value range is used instead
    original = -np.abs(field) - 1
    reconstructed = original + 0.01
    metrics = Evaluator.quality(DataWrapper(original), DataWrapper(reconstructed))

    value_range = float(original.max() - original.min())
    assert metrics["mse"] == pytest.approx(1e-4, rel=1e-3)
    assert metrics["psnr"] == pytest.approx(10 * math.log10(value_range ** 2 / 1e-4), abs=1e-2)
    assert 0 < metrics["ssim"] <= 1


def test_constant_data():
    original = np.zeros((8, 8, 8), dtype=np.float32)
    metrics = Evaluator.quality(DataWrapper(original), DataWrapper(original))
    assert metrics["mse"] == 0


def test_2d_data():
    rng = np.random.default_rng(0)
    original = rng.random((32, 32)).astype(np.float32)
    metrics = Evaluator.quality(DataWrapper(original), DataWrapper(original * 0.9))
    assert 0 < metrics["ssim"] < 1


def test_1d_data_has_no_ssim():
    original = np.linspace(0, 1, 100, dtype=np.float32)
    with pytest.warns(UserWarning, match="SSIM"):
        metrics = Evaluator.quality(DataWrapper(original), DataWrapper(original + 0.1))

    assert math.isnan(metrics["ssim"])
    assert metrics["mse"] == pytest.approx(0.01)
    assert metrics["l1"] == pytest.approx(0.1)


def test_shape_mismatch(field):
    with pytest.raises(ValueError, match="same shape"):
        Evaluator.quality(DataWrapper(field), DataWrapper(field[:-1]))


def test_requires_data_wrappers(field):
    with pytest.raises(TypeError):
        Evaluator.quality(field, field)


def test_size():
    metrics = Evaluator.size(b"x" * 250, original_size=1000)
    assert metrics == {"original_size": 1000, "compressed_size": 250, "compression_ratio": 4.0}


def test_size_of_a_blob():
    blob_bytes = BytesBlob(b"0123456789").to_bytes()
    assert Evaluator.size(blob_bytes, original_size=40)["compression_ratio"] == 4.0


def test_methods_work_on_an_instance(field):
    evaluator = Evaluator()
    assert evaluator.size(b"abcd", 8)["compression_ratio"] == 2.0
    assert evaluator.quality(DataWrapper(field), DataWrapper(field))["mse"] == 0
