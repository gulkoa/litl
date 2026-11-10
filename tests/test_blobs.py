import numpy as np
import pytest
import torch

from litl.blobs import ArrayBlob, Blob, BytesBlob, ModelBlob


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int64, np.float32, np.float64])
def test_array_blob_round_trip(dtype):
    array = (np.arange(24).reshape(2, 3, 4) * 3).astype(dtype)
    restored = ArrayBlob.from_bytes(ArrayBlob(array).to_bytes())

    assert isinstance(restored, ArrayBlob)
    assert restored.array.dtype == array.dtype
    assert restored.array.shape == array.shape
    np.testing.assert_array_equal(restored.array, array)


def test_array_blob_rejects_other_types():
    with pytest.raises(ValueError):
        ArrayBlob([1, 2, 3])


def test_bytes_blob_round_trip():
    data = bytes(range(256)) * 4
    restored = BytesBlob.from_bytes(BytesBlob(data).to_bytes())

    assert isinstance(restored, BytesBlob)
    assert restored.data == data


def test_model_blob_round_trip():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    restored = ModelBlob.from_bytes(ModelBlob(model).to_bytes())

    assert isinstance(restored, ModelBlob)
    x = torch.randn(5, 3)
    torch.testing.assert_close(restored.model(x), model(x))
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], parameter)


def test_blob_is_an_interface():
    with pytest.raises(NotImplementedError):
        Blob()
    with pytest.raises(NotImplementedError):
        Blob.from_bytes(b"")


@pytest.mark.parametrize("blob_class", [ArrayBlob, BytesBlob, ModelBlob])
def test_built_in_blobs_are_blobs(blob_class):
    assert issubclass(blob_class, Blob)
