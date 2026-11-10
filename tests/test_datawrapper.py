import os

import numpy as np
import pytest
import torch

from litl import DataWrapper


def test_wraps_a_copy_of_a_numpy_array(field):
    wrapper = DataWrapper(field)
    assert wrapper.numpy() is not field
    np.testing.assert_array_equal(wrapper.numpy(), field)
    assert wrapper.shape() == field.shape
    assert wrapper.dim() == 3


def test_wraps_a_tensor(field):
    wrapper = DataWrapper(torch.from_numpy(field))
    np.testing.assert_array_equal(wrapper.numpy(), field)
    assert isinstance(wrapper.tensor(), torch.Tensor)
    assert wrapper.tensor().dtype == torch.float32


def test_wraps_a_tensor_that_requires_grad():
    tensor = torch.linspace(0, 1, 6, requires_grad=True).reshape(2, 3) * 2
    wrapper = DataWrapper(tensor)
    np.testing.assert_allclose(wrapper.numpy(), tensor.detach().numpy())


def test_rejects_other_types():
    with pytest.raises(ValueError):
        DataWrapper([1.0, 2.0, 3.0])


@pytest.mark.parametrize("extension", ["npy", "np", "pt"])
def test_file_round_trip(tmp_path, field, extension):
    path = str(tmp_path / f"field.{extension}")
    DataWrapper(field).save_to_file(path)

    # the file is written exactly where it was asked for
    assert os.listdir(tmp_path) == [f"field.{extension}"]

    restored = DataWrapper.from_file(path)
    assert restored.numpy().dtype == field.dtype
    np.testing.assert_array_equal(restored.numpy(), field)


def test_unsupported_extensions(tmp_path, field):
    with pytest.raises(ValueError, match="Unsupported file format"):
        DataWrapper(field).save_to_file(str(tmp_path / "field.csv"))
    with pytest.raises(ValueError, match="Unsupported file format"):
        DataWrapper.from_file(str(tmp_path / "field.csv"))


def test_netcdf_round_trip(tmp_path, field):
    netCDF4 = pytest.importorskip("netCDF4")
    path = str(tmp_path / "field.nc")
    DataWrapper(field).save_to_file(path)

    with netCDF4.Dataset(path) as nc:
        # a 3D array uses the last three default dimension names
        assert nc.variables["data"].dimensions == ("z", "y", "x")
        assert nc.variables["data"].shape == field.shape

    restored = DataWrapper.from_file(path)
    assert restored.numpy().dtype == field.dtype
    np.testing.assert_array_equal(restored.numpy(), field)


def test_netcdf_custom_dimension_names(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    array = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.float64).reshape(2, 3, 4, 5, 6)
    path = str(tmp_path / "five_dims.nc")

    with pytest.raises(ValueError, match="more than 4 dimensions"):
        DataWrapper(array).save_netcdf(path)
    with pytest.raises(ValueError, match="only 2 names"):
        DataWrapper(array).save_netcdf(path, dimnames=["a", "b"])

    DataWrapper(array).save_netcdf(path, dimnames=["a", "b", "c", "d", "e"])
    with netCDF4.Dataset(path) as nc:
        assert nc.variables["data"].dimensions == ("a", "b", "c", "d", "e")
    np.testing.assert_array_equal(DataWrapper.from_netcdf(path).numpy(), array)


def test_netcdf_reads_the_first_variable_without_a_data_variable(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    path = str(tmp_path / "other.nc")
    temperature = np.linspace(250, 300, 12, dtype=np.float32).reshape(3, 4)
    pressure = temperature * 2
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("y", 3)
        nc.createDimension("x", 4)
        nc.createVariable("temperature", "f4", ("y", "x"))[...] = temperature
        nc.createVariable("pressure", "f4", ("y", "x"))[...] = pressure

    np.testing.assert_array_equal(DataWrapper.from_file(path).numpy(), temperature)
    np.testing.assert_array_equal(DataWrapper.from_netcdf(path, variable="pressure").numpy(), pressure)
    with pytest.raises(ValueError, match="does not have a variable"):
        DataWrapper.from_netcdf(path, variable="humidity")
