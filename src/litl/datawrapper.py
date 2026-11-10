import os

import numpy as np
import torch

class DataWrapper():
  """
  A wrapper class for structured datasets

  Args:
    data (torch.Tensor or np.ndarray): The data to be wrapped. It can be a torch tensor or a numpy array.

  """
  def __init__(self, data: torch.Tensor | np.ndarray):
    if isinstance(data, np.ndarray):
      self.array = data.copy()
    elif isinstance(data, torch.Tensor):
      # detach and move to cpu first, so tensors on a GPU or tracked by autograd are accepted too
      self.array = data.detach().cpu().numpy().copy()
    else:
      raise ValueError(f'data must be a `torch.Tensor` or `numpy.ndarray`, but got {type(data)}')

  @classmethod
  def from_file(cls, path: str):
    """
    Create a new `DataWrapper` instance from a file. The file format can be .npy, .np, .pt, or .nc

    Args:
      path (str): The path to the file to load data from

    """
    extension = os.path.splitext(path)[-1][1:].lower()
    if extension == 'npy' or extension == 'np':
      array = np.load(path, allow_pickle=False)
      return cls(array)
    elif extension == 'pt':
      array = torch.load(path, weights_only=False)
      return cls(array)
    elif extension == 'nc':
      return cls.from_netcdf(path)
    else:
      raise ValueError(f'Unsupported file format to load from: .{extension}')

  @classmethod
  def from_netcdf(cls, path: str, variable: str=None):
    """
    Create a new `DataWrapper` instance from a NetCDF file.

    This function requires netCDF4, which is an optional dependency. Please install it with `pip install litl[netcdf]`

    Args:
      path (str): The path to the NetCDF file to load data from

      variable (str, optional): The name of the variable to load. If not provided, the variable named `data` is used \
      (this is what `save_netcdf` writes), or the first variable in the file if there is no `data` variable.
    """
    Dataset = _import_netcdf_dataset()

    with Dataset(path, 'r') as nc:
      if variable is None:
        if len(nc.variables) == 0:
          raise ValueError(f'NetCDF file {path} does not contain any variables')
        variable = 'data' if 'data' in nc.variables else next(iter(nc.variables))
      elif variable not in nc.variables:
        raise ValueError(f'NetCDF file {path} does not have a variable named "{variable}". Available variables: {list(nc.variables.keys())}')

      # netCDF4 returns a masked array when the variable has missing values, keep the raw values
      array = np.ma.getdata(nc.variables[variable][...])

    return cls(np.asarray(array))

  def save_to_file(self, path: str):
    """
    Save the data to a file. The file format can be .npy, .np, .pt, or .nc

    Args:
      path (str): The path to the file to save data to
    """
    extension = os.path.splitext(path)[-1][1:].lower()
    if extension == 'npy' or extension == 'np':
      # write through a file object: np.save would otherwise append ".npy" to a ".np" path
      with open(path, 'wb') as f:
        np.save(f, self.array, allow_pickle=False)
    elif extension == 'pt':
      torch.save(self.tensor(), path, _use_new_zipfile_serialization=False)
    elif extension == 'nc':
      self.save_netcdf(path)
    else:
      raise ValueError(f'Unsupported file format to save as: ".{extension}"')

  def save_netcdf(self, path: str, dimnames: list[str]=None):
    """
    Save the data to a NetCDF file, as a variable named `data`.

    This function requires netCDF4, which is an optional dependency. Please install it with `pip install litl[netcdf]`

    Args:
      path (str): The path to the NetCDF file to save data to

      dimnames (list[str], optional): The names of the dimensions. If not provided, default names will be used. \
      If the dataset has more than 4 dimensions, you must provide the names manually. \
      Note, that if you provide more names than dimensions, the first `len(dimnames) - len(self.array.shape)` names will be ignored.

    """
    Dataset = _import_netcdf_dataset()

    # come up with dimnames
    default_dimnames = ['t', 'z', 'y', 'x']
    if dimnames is None and len(self.array.shape) > len(default_dimnames):
      raise ValueError(f'Dataset has more than {len(default_dimnames)} dimensions, please provide their names manually through `dimnames` argument')
    elif dimnames is None:
      dimnames = default_dimnames
    elif len(dimnames) < len(self.array.shape):
      raise ValueError(f'Dataset has {len(self.array.shape)} dimensions, but only {len(dimnames)} names were provided through `dimnames` argument')

    lendiff = len(dimnames) - len(self.array.shape)
    used_dimnames = tuple(dimnames[lendiff:])
    with Dataset(path, 'w', format='NETCDF4') as nc:
      # create dimensions
      for name, dim in zip(used_dimnames, self.array.shape):
        nc.createDimension(name, dim)

      var = nc.createVariable('data', self.array.dtype, used_dimnames)
      var[...] = self.array

      var.units = 'unknown'
      var.description = 'Data saved by LITL'

  def tensor(self) -> torch.Tensor:
    """
    Convert the data to a torch tensor
    """
    return torch.from_numpy(self.array)

  def numpy(self) -> np.ndarray:
    return self.array

  def dim(self) -> int:
    return len(self.array.shape)

  def shape(self) -> tuple:
    return self.array.shape


def _import_netcdf_dataset():
  """
  Import netCDF4 only when it is needed, since it is an optional dependency
  """
  try:
    from netCDF4 import Dataset
  except ImportError:
    raise ImportError('This function requires netCDF4, which is an optional dependency. Please install it with `pip install litl[netcdf]`')
  return Dataset
