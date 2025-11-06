# __package__ = "litl.datawrapper"

import numpy as np
import torch
import os

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
      self.array = data.numpy().copy()
    else:
      raise ValueError(f'data must be a `torch.Tensor` or `numpy.ndarray`, but got {type(data)}')
    
    self.array = self.array.astype(np.float32)

  @classmethod
  def from_file(cls, path: str):
    """
    Create a new `DataWrapper` instance from a file. The file format can be either .npy, .np, or .pt

    Args:
      path (str): The path to the file to load data from
      
    """
    extension = os.path.splitext(path)[-1][1:]
    if extension == 'npy' or extension == 'np':
      array = np.load(path, allow_pickle=False)
      return cls(array)
    elif extension == 'pt':
      array = torch.load(path, weights_only=False)
      return cls(array)
    else:
      raise ValueError(f'Unsupported file format to load from: .{extension}')

  def save_to_file(self, path: str):
    """
    Save the data to a file. The file format can be either .npy, .np, or .pt

    Args:
      path (str): The path to the file to save data to
    """
    extension = os.path.splitext(path)[-1][1:]
    if extension == 'npy' or extension == 'np':
      np.save(path, self.array, allow_pickle=False)
    elif extension == 'pt':
      torch.save(self.tensor(), path, _use_new_zipfile_serialization=False)
    elif extension == 'nc':
      self.save_netcdf(path)
    else:
      raise ValueError(f'Unsupported file format to save as: ".{extension}"')
    
  def save_netcdf(self, path: str, dimnames: list[str]=None):
    """
    Save the data to a NetCDF file.

    This function requires netCDF4, which is an optional dependency. Please install it with `pip install netCDF4`

    Args:
      path (str): The path to the NetCDF file to save data to

      dimnames (list[str], optional): The names of the dimensions. If not provided, default names will be used. \
      If the dataset has more than 4 dimensions, you must provide the names manually. \
      Note, that if you provide more names than dimensions, the first `len(dimnames) - len(self.array.shape)` names will be ignored.

    """
    # import netCDF4 only if needed
    try:
      from netCDF4 import Dataset
    except ImportError:
      raise ImportError('This function requires netCDF4, which is an optional dependency. Please install it with `pip install netCDF4`')

    # come up with dimnames
    default_dimnames = ['t', 'z', 'y', 'x']
    if dimnames is None and len(self.array.shape) > len(default_dimnames):
      raise ValueError(f'Dataset has more than {len(default_dimnames)} dimensions, please provide their names manually through `dimnames` argument')
    elif dimnames is None:
      dimnames = default_dimnames
    elif len(dimnames) < len(self.array.shape):
      raise ValueError(f'Dataset has {len(self.array.shape)} dimensions, but only {len(dimnames)} names were provided through `dimnames` argument')
      
    lendiff = len(dimnames) - len(self.array.shape)
    with Dataset(path, 'w', format='NETCDF4') as nc:
      # create dimensions
      for i, dim in enumerate(self.array.shape):
        name = dimnames[lendiff + i]
        nc.createDimension(name, dim)

      var = nc.createVariable('data', self.array.dtype, ('dim',))
      var[:] = self.array.flatten()

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

  