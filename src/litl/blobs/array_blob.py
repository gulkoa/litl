import io

import numpy as np

from . import Blob

class ArrayBlob(Blob):
  def __init__(self, data: np.ndarray) -> None:
    if isinstance(data, np.ndarray):
      self.array = data
    else:
      raise ValueError(f'Unsupported data type {type(data)}. Expected numpy.ndarray')
    
  @classmethod
  def from_bytes(cls, data: bytes) -> 'ArrayBlob':
    """
    Create a new ArrayBlob instance from bytes
    """
    # load the numpy array from bytes, keep the original dtype and shape

    return cls(np.load(io.BytesIO(data), allow_pickle=True))
  
  def to_bytes(self) -> bytes:
    """
    Convert the ArrayBlob to bytes
    """
    with io.BytesIO() as buffer:
      # Save the numpy array to bytes
      np.save(buffer, self.array, allow_pickle=True)
      data = buffer.getvalue()

    return data