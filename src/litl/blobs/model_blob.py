import io

import torch

from . import Blob

class ModelBlob(Blob):
  def __init__(self, model: torch.nn.Module) -> None:
    self.model = model

  @classmethod
  def from_bytes(cls, data: bytes) -> 'ModelBlob':
    """
    Create a new ModelBlob instance from bytes

    Args:
      data (bytes): The bytes to be loaded into the model blob
    """
    # Load the model state dict from the bytes
    buffer = io.BytesIO(data)
    model = torch.load(buffer, map_location='cpu', weights_only=False)
    return cls(model)
  
  def to_bytes(self) -> bytes:
    """
    Convert the ModelBlob to bytes
    """
    buffer = io.BytesIO()
    torch.save(self.model, buffer)
    return buffer.getvalue()
