from . import Blob

class BytesBlob(Blob):
  def __init__(self, data: bytes) -> None:
    """
    Initialize a BytesBlob with the given data
    
    Args:
      data (bytes): The data to be stored in the blob
    """
    self.data = data

  @classmethod
  def from_bytes(cls, data: bytes) -> 'BytesBlob':
    """
    Create a new BytesBlob instance from bytes
    
    Args:
      data (bytes): The data to be stored in the blob
    
    Returns:
      BytesBlob: A new instance of BytesBlob containing the data
    """
    return cls(data)
  
  def to_bytes(self) -> bytes:
    """
    Convert the BytesBlob to bytes
    
    Returns:
      bytes: The data stored in the blob
    """
    return self.data
