class Blob():
  """
  Blob is an abstract class that defines the interface for compressed data

  You don't need to store metadata in the blob, return it from compress method along with blob instead
  """

  def __init__(self, *args, **kwargs) -> None:
    """
    Initialize a blob with the compressed data
    """
    raise NotImplementedError()

  @classmethod
  def from_bytes(cls, data: bytes) -> 'Blob':
    """
    Create a new blob instance from bytes
    """
    raise NotImplementedError()
  
  def to_bytes(self) -> bytes:
    """
    Convert the blob to bytes.
    """
    raise NotImplementedError()