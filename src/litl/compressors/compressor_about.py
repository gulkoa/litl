from typing import Optional

import pydantic

class CompressorAbout(pydantic.BaseModel):
  """
  Pydantic model that defines the metadata for a compressor.
  """
  name: str
  description: str
  version: str
  author: Optional[str] = None
  license: Optional[str] = None
  url: Optional[str] = None
