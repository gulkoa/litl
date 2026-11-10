import hashlib
import json
import os
import struct
import warnings
from importlib import metadata as importlib_metadata

import pydantic
import zstd

class LitlMeta(pydantic.BaseModel):
    litl_version: str
    compressor_id: str
    compressor_version: str
    compressor_meta: dict

    def serialize_value(self) -> str:
        """
        Serialize the metadata to a string
        """
        li = list(self.model_dump().values())
        return json.dumps(li)
    
    @classmethod
    def deserialize_value(cls, value: str) -> 'LitlMeta':
        """
        Deserialize the metadata from a string
        """
        try:
            values_list = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to deserialize .litl metadata: {e}")
        
        if len(values_list) != len(cls.model_fields):
            raise ValueError(f"Failed to deserialize .litl metadata: expected {len(cls.model_fields)} values, got {len(values_list)}")
        
        values = dict(zip(cls.model_fields.keys(), values_list))
        return cls(**values)

class DotLitl:
    """
    A class to save and read .litl files
    """
    version = 1

    @classmethod
    def zstd_compress_bytes(cls, data: bytes) -> bytes:
        """
        Compress the data using Zstandard
        """
        level = 3
        compressed = zstd.compress(data, level)
        return compressed
    
    @classmethod
    def zstd_decompress_bytes(cls, data: bytes) -> bytes:
        """
        Decompress the data using Zstandard
        """
        return zstd.decompress(data)
    

    @classmethod
    def make_meta(cls, meta: dict, compressor_id: str, compressor_version: str) -> str:
        """
        Create the serialized metadata string that is stored in the .litl file
        """
        litl_version = importlib_metadata.version("litl")
        
        litl_meta = LitlMeta(
            litl_version=litl_version,
            compressor_id=compressor_id,
            compressor_version=compressor_version,
            compressor_meta=meta
        ).serialize_value()
        return litl_meta
    
    @classmethod
    def checksum(cls, blob_bytes: bytes, meta_bytes: bytes) -> int:
        """
        Calculate a checksum for the data
        """
        hasher = hashlib.sha256(usedforsecurity=False)
        hasher.update(blob_bytes)
        hasher.update(meta_bytes)
        checksum = hasher.digest()[:4]
        return int.from_bytes(checksum, byteorder="little")

    @classmethod
    def save(cls, path: str, blob_bytes: bytes, meta: dict, compressor_name: str, compressor_version: str) -> int:
        """
        Save the blob to a .litl file

        Byte structure is as follows:
        - 4 bytes: magic number (LITL)
        - 1 byte: version
        - 4 bytes: checksum
        - 4 bytes: length of meta
        - meta: metadata (as bytes)
        - blob: blob data (as bytes)
        
        Args:
            path (str): The path to save the .litl file
            blob_bytes (bytes): The compressed blob data
            meta (dict): Metadata to be saved
            compressor_name (str): Name of the compressor used
            compressor_version (str): Version of the compressor used
            
        Returns:
            int: The total size of the saved .litl file
        """
        meta_bytes = cls.zstd_compress_bytes(cls.make_meta(meta, compressor_name, compressor_version).encode('utf-8'))
        blob_bytes = cls.zstd_compress_bytes(blob_bytes)
        checksum = cls.checksum(blob_bytes, meta_bytes)

        with open(path, 'wb') as f:
            f.write(b"LITL")
            f.write(struct.pack("<B", cls.version))
            f.write(struct.pack("<I", checksum))
            f.write(struct.pack("<I", len(meta_bytes)))
            f.write(meta_bytes)
            f.write(blob_bytes)

        total_size = os.path.getsize(path)

        return total_size

    @classmethod
    def read(cls, path: str) -> tuple[bytes, str, str, dict, str]:
        """
        Read the blob and metadata from a .litl file
        """
        try:
            with open(path, 'rb') as f:
                magic = f.read(4)
                if magic != b"LITL":
                    raise ValueError("Invalid file format")

                # version (1 byte), checksum (4 bytes), length of meta (4 bytes)
                header = f.read(9)
                if len(header) != 9:
                    raise ValueError("Invalid file format: file is truncated")

                version = struct.unpack("<B", header[0:1])[0]
                checksum = struct.unpack("<I", header[1:5])[0]

                if version != cls.version:
                    raise ValueError(f"Unsupported version: {version}")
                
                meta_length = struct.unpack("<I", header[5:9])[0]
                meta_bytes = f.read(meta_length)
                if len(meta_bytes) != meta_length:
                    raise ValueError("Invalid file format: file is truncated")
                blob_bytes = f.read()

            # checksum validation
            new_checksum = cls.checksum(blob_bytes, meta_bytes)
            if new_checksum != checksum:
                 warnings.warn(
                    f"Checksum mismatch in file {path}: expected {checksum:08x}, "
                    f"got {new_checksum:08x}. File may be corrupted or tampered with.",
                    UserWarning
                )
            
            # decompress meta
            blob_bytes = cls.zstd_decompress_bytes(blob_bytes)
            meta_bytes = cls.zstd_decompress_bytes(meta_bytes)

            meta = LitlMeta.deserialize_value(meta_bytes.decode('utf-8'))

            return blob_bytes, meta.compressor_id, meta.compressor_version, meta.compressor_meta, meta.litl_version
        except IOError as e:
            raise ValueError(f"Failed to read file {path}: {e}")