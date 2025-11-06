# litl 🗜️

**L**ightweight **I**n**T**erface for **L**ossy compression

A flexible Python framework for implementing, testing, and deploying lossy compression algorithms.

```
  █                    █                  
▄▄█                    █▄▄                
   ██    ██ █████ ██                      
   ██       ██    ██                      
   ██    ██ ██    ██                      
   ██    ██ ██    ██                      
   ██    ██ ██    ██                      
   █████ ██ ██    █████                   
▀▀█                    █▀▀                
  █                    █             
```

## Features

- 🔧 **Simple Compressors Interface**: Simple and universal interface for implementing and sharing compressors
- 📊 **Built-in Evaluation**: Comprehensive quality metrics with batteries included (MSE, PSNR, SSIM, L1)
- 💾 **Standardized Storage Format**: Custom `.litl` file format for any compressor with user-defined metadata
- 🎯 **Multiple Data Formats**: Support for NumPy, PyTorch, and NetCDF
- 📈 **Rich CLI**: Beautiful command-line interface, in addition to rich Python API

## Installation

```bash
pip install litl
```

### Optional Dependencies

For NetCDF support:

```bash
pip install litl[netcdf]
```

## CLI

### 1. Compress Data

```bash
litl compress <path_to_compressor.py:CompressorClassName> <data_to_compress.npy> <config.json> <output.litl>
```

### 2. Decompress Data

```bash
litl decompress <output.litl> <decompressed.npy>
```

### 3. Evaluate Compression (compress and decompress, recording quality metrics)

```bash
litl evaluate <path_to_compressor.py:CompressorClassName> <data_to_compress.npy> <config.json>
```

### 4. Inspect Compressed Files

```bash
litl info <output.litl>
```

## Python API

### Compress Data

```python
import litl

compressed_blob, meta, metrics = litl.compress(
    compressor_name="path/to/compressor.py:ClassName",
    original_data_path="data/hurricane.npy",
    litl_file_path="path/to/compressed.litl"
)

print(f"Compressed data saved to {compressed_path}")
print(f"Metadata: {meta}")
print(f"Metrics: {metrics}")

litl.DotLitl.save(
    litl_file_path="path/to/compressed.litl",
    blob=compressed_blob,
    compressor_name="path/to/compressor.py:ClassName",
    meta=meta
)
```

### Decompress Data

```python
import litl

decompressed_data = litl.decompress(
    litl_file_path="path/to/compressed.litl",
    decompressed_path="data/decompressed.npy"
)

print(f"Decompressed data saved to {decompressed_path}")

data_numpy = decompressed_data.numpy()
```

## Creating a Custom Compressor

Implement the [`Compressor`](src/litl/compressors/compressor.py) interface:

```python
from litl.compressors import Compressor, CompressorAbout
from litl.datawrapper import DataWrapper
from litl.blobs import ArrayBlob, Blob
import pydantic

class MyCompressor(Compressor):
    
    @classmethod
    def about(cls) -> CompressorAbout:
        return CompressorAbout(
            name="MyCompressor",
            description="A simple lossy compressor",
            version="1.0.0",
            author="Your Name"
        )
    
    @classmethod
    def blob_class(cls):
        # Specify the blob class used for compressed data. This type will be passed into your decompressed function
        return ArrayBlob
    
    @classmethod
    def compress(cls, data: DataWrapper, config: dict) -> tuple[Blob, dict]:
        # Your compression logic here

        # Return a blob containing compressed data.
        # Built-ins are:
        # - ArrayBlob for NumPy and PyTorch arrays
        # - ModelBlob for PyTorch models
        # - ByteBlob for byte arrays
        # If none of these work - extend the Blob class and implement a custom blob

        meta = {}  # Metadata required to reconstruct the data, can be any serializable dictionary
        return ArrayBlob(compressed_array), meta
    
    @classmethod
    def decompress(cls, blob: ArrayBlob, meta: dict) -> DataWrapper:
        # Your decompression logic here

        # Return Reconstructed data wrapped in DataWrapper
        return DataWrapper(decompressed)
```

## DataWrapper

The [`DataWrapper`](src/litl/datawrapper.py) class supports multiple formats for loading and saving data:

- **NumPy**: `.npy`, `.np` files
- **PyTorch**: `.pt` files  
- **NetCDF**: `.nc` files (requires `netCDF4`)

```python
from litl import DataWrapper

# Load data
data = DataWrapper.from_file("data.npy")

# Save data
data.save_to_file("output.nc")

# Access underlying arrays
numpy_array = data.numpy()
torch_tensor = data.tensor()
```

## Built-in Blob Types

Choose the appropriate blob type for your compressed data:

- [`ArrayBlob`](src/litl/blobs/array_blob.py): For NumPy arrays
- [`BytesBlob`](src/litl/blobs/bytes_blob.py): For raw bytes
- [`ModelBlob`](src/litl/blobs/model_blob.py): For PyTorch models

## Evaluation Metrics

The [`Evaluator`](src/litl/evaluator.py) provides standard compression metrics:

**Quality Metrics:**

- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- L1 Loss

**Size Metrics:**

- Original size
- Compressed size
- Compression ratio

## File Format

The `.litl` format ([`DotLitl`](src/litl/dot_litl.py)) stores:

- Compressed data with Zstandard compression
- Compressor metadata and version
- Integrity checksums
- Configuration parameters

## License

MIT License - see [LICENSE](LICENSE) for details.
