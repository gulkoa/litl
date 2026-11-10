# litl 🗜️

[![PyPI](https://img.shields.io/pypi/v/litl)](https://pypi.org/project/litl/)
[![Tests](https://github.com/GulkoA/litl/actions/workflows/tests.yml/badge.svg)](https://github.com/GulkoA/litl/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**L**ightweight **I**n**T**erface for **L**ossy compression

A flexible Python framework for implementing, testing, and deploying lossy compression algorithms.

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

litl requires Python 3.10 or newer.

### Optional Dependencies

For NetCDF support:

```bash
pip install litl[netcdf]
```

## Quickstart

A compressor is a single Python class. [`examples/quantize.py`](examples/quantize.py) is a tiny one: it rounds every value to one of 2<sup>bits</sup> evenly spaced levels. Let's try it on a smooth 3D field:

```bash
git clone https://github.com/GulkoA/litl
cd litl

# some data to compress, and a config for the compressor
python -c "
import numpy as np
x, y, z = np.meshgrid(*[np.linspace(0, 2 * np.pi, 64)] * 3, indexing='ij')
np.save('field.npy', (np.sin(x) * np.cos(y) + 0.5 * np.sin(2 * z)).astype(np.float32))
"
echo '{"bits": 8}' > config.json

litl evaluate examples/quantize.py:Quantize field.npy config.json
```

`litl evaluate` compresses the data, decompresses it back, and reports how it went:

```text
┓• ┓
┃┓╋┃
┗┗┗┗
Running litl evaluate
Metrics:
{
    'original_size': 1048704,
    'compressed_size': 262272,
    'compression_ratio': 3.998535871156662,
    'compression_time': 0.004363536834716797,
    'io_time': 0.0001246929168701172,
    'decompression_time': 0.0013813972473144531,
    'mse': 1.1364933925506193e-05,
    'psnr': 52.96345520019531,
    'ssim': 0.9994416236877441,
    'l1': 0.002910611219704151
}
```

## CLI

`<compressor>` is either a path to a Python file and a class name (`path/to/compressor.py:ClassName`), or the name of an [installed compressor](#sharing-a-compressor). `<config.json>` is a JSON file that is passed to the compressor as a dictionary.

### 1. Compress Data

```bash
litl compress <compressor> <data_to_compress.npy> <config.json> <output.litl>
```

### 2. Decompress Data

```bash
litl decompress <output.litl> <decompressed.npy>
```

Add `-o <original.npy>` to also compare the result against the original data.

### 3. Evaluate Compression (compress and decompress, recording quality metrics)

```bash
litl evaluate <compressor> <data_to_compress.npy> <config.json>
```

Add `-c <output.litl>` to keep the compressed file, and `-d <decompressed.npy>` to keep the decompressed data.

### 4. Inspect Compressed Files

```bash
litl info <output.litl>
```

### Saving Metrics

Put `-m <metrics.json>` before any command to save its metrics as JSON:

```bash
litl -m metrics.json evaluate <compressor> <data_to_compress.npy> <config.json>
```

## Python API

Everything the CLI does is available from Python. These snippets continue the quickstart above.

### Compress Data

```python
import litl

compressed_blob, meta, metrics = litl.compress(
    compressor_name="examples/quantize.py:Quantize",
    data_path="field.npy",
    config={"bits": 8},
    litl_file_path="field.litl",  # optional, saves the result to a .litl file
)

print(f"Metadata: {meta}")
print(f"Metrics: {metrics}")
```

### Decompress Data

```python
import litl

decompressed_data, metrics = litl.decompress(
    litl_file_path="field.litl",
    decompressed_path="decompressed.npy",  # optional, saves the decompressed data
    original_data_path="field.npy",  # optional, adds quality metrics
)

print(f"Metrics: {metrics}")

data_numpy = decompressed_data.numpy()
```

### Evaluate Compression

```python
import litl

decompressed_data, metrics = litl.evaluate(
    compressor_name="examples/quantize.py:Quantize",
    data_path="field.npy",
    config={"bits": 8},
)

print(f"PSNR: {metrics['psnr']:.2f}, compression ratio: {metrics['compression_ratio']:.2f}")
```

## Creating a Custom Compressor

Implement the [`Compressor`](src/litl/compressors/compressor.py) interface:

```python
import numpy as np

from litl import DataWrapper
from litl.blobs import ArrayBlob, Blob
from litl.compressors import Compressor, CompressorAbout

class MyCompressor(Compressor):

    @classmethod
    def about(cls) -> CompressorAbout:
        return CompressorAbout(
            name="MyCompressor",
            description="A simple lossy compressor that stores values as 16-bit floats",
            version="1.0.0",
            author="Your Name"
        )

    @classmethod
    def blob_class(cls):
        # Specify the blob class used for compressed data. A blob of this type will be passed into your decompress method
        return ArrayBlob

    @classmethod
    def compress(cls, data: DataWrapper, config: dict) -> tuple[Blob, dict]:
        # Your compression logic here
        array = data.numpy()
        compressed_array = array.astype(np.float16)

        # Return a blob containing compressed data.
        # Built-ins are:
        # - ArrayBlob for NumPy arrays
        # - ModelBlob for PyTorch models
        # - BytesBlob for raw bytes
        # If none of these work - extend the Blob class and implement a custom blob

        meta = {"dtype": str(array.dtype)}  # Metadata required to reconstruct the data, can be any JSON-serializable dictionary
        return ArrayBlob(compressed_array), meta

    @classmethod
    def decompress(cls, blob: ArrayBlob, meta: dict) -> DataWrapper:
        # Your decompression logic here
        decompressed = blob.array.astype(meta["dtype"])

        # Return reconstructed data wrapped in DataWrapper
        return DataWrapper(decompressed)
```

Save it as `my_compressor.py` and it is ready to use:

```bash
litl evaluate my_compressor.py:MyCompressor field.npy config.json
```

The version returned by `about()` is saved in every `.litl` file, and a file can only be decompressed by the same version of the compressor.

### Sharing a Compressor

To use a compressor by name, package it and register the class in the `litl.compressors` entry point group of your `pyproject.toml`:

```toml
[project.entry-points."litl.compressors"]
my_compressor = "my_package.my_compressor:MyCompressor"
```

Once the package is installed, `litl evaluate my_compressor field.npy config.json` works from anywhere.

## DataWrapper

The [`DataWrapper`](src/litl/datawrapper.py) class supports multiple formats for loading and saving data:

- **NumPy**: `.npy`, `.np` files
- **PyTorch**: `.pt` files  
- **NetCDF**: `.nc` files (requires `netCDF4`)

```python
from litl import DataWrapper

# Load data
data = DataWrapper.from_file("field.npy")

# Save data
data.save_to_file("field.pt")

# Access underlying arrays
numpy_array = data.numpy()
torch_tensor = data.tensor()
```

A `DataWrapper` can also be created directly from a NumPy array or a PyTorch tensor: `DataWrapper(array)`.

NetCDF files are written with a single variable named `data`. When reading, the `data` variable is used if there is one, and the first variable in the file otherwise.

## Built-in Blob Types

Choose the appropriate blob type for your compressed data:

- [`ArrayBlob`](src/litl/blobs/array_blob.py): For NumPy arrays
- [`BytesBlob`](src/litl/blobs/bytes_blob.py): For raw bytes
- [`ModelBlob`](src/litl/blobs/model_blob.py): For PyTorch models

For anything else, extend [`Blob`](src/litl/blobs/blob.py) and implement `to_bytes` and `from_bytes`.

## Evaluation Metrics

The [`Evaluator`](src/litl/evaluator.py) provides standard compression metrics:

**Quality Metrics:**

- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- L1 Loss

**Size Metrics:**

- Original size (of the input file, in bytes)
- Compressed size (of the blob, in bytes)
- Compression ratio

On top of these, `compress`, `decompress`, and `evaluate` report the compression and decompression time in seconds, and the total size of the `.litl` file when one is saved.

PSNR and SSIM use the maximum value of the original data as the data range. SSIM is computed for data with 2 to 5 dimensions.

## File Format

The `.litl` format ([`DotLitl`](src/litl/dot_litl.py)) stores:

- Compressed data with Zstandard compression
- Compressor name and version
- A checksum to detect corrupted files
- Metadata returned by the compressor

| Bytes | Content |
| --- | --- |
| 4 | Magic number `LITL` |
| 1 | Format version |
| 4 | Checksum of the two sections below |
| 4 | Length of the metadata section |
| variable | Metadata: litl version, compressor name and version, and the `meta` dictionary returned by the compressor, as Zstandard-compressed JSON |
| the rest | Blob: the compressed data, with Zstandard compression on top |

Decompressing a `.litl` file imports the compressor that is named in the file, and the built-in array and model blobs are loaded with pickle enabled. Only open `.litl` files that come from a source you trust.

## Development

```bash
pip install -e ".[test,netcdf]"
pytest
```

## License

MIT License - see [LICENSE](LICENSE) for details.
