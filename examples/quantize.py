"""
A tiny example compressor for litl: uniform quantization.

Every value is mapped to one of 2^bits evenly spaced levels between the minimum and the maximum of the data,
and stored as an unsigned integer. The reconstruction error is at most half of the distance between two levels.

Values are stored in whole bytes (uint8 for up to 8 bits, uint16 for up to 16 bits). With fewer bits there are
fewer distinct values, which the Zstandard pass of the .litl format then compresses further.

Try it from the root of the repository:

    echo '{"bits": 8}' > config.json
    litl evaluate examples/quantize.py:Quantize data.npy config.json
"""
import numpy as np

from litl import DataWrapper
from litl.blobs import ArrayBlob
from litl.compressors import Compressor, CompressorAbout


class Quantize(Compressor):

    @classmethod
    def about(cls) -> CompressorAbout:
        return CompressorAbout(
            name="Quantize",
            description="Uniform quantization to a fixed number of bits per value",
            version="1.0.0",
            license="MIT",
            url="https://github.com/GulkoA/litl",
        )

    @classmethod
    def optuna_suggest_config(cls, trial):
        # optional: lets a hyperparameter search pick the config
        return {"bits": trial.suggest_int("bits", 1, 16)}

    @classmethod
    def blob_class(cls):
        return ArrayBlob

    @classmethod
    def compress(cls, data: DataWrapper, config: dict) -> tuple[ArrayBlob, dict]:
        bits = int(config.get("bits", 8))
        if not 1 <= bits <= 16:
            raise ValueError(f"bits must be between 1 and 16, but got {bits}")

        array = data.numpy()
        low, high = float(array.min()), float(array.max())
        levels = 2 ** bits - 1
        step = (high - low) / levels if high > low else 1.0

        quantized = np.round((array.astype(np.float64) - low) / step)
        quantized = quantized.astype(np.uint8 if bits <= 8 else np.uint16)

        # everything decompress needs besides the blob itself, saved in the .litl file as JSON
        meta = {"bits": bits, "min": low, "max": high, "dtype": str(array.dtype)}
        return ArrayBlob(quantized), meta

    @classmethod
    def decompress(cls, blob: ArrayBlob, meta: dict) -> DataWrapper:
        levels = 2 ** meta["bits"] - 1
        step = (meta["max"] - meta["min"]) / levels if meta["max"] > meta["min"] else 1.0

        array = blob.array.astype(np.float64) * step + meta["min"]

        dtype = np.dtype(meta["dtype"])
        if np.issubdtype(dtype, np.integer):
            array = np.round(array)
        return DataWrapper(array.astype(dtype))
