import argparse
import os
from functools import partial
import sys
import time

import numpy as np
import torch

from .compressors import Compressor
from .datawrapper import DataWrapper
from .blobs import Blob
from .evaluator import Evaluator
from .dot_litl import DotLitl

import importlib

def _discover_compressors():
    """
    Discover all compressors in the entry points
    """
    return {
        ep.name: ep.load()
        for ep in importlib.metadata.entry_points(group="litl.compressors")
    }

global _AVAILABLE_COMPRESSORS
_AVAILABLE_COMPRESSORS = None

def get_compressor(name: str) -> Compressor:
    """
    Get a compressor by name
    """
    global _AVAILABLE_COMPRESSORS
    if not _AVAILABLE_COMPRESSORS:
        _AVAILABLE_COMPRESSORS = _discover_compressors()

    if name in _AVAILABLE_COMPRESSORS:
        # if the compressor is in the entry points, return it
        return _AVAILABLE_COMPRESSORS[name]
    elif ":" in name:
        # if the compressor is not in the entry points, try to load it from a file
        try:
            path, classname = name.rsplit(':', 1)

            sys.path.append(os.getcwd())

        #     absolute_path = os.path.abspath(os.path.expanduser(path))
            
        #     if not os.path.exists(absolute_path):
        #         raise ValueError(f"Compressor script file at path {path} does not exist")
            
        #     dirname = os.path.dirname(absolute_path)
        #     if dirname not in sys.path:
        #         sys.path.insert(0, dirname)
                
        #     spec = importlib.util.spec_from_file_location(path, absolute_path)
        #     mod  = importlib.util.module_from_spec(spec)
        #     spec.loader.exec_module(mod)
        # except ValueError as e:
        #   raise ImportError(f"Failed to load compressor script file at path {path}, possibly it doesn't exist") from e
          

            # import using module path (e.g., python -m package.module)
            mod = importlib.import_module(path)
        except Exception as e:
            raise e
            # raise ImportError(f"Failed to import compressor module '{path}'. Ensure it's importable (on PYTHONPATH) and spelled correctly: {e}") from e
          

        try:
          cls = getattr(mod, classname)
        except AttributeError:
            raise ValueError(f"Compressor {name} does not have class {classname}")
        
        if not issubclass(cls, Compressor):
            raise ValueError(f"Compressor {name} is not a subclass of Compressor")
        return cls
    
    else:
        raise ValueError(f"Compressor {name} not found in entry points. Available compressors in pip: {list(_AVAILABLE_COMPRESSORS.keys())} or please provide a path to a compressor file in the format 'path/to/file.py:CompressorClassName'")
        

def compress(compressor_name: str, data_path: str, config: dict, litl_file_path: str=None) -> tuple[Blob, dict, dict]:
    """
    Compress data using the specified compressor
    Args:
        compressor_name (str): The name of the compressor to use
        data_path (str): The path to the data file to compress
        config (dict): Configuration for the compressor
        litl_file_path (str, optional): If provided, save the compressed data to a .litl file
        
    Returns:
        tuple: A tuple containing the compressed blob, metadata, and metrics
    """
    # load data
    compressor_class = get_compressor(compressor_name)

    data = DataWrapper.from_file(data_path)
    original_size = os.path.getsize(data_path)

    start_time = time.time()
    compressed_blob, meta = compressor_class.compress(data, config)
    compression_time = time.time() - start_time

    del data

    start_time = time.time()
    compressed_bytes = compressed_blob.to_bytes()
    io_time = time.time() - start_time
    
    metrics = Evaluator.size(compressed_bytes, original_size)

    metrics.update({
        'compression_time': compression_time,
        'io_time': io_time,
    })

    if litl_file_path is not None:
        # save to litl file
        total_size = DotLitl.save(litl_file_path, compressed_bytes, meta, compressor_name, compressor_class.about().version)
        metrics.update({
            'total_litl_size': total_size,

        })

    return compressed_blob, meta, metrics

def decompress(compressor_name=None, original_data_path: str=None, compressed_bytes: bytes=None, meta: dict=None, litl_file_path: str=None, decompressed_path: str=None) -> tuple[DataWrapper, dict]:
    """
    Decompress data using the specified compressor
    Args:
        compressor_name (str): The name of the compressor to use
        original_data_path (str): The path to the original data file for quality evaluation
        compressed_bytes (bytes): The compressed data bytes
        meta (dict): Metadata for the compressed data
        
        litl_file_path (str): Provide instead of compressor_name, original_data_bytes, and meta to read everything the compressed data from a .litl file
        
        decompressed_path (str, optional): If provided, save the decompressed data to this path
        
    Returns:
        tuple: A tuple containing the decompressed data and metrics
    """

    if litl_file_path is not None:
        # read from litl file
        compressed_bytes, compressor_name, compressor_version, meta, litl_version = DotLitl.read(litl_file_path)
        compressor_class = get_compressor(compressor_name)

        if compressor_class.about().version != compressor_version:
            raise ValueError(f"Compressor version mismatch: loaded compressor is v{compressor_class.about().version}, originally compressed with v{compressor_version}")

        compressed_blob_class = compressor_class.blob_class()
        compressed_blob = compressed_blob_class.from_bytes(compressed_bytes)
    elif compressed_bytes is not None and meta is not None:
        # read from bytes
        compressor_class = get_compressor(compressor_name)
        compressed_blob_class = compressor_class.blob_class()
        compressed_blob = compressed_blob_class.from_bytes(compressed_bytes)
    else:
        raise ValueError("Either compressed_bytes with meta or litl_file_path must be provided")
    
    start_time = time.time()
    decompressed_data = compressor_class.decompress(compressed_blob, meta)
    decompression_time = time.time() - start_time

    metrics = {
        'decompression_time': decompression_time,
    }

    if original_data_path is not None:
        original_data = DataWrapper.from_file(original_data_path)
    
        metrics.update(Evaluator.quality(original_data, decompressed_data))
        metrics.update(Evaluator.size(compressed_bytes, os.path.getsize(original_data_path)))
        
        del original_data

    if decompressed_path is not None:
        # save decompressed data
        decompressed_data.save_to_file(decompressed_path)

    return decompressed_data, metrics


def evaluate(compressor_name: str, data_path: str, config, decompressed_path: str=None, litl_file_path: str=None):
    """
    Evaluate the compression and decompression process by compressing and decompressing the data.
    Args:
        compressor_name (str): The name of the compressor to use
        data_path (str): The path to the data file to compress
        config (dict): Configuration for the compressor
        
        decompressed_path (str, optional): If provided, save the decompressed data to this path
        litl_file_path (str, optional): If provided, save the compressed data to a .litl file
        
    Returns:
        tuple: A tuple containing the decompressed data and metrics
    """

    # make deterministic
    torch.manual_seed(0)
    np.random.seed(0)

    # optimize cuda
    torch.set_float32_matmul_precision("medium")

    compressor = get_compressor(compressor_name)

    metrics = {}

    # compress
    compressed_blob, meta, compress_metrics = compress(compressor_name, data_path, config, litl_file_path)
    metrics.update(compress_metrics)

    # decompress
    decompressed_data, decompress_metrics = decompress(compressor_name, data_path, compressed_blob.to_bytes(), meta, decompressed_path=decompressed_path)
    metrics.update(decompress_metrics)

    return decompressed_data, metrics
