import warnings

import torch
import torcheval.metrics as metrics
from pytorch_msssim import ssim

from .datawrapper import DataWrapper

class Evaluator():

  @staticmethod
  def quality(original: DataWrapper, reconstructed: DataWrapper) -> dict:
    """
    Returns the quality metrics of the reconstructed data compared to the original data

    PSNR and SSIM use the maximum value of the original data as the data range. If that maximum is not positive
    (for example, when all values are negative), the value range (max - min) of the original data is used instead.

    SSIM is only defined for data with 2 to 5 dimensions, it is reported as NaN otherwise.
    """
    if not isinstance(original, DataWrapper) or not isinstance(reconstructed, DataWrapper):
      raise TypeError("Both original and reconstructed must be instances of DataWrapper")

    if original.shape() != reconstructed.shape():
      raise ValueError(f"Original and reconstructed data must have the same shape. Original: {original.shape()}, Reconstructed: {reconstructed.shape()}")


    original_data = original.tensor()
    reconstructed_data = reconstructed.tensor()

    # metrics are computed in floating point, so integer (and mismatched) dtypes are converted first
    dtype = torch.promote_types(original_data.dtype, reconstructed_data.dtype)
    if not dtype.is_floating_point:
      dtype = torch.float64
    elif dtype == torch.float16:
      dtype = torch.float32
    original_data = original_data.to(dtype)
    reconstructed_data = reconstructed_data.to(dtype)

    data_range = original_data.max().item()
    if data_range <= 0:
      data_range = (original_data.max() - original_data.min()).item()
    if data_range <= 0:
      # constant data, any positive range works
      data_range = 1.0

    mse = metrics.MeanSquaredError()
    psnr = metrics.PeakSignalNoiseRatio(data_range=data_range)

    mse_metric = mse.update(reconstructed_data.flatten(), original_data.flatten()).compute().item()
    del mse

    psnr_metric = psnr.update(reconstructed_data.flatten(), original_data.flatten()).compute().item()
    del psnr

    l1_metric = torch.mean(torch.abs(reconstructed_data - original_data)).item()

    if 2 <= original_data.dim() <= 5:
      # extend to 4d
      while reconstructed_data.dim() < 4:
        reconstructed_data = reconstructed_data.unsqueeze(0)
        original_data = original_data.unsqueeze(0)
      ssim_metric = ssim(reconstructed_data, original_data, data_range=data_range).item()
    else:
      warnings.warn(f"SSIM is only supported for data with 2 to 5 dimensions, but got {original_data.dim()}. Reporting SSIM as NaN.", UserWarning)
      ssim_metric = float('nan')

    return {
      'mse': mse_metric,
      'psnr': psnr_metric,
      'ssim': ssim_metric,
      'l1': l1_metric,
    }

  @staticmethod
  def size(compressed_bytes: bytes, original_size: int) -> dict:
    """
    Returns the size metrics of the blob
    """
    blob_size = len(compressed_bytes)

    return {
      'original_size': original_size,
      'compressed_size': blob_size,
      'compression_ratio': original_size / blob_size if blob_size > 0 else float('inf'),
    }
