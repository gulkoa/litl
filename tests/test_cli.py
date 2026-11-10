import json
import subprocess
import sys
from importlib import metadata as importlib_metadata

import numpy as np
import pytest

from litl.cli import main


def test_evaluate(tmp_path, capsys, quantize, field_path, config_path):
    metrics_path = tmp_path / "metrics.json"
    litl_path = tmp_path / "field.litl"
    decompressed_path = tmp_path / "decompressed.npy"

    exit_code = main(["-m", str(metrics_path), "evaluate", quantize, field_path, config_path, "-c", str(litl_path), "-d", str(decompressed_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Running litl evaluate" in output
    assert "Metrics:" in output
    assert litl_path.exists()
    assert np.load(decompressed_path).shape == np.load(field_path).shape

    # the metrics file is real JSON
    metrics = json.loads(metrics_path.read_text())
    assert {"mse", "psnr", "ssim", "l1", "original_size", "compressed_size", "compression_ratio", "total_litl_size"} <= set(metrics)
    assert metrics["psnr"] > 40


def test_compress_info_decompress(tmp_path, capsys, quantize, field_path, config_path):
    litl_path = tmp_path / "field.litl"
    decompressed_path = tmp_path / "decompressed.npy"

    assert main(["compress", quantize, field_path, config_path, str(litl_path)]) == 0
    assert litl_path.exists()
    assert "Compressed data saved to" in capsys.readouterr().out

    assert main(["info", str(litl_path)]) == 0
    output = capsys.readouterr().out.replace("\n", "")
    assert "was compressed with" in output
    assert f"using litl v{importlib_metadata.version('litl')}" in output
    assert "Metadata:" in output and "'bits': 8" in output

    assert main(["decompress", str(litl_path), str(decompressed_path), "-o", field_path]) == 0
    assert "Decompressed data saved to" in capsys.readouterr().out
    original = np.load(field_path)
    decompressed = np.load(decompressed_path)
    assert decompressed.shape == original.shape
    assert np.abs(decompressed - original).max() < 0.01


def test_info_with_a_metrics_path(tmp_path, capsys, quantize, field_path, config_path):
    litl_path = tmp_path / "field.litl"
    info_path = tmp_path / "info.json"
    main(["compress", quantize, field_path, config_path, str(litl_path)])

    assert main(["-m", str(info_path), "info", str(litl_path)]) == 0
    info = json.loads(info_path.read_text())
    assert info["compressor"] == quantize
    assert info["compressor_version"] == "1.0.0"
    assert info["litl_version"] == importlib_metadata.version("litl")
    assert info["meta"]["bits"] == 8


def test_errors_exit_with_a_non_zero_code(tmp_path, capsys, quantize, field_path):
    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate", quantize, field_path, str(tmp_path / "missing_config.json")])
    assert exit_info.value.code == 1


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"litl v{importlib_metadata.version('litl')}"


def test_cli_in_a_subprocess(tmp_path, quantize, field_path, config_path):
    # same entry point as the `litl` console script, reading its arguments from sys.argv
    command = [sys.executable, "-c", "import sys; from litl.cli import main; sys.exit(main())"]
    litl_path = tmp_path / "field.litl"

    result = subprocess.run(command + ["compress", quantize, field_path, config_path, str(litl_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert litl_path.exists()

    result = subprocess.run(command + ["info", str(litl_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "was compressed with" in result.stdout.replace("\n", "")

    result = subprocess.run(command + ["info", str(tmp_path / "missing.litl")], capture_output=True, text=True)
    assert result.returncode == 1
