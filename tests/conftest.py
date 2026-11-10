import json
import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def quantize():
    """Name of the example compressor, in the 'path/to/file.py:ClassName' format"""
    return os.path.join(REPO_ROOT, "examples", "quantize.py") + ":Quantize"


@pytest.fixture
def field():
    """A smooth 3D float32 field"""
    x, y, z = np.meshgrid(*[np.linspace(0, 2 * np.pi, 24)] * 3, indexing="ij")
    return (np.sin(x) * np.cos(y) + 0.5 * np.sin(2 * z)).astype(np.float32)


@pytest.fixture
def field_path(tmp_path, field):
    path = tmp_path / "field.npy"
    np.save(path, field)
    return str(path)


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"bits": 8}))
    return str(path)
