"""
Tests for Project Foundation: Package Structure, Config Loading, and Reproducibility.
"""

from pathlib import Path
import pytest
import numpy as np

import aesw
from aesw.utils.config import load_config
from aesw.utils.reproducibility import seed_everything, create_rng


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_package_import():
    """Verify that the aesw package and subpackages import cleanly."""
    assert aesw.__version__ == "0.1.0"

    import aesw.graph as graph
    import aesw.dynamics as dynamics
    import aesw.environment as environment
    import aesw.walkers as walkers
    import aesw.baselines as baselines
    import aesw.aesw as aesw_core
    import aesw.memory as memory
    import aesw.evaluation as evaluation
    import aesw.utils as utils

    assert graph is not None
    assert dynamics is not None
    assert environment is not None
    assert walkers is not None
    assert baselines is not None
    assert aesw_core is not None
    assert memory is not None
    assert evaluation is not None
    assert utils is not None


def test_config_files_exist():
    """Verify that all required YAML configuration files exist."""
    required_configs = ["default.yaml", "graph.yaml", "dynamics.yaml", "experiments.yaml"]
    for config_name in required_configs:
        config_path = CONFIGS_DIR / config_name
        assert config_path.is_file(), f"Missing configuration file: {config_path}"


def test_default_config_loading():
    """Verify that default.yaml loads correctly and has required keys."""
    config = load_config(CONFIGS_DIR / "default.yaml")
    assert isinstance(config, dict)
    assert "project" in config
    assert "reproducibility" in config
    assert config["reproducibility"]["seed"] == 42
    assert config["reproducibility"]["deterministic"] is True


def test_graph_config_loading():
    """Verify that graph.yaml loads correctly and contains graph types."""
    config = load_config(CONFIGS_DIR / "graph.yaml")
    assert isinstance(config, dict)
    assert "graph" in config
    assert "type" in config["graph"]
    assert "num_nodes" in config["graph"]
    assert "seed" in config["graph"]


def test_dynamics_config_loading():
    """Verify that dynamics.yaml loads correctly and contains dynamic parameters."""
    config = load_config(CONFIGS_DIR / "dynamics.yaml")
    assert isinstance(config, dict)
    assert "dynamics" in config
    assert "regime" in config["dynamics"]
    assert "p_on" in config["dynamics"]
    assert "p_off" in config["dynamics"]
    assert "regimes" in config["dynamics"]


def test_experiments_config_loading():
    """Verify that experiments.yaml loads correctly and lists candidate algorithms."""
    config = load_config(CONFIGS_DIR / "experiments.yaml")
    assert isinstance(config, dict)
    assert "experiments" in config
    assert "algorithms" in config["experiments"]
    algorithms = config["experiments"]["algorithms"]
    assert "random_walk" in algorithms
    assert "aesw" in algorithms


def test_config_loader_errors():
    """Verify load_config raises appropriate exceptions for invalid paths."""
    with pytest.raises(FileNotFoundError):
        load_config(CONFIGS_DIR / "non_existent.yaml")


def test_reproducibility_seed_everything():
    """Verify that seed_everything produces deterministic pseudo-random sequences."""
    seed_everything(12345)
    r1 = [np.random.rand() for _ in range(5)]

    seed_everything(12345)
    r2 = [np.random.rand() for _ in range(5)]

    assert r1 == r2

    seed_everything(99999)
    r3 = [np.random.rand() for _ in range(5)]
    assert r1 != r3


def test_reproducibility_rng():
    """Verify that create_rng produces independent, reproducible generator streams."""
    rng1 = create_rng(42)
    stream1 = rng1.uniform(0, 1, size=10)

    rng2 = create_rng(42)
    stream2 = rng2.uniform(0, 1, size=10)

    np.testing.assert_array_equal(stream1, stream2)
