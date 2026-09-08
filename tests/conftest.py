"""Shared fixtures for unit tests."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src/ to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vae import VAE
from attacker import MLPAttacker


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Fix all random seeds before every test."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def small_vae():
    """A small VAE for fast unit tests (50 genes, z=8)."""
    model = VAE(
        input_dim=50,
        hidden_dims=[32, 16],
        latent_dim=8,
        likelihood="nb",
        dropout=0.0,
        use_layer_norm=True,
    )
    model.eval()
    return model


@pytest.fixture
def small_vae_gaussian():
    """A small Gaussian VAE for fast unit tests."""
    model = VAE(
        input_dim=50,
        hidden_dims=[32, 16],
        latent_dim=8,
        likelihood="gaussian",
        dropout=0.0,
        use_layer_norm=True,
    )
    model.eval()
    return model


@pytest.fixture
def small_attacker():
    """A small MLP attacker for testing."""
    return MLPAttacker(
        input_dim=21,  # 2*8 + 5 = 21 for z=8 v1 features
        hidden_dims=[16, 16],
        dropout=0.0,
        use_spectral_norm=False,
    )


@pytest.fixture
def dummy_batch():
    """A batch of 16 fake gene-expression samples (50 genes)."""
    x = torch.rand(16, 50) * 5  # simulate log-normalized expression
    library_size = x.sum(dim=1, keepdim=True)
    return x, library_size
