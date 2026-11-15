"""Unit tests for the 3D chord autoencoder and TPS regularization."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from autoencoder import ChordAutoencoder, reconstruction_loss, tps_distance_loss


def test_autoencoder_forward_preserves_window_shapes() -> None:
    """The autoencoder should preserve [batch, window, feature] shapes."""
    model = ChordAutoencoder()
    x = torch.rand(2, 8, 15)

    x_hat, z = model(x)

    assert x_hat.shape == (2, 8, 15)
    assert z.shape == (2, 8, 3)
    assert torch.all(x_hat[..., :12] >= 0.0)
    assert torch.all(x_hat[..., :12] <= 1.0)

    circular_norm = torch.linalg.norm(x_hat[..., 12:14], dim=-1)
    assert torch.allclose(circular_norm, torch.ones_like(circular_norm), atol=1e-5)


def test_reconstruction_loss_returns_finite_components() -> None:
    """Reconstruction loss should be finite on valid feature tensors."""
    model = ChordAutoencoder()
    x = torch.rand(1, 8, 15)
    x_hat, _ = model(x)

    breakdown = reconstruction_loss(x_hat, x)

    assert breakdown.total.item() >= 0.0
    assert breakdown.chroma.item() >= 0.0
    assert breakdown.circular.item() >= 0.0
    assert breakdown.duration.item() >= 0.0


def test_tps_distance_loss_matches_target_distances() -> None:
    """TPS loss should be zero when latent distances exactly match the target."""
    z = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    tps_dist = torch.ones(1, 7, dtype=torch.float32)

    loss = tps_distance_loss(z, tps_dist)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
