"""3D autoencoder components for cognitively constrained chord embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AutoencoderLossBreakdown:
    """Structured loss values for logging and checkpointing."""

    total: torch.Tensor
    chroma: torch.Tensor
    circular: torch.Tensor
    duration: torch.Tensor
    tps: torch.Tensor


class ChordAutoencoder(nn.Module):
    """Minimal MLP autoencoder for [B, 8, 15] chord windows."""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 64, latent_dim: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        if input_dim != 15:
            raise ValueError(f"input_dim must be 15, got {input_dim}")
        if latent_dim != 3:
            raise ValueError(f"latent_dim must be 3, got {latent_dim}")

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        encoder_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            encoder_layers.append(nn.Dropout(dropout))
        encoder_layers.extend(
            [
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            ]
        )
        if dropout > 0:
            encoder_layers.append(nn.Dropout(dropout))
        encoder_layers.append(nn.Linear(hidden_dim // 2, latent_dim))

        decoder_layers: list[nn.Module] = [
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
        ]
        if dropout > 0:
            decoder_layers.append(nn.Dropout(dropout))
        decoder_layers.extend(
            [
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
            ]
        )
        if dropout > 0:
            decoder_layers.append(nn.Dropout(dropout))
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of chord windows into 3D latent coordinates."""
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dimension {self.input_dim}, got {x.shape[-1]}")
        flat_x = x.reshape(-1, self.input_dim)
        flat_z = self.encoder(flat_x)
        return flat_z.reshape(*x.shape[:-1], self.latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode 3D latent coordinates back to the 15D chord representation."""
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected last dimension {self.latent_dim}, got {z.shape[-1]}")
        flat_z = z.reshape(-1, self.latent_dim)
        flat_x = self.decoder(flat_z).reshape(*z.shape[:-1], self.input_dim)

        chroma = torch.sigmoid(flat_x[..., :12])
        circular = F.normalize(flat_x[..., 12:14], dim=-1, eps=1e-8)
        duration = flat_x[..., 14:15]
        return torch.cat([chroma, circular, duration], dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstructed windows and their latent codes."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> AutoencoderLossBreakdown:
    """Compute the split reconstruction loss used by the outline."""
    if x_hat.shape != x.shape:
        raise ValueError(f"Shape mismatch: {tuple(x_hat.shape)} vs {tuple(x.shape)}")

    chroma_loss = F.binary_cross_entropy(x_hat[..., :12], x[..., :12])

    pred_circular = F.normalize(x_hat[..., 12:14], dim=-1, eps=1e-8)
    target_circular = F.normalize(x[..., 12:14], dim=-1, eps=1e-8)
    circular_loss = 1.0 - F.cosine_similarity(pred_circular, target_circular, dim=-1).mean()

    duration_loss = F.mse_loss(x_hat[..., 14], x[..., 14])
    total = chroma_loss + circular_loss + duration_loss
    zero = total.new_tensor(0.0)
    return AutoencoderLossBreakdown(
        total=total,
        chroma=chroma_loss,
        circular=circular_loss,
        duration=duration_loss,
        tps=zero,
    )


def latent_tps_distances(z: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distances between consecutive latent codes in a window."""
    if z.ndim != 3:
        raise ValueError(f"Expected z with shape [batch, window, latent_dim], got {tuple(z.shape)}")
    return torch.linalg.norm(z[:, 1:, :] - z[:, :-1, :], dim=-1)


def tps_distance_loss(z: torch.Tensor, tps_dist: torch.Tensor) -> torch.Tensor:
    """Regress latent distances toward the precomputed TPS targets."""
    latent_dist = latent_tps_distances(z)
    target = tps_dist.to(dtype=latent_dist.dtype, device=latent_dist.device)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if target.shape != latent_dist.shape:
        raise ValueError(f"Shape mismatch: latent {tuple(latent_dist.shape)} vs target {tuple(target.shape)}")
    return F.mse_loss(latent_dist, target)
