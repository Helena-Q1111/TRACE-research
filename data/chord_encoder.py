"""Chord feature encoding utilities for Hooktheory harmony annotations."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


def normalized_root_pitch_class(chord: Mapping[str, Any], tonic_pitch_class: int = 0) -> int:
    """Return the chord root after transposition into the target tonic frame."""
    return (int(chord["root_pitch_class"]) - tonic_pitch_class) % 12


def chord_pitch_classes(chord: Mapping[str, Any], tonic_pitch_class: int = 0) -> list[int]:
    """Expand a Hooktheory chord entry into normalized pitch classes."""
    root = normalized_root_pitch_class(chord, tonic_pitch_class)
    intervals = [int(interval) for interval in chord.get("root_position_intervals", [])]

    pitch_classes = [root]
    current = root
    for interval in intervals:
        current = (current + interval) % 12
        pitch_classes.append(current)
    return pitch_classes


def chord_to_type_token(chord: Mapping[str, Any], tonic_pitch_class: int = 0) -> str:
    """Build a reproducible chord identity token for vocabulary lookup.

    The token is computed after relative-major normalization so the index space is
    shared across songs and tonal segments.
    """
    root = normalized_root_pitch_class(chord, tonic_pitch_class)
    intervals = tuple(int(interval) for interval in chord.get("root_position_intervals", []))
    inversion = int(chord.get("inversion", 0))
    interval_str = ",".join(str(interval) for interval in intervals)
    return f"root={root}|intervals={interval_str}|inv={inversion}"


def chord_to_vector(
    chord: Mapping[str, Any],
    tonic_pitch_class: int = 0,
) -> tuple[np.ndarray, int, int]:
    """Convert a Hooktheory chord entry into the 15D feature vector.

    Args:
        chord: Harmony annotation containing root, intervals, inversion and timing.
        tonic_pitch_class: Transposition amount so the relative major tonic maps to C.

    Returns:
        A tuple of `(vector, root_pitch_class, bass_pitch_class)` where `vector` is a
        NumPy array of shape `[15]` and dtype `float32`.
    """
    root = normalized_root_pitch_class(chord, tonic_pitch_class)
    pitch_classes = chord_pitch_classes(chord, tonic_pitch_class)
    inversion = int(chord.get("inversion", 0))

    chroma = np.zeros(12, dtype=np.float32)
    for pitch_class in pitch_classes:
        chroma[pitch_class] = 0.5
    chroma[root] = 1.0

    bass_pitch_class = pitch_classes[inversion] if 0 <= inversion < len(pitch_classes) else root
    root_angle = (2.0 * np.pi * root) / 12.0
    root_sin = np.sin(root_angle, dtype=np.float32)
    root_cos = np.cos(root_angle, dtype=np.float32)

    onset = float(chord.get("onset", 0.0))
    offset = float(chord.get("offset", onset))
    duration = max(0.0, offset - onset)
    log_duration = np.log1p(duration, dtype=np.float32)

    vector = np.concatenate(
        [
            chroma,
            np.asarray([root_sin, root_cos, log_duration], dtype=np.float32),
        ]
    ).astype(np.float32)
    return vector, root, bass_pitch_class


def decode_bass_from_circular(sin_val: torch.Tensor, cos_val: torch.Tensor) -> torch.Tensor:
    """Recover the circularly encoded pitch class from the unit circle."""
    angle = torch.atan2(sin_val, cos_val)
    angle = torch.where(angle < 0, angle + (2 * np.pi), angle)
    pitch_class = torch.round(angle * 12 / (2 * np.pi)) % 12
    return pitch_class.long()