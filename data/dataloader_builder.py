"""Utilities to prepare Hooktheory window datasets and PyTorch dataloaders."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from .chord_dataset import ChordPairDataset, build_chord_type_vocabulary
from .hooktheory_loader import build_harmony_sequences_for_song_ids, load_dataset, split_by_song


def prepare_datasets(
    dataset_path: str,
    seed: int = 42,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    chord_type_min_count: int = 1,
    window_size: int = 8,
    stride: int = 1,
) -> tuple[dict[str, ChordPairDataset], dict[str, Any]]:
    """Prepare train/valid/test window datasets from the raw Hooktheory dump."""
    raw_dataset = load_dataset(dataset_path)
    split_song_ids = split_by_song(
        song_ids=sorted(raw_dataset.keys()),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    split_sequences = {
        split_name: build_harmony_sequences_for_song_ids(raw_dataset, split_ids)
        for split_name, split_ids in split_song_ids.items()
    }

    chord_type_to_index = build_chord_type_vocabulary(
        split_sequences["train"],
        window_size=window_size,
        stride=stride,
        min_count=chord_type_min_count,
    )
    datasets = {
        split_name: ChordPairDataset(
            sequences,
            chord_type_to_index=chord_type_to_index,
            window_size=window_size,
            stride=stride,
        )
        for split_name, sequences in split_sequences.items()
    }

    metadata = {
        "song_splits": split_song_ids,
        "sequence_counts": {name: len(sequences) for name, sequences in split_sequences.items()},
        "window_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "chord_type_to_index": chord_type_to_index,
        "window_size": window_size,
        "stride": stride,
    }
    return datasets, metadata


def build_dataloaders(
    dataset_path: str,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    chord_type_min_count: int = 1,
    window_size: int = 8,
    stride: int = 1,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    """Build train/valid/test dataloaders for cognitive-window modeling."""
    datasets, metadata = prepare_datasets(
        dataset_path=dataset_path,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        chord_type_min_count=chord_type_min_count,
        window_size=window_size,
        stride=stride,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return dataloaders, metadata