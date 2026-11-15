"""PyTorch dataset for normalized Hooktheory cognitive chord windows."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .chord_encoder import chord_pitch_classes, chord_to_type_token, chord_to_vector, normalized_root_pitch_class
from .hooktheory_loader import HarmonySequence, infer_local_relative_major_tonic


UNKNOWN_CHORD_TYPE = "<UNK>"


def build_chord_type_vocabulary(
    sequences: Sequence[HarmonySequence],
    window_size: int = 8,
    stride: int = 1,
    min_count: int = 1,
) -> dict[str, int]:
    """Build a reproducible chord-type vocabulary from normalized windows."""
    counter: Counter[str] = Counter()
    for sequence in sequences:
        chords = sequence.chords
        if len(chords) < window_size:
            continue

        for start_index in range(0, len(chords) - window_size + 1, stride):
            window_chords = chords[start_index : start_index + window_size]
            local_tonic = infer_local_relative_major_tonic(window_chords)
            for chord in window_chords:
                token = chord_to_type_token(chord, tonic_pitch_class=local_tonic)
                counter[token] += 1

    vocabulary = {UNKNOWN_CHORD_TYPE: 0}
    for token in sorted(counter):
        if counter[token] >= min_count:
            vocabulary[token] = len(vocabulary)
    return vocabulary


class ChordPairDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Dataset that returns 8-chord windows and symbolic metadata."""

    def __init__(
        self,
        sequences: Sequence[HarmonySequence],
        chord_type_to_index: Mapping[str, int] | None = None,
        window_size: int = 8,
        stride: int = 1,
    ) -> None:
        self.window_size = int(window_size)
        self.stride = int(stride)
        if self.window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {self.window_size}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")

        self.sequences = [sequence for sequence in sequences if len(sequence.chords) >= self.window_size]
        if chord_type_to_index is None:
            chord_type_to_index = build_chord_type_vocabulary(
                self.sequences,
                window_size=self.window_size,
                stride=self.stride,
            )
        self.chord_type_to_index = dict(chord_type_to_index)
        self.index_to_chord_type = {index: token for token, index in self.chord_type_to_index.items()}

        window_records: list[tuple[torch.Tensor, ...]] = []
        for sequence in self.sequences:
            chords = sequence.chords
            for start_index in range(0, len(chords) - self.window_size + 1, self.stride):
                window_chords = chords[start_index : start_index + self.window_size]
                local_tonic = infer_local_relative_major_tonic(window_chords)

                vectors: list[torch.Tensor] = []
                roots: list[int] = []
                basses: list[int] = []
                chord_types: list[int] = []

                for chord in window_chords:
                    vector, root_pitch_class, bass_pitch_class = chord_to_vector(
                        chord,
                        tonic_pitch_class=local_tonic,
                    )
                    token = chord_to_type_token(chord, tonic_pitch_class=local_tonic)
                    chord_type_index = self.chord_type_to_index.get(token, self.chord_type_to_index[UNKNOWN_CHORD_TYPE])

                    vectors.append(torch.from_numpy(vector))
                    roots.append(root_pitch_class)
                    basses.append(bass_pitch_class)
                    chord_types.append(chord_type_index)

                tps_distances = [
                    _simple_tps_distance(window_chords[index], window_chords[index + 1], local_tonic)
                    for index in range(self.window_size - 1)
                ]

                window_records.append(
                    (
                        torch.stack(vectors, dim=0),
                        torch.tensor(roots, dtype=torch.long),
                        torch.tensor(basses, dtype=torch.long),
                        torch.tensor(chord_types, dtype=torch.long),
                        torch.tensor(tps_distances, dtype=torch.float32),
                        torch.tensor(local_tonic, dtype=torch.long),
                    )
                )

        self._window_records = window_records

    def __len__(self) -> int:
        """Return the number of cognitive windows in the dataset."""
        return len(self._window_records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Return one 8-chord window and its metadata tensors."""
        return self._window_records[index]


def _circle_of_fifths_distance(root_a: int, root_b: int) -> int:
    """Compute minimum clockwise/counterclockwise circle-of-fifths steps."""
    pos_a = (7 * root_a) % 12
    pos_b = (7 * root_b) % 12
    diff = abs(pos_a - pos_b)
    return min(diff, 12 - diff)


def _simple_tps_distance(chord_a: Mapping[str, object], chord_b: Mapping[str, object], tonic_pitch_class: int) -> float:
    """Approximate TPS distance as fifth-steps plus pitch-class change count."""
    root_a = normalized_root_pitch_class(chord_a, tonic_pitch_class=tonic_pitch_class)
    root_b = normalized_root_pitch_class(chord_b, tonic_pitch_class=tonic_pitch_class)

    pcs_a = set(chord_pitch_classes(chord_a, tonic_pitch_class=tonic_pitch_class))
    pcs_b = set(chord_pitch_classes(chord_b, tonic_pitch_class=tonic_pitch_class))

    fifth_steps = _circle_of_fifths_distance(root_a, root_b)
    semitone_changes = len(pcs_a.symmetric_difference(pcs_b))
    return float(fifth_steps + semitone_changes)


ChordWindowDataset = ChordPairDataset