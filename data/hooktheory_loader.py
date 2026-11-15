"""Load and segment Hooktheory annotations for chord transition modeling."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DIATONIC_MODE_TO_RELATIVE_MAJOR_OFFSET = {
    (2, 2, 1, 2, 2, 2, 1): 0,
    (2, 1, 2, 2, 2, 1, 2): 2,
    (1, 2, 2, 2, 1, 2, 2): 4,
    (2, 2, 2, 1, 2, 2, 1): 5,
    (2, 2, 1, 2, 2, 1, 2): 7,
    (2, 1, 2, 2, 1, 2, 2): 9,
    (1, 2, 2, 1, 2, 2, 2): 11,
}


@dataclass(frozen=True)
class TonalSegment:
    """A contiguous harmony span with a single key annotation."""

    song_id: str
    segment_index: int
    start_beat: float
    end_beat: float | None
    tonic_pitch_class: int
    relative_major_tonic: int
    scale_degree_intervals: tuple[int, ...]
    chords: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class HarmonySequence:
    """Song-level harmony timeline used for local cognitive windows."""

    song_id: str
    chords: tuple[dict[str, Any], ...]


def load_dataset(dataset_path: str | Path) -> dict[str, Any]:
    """Load the raw Hooktheory JSON or JSON.GZ dataset."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Hooktheory dataset not found: {path}")

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_by_song(
    song_ids: Sequence[str],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create reproducible train/valid/test splits at the song level."""
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    shuffled_song_ids = list(song_ids)
    random.Random(seed).shuffle(shuffled_song_ids)

    total = len(shuffled_song_ids)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    return {
        "train": shuffled_song_ids[:train_end],
        "valid": shuffled_song_ids[train_end:valid_end],
        "test": shuffled_song_ids[valid_end:],
    }


def infer_relative_major_tonic(key_annotation: Mapping[str, Any]) -> int:
    """Infer the relative-major tonic for a Hooktheory key annotation.

    Hooktheory provides the diatonic scale via `scale_degree_intervals`. We match the
    completed seven-interval pattern against the church modes and map each mode back to
    its relative major anchor. Unknown scales fall back to the annotated tonic.
    """
    tonic_pitch_class = int(key_annotation.get("tonic_pitch_class", 0))
    intervals = tuple(int(value) for value in key_annotation.get("scale_degree_intervals", []))
    if not intervals:
        return tonic_pitch_class

    if len(intervals) == 6:
        closing_interval = (12 - (sum(intervals) % 12)) % 12
        completed_intervals = intervals + ((closing_interval or 12),)
    else:
        completed_intervals = intervals[:7]

    offset = DIATONIC_MODE_TO_RELATIVE_MAJOR_OFFSET.get(completed_intervals)
    if offset is None:
        return tonic_pitch_class
    return (tonic_pitch_class - offset) % 12


def extract_tonal_segments(song_id: str, song_payload: Mapping[str, Any]) -> list[TonalSegment]:
    """Split one song into tonal segments using key change boundaries."""
    annotations = song_payload.get("annotations", {})
    harmony_entries = annotations.get("harmony") or []
    harmony = sorted(harmony_entries, key=lambda item: float(item.get("onset", 0.0)))
    if not harmony:
        return []

    key_entries = annotations.get("keys") or []
    keys = sorted(key_entries, key=lambda item: float(item.get("beat", 0.0)))
    if not keys:
        keys = [{"beat": 0.0, "tonic_pitch_class": 0, "scale_degree_intervals": [2, 2, 1, 2, 2, 2]}]

    segments: list[TonalSegment] = []
    for index, key_annotation in enumerate(keys):
        start_beat = float(key_annotation.get("beat", 0.0))
        end_beat = None
        if index + 1 < len(keys):
            end_beat = float(keys[index + 1].get("beat", start_beat))

        segment_chords = []
        for chord in harmony:
            onset = float(chord.get("onset", 0.0))
            if onset < start_beat:
                continue
            if end_beat is not None and onset >= end_beat:
                continue
            segment_chords.append(dict(chord))

        if not segment_chords:
            continue

        scale_intervals = tuple(int(value) for value in key_annotation.get("scale_degree_intervals", []))
        segments.append(
            TonalSegment(
                song_id=song_id,
                segment_index=index,
                start_beat=start_beat,
                end_beat=end_beat,
                tonic_pitch_class=int(key_annotation.get("tonic_pitch_class", 0)),
                relative_major_tonic=infer_relative_major_tonic(key_annotation),
                scale_degree_intervals=scale_intervals,
                chords=tuple(segment_chords),
            )
        )

    return segments


def build_segments_for_song_ids(dataset: Mapping[str, Any], song_ids: Iterable[str]) -> list[TonalSegment]:
    """Extract tonal segments for a collection of song identifiers."""
    segments: list[TonalSegment] = []
    for song_id in song_ids:
        payload = dataset.get(song_id)
        if payload is None:
            continue
        segments.extend(extract_tonal_segments(song_id, payload))
    return segments


def extract_harmony_sequence(song_id: str, song_payload: Mapping[str, Any]) -> HarmonySequence | None:
    """Extract a song-level harmony sequence sorted by onset."""
    annotations = song_payload.get("annotations", {})
    harmony_entries = annotations.get("harmony") or []
    harmony = sorted(harmony_entries, key=lambda item: float(item.get("onset", 0.0)))
    if not harmony:
        return None
    return HarmonySequence(song_id=song_id, chords=tuple(dict(chord) for chord in harmony))


def build_harmony_sequences_for_song_ids(
    dataset: Mapping[str, Any],
    song_ids: Iterable[str],
) -> list[HarmonySequence]:
    """Build song-level harmony sequences for a collection of songs."""
    sequences: list[HarmonySequence] = []
    for song_id in song_ids:
        payload = dataset.get(song_id)
        if payload is None:
            continue
        sequence = extract_harmony_sequence(song_id, payload)
        if sequence is not None:
            sequences.append(sequence)
    return sequences


MAJOR_SCALE_INTERVALS = (0, 2, 4, 5, 7, 9, 11)


def infer_local_relative_major_tonic(chords: Sequence[Mapping[str, Any]]) -> int:
    """Infer local relative-major tonic from a short chord window.

    The score favors pitch classes strongly supported by chord tones and roots,
    then selects the major scale that best explains those notes.
    """
    if not chords:
        return 0

    pitch_energy = [0.0] * 12
    root_energy = [0.0] * 12

    for chord in chords:
        root = int(chord.get("root_pitch_class", 0)) % 12
        root_energy[root] += 1.0

        intervals = [int(interval) for interval in chord.get("root_position_intervals", [])]
        pitch_energy[root] += 2.0
        current = root
        for interval in intervals:
            current = (current + interval) % 12
            pitch_energy[current] += 1.0

    best_tonic = 0
    best_score = float("-inf")
    for tonic in range(12):
        in_scale_score = sum(pitch_energy[(tonic + interval) % 12] for interval in MAJOR_SCALE_INTERVALS)
        score = in_scale_score + (0.05 * root_energy[tonic])
        if score > best_score:
            best_score = score
            best_tonic = tonic

    return best_tonic