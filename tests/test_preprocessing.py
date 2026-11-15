"""Unit tests for the synthetic preprocessing invariants of the CoLa data pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.chord_dataset import ChordPairDataset
from data.chord_encoder import chord_to_vector
from data.hooktheory_loader import HarmonySequence, extract_tonal_segments, infer_local_relative_major_tonic, infer_relative_major_tonic


def make_chord(
    root_pitch_class: int,
    root_position_intervals: list[int],
    inversion: int = 0,
    onset: float = 0.0,
    offset: float = 1.0,
) -> dict[str, object]:
    """Create a minimal synthetic Hooktheory chord object."""
    return {
        "root_pitch_class": root_pitch_class,
        "root_position_intervals": root_position_intervals,
        "inversion": inversion,
        "onset": onset,
        "offset": offset,
    }


@pytest.mark.parametrize(
    ("name", "key_annotation", "relative_major_root", "expected_normalized_root"),
    [
        (
            "A minor",
            {"beat": 0, "tonic_pitch_class": 9, "scale_degree_intervals": [2, 1, 2, 2, 1, 2]},
            0,
            0,
        ),
        (
            "E major",
            {"beat": 0, "tonic_pitch_class": 4, "scale_degree_intervals": [2, 2, 1, 2, 2, 2]},
            4,
            0,
        ),
        (
            "D dorian",
            {"beat": 0, "tonic_pitch_class": 2, "scale_degree_intervals": [2, 1, 2, 2, 2, 1]},
            0,
            0,
        ),
    ],
)
def test_relative_major_normalization_maps_relative_major_tonic_to_zero(
    name: str,
    key_annotation: dict[str, object],
    relative_major_root: int,
    expected_normalized_root: int,
) -> None:
    """Relative-major normalization should map the segment's relative-major tonic to C."""
    relative_major_tonic = infer_relative_major_tonic(key_annotation)
    assert relative_major_tonic == relative_major_root, name

    relative_major_chord = make_chord(
        root_pitch_class=relative_major_root,
        root_position_intervals=[4, 3],
    )
    _, normalized_root, _ = chord_to_vector(
        relative_major_chord,
        tonic_pitch_class=relative_major_tonic,
    )
    assert normalized_root == expected_normalized_root, name


def test_chord_encoding_sets_weighted_chroma_and_unit_circle_root() -> None:
    """Chord encoding should use the expected weighted chroma and circular root encoding."""
    c_major = make_chord(root_pitch_class=0, root_position_intervals=[4, 3], inversion=0)

    vector, root_pitch_class, bass_pitch_class = chord_to_vector(c_major)

    assert root_pitch_class == 0
    assert bass_pitch_class == 0
    assert vector[0] == pytest.approx(1.0)
    assert vector[4] == pytest.approx(0.5)
    assert vector[7] == pytest.approx(0.5)
    assert vector[1] == pytest.approx(0.0)
    assert vector[11] == pytest.approx(0.0)

    root_sin = vector[12]
    root_cos = vector[13]
    assert root_sin**2 + root_cos**2 == pytest.approx(1.0, abs=1e-6)
    assert root_sin == pytest.approx(0.0, abs=1e-6)
    assert root_cos == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    ("inversion", "expected_bass_pitch_class"),
    [
        (1, 4),
        (2, 7),
    ],
)
def test_inversion_correctly_selects_bass_pitch_class(
    inversion: int,
    expected_bass_pitch_class: int,
) -> None:
    """Inversions should still select the correct bass note from the chord tones."""
    c_major_inversion = make_chord(
        root_pitch_class=0,
        root_position_intervals=[4, 3],
        inversion=inversion,
    )

    vector, _, bass_pitch_class = chord_to_vector(c_major_inversion)
    assert bass_pitch_class == expected_bass_pitch_class
    assert vector[12] == pytest.approx(0.0, abs=1e-6)
    assert vector[13] == pytest.approx(1.0, abs=1e-6)


def test_duration_encoding_uses_log1p_of_offset_minus_onset() -> None:
    """Duration encoding should be exactly log1p(offset - onset)."""
    chord = make_chord(
        root_pitch_class=7,
        root_position_intervals=[4, 3],
        onset=1.25,
        offset=3.75,
    )

    vector, _, _ = chord_to_vector(chord)
    expected_duration = np.log1p(3.75 - 1.25)
    assert vector[14] == pytest.approx(expected_duration)


def test_segment_splitting_respects_key_boundaries_exactly() -> None:
    """Synthetic songs with key changes should split into segments at the declared beats."""
    song_payload = {
        "annotations": {
            "keys": [
                {"beat": 0, "tonic_pitch_class": 9, "scale_degree_intervals": [2, 1, 2, 2, 1, 2]},
                {"beat": 4, "tonic_pitch_class": 4, "scale_degree_intervals": [2, 2, 1, 2, 2, 2]},
                {"beat": 8, "tonic_pitch_class": 2, "scale_degree_intervals": [2, 1, 2, 2, 2, 1]},
            ],
            "harmony": [
                make_chord(9, [3, 4], onset=0.0, offset=1.0),
                make_chord(0, [4, 3], onset=2.0, offset=3.0),
                make_chord(4, [4, 3], onset=4.0, offset=5.0),
                make_chord(11, [4, 3], onset=6.0, offset=7.0),
                make_chord(2, [3, 4], onset=8.0, offset=9.0),
                make_chord(7, [4, 3], onset=10.0, offset=11.0),
            ],
        }
    }

    segments = extract_tonal_segments("synthetic-song", song_payload)

    assert len(segments) == 3

    first_segment, second_segment, third_segment = segments
    assert first_segment.start_beat == pytest.approx(0.0)
    assert first_segment.end_beat == pytest.approx(4.0)
    assert [chord["onset"] for chord in first_segment.chords] == [0.0, 2.0]

    assert second_segment.start_beat == pytest.approx(4.0)
    assert second_segment.end_beat == pytest.approx(8.0)
    assert [chord["onset"] for chord in second_segment.chords] == [4.0, 6.0]

    assert third_segment.start_beat == pytest.approx(8.0)
    assert third_segment.end_beat is None
    assert [chord["onset"] for chord in third_segment.chords] == [8.0, 10.0]


def test_local_window_tonic_inference_prefers_c_major_for_diatonic_window() -> None:
    """A C-major diatonic window should infer C as local relative-major tonic."""
    chords = [
        make_chord(0, [4, 3]),   # C
        make_chord(7, [4, 3]),   # G
        make_chord(9, [3, 4]),   # Am
        make_chord(5, [4, 3]),   # F
        make_chord(2, [3, 4]),   # Dm
        make_chord(7, [4, 3]),   # G
        make_chord(0, [4, 3]),   # C
        make_chord(4, [3, 4]),   # Em
    ]

    tonic = infer_local_relative_major_tonic(chords)
    assert tonic == 0


def test_window_dataset_returns_expected_shapes() -> None:
    """Window dataset should return [8,15] features and 7-step TPS labels."""
    sequence = HarmonySequence(
        song_id="synthetic",
        chords=tuple(
            make_chord(root, [4, 3], onset=float(index), offset=float(index + 1))
            for index, root in enumerate([0, 7, 9, 5, 2, 7, 0, 4, 11])
        ),
    )

    dataset = ChordPairDataset([sequence], window_size=8, stride=1)
    assert len(dataset) == 2

    x_window, root_window, bass_window, chord_type_window, tps_dist, local_tonic = dataset[0]
    assert tuple(x_window.shape) == (8, 15)
    assert tuple(root_window.shape) == (8,)
    assert tuple(bass_window.shape) == (8,)
    assert tuple(chord_type_window.shape) == (8,)
    assert tuple(tps_dist.shape) == (7,)
    assert local_tonic.ndim == 0
    assert (tps_dist >= 0).all().item() is True