"""Bigand & Parncutt (1999) Exp. 1: 31-chord sequence (C major -> F major -> B major) and tension ratings.

Chords: reconstructed from Table 2 (roman numerals per section) and Fig. 2 (score). A
seventh is added where Table 2's local-stability score includes the +3 "non-triadic tone"
term (chords 3-6, 9, 12, 14, 18, 22, 24, 29), which also matches the accidentals legible
in the score (F# at 6, Bb at 12/16/18, sharps at 22/26). Inversions are ignored (the
15D input has no bass), so every chord is encoded in root position.

Ratings: nonmusicians' mean tension (0-10) for chords 2-31, digitized from Fig. 3
(1670x1198 raster; axis calibrated on the 0-10 ticks with piecewise-linear
interpolation). Typical error ±0.2; chords 7, 11, 18, 31 were read by eye after
automatic detection was confounded by error bars (±0.5). Musicians were not digitized;
the paper reports r = .94 between the two groups' means.

Baseline predictors are copied from the paper: Lerdahl sequential distance and local
stability (Table 2), hierarchic value (Table 1).
"""

from __future__ import annotations

MAJ, MIN, DIM, DOM7, MIN7, HDIM7 = [4, 3], [3, 4], [3, 3], [4, 3, 3], [3, 4, 3], [3, 3, 4]

# (label, root pitch class, root_position_intervals) - absolute pitch
CHORDS = [
    ("C: I", 0, MAJ), ("vi", 9, MIN), ("vii7", 11, HDIM7), ("iii7", 4, MIN7), ("vi7", 9, MIN7),
    ("V7/V", 2, DOM7), ("V", 7, MAJ), ("ii/V (Am)", 9, MIN), ("ii7/I", 2, MIN7), ("V", 7, MAJ),
    ("I", 0, MAJ),
    ("F: V7/IV", 0, DOM7), ("I", 5, MAJ), ("V7", 0, DOM7), ("I", 5, MAJ), ("IV", 10, MAJ),
    ("V", 0, MAJ), ("V7/IV", 0, DOM7), ("IV", 10, MAJ), ("V", 0, MAJ), ("I", 5, MAJ),
    ("B: V7/#IV", 6, DOM7), ("I", 11, MAJ), ("V7", 6, DOM7), ("I", 11, MAJ), ("IV", 4, MAJ),
    ("V", 6, MAJ), ("vi", 8, MIN), ("ii7", 1, MIN7), ("V", 6, MAJ), ("I", 11, MAJ),
]
SECTION_TONIC = [0] * 11 + [5] * 10 + [11] * 10

# chords 2..31
NONMUSICIAN_TENSION = [
    6.50, 7.05, 6.09, 4.68, 6.54, 1.90, 6.41, 7.00, 6.18, 1.20,
    7.56, 6.62, 7.29, 5.42, 8.22, 6.10, 8.60, 5.62, 5.58, 2.27,
    8.93, 7.40, 7.72, 4.85, 6.70, 6.17, 5.13, 6.54, 6.38, 1.10,
]

LERDAHL_SEQUENTIAL = [7, 8, 5, 5, 6, 5, 8, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 5, 8, 5, 36, 5, 5, 5, 5, 8, 8, 5, 5, 5]
HIERARCHIC_VALUE = [7, 15, 12, 7, 12, 7, 15, 10, 5, 0, 12, 7, 12, 7, 20, 12, 25, 20, 12, 7, 42, 37, 42, 37, 50, 42, 50, 47, 42, 37]
LOCAL_STABILITY = [1, 4, 4, 4, 4, 1, 1, 7, 1, 0, 7, 1, 7, 4, 1, 0, 7, 4, 0, 1, 4, 0, 7, 0, 4, 1, 1, 7, 1, 0]

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITY_SUFFIX = {tuple(MAJ): "", tuple(MIN): "m", tuple(DIM): "dim", tuple(DOM7): "7", tuple(MIN7): "m7", tuple(HDIM7): "m7b5"}


def as_hooktheory(i: int, beat: float = 1.0) -> dict:
    """Chord i in the Hooktheory dict format expected by data/ (one beat long)."""
    _, root, intervals = CHORDS[i]
    return {"root_pitch_class": root, "root_position_intervals": list(intervals), "inversion": 0,
            "onset": float(i) * beat, "offset": float(i + 1) * beat}


def symbol(i: int, tonic: int = 0) -> str:
    """Chord-symbol name (Skip-gram vocabulary format) after transposing tonic -> C."""
    _, root, intervals = CHORDS[i]
    return PC_NAMES[(root - tonic) % 12] + QUALITY_SUFFIX[tuple(intervals)]
