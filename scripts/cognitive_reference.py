"""Reference matrices and scoring for the 7 C-major diatonic triads.

Order everywhere: I, ii, iii, IV, V, vi, vii°.

Sources (PDFs in the project root):
  Krumhansl & Kessler (1982), Psych. Review 89:334-368  -> KK_MAJOR / KK_MINOR probe-tone
      profiles. The paper plots them (Fig. 2) without printing numbers; the values below
      are the standard published ones and match the project's
      Krumhansl-Kesslers-1982-...webp figure to r = .997 (pixel measurement).
  Bigand & Parncutt (1999), Psych. Research 62:237-254  -> Lerdahl (1988) TPS chord
      distances used as tension predictors (Tables 1-2); lerdahl_tps() reproduces all six
      diatonic pairs listed there (I-V 5, I-ii 8, I-vi 7, IV-V 8, vi-ii 5, V-vi 8).
  Bharucha & Krumhansl (1983), Cognition 13:63-102  -> Fig. 2 clustering (no numeric
      matrix is published); kept for comparison only.

Human-derived references (empirical data enters through the KK profiles):
  kk_key        1 - r(KK profile of the key each chord is tonic of); vii° has no key,
                so 6 chords / 15 pairs. This is KK's interkey-distance measure.
  kk_chordkey   each chord's 24-d vector of correlations with the 24 KK key profiles
                (KK's chord-key analysis, with a chord-tone template standing in for
                the measured chord profiles); distance = 1 - r between vectors.
Theory references (TPS is the training target, so these are circular for λ>0):
  lerdahl_tps   Lerdahl (1988) δ = j + k with i = 0 (single region)
  simple_tps    the simplified j + k used as training target (chord_dataset.py)
Other:
  bk1983_fig2   cophenetic distance from B&K (1983) Fig. 2 (7 - merge height)
  legacy_hand   hand-entered matrix in eval_cognitive_comparison.get_krumhansl_reference()
"""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

CHORD_NAMES = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
TRIADS = {  # root, pitch classes (root, third, fifth)
    "I": (0, (0, 4, 7)), "ii": (2, (2, 5, 9)), "iii": (4, (4, 7, 11)), "IV": (5, (5, 9, 0)),
    "V": (7, (7, 11, 2)), "vi": (9, (9, 0, 4)), "vii°": (11, (11, 2, 5)),
}
TONIC_KEY = {"I": (0, "maj"), "ii": (2, "min"), "iii": (4, "min"), "IV": (5, "maj"), "V": (7, "maj"), "vi": (9, "min")}

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_profile(tonic: int, mode: str) -> np.ndarray:
    return np.roll(KK_MAJOR if mode == "maj" else KK_MINOR, tonic)


def chord_template(name: str) -> np.ndarray:
    t = np.zeros(12)
    t[list(TRIADS[name][1])] = 1.0
    return t


def kk_key() -> np.ndarray:
    names = [n for n in CHORD_NAMES if n in TONIC_KEY]
    profiles = [key_profile(*TONIC_KEY[n]) for n in names]
    return 1.0 - np.corrcoef(profiles)


def kk_chordkey_vectors() -> np.ndarray:
    keys = [key_profile(t, m) for m in ("maj", "min") for t in range(12)]
    return np.array([[np.corrcoef(chord_template(n), k)[0, 1] for k in keys] for n in CHORD_NAMES])


def kk_chordkey() -> np.ndarray:
    return 1.0 - np.corrcoef(kk_chordkey_vectors())


def kk_key_fit() -> np.ndarray:
    """r(chord template, C-major KK profile): how well each chord fits the key (stability)."""
    return np.array([np.corrcoef(chord_template(n), KK_MAJOR)[0, 1] for n in CHORD_NAMES])


_DIATONIC_FIFTHS = [0, 7, 2, 9, 4, 11, 5]  # C G D A E B F: Lerdahl's diatonic chord circle


def _basic_space(name: str) -> list[set[int]]:
    root, (r, third, fifth) = TRIADS[name]
    return [{root}, {root, fifth}, {r, third, fifth}, {0, 2, 4, 5, 7, 9, 11}, set(range(12))]


def lerdahl_tps() -> np.ndarray:
    n = len(CHORD_NAMES)
    d = np.zeros((n, n))
    for i, a in enumerate(CHORD_NAMES):
        for j, b in enumerate(CHORD_NAMES):
            if i == j:
                continue
            pa, pb = _DIATONIC_FIFTHS.index(TRIADS[a][0]), _DIATONIC_FIFTHS.index(TRIADS[b][0])
            steps = abs(pa - pb)
            jd = min(steps, 7 - steps)
            kd = sum(len(lb - la) for la, lb in zip(_basic_space(a), _basic_space(b)))
            d[i, j] = jd + kd
    return d


def simple_tps() -> np.ndarray:
    n = len(CHORD_NAMES)
    d = np.zeros((n, n))
    for i, a in enumerate(CHORD_NAMES):
        for j, b in enumerate(CHORD_NAMES):
            ra, rb = TRIADS[a][0], TRIADS[b][0]
            diff = abs(7 * ra % 12 - 7 * rb % 12)
            d[i, j] = min(diff, 12 - diff) + len(set(TRIADS[a][1]) ^ set(TRIADS[b][1]))
    return d


_FIG2_LEVELS = [
    ({"I", "V"}, 6.03), ({"I", "V", "IV"}, 5.35), ({"ii", "vi"}, 4.97),
    ({"I", "V", "IV", "ii", "vi"}, 4.87), ({"I", "V", "IV", "ii", "vi", "vii°"}, 4.30), (set(CHORD_NAMES), 4.08),
]


def bk1983_fig2_cophenetic() -> np.ndarray:
    n = len(CHORD_NAMES)
    d = np.zeros((n, n))
    for i, a in enumerate(CHORD_NAMES):
        for j, b in enumerate(CHORD_NAMES):
            if i != j:
                d[i, j] = 7.0 - next(h for members, h in _FIG2_LEVELS if {a, b} <= members)
    return d


def legacy_hand_estimated() -> np.ndarray:
    return np.array([
        [0.00, 0.70, 0.80, 0.45, 0.30, 0.50, 0.95],
        [0.70, 0.00, 0.65, 0.75, 0.60, 0.65, 0.75],
        [0.80, 0.65, 0.00, 0.85, 0.80, 0.60, 0.80],
        [0.45, 0.75, 0.85, 0.00, 0.65, 0.60, 0.90],
        [0.30, 0.60, 0.80, 0.65, 0.00, 0.70, 0.50],
        [0.50, 0.65, 0.60, 0.60, 0.70, 0.00, 0.90],
        [0.95, 0.75, 0.80, 0.90, 0.50, 0.90, 0.00],
    ])


# name -> (builder, chord subset it covers)
REFERENCES = {
    "kk_key": (kk_key, [n for n in CHORD_NAMES if n in TONIC_KEY]),
    "kk_chordkey": (kk_chordkey, CHORD_NAMES),
    "lerdahl_tps": (lerdahl_tps, CHORD_NAMES),
    "simple_tps": (simple_tps, CHORD_NAMES),
    "bk1983_fig2": (bk1983_fig2_cophenetic, CHORD_NAMES),
    "legacy_hand": (legacy_hand_estimated, CHORD_NAMES),
}


def subset(dist: np.ndarray, names: list[str]) -> np.ndarray:
    idx = [CHORD_NAMES.index(n) for n in names]
    return dist[np.ix_(idx, idx)]


def _upper(d: np.ndarray) -> np.ndarray:
    return d[np.triu_indices(d.shape[0], k=1)]


def mantel_exact(dist: np.ndarray, ref: np.ndarray, method: str = "spearman") -> tuple[float, float]:
    """Correlation of two distance matrices with an exact one-sided Mantel p-value.

    The pairwise distances are not independent (each chord enters n-1 pairs), so
    parametric p-values are invalid; this permutes the chord labels of `dist` over all
    n! orderings (5040 for 7 chords).
    """
    iu = np.triu_indices(dist.shape[0], k=1)
    ref_v = rankdata(ref[iu]) if method == "spearman" else ref[iu].astype(float)
    ref_v = (ref_v - ref_v.mean()) / np.linalg.norm(ref_v - ref_v.mean())

    def corr(d: np.ndarray) -> float:
        v = rankdata(d[iu]) if method == "spearman" else d[iu].astype(float)
        v = v - v.mean()
        norm = np.linalg.norm(v)
        return float(v @ ref_v / norm) if norm > 0 else float("nan")

    observed = corr(dist)
    perms = np.array([corr(dist[np.ix_(p, p)]) for p in map(list, permutations(range(dist.shape[0])))])
    return observed, float(np.mean(perms >= observed - 1e-12))


def triplet_agreement(dist: np.ndarray, ref: np.ndarray) -> float:
    """Share of anchored triplets (a; b, c) where dist and ref agree on whether b or c is closer to a.

    Triplets tied in the reference are skipped. Chance = 0.5.
    """
    n, hit, tot = dist.shape[0], 0, 0
    for a in range(n):
        for b, c in combinations([x for x in range(n) if x != a], 2):
            r = np.sign(ref[a, b] - ref[a, c])
            if r == 0:
                continue
            tot += 1
            hit += np.sign(dist[a, b] - dist[a, c]) == r
    return hit / tot


def score(dist7: np.ndarray, ref_name: str) -> dict[str, float]:
    """All matrix metrics of a 7x7 latent distance matrix against one reference."""
    builder, names = REFERENCES[ref_name]
    ref, d = builder(), subset(dist7, names)
    rho, p = mantel_exact(d, ref)
    tau = kendalltau(_upper(d), _upper(ref)).statistic
    return {"rho": rho, "p": p, "tau_b": float(tau), "triplet": triplet_agreement(d, ref)}


def hierarchy_score(dist7: np.ndarray) -> dict[str, float]:
    """Tonal hierarchy: is distance from I ordered by (lack of) KK key fit? 6 chords, exact p over 720 perms."""
    fit = kk_key_fit()[1:]
    from_tonic = dist7[0, 1:]
    rho = spearmanr(from_tonic, -fit).statistic
    perms = [spearmanr(from_tonic[list(p)], -fit).statistic for p in permutations(range(6))]
    return {"rho": float(rho), "p": float(np.mean(np.array(perms) >= rho - 1e-12))}
