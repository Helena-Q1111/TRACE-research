"""Multi-metric evaluation of every TRACE model against the references in the project PDFs.

Models: the 90 ablation checkpoints (outputs/ablation_15d/checkpoints), the original
M0 / M2 checkpoints, and the Skip-gram (PCA 256 -> 3D, as in eval_cognitive_comparison).

Metric families
  A  human-derived, chord level (Krumhansl & Kessler 1982 profiles; see cognitive_reference.py)
       kk_key       Spearman rho + exact Mantel p, Kendall tau-b, triplet agreement (6 chords)
       kk_chordkey  same (7 chords)
       hierarchy    Spearman(distance from I, 1 - KK key fit), exact p over 6! perms
  B  human behavioural, sequence level (Bigand & Parncutt 1999 Exp. 1, bp1999_sequence.py)
       bp_tonic     Spearman(|z_t - z_I|, nonmusician tension), chords 2-31
       bp_seq       Spearman(|z_t - z_{t-1}|, tension)
     Each chord is encoded like the training pipeline: the 8-chord window ending at t
     sets the local relative-major frame (data.hooktheory_loader).
  C  theory, circular for TPS-trained models: lerdahl_tps rho, simple_tps rho
  D  legacy hand matrix rho, B&K (1983) Fig. 2 rho (reported in CSV only)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import bp1999_sequence as bp
from ablation_15d import VARIANTS, AblationAutoencoder, FeatureConfig, build_features
from autoencoder import ChordAutoencoder
from cognitive_reference import CHORD_NAMES, TRIADS, hierarchy_score, score
from data.chord_encoder import chord_to_vector
from data.hooktheory_loader import infer_local_relative_major_tonic

OUT = PROJECT_ROOT / "results" / "multi_metric"
ORDER = ["full15", "no_duration", "no_root_circle", "binary_chroma", "chroma_only",
         "binary_chroma_only", "no_chroma", "plus_bass17", "fifths_root15"]


def triad_dict(name: str) -> dict:
    root, pcs = TRIADS[name]
    third, fifth = (pcs[1] - pcs[0]) % 12, (pcs[2] - pcs[1]) % 12
    return {"root_pitch_class": root, "root_position_intervals": [third, fifth], "inversion": 0, "onset": 0.0, "offset": 1.0}


def encode_15d(chords: list[dict], tonics: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    rows, basses = [], []
    for c, t in zip(chords, tonics):
        v, _, bass = chord_to_vector(c, tonic_pitch_class=t)
        rows.append(torch.from_numpy(v))
        basses.append(bass)
    return torch.stack(rows), torch.tensor(basses)


# ---- sequence frames (shared by all models) ------------------------------------------
SEQ = [bp.as_hooktheory(i) for i in range(len(bp.CHORDS))]
WINDOW_TONIC = [infer_local_relative_major_tonic(SEQ[max(0, t - 7) : t + 1]) for t in range(len(SEQ))]
# chord t and t-1 are both encoded in the frame of the window ending at t
CUR = [(t, WINDOW_TONIC[t]) for t in range(1, len(SEQ))]
PREV = [(t - 1, WINDOW_TONIC[t]) for t in range(1, len(SEQ))]


def ae_encoder(model, cfg: FeatureConfig | None):
    def enc(chords: list[dict], tonics: list[int]) -> np.ndarray:
        x15, bass = encode_15d(chords, tonics)
        x = build_features(x15, bass, cfg) if cfg is not None else x15
        with torch.no_grad():
            return model.encode(x).numpy()
    return enc


def skipgram_encoder():
    emb = np.load(PROJECT_ROOT / "results/skipgram_baseline/chord_embeddings.npy")
    vocab = json.loads((PROJECT_ROOT / "results/skipgram_baseline/vocabulary.json").read_text(encoding="utf-8"))["chord2idx"]
    pca = PCA(n_components=3).fit(emb)

    def enc(chords: list[dict], tonics: list[int]) -> np.ndarray:
        out = []
        for c, t in zip(chords, tonics):
            sym = bp.PC_NAMES[(c["root_pitch_class"] - t) % 12] + bp.QUALITY_SUFFIX[tuple(c["root_position_intervals"])]
            out.append(emb[vocab.get(sym, vocab["<UNK>"])])
        return pca.transform(np.array(out))
    return enc


def evaluate_distances(d7: np.ndarray | None, tonic_dist, step_dist) -> dict[str, float]:
    m: dict[str, float] = {}
    if d7 is not None:
        for ref in ["kk_key", "kk_chordkey", "lerdahl_tps", "simple_tps", "legacy_hand", "bk1983_fig2"]:
            for k, v in score(d7, ref).items():
                m[f"{ref}_{k}"] = v
        h = hierarchy_score(d7)
        m["hierarchy_rho"], m["hierarchy_p"] = h["rho"], h["p"]
    tension = np.array(bp.NONMUSICIAN_TENSION)
    m["bp_tonic_rho"] = float(spearmanr(tonic_dist, tension).statistic) if tonic_dist is not None else float("nan")
    m["bp_seq_rho"] = float(spearmanr(step_dist, tension).statistic) if step_dist is not None else float("nan")
    return m


def evaluate(enc) -> dict[str, float]:
    z7 = enc([triad_dict(n) for n in CHORD_NAMES], [0] * 7)
    z_cur = enc([SEQ[i] for i, _ in CUR], [t for _, t in CUR])
    z_prev = enc([SEQ[i] for i, _ in PREV], [t for _, t in PREV])
    z_tonic = enc([triad_dict("I")], [0])[0]
    return evaluate_distances(np.linalg.norm(z7[:, None] - z7[None], axis=-1),
                              np.linalg.norm(z_cur - z_tonic, axis=1), np.linalg.norm(z_cur - z_prev, axis=1))


def _pcs(chord: dict, tonic: int) -> tuple[int, set[int]]:
    root = (chord["root_pitch_class"] - tonic) % 12
    pcs, cur = {root}, root
    for iv in chord["root_position_intervals"]:
        cur = (cur + iv) % 12
        pcs.add(cur)
    return root, pcs


def _simple_tps(a: tuple[int, set[int]], b: tuple[int, set[int]]) -> float:
    diff = abs(7 * a[0] % 12 - 7 * b[0] % 12)
    return min(diff, 12 - diff) + len(a[1] ^ b[1])


def oracle_rows() -> list[dict]:
    """Score the references themselves, to tell what a model adds beyond its target."""
    from cognitive_reference import KK_MAJOR, lerdahl_tps, simple_tps

    cur = [_pcs(SEQ[i], t) for i, t in CUR]
    prev = [_pcs(SEQ[i], t) for i, t in PREV]
    tonic = (0, {0, 4, 7})
    tps_tonic = [_simple_tps(c, tonic) for c in cur]
    tps_step = [_simple_tps(c, p) for c, p in zip(cur, prev)]

    def kk_misfit(p: set[int]) -> float:
        t = np.zeros(12)
        t[list(p)] = 1
        return 1 - np.corrcoef(t, KK_MAJOR)[0, 1]

    base = {"lambda_tps": None, "seed": None}
    return [
        {"model": "[target] simple TPS j+k", **base, **evaluate_distances(simple_tps(), tps_tonic, tps_step)},
        {"model": "[theory] Lerdahl TPS", **base, **evaluate_distances(lerdahl_tps(), None, None)},
        {"model": "[human] KK key misfit", **base, **evaluate_distances(None, [kk_misfit(c[1]) for c in cur], None)},
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ckpt in sorted((PROJECT_ROOT / "results/ablation_15d/checkpoints").glob("*.pt")):
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = FeatureConfig(**c["feature_config"])
        model = AblationAutoencoder(cfg)
        model.load_state_dict(c["model_state_dict"])
        model.eval()
        rows.append({"model": cfg.name, "lambda_tps": c["lambda_tps"], "seed": c["seed"], **evaluate(ae_encoder(model, cfg))})
    for name, path in [("M0 original", "results/original_models/output_ae_vanilla/best_model.pt"), ("M2 original", "results/original_models/output_ae/best_model.pt")]:
        model = ChordAutoencoder()
        model.load_state_dict(torch.load(PROJECT_ROOT / path, map_location="cpu", weights_only=False)["model_state_dict"])
        model.eval()
        rows.append({"model": name, "lambda_tps": None, "seed": None, **evaluate(ae_encoder(model, None))})
    rows.append({"model": "Skip-gram PCA3D", "lambda_tps": None, "seed": None, **evaluate(skipgram_encoder())})
    rows += oracle_rows()

    (OUT / "per_model.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    keys = [k for k in rows[0] if k not in ("model", "lambda_tps", "seed")]
    rows = [{k: r.get(k, float("nan")) for k in ["model", "lambda_tps", "seed", *keys]} for r in rows]
    (OUT / "per_model.csv").write_text(
        "\n".join([",".join(["model", "lambda_tps", "seed", *keys])]
                  + [",".join(str(r[k]) for k in ["model", "lambda_tps", "seed", *keys]) for r in rows]), encoding="utf-8")
    write_summary(rows)


def write_summary(rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["lambda_tps"])].append(r)

    def cell(rs, key, sig=None):
        a = np.array([r.get(key, np.nan) for r in rs], dtype=float)
        if np.all(np.isnan(a)):
            return "—"
        s = f"{np.nanmean(a):+.2f}" + (f" ± {np.nanstd(a, ddof=1):.2f}" if len(a) > 1 else "")
        if sig:
            s += f" ({sum(r.get(sig, 1) < 0.05 for r in rs)}/{len(rs)})"
        return s

    cols = [("KK key ρ", "kk_key_rho", "kk_key_p"), ("KK key triplet", "kk_key_triplet", None),
            ("KK chord-key ρ", "kk_chordkey_rho", "kk_chordkey_p"), ("hierarchy ρ", "hierarchy_rho", "hierarchy_p"),
            ("B&P tension (tonic dist) ρ", "bp_tonic_rho", None), ("B&P tension (step) ρ", "bp_seq_rho", None),
            ("Lerdahl TPS ρ*", "lerdahl_tps_rho", None), ("legacy ρ", "legacy_hand_rho", "legacy_hand_p")]
    lines = ["# Multi-metric evaluation", "",
             "Mean ± SD over 5 seeds for ablation groups; (k/n) = runs with exact permutation p < 0.05. "
             "* = circular for TPS-trained models (TPS is the training target). "
             "B&P tension: Spearman with nonmusicians' ratings, n = 30 chords; |ρ| > 0.36 is nominally p < .05, "
             "but ratings are serially dependent, so treat as descriptive.", "",
             "| model | λ | " + " | ".join(c[0] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    order = [(v, 1.0) for v in ORDER] + [(v, 0.0) for v in ORDER] + [("M2 original", None), ("M0 original", None), ("Skip-gram PCA3D", None),
                                                                      ("[target] simple TPS j+k", None), ("[theory] Lerdahl TPS", None), ("[human] KK key misfit", None)]
    for key in order:
        rs = groups.get(key)
        if rs:
            lam = "—" if key[1] is None else f"{key[1]:g}"
            lines.append(f"| {key[0]} | {lam} | " + " | ".join(cell(rs, k, s) for _, k, s in cols) + " |")
    lines += ["", "Baselines on the B&P sequence (paper's own predictors, same 30 chords): "
              "Lerdahl sequential ρ = +0.18, hierarchic value ρ = +0.33, local stability ρ = +0.55."]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

