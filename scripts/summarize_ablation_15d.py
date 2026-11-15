"""Aggregate outputs/ablation_15d/runs/*.json into summary tables.

Cognitive alignment is re-scored from the stored 7-chord latents (no torch needed):
    kk_key       Krumhansl & Kessler (1982) interkey distance of the keys each chord is
                 tonic of (6 chords; primary, see cognitive_reference.py)
    legacy_hand  hand-entered matrix used by eval_cognitive_comparison.py
The full multi-metric table is produced by eval_multi_metric.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cognitive_reference import CHORD_NAMES, REFERENCES, mantel_exact, subset

ORDER = ["full15", "no_duration", "no_root_circle", "binary_chroma", "chroma_only",
         "binary_chroma_only", "no_chroma", "plus_bass17", "fifths_root15"]
MASK = np.triu(np.ones((7, 7), dtype=bool), k=1)
REF = {name: (fn(), names) for name, (fn, names) in REFERENCES.items()}


def rho(run: dict, ref: str) -> tuple[float, float]:
    """(Spearman rho, exact one-sided Mantel p) of the run's 7-chord latent distances."""
    z = np.array([run["cognitive"]["latents"][n] for n in CHORD_NAMES])
    d = np.linalg.norm(z[:, None] - z[None], axis=-1)
    ref_mat, names = REF[ref]
    return mantel_exact(subset(d, names), ref_mat) if np.var(d[MASK]) > 1e-10 else (float("nan"), float("nan"))


def ms(values, fmt: str = "{:+.3f}") -> str:
    a = np.asarray(values, dtype=float)
    return f"{fmt.format(np.nanmean(a))} ± {np.nanstd(a, ddof=1) if len(a) > 1 else 0:.3f}"


def compare(a, b) -> tuple[str, str, str]:
    if len(a) < 2 or len(b) < 2:
        return "—", "—", "—"
    return (f"{np.mean(a) - np.mean(b):+.3f}", f"{ttest_ind(a, b, equal_var=False).pvalue:.3g}",
            f"{mannwhitneyu(a, b, alternative='two-sided').pvalue:.3g}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=str(Path("outputs") / "ablation_15d"))
    out = Path(p.parse_args().output_dir)

    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((out / "runs").glob("*.json"))]
    for r in runs:
        r["rho_kk"], r["p_kk"] = rho(r, "kk_key")
        r["rho_legacy"], r["p_legacy"] = rho(r, "legacy_hand")
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["variant"], r["lambda_tps"])].append(r)
    col = lambda rs, key: [r[key] for r in rs]
    test = lambda rs, key: [r["test"][key] for r in rs]
    sig = lambda rs, key: f"{sum(r[key] < 0.05 for r in rs)}/{len(rs)}"

    lines = ["# 15D input ablation results", "",
             f"{len(runs)} runs. Mean ± sample std over init seeds (song split fixed at seed 42). "
             "**ρ KK** = Spearman(latent distance, Krumhansl & Kessler 1982 interkey distance) over the 15 pairs of "
             "I ii iii IV V vi (primary). **ρ legacy** = same against the hand-entered matrix of "
             "eval_cognitive_comparison.py (21 pairs). Δ/p compare each variant with full15 on ρ KK "
             "(Welch t-test, Mann-Whitney U). **sig** = runs whose own exact Mantel test (all chord-label "
             "permutations, one-sided) gives p < 0.05. Test TPS ρ = Spearman(latent step length, TPS target) over all "
             "adjacent transitions of the held-out test split.", ""]
    csv = ["variant,lambda_tps,input_dim,n,rho_kk_mean,rho_kk_std,rho_legacy_mean,rho_legacy_std,"
           "test_recon,test_tps_mse,test_tps_spearman"]

    for lam in sorted({k[1] for k in groups}, reverse=True):
        base = col(groups.get(("full15", lam), []), "rho_kk")
        lines += [f"## λ_tps = {lam:g}", "",
                  "| variant | dim | n | ρ KK | sig KK | ρ legacy | sig legacy | test recon | test TPS MSE | test TPS ρ | Δρ KK vs full | p Welch | p MWU |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in ORDER:
            rs = groups.get((v, lam))
            if not rs:
                continue
            f2, lg = col(rs, "rho_kk"), col(rs, "rho_legacy")
            d, pw, pm = compare(f2, base) if v != "full15" else ("—", "—", "—")
            lines.append(f"| {v} | {rs[0]['input_dim']} | {len(rs)} | {ms(f2)} | {sig(rs, 'p_kk')} | {ms(lg)} | {sig(rs, 'p_legacy')} | "
                         f"{ms(test(rs, 'recon'), '{:.4f}')} | {ms(test(rs, 'tps'), '{:.3f}')} | "
                         f"{ms(test(rs, 'tps_spearman'))} | {d} | {pw} | {pm} |")
            csv.append(",".join(str(x) for x in [
                v, lam, rs[0]["input_dim"], len(rs), np.mean(f2), np.std(f2, ddof=1) if len(f2) > 1 else 0,
                np.mean(lg), np.std(lg, ddof=1) if len(lg) > 1 else 0, np.mean(test(rs, "recon")),
                np.mean(test(rs, "tps")), np.mean(test(rs, "tps_spearman"))]))
        lines.append("")

    if {0.0, 1.0} <= {k[1] for k in groups}:
        lines += ["## TPS effect per variant (λ=1 vs λ=0, ρ KK)", "",
                  "| variant | λ=1 | λ=0 | Δ | p Welch | p MWU |", "|---|---|---|---|---|---|"]
        for v in ORDER:
            a, b = groups.get((v, 1.0)), groups.get((v, 0.0))
            if a and b:
                d, pw, pm = compare(col(a, "rho_kk"), col(b, "rho_kk"))
                lines.append(f"| {v} | {ms(col(a, 'rho_kk'))} | {ms(col(b, 'rho_kk'))} | {d} | {pw} | {pm} |")
        lines.append("")

    inv = groups.get(("plus_bass17", 1.0))
    if inv:
        lines += ["## Inversion offsets (plus_bass17, λ=1)", "",
                  f"‖z(C/E) − z(C)‖ = {ms([r['inversion']['C_to_C/E'] for r in inv], '{:.3f}')}, "
                  f"‖z(C/G) − z(C)‖ = {ms([r['inversion']['C_to_C/G'] for r in inv], '{:.3f}')} "
                  "(adjacent-chord TPS targets are ~1–8, so compare against that scale)", ""]

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.csv").write_text("\n".join(csv), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
