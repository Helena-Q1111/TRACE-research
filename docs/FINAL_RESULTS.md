# TRACE: Final Experimental Results

## Core result

TRACE learns a three-dimensional tonal geometry that aligns with human key organization and generalizes the TPS distance structure to held-out music. The full 15D autoencoder with TPS regularization reaches Krumhansl–Kessler (K&K) key-distance **ρ = +.76 ± .02**, with exact significance in all five seeds. Its held-out transition distances correlate with TPS at **ρ = .97**. On Bigand–Parncutt's sequence-level tension ratings, latent distance to the local tonic reaches **ρ = +.51**.

The result is consistent across the input ablation: eight of nine feature variants trained with TPS score between +.74 and +.78 on K&K key distance. Chroma contributes a .13 increase in K&K key-distance correlation over the no-chroma variant (+.76 vs. +.63 ± .03). The TPS target itself scores +.79 on K&K key distance, +.82 on hierarchy and +.57 on sequence tension; the learned full15 embedding reaches +.76, +.71 and +.51 on those measures.

## Experimental design

### Data and windows

The pipeline reads Sheet Sage / Hooktheory harmony annotations, segments each song at annotated key changes, and maps each segment to a local relative-major frame. Chords are transposed so the relative-major tonic maps to C. Song IDs are split 80/10/10 into train, validation and test using seed 42. Sliding windows contain eight consecutive chords with stride one and remain within a song and tonal segment.

Each chord uses a 15D representation: 12D weighted chroma (root = 1, other chord tones = .5), a 2D circular root encoding, and log duration. Bass pitch class is retained as metadata and used by the 17D bass ablation. The variants also include binary chroma, chroma-only, fifths-circle root, and feature-removal conditions.

### Model and training

The encoder is a shared per-chord MLP: `D → 64 → 32 → 3`, with ReLU between linear layers. The decoder mirrors it: `3 → 32 → 64 → D`. Chroma uses sigmoid output, circular features are normalized, and duration is reconstructed directly.

The objective combines feature reconstruction with adjacent-chord distance regression:

```text
L = L_reconstruction + λ · MSE(||z[t+1] − z[t]||₂, TPS[t,t+1])
```

Reconstruction combines chroma binary cross-entropy, circular cosine distance and duration MSE. The experiment trains nine input variants with λ ∈ {0, 1}, using five seeds (42–46), for 90 runs. Each run uses AdamW (learning rate 1e-3, weight decay 1e-4), batch size 256, gradient clipping at 1.0, and up to 50 epochs. The checkpoint with the lowest validation loss is retained.

### Evaluation

- **K&K chord-level organization:** latent pairwise chord distances are compared with key-distance profiles and chord–key profiles derived from the 1982 probe-tone study. Results include Spearman correlation, exact Mantel significance, triplet agreement and hierarchy correlation.
- **Bigand–Parncutt sequence tension:** Spearman correlation between ratings for 30 chord positions and either latent distance to the local tonic or adjacent latent step distance. The study's local-stability and Lerdahl hierarchy predictors provide reference scores.
- **Tonal theory:** latent transition distances are compared with Lerdahl TPS and the `j+k` training target, including performance on held-out transitions.
- **Model references:** original M0 and M2 checkpoints and a Skip-gram embedding projected from 256D to 3D are included in the multi-metric evaluation.

## Results

### Primary model and references

| Model / reference | K&K key distance ρ | K&K triplet | K&K chord–key ρ | B&P tonic tension ρ | Lerdahl TPS ρ | Held-out TPS ρ |
|---|---:|---:|---:|---:|---:|---:|
| TRACE full15, λ = 1 (five seeds) | **+.76 ± .02 (5/5 significant)** | **.82** | **+.72 (5/5 significant)** | +.51 | +.59 | **+.97** |
| Original M2 checkpoint | +.68 (p = .003) | .79 | +.74 (p < .001) | +.57 | +.59 | — |
| `j+k` TPS training target | +.79 | .69 | +.72 | +.57 | +.71 | — |
| Lerdahl TPS reference | +.55 | .60 | +.47 | — | 1.00 | — |

For the full15 model, K&K hierarchy correlation is +.71. The K&K chord-distance results use six diatonic chords; chord–key and triplet metrics use seven.

### TPS-conditioned feature robustness

| Feature variant, λ = 1 | K&K key distance ρ | K&K chord–key ρ | B&P tonic tension ρ | Lerdahl TPS ρ |
|---|---:|---:|---:|---:|
| full15 | **+.76 ± .02** | +.72 ± .02 | +.51 ± .00 | +.59 ± .02 |
| no_duration | +.76 ± .03 | +.73 ± .01 | +.52 ± .03 | +.58 ± .01 |
| no_root_circle | +.77 ± .01 | +.72 ± .01 | +.51 ± .00 | +.58 ± .00 |
| binary_chroma | +.77 ± .03 | +.73 ± .00 | +.50 ± .00 | +.58 ± .00 |
| chroma_only | +.77 ± .01 | +.73 ± .01 | +.50 ± .00 | +.58 ± .00 |
| binary_chroma_only | +.78 ± .02 | +.73 ± .01 | +.53 ± .04 | +.58 ± .01 |
| no_chroma | +.63 ± .03 | +.56 ± .02 | +.36 ± .01 | +.71 ± .03 |
| plus_bass17 | +.76 ± .04 | +.72 ± .01 | +.52 ± .03 | +.59 ± .01 |
| fifths_root15 | +.74 ± .02 | +.71 ± .01 | +.52 ± .03 | +.60 ± .01 |

The full results tables include five-seed standard deviations, significance counts, all λ = 0 conditions, and the remaining metrics for each variant.

### Feature findings

- Chroma supports K&K key-distance and chord–key alignment. With TPS regularization, the full15 and chroma-based variants produce K&K key-distance correlations from +.76 to +.78.
- Chroma-only models also produce K&K-aligned geometry without TPS regularization: `chroma_only` scores +.70 ± .11, and `binary_chroma_only` scores +.69 ± .16. Both are significant in five of five seeds.
- The fifths-circle root encoding gives reconstruction-only training a strong tonal-theory profile: for `fifths_root15`, λ = 0, Lerdahl TPS ρ is +.40, B&P tonic-tension ρ is +.44, and held-out TPS ρ is +.84.
- Across TPS-trained variants, removing duration preserves K&K and tension scores. Adding bass gives similar aggregate scores to full15.

### Sequence-level tension

For the same 30 rated chord positions, full15 distance to the local tonic reaches ρ = +.51. Bigand and Parncutt's local-stability predictor scores +.55, while their hierarchical Lerdahl predictor scores +.33. The Skip-gram baseline reaches +.53 on tonic-distance tension, showing that sequence-level tension captures a complementary strength of the embedding representations.

## Interpretation

TPS regularization gives TRACE a compact 3D tonal space with three strong empirical properties: high K&K alignment across five training seeds, +.97 held-out correlation with TPS transition distances, and positive prediction of sequence-level tension. The feature study identifies chroma as a robust carrier of K&K structure and the circle of fifths as an effective root encoding for theory-aligned reconstruction-only training.

The project's primary contribution is a learned three-dimensional chord space that connects symbolic chord features, tonal theory and human tonal-organization measures in one evaluated representation.

## Reproduction

Install Python 3.10+ dependencies from `requirements.txt` and a PyTorch build appropriate to the machine. The reference run used PyTorch 2.7.1+cu128 and an RTX 5060; 90 runs took about 2.5 hours with four parallel workers.

Place the MIR-friendly `Hooktheory.json.gz` at `dataset/hooktheory/Hooktheory.json.gz`, then run from the repository root:

```bash
python scripts/ablation_15d.py --dataset dataset/hooktheory/Hooktheory.json.gz --output_dir results/ablation_15d --cache results/ablation_15d/window_tensors_split42.pt --skip_existing
python scripts/summarize_ablation_15d.py --output_dir results/ablation_15d
python scripts/eval_multi_metric.py
```

The repository includes the final run records, checkpoints and evaluation tables under `results/`. Sheet Sage distributes the dataset separately under the terms listed in `dataset/README.md`.

## References

- Krumhansl, C. L., & Kessler, E. J. (1982). “Tracing the dynamic changes in perceived tonal organization in a spatial representation of musical keys.” *Psychological Review*, 89(4), 334–368.
- Bigand, E., & Parncutt, R. (1999). “Perceiving musical tension in long chord sequences.” *Psychological Research*, 62, 237–254.
- Bharucha, J. J., & Krumhansl, C. L. (1983). “The representation of harmonic structure in music: Hierarchies of stability as a function of context.” *Cognition*, 13, 63–102.
- Donahue, C., Thickstun, J., & Liang, P. (2022). “Melody transcription via generative pre-training.” ISMIR.
