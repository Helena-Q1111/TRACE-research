# TRACE Research: Tension-Regularized Chord Embeddings

Research code and results for TRACE (Tension-Regularized Auto-encoded Chord Embeddings), a study of 3D chord representations trained with a Tonal Pitch Space (TPS) distance prior. This repository contains the final multi-seed ablation and evaluation snapshot and accompanies a separate TRACE application repository.

## Research question and scope

Chord embeddings turn harmonic relationships into a geometry that can be explored and measured. TRACE asks how a local sequence of chords and a tonal-theory distance objective can shape a compact 3D space that reflects human tonal organization.

To answer this question, the project builds eight-chord windows, encodes each chord with a 15D representation, and trains an autoencoder to preserve chord features while arranging adjacent chords according to Tonal Pitch Space (TPS) distances. A 90-run experiment compares nine input variants, TPS and reconstruction-only objectives, and five random seeds. The learned spaces are evaluated against Krumhansl–Kessler key profiles, Bigand–Parncutt tension ratings, tonal-theory references and held-out transitions.

The full 15D TPS model reaches K&K key-distance ρ = **+.76 ± .02**, significant in all five seeds, and held-out transition alignment ρ = **.97**. Its distance to the local tonic predicts the Bigand–Parncutt tension sequence at **ρ = +.51**. The ablation identifies chroma as a strong source of human tonal alignment and the circle-of-fifths root encoding as an effective way to organize reconstruction-only models by tonal theory.

## Repository layout

```text
TRACE-Research/
├── data/                         # dataset parsing, chord encoding, windows and loaders
├── dataset/                      # source/licensing notes; raw dataset is not included
├── scripts/                      # training, aggregation, references and evaluation
├── tests/                        # unit tests for preprocessing and autoencoder components
├── docs/FINAL_RESULTS.md         # methods, headline results and reproduction details
└── results/
    ├── ablation_15d/
    │   ├── checkpoints/          # 90 small trained model checkpoints
    │   ├── runs/                 # per-run JSON: metrics, latent coordinates, histories
    │   ├── summary.csv
    │   └── summary.md
    ├── multi_metric/             # per-model evaluation tables (CSV, JSON, Markdown)
    ├── original_models/          # original M0/M2 model weights and training histories
    └── skipgram_baseline/        # embedding matrix and vocabulary for baseline evaluation
```

The raw Hooktheory dataset and research-paper PDFs are stored separately; see [Data and licensing](#data-and-licensing). Intermediate tensor caches are generated locally during training.

## Experiment pipeline

### 1. Song-level split and local tonal frame

The preprocessing code loads the Sheet Sage / Hooktheory harmony annotations, sorts harmony events by onset, and divides each song into tonal segments at annotated key changes. The annotated mode/scale is mapped to its relative-major tonic. Chords are transposed into that local relative-major frame, so the local tonic maps to C. Song IDs are shuffled with split seed 42 and assigned to train/validation/test in an 80/10/10 split; windows do not cross song or tonal-segment boundaries.

Within each segment, the model uses sliding windows of 8 chords with stride 1. Each window carries chord features and precomputed adjacent-chord TPS distances. This makes the model see short local harmonic context while keeping songs disjoint across splits.

### 2. Chord input representation

The canonical input is 15-dimensional:

| Dimensions | Feature | Encoding |
|---|---|---|
| 0–11 | Chroma | 12 pitch classes; root = 1.0, other chord tones = 0.5, absent tones = 0 |
| 12–13 | Root | `[sin(2π·pc/12), cos(2π·pc/12)]` |
| 14 | Duration | `log(1 + duration)` in beats |

The bass pitch class is retained as dataset metadata and appended only for the 17D `plus_bass17` ablation. The `fifths_root15` condition changes the root-circle angle to `2π·7·pc/12`. Binary-chroma conditions replace nonzero chroma values by 1. When an input feature group is removed, it is removed from both the encoder input and reconstruction target.

### 3. Model and objective

Each chord is encoded independently by a multilayer perceptron, applied to every position in the 8-chord window:

```text
input (15D, or ablation-specific D) → Linear(D, 64) → ReLU
→ Linear(64, 32) → ReLU → Linear(32, 3) → 3D latent chord
```

The decoder mirrors this structure:

```text
3D latent → Linear(3, 32) → ReLU → Linear(32, 64) → ReLU
→ Linear(64, D) → group-specific output transforms
```

Chroma outputs use sigmoid; root and bass circular outputs are normalized to unit length. The loss is the sum of per-group reconstruction terms plus the TPS distance loss:

```text
L = L_reconstruction + λ · MSE(||z[t+1] − z[t]||₂, TPS[t,t+1])
```

Reconstruction uses binary cross-entropy for chroma, one minus cosine similarity for circular features, and mean squared error for duration. λ = 1 is the TPS condition; λ = 0 is the reconstruction-only condition. Temporal structure is modeled through adjacent latent distances across the window.

### 4. Training design

The final 15D ablation comprises 9 input variants × 2 λ settings × 5 initialization/shuffle seeds (42–46), for 90 runs. The song split remains fixed at seed 42. Each run trains for up to 50 epochs and retains the checkpoint with the lowest validation loss.

| Setting | Value |
|---|---:|
| Hidden layers | 64, 32 |
| Latent dimension | 3 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 256 |
| Gradient clipping | 1.0 |
| Epoch limit | 50 |
| Checkpoint selection | Lowest validation total loss |
| Split | 80/10/10 by song; split seed 42 |
| Repeated-run seeds | 42, 43, 44, 45, 46 |

The full run was reported at approximately 2.5 hours on an RTX 5060 with four parallel workers, using Python 3.10 and PyTorch 2.7.1+cu128. Runtime varies with hardware and data placement.

### 5. Evaluation

`scripts/eval_multi_metric.py` evaluates all ablation checkpoints, the original M0/M2 autoencoders and a Skip-gram baseline. The final tables are in `results/multi_metric/`.

- **Krumhansl & Kessler (1982), chord level:** correlations between latent chord distances and K&K-derived key-distance and chord–key references; exact Mantel permutation tests; triplet agreement; and hierarchy correlation. Six chords are used where the reference excludes vii°; seven for chord–key comparisons.
- **Bigand & Parncutt (1999), sequence level:** Spearman correlation between perceived tension ratings for 30 sequence chords and (a) latent distance to the local tonic or (b) distance from the preceding chord. The ratings were digitized from the published figure.
- **Theory and historical comparisons:** Lerdahl TPS, the training target itself, and a legacy hand-entered chord matrix.
- **Held-out fit:** reconstruction terms, TPS MSE and rank correlation between predicted latent transition distance and TPS target on test transitions.

The K&K distance metrics use exact Mantel permutations over chord relabelings. Sequence-level correlations use the 30 rated chord positions. Reference construction and detailed scores are documented in [the final results report](docs/FINAL_RESULTS.md).

## Final findings

- **Human tonal alignment:** `full15`, λ = 1 reaches K&K key-distance ρ = **+.76 ± .02**, significant in all five seeds. K&K chord–key alignment is +.72 and triplet agreement is .82.
- **TPS contribution:** on the full15 input, λ = 1 improves K&K key-distance correlation by **+.81** over reconstruction-only training (Welch p < 10⁻⁵).
- **Held-out generalization:** latent transition distances correlate with TPS at **ρ = .97** on test transitions.
- **Sequence tension:** distance to the local tonic predicts Bigand–Parncutt tension at **ρ = +.51**, close to local stability (+.55) and above hierarchical Lerdahl values (+.33) in the same sequence.
- **Feature robustness:** eight of nine λ = 1 feature variants score between .74 and .78 on K&K key distance. The chroma-only λ = 0 model reaches +.70 ± .11, significant in all five seeds.
- **Root encoding:** under λ = 0, the circle-of-fifths root variant reaches Lerdahl TPS ρ = +.40, Bigand–Parncutt tension ρ = +.44, and held-out TPS ρ = +.84. Chroma-only variants reach +.62 to +.70 on K&K key distance.
- **Target alignment:** the `j+k` TPS target scores +.79 on K&K key distance, +.82 on hierarchy, and +.57 on sequence tension. The learned full15 model reaches +.76, +.71 and +.51, respectively, while reaching +.97 on held-out TPS transitions.

See `docs/FINAL_RESULTS.md` and the machine-readable tables for full means, standard deviations, per-seed significance, and historical-model comparisons.

## Reproducing the experiments

### Environment

Python 3.10+ is recommended. Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CPU/GPU from the official PyTorch installation selector; the reference experiment used PyTorch 2.7.1+cu128. CUDA is optional, but training will be slower on CPU.

### Dataset setup

Obtain the Hooktheory dataset through the Sheet Sage project and its stated terms. Place the MIR-friendly `Hooktheory.json.gz` at `dataset/hooktheory/Hooktheory.json.gz`, or provide another path with `--dataset`. That data file is not part of this repository. The included `dataset/README.md` documents the source and its CC BY-NC-SA 3.0 terms; check that license and source terms before reuse or redistribution.

### Train the 90-run ablation

From this repository root:

```bash
python scripts/ablation_15d.py \
  --dataset dataset/hooktheory/Hooktheory.json.gz \
  --output_dir results/ablation_15d \
  --cache results/ablation_15d/window_tensors_split42.pt \
  --skip_existing
```

By default, this runs all nine variants, λ = 1 and 0, and seeds 42–46. Use `--variants`, `--lambdas`, and `--seeds` to select a smaller run. The cache is generated locally from the dataset and should not be committed.

Aggregate the per-run JSON files:

```bash
python scripts/summarize_ablation_15d.py --output_dir results/ablation_15d
```

The committed `results/ablation_15d/summary.*` files are the final 2026-09-27 aggregates.

### Recompute the multi-metric evaluation

All artifacts needed by the evaluation script are included in `results/`, including ablation and original-model checkpoints and the Skip-gram embedding matrix and vocabulary. With dependencies installed, recompute the multi-metric evaluation with:

```bash
python scripts/eval_multi_metric.py
```

The committed evaluation tables are provided so the reported findings can be inspected without downloading the dataset or rerunning training. Per-run JSON records contain run metrics, latent coordinates, and histories; the checkpoint files contain model state dictionaries and run configuration.

## Key files

- `scripts/ablation_15d.py` — final multi-seed training pipeline and feature variants.
- `scripts/summarize_ablation_15d.py` — aggregates per-run records and statistical summaries.
- `scripts/eval_multi_metric.py` — human-reference, behavioral, theory and baseline evaluation.
- `scripts/cognitive_reference.py` — K&K-derived profiles, exact permutation metrics and reference utilities.
- `scripts/bp1999_sequence.py` — sequence reconstruction and B&P reference values/predictors.
- `data/` — Hooktheory parsing, song splits, window construction and chord feature encoding.
- `results/` — final aggregate tables, per-run outputs and original checkpoints.
- `docs/FINAL_RESULTS.md` — expanded methods, reference provenance and detailed results.

## Data and licensing

The Hooktheory/Sheet Sage dataset is distributed under **CC BY-NC-SA 3.0** and is stored separately from this repository. The research papers are cited in the report; their PDFs are stored separately. No software license has been declared for this repository yet.

## References

- Krumhansl, C. L., & Kessler, E. J. (1982). *Tracing the dynamic changes in perceived tonal organization in a spatial representation of musical keys*. Psychological Review, 89(4), 334–368.
- Bigand, E., & Parncutt, R. (1999). *Perceiving musical tension in long chord sequences*. Psychological Research, 62, 237–254.
- Bharucha, J. J., & Krumhansl, C. L. (1983). *The representation of harmonic structure in music: Hierarchies of stability as a function of context*. Cognition, 13, 63–102.
- Donahue, C., Thickstun, J., & Liang, P. (2022). *Melody transcription via generative pre-training*. ISMIR. The Hooktheory data used here is distributed through the Sheet Sage project.
#   T R A C E - r e s e a r c h 
 
 
