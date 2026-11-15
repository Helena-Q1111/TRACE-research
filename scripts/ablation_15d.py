"""Ablation study of the 15D chord input representation.

The 15D vector is three feature groups:
    chroma   dims 0-11   weighted chroma (root=1.0, other chord tones=0.5)
    root     dims 12-13  root circular encoding [sin, cos]
    duration dim  14     log(1 + duration)

Each ablation variant rebuilds the input from those groups (dropping groups or
binarizing the chroma), so removed features are absent from both the encoder input
and the reconstruction target. One optional additive variant appends the bass
circular encoding kept in the dataset metadata.

All other settings follow the M2 recipe in train_ae.py: MLP 64/32/3, AdamW lr=1e-3,
weight_decay=1e-4, batch 256, grad clip 1.0, 50 epochs, best checkpoint by valid
loss. The song split is fixed (split_seed=42, same as M2); only the init/shuffle
seed varies across repeats.

Evaluation per run:
    - cognitive alignment: Spearman/Pearson between latent distances of the 7
      C-major diatonic triads and the reference matrix in eval_cognitive_comparison
    - held-out test: reconstruction terms, TPS MSE and Spearman(latent dist, TPS)
      over all adjacent test transitions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataloader_builder import prepare_datasets
from scripts.eval_cognitive_comparison import get_c_major_chords, get_krumhansl_reference


@dataclass(frozen=True)
class FeatureConfig:
    """Which input feature groups a variant keeps."""

    name: str
    chroma: str = "weighted"  # "weighted" | "binary" | "none"
    root_circle: bool = True
    duration: bool = True
    bass_circle: bool = False
    circle: str = "chromatic"  # root/bass circle: "chromatic" (angle 2π·pc/12) | "fifths" (2π·7pc/12)

    def groups(self) -> list[tuple[str, int]]:
        layout: list[tuple[str, int]] = []
        if self.chroma != "none":
            layout.append(("chroma", 12))
        if self.root_circle:
            layout.append(("root", 2))
        if self.duration:
            layout.append(("duration", 1))
        if self.bass_circle:
            layout.append(("bass", 2))
        return layout

    @property
    def input_dim(self) -> int:
        return sum(size for _, size in self.groups())


VARIANTS: dict[str, FeatureConfig] = {
    cfg.name: cfg
    for cfg in [
        FeatureConfig("full15"),
        FeatureConfig("no_duration", duration=False),
        FeatureConfig("no_root_circle", root_circle=False),
        FeatureConfig("binary_chroma", chroma="binary"),
        FeatureConfig("chroma_only", root_circle=False, duration=False),
        FeatureConfig("binary_chroma_only", chroma="binary", root_circle=False, duration=False),
        FeatureConfig("no_chroma", chroma="none"),
        FeatureConfig("plus_bass17", bass_circle=True),
        FeatureConfig("fifths_root15", circle="fifths"),
    ]
}


def circle(pc: torch.Tensor, kind: str = "chromatic") -> torch.Tensor:
    step = 7 if kind == "fifths" else 1
    angle = 2.0 * np.pi * ((step * pc) % 12).float() / 12.0
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def root_pc_from_circle(sin_cos: torch.Tensor) -> torch.Tensor:
    angle = torch.atan2(sin_cos[..., 0], sin_cos[..., 1])
    return torch.round(angle * 12 / (2 * np.pi)).long() % 12


def build_features(x15: torch.Tensor, bass: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """Map the canonical 15D tensor (+ bass pitch classes) to the variant's input."""
    parts: list[torch.Tensor] = []
    if cfg.chroma == "weighted":
        parts.append(x15[..., :12])
    elif cfg.chroma == "binary":
        parts.append((x15[..., :12] > 0).float())
    if cfg.root_circle:
        parts.append(x15[..., 12:14] if cfg.circle == "chromatic" else circle(root_pc_from_circle(x15[..., 12:14]), cfg.circle))
    if cfg.duration:
        parts.append(x15[..., 14:15])
    if cfg.bass_circle:
        parts.append(circle(bass, cfg.circle))
    return torch.cat(parts, dim=-1)


class AblationAutoencoder(nn.Module):
    """Same MLP as ChordAutoencoder, with a variable input layout."""

    def __init__(self, cfg: FeatureConfig, hidden_dim: int = 64, latent_dim: int = 3) -> None:
        super().__init__()
        self.cfg = cfg
        self.layout = cfg.groups()
        d = cfg.input_dim
        self.encoder = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, d),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.decoder(z)
        out, i = [], 0
        for group, size in self.layout:
            chunk = raw[..., i : i + size]
            if group == "chroma":
                chunk = torch.sigmoid(chunk)
            elif group in ("root", "bass"):
                chunk = F.normalize(chunk, dim=-1, eps=1e-8)
            out.append(chunk)
            i += size
        return torch.cat(out, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def recon_terms(x_hat: torch.Tensor, x: torch.Tensor, layout: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
    """Per-group reconstruction losses matching autoencoder.reconstruction_loss."""
    terms, i = {}, 0
    for group, size in layout:
        p, t = x_hat[..., i : i + size], x[..., i : i + size]
        if group == "chroma":
            terms[group] = F.binary_cross_entropy(p, t)
        elif group in ("root", "bass"):
            terms[group] = 1.0 - F.cosine_similarity(p, F.normalize(t, dim=-1, eps=1e-8), dim=-1).mean()
        else:
            terms[group] = F.mse_loss(p[..., 0], t[..., 0])
        i += size
    return terms


def latent_steps(z: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(z[:, 1:, :] - z[:, :-1, :], dim=-1)


def run_split(model, X, TPS, lam, batch_size, optimizer=None, grad_clip=1.0, generator=None):
    train = optimizer is not None
    model.train(train)
    n = X.shape[0]
    order = torch.randperm(n, device=X.device, generator=generator) if train else torch.arange(n, device=X.device)
    sums: dict[str, float] = {}
    lat_all, tps_all = [], []
    with torch.set_grad_enabled(train):
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            x, tps = X[idx], TPS[idx]
            x_hat, z = model(x)
            terms = recon_terms(x_hat, x, model.layout)
            recon = sum(terms.values())
            steps = latent_steps(z)
            tps_loss = F.mse_loss(steps, tps)
            loss = recon + lam * tps_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            else:
                lat_all.append(steps.detach().flatten())
                tps_all.append(tps.flatten())
            bs = float(x.shape[0])
            for key, value in {"loss": loss, "recon": recon, "tps": tps_loss, **terms}.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * bs
    metrics = {key: value / n for key, value in sums.items()}
    if not train:
        lat = torch.cat(lat_all).cpu().numpy()
        tgt = torch.cat(tps_all).cpu().numpy()
        metrics["tps_spearman"] = float(spearmanr(lat, tgt)[0])
    return metrics


def diatonic_inputs(cfg: FeatureConfig, device: torch.device) -> tuple[list[str], torch.Tensor]:
    """7 C-major root-position triads, built exactly like eval_cognitive_comparison."""
    names, rows, basses = [], [], []
    for name, (root, pcs) in get_c_major_chords().items():
        v = torch.zeros(15)
        v[pcs] = 0.5
        v[root] = 1.0
        ang = 2 * np.pi * root / 12
        v[12], v[13], v[14] = np.sin(ang), np.cos(ang), np.log(2.0)
        names.append(name)
        rows.append(v)
        basses.append(root)
    x15 = torch.stack(rows).to(device)
    return names, build_features(x15, torch.tensor(basses, device=device), cfg)


def inversion_offsets(model: AblationAutoencoder, device: torch.device) -> dict[str, float]:
    """Latent distance from C to C/E and C/G (only nonzero when bass is an input)."""
    v = torch.zeros(15)
    v[[0, 4, 7]] = 0.5
    v[0] = 1.0
    v[12], v[13], v[14] = 0.0, 1.0, np.log(2.0)
    x15 = v.unsqueeze(0).repeat(3, 1).to(device)
    x = build_features(x15, torch.tensor([0, 4, 7], device=device), model.cfg)
    with torch.no_grad():
        z = model.encode(x)
    return {"C_to_C/E": float(torch.norm(z[1] - z[0])), "C_to_C/G": float(torch.norm(z[2] - z[0]))}


def cognitive_alignment(model: AblationAutoencoder, device: torch.device) -> dict[str, float]:
    names, x = diatonic_inputs(model.cfg, device)
    model.eval()
    with torch.no_grad():
        z = model.encode(x).cpu().numpy()
    dist = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    mask = np.triu(np.ones_like(dist, dtype=bool), k=1)
    lat, ref = dist[mask], get_krumhansl_reference()[mask]
    if np.var(lat) < 1e-10:
        return {"spearman_rho": float("nan"), "spearman_p": float("nan"), "pearson_r": float("nan")}
    rho, p = spearmanr(lat, ref)
    r, _ = pearsonr(lat, ref)
    return {"spearman_rho": float(rho), "spearman_p": float(p), "pearson_r": float(r),
            "latents": {n: z[i].tolist() for i, n in enumerate(names)}}


def load_tensors(args, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    cache = Path(args.cache)
    if cache.exists():
        splits = torch.load(cache)
    else:
        t0 = time.time()
        datasets, _ = prepare_datasets(dataset_path=args.dataset, seed=args.split_seed)
        splits = {}
        for name, ds in datasets.items():
            recs = ds._window_records
            splits[name] = {
                "x15": torch.stack([r[0] for r in recs]),
                "bass": torch.stack([r[2] for r in recs]),
                "tps": torch.stack([r[4] for r in recs]),
            }
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(splits, cache)
        print(f"Built dataset tensors in {time.time() - t0:.1f}s -> {cache}")
    return {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in splits.items()}


def train_one(cfg, lam, seed, data, args, device, out_dir: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    X = {s: build_features(d["x15"], d["bass"], cfg) for s, d in data.items()}
    T = {s: d["tps"] for s, d in data.items()}

    model = AblationAutoencoder(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        tr = run_split(model, X["train"], T["train"], lam, args.batch_size, opt, args.grad_clip, gen)
        va = run_split(model, X["valid"], T["valid"], lam, args.batch_size)
        history.append({"epoch": epoch, "train_loss": tr["loss"], "valid_loss": va["loss"], "valid_tps": va["tps"]})
        if va["loss"] < best:
            best, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = run_split(model, X["test"], T["test"], lam, args.batch_size)
    cog = cognitive_alignment(model, device)
    result = {
        "variant": cfg.name, "feature_config": asdict(cfg), "input_dim": cfg.input_dim,
        "lambda_tps": lam, "seed": seed, "best_epoch": best_epoch, "best_valid_loss": best,
        "test": test, "cognitive": cog, "inversion": inversion_offsets(model, device), "history": history,
    }
    tag = f"{cfg.name}__lam{lam:g}__s{seed}"
    torch.save({"model_state_dict": best_state, "feature_config": asdict(cfg), "lambda_tps": lam, "seed": seed},
               out_dir / "checkpoints" / f"{tag}.pt")
    (out_dir / "runs" / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="15D input feature ablation for the TRACE chord AE")
    p.add_argument("--dataset", default=str(Path("dataset") / "hooktheory" / "Hooktheory.json.gz"))
    p.add_argument("--output_dir", default=str(Path("outputs") / "ablation_15d"))
    p.add_argument("--cache", default=str(Path("outputs") / "ablation_15d" / "window_tensors_split42.pt"))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 0.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    data = load_tensors(args, device)
    print({s: tuple(d["x15"].shape) for s, d in data.items()})

    jobs = [(v, lam, s) for lam in args.lambdas for v in args.variants for s in args.seeds]
    for i, (v, lam, s) in enumerate(jobs, 1):
        tag = f"{v}__lam{lam:g}__s{s}"
        if args.skip_existing and (out_dir / "runs" / f"{tag}.json").exists():
            continue
        t0 = time.time()
        r = train_one(VARIANTS[v], lam, s, data, args, device, out_dir)
        print(f"[{i}/{len(jobs)}] {tag:<40} rho={r['cognitive']['spearman_rho']:+.4f} "
              f"test_recon={r['test']['recon']:.4f} test_tps={r['test']['tps']:.4f} "
              f"tps_rho={r['test']['tps_spearman']:+.3f} best_ep={r['best_epoch']} ({time.time() - t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
