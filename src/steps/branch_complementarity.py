"""
branch_complementarity.py — Table 10 + Figure 4.

Pearson/Spearman correlation between s_text and s_img on pooled OOF predictions.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json


def load_fusion_predictions(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "text_prob" not in row or "image_prob" not in row:
                raise ValueError(f"Missing text_prob/image_prob in {csv_path}")
            rows.append({
                "y_true": int(row["y_true"]),
                "s_text": float(row["text_prob"]),
                "s_img":  float(row["image_prob"]),
            })
    return rows


def bootstrap_corr(x: np.ndarray, y: np.ndarray, n: int = None, seed: int = None):
    if n is None:
        n = CFG.n_bootstrap
    if seed is None:
        seed = CFG.seed
    rng = np.random.RandomState(seed)
    p_boot, s_boot = [], []
    for _ in range(n):
        idx = rng.choice(len(x), len(x), replace=True)
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            continue
        p_boot.append(pearsonr(xb, yb)[0])
        s_boot.append(spearmanr(xb, yb)[0])
    p_ci = (round(float(np.percentile(p_boot, 2.5)), 2),
            round(float(np.percentile(p_boot, 97.5)), 2))
    s_ci = (round(float(np.percentile(s_boot, 2.5)), 2),
            round(float(np.percentile(s_boot, 97.5)), 2))
    return p_ci, s_ci


def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir = base_dir / "branch_complementarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / "fusion" / "late_fusion_soft_voting" / "predictions.csv"
    if not csv_path.exists():
        print(f"[error] {csv_path} not found. Run training first.")
        return

    rows = load_fusion_predictions(csv_path)
    s_text = np.array([r["s_text"] for r in rows])
    s_img  = np.array([r["s_img"]  for r in rows])
    y_true = np.array([r["y_true"] for r in rows])

    results = {}
    fig, ax = plt.subplots(figsize=(7, 6))

    for subset_name, mask in [
        ("Positives (y=1)", y_true == 1),
        ("Negatives (y=0)", y_true == 0),
        ("Pooled",          np.ones(len(y_true), dtype=bool)),
    ]:
        x, y = s_text[mask], s_img[mask]
        if len(x) < 3:
            continue
        rp = round(float(pearsonr(x, y)[0]), 2)
        rs = round(float(spearmanr(x, y)[0]), 2)
        p_ci, s_ci = bootstrap_corr(x, y)
        results[subset_name] = {
            "n": int(mask.sum()),
            "rho_pearson": rp,
            "pearson_ci95": list(p_ci),
            "rho_spearman": rs,
            "spearman_ci95": list(s_ci),
        }

    pos_mask = y_true == 1
    ax.scatter(s_text[~pos_mask], s_img[~pos_mask], marker="o", alpha=0.5,
               color="blue", label="Negative (y=0)", s=20)
    ax.scatter(s_text[pos_mask],  s_img[pos_mask],  marker="^", alpha=0.7,
               color="red",  label="Positive (y=1)", s=30)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Text-Only / Image-Only τ=0.5")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.fill_between([0, 0.5], [0.5, 0.5], [1, 1], alpha=0.05, color="orange")
    ax.fill_between([0.5, 1], [0, 0], [0.5, 0.5], alpha=0.05, color="orange")
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, 1.0 - x_line, color="green", linestyle="-", linewidth=1.5,
            label="Soft Voting boundary (α=0.5, τ=0.5)")
    ax.set_xlabel("Text branch score s_text")
    ax.set_ylabel("Image branch score s_img")
    ax.set_title("Figure 4: Text vs Image branch scores (pooled OOF, N=300)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(out_dir / "figure4_branch_scatter.png", dpi=150)
    plt.close(fig)

    write_json(out_dir / "table10_correlations.json", results)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
