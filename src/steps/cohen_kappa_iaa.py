"""cohen_kappa_iaa.py — Table 1: Inter-annotator agreement (N=100 apps)."""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG

# ── Input CSV format ───────────────────────────────────────────────────────────
# data/inter_annotator.csv  (path: CFG.iaa_csv) with columns:
#   app_id       : string
#   annotator1   : int (0 or 1)
#   annotator2   : int (0 or 1)
#   final_label  : int (0 or 1) — after adjudication (optional)


def bootstrap_kappa_ci(labels1: np.ndarray, labels2: np.ndarray):
    rng = np.random.default_rng(CFG.seed)
    kappas = [
        cohen_kappa_score(
            labels1[rng.choice(len(labels1), size=len(labels1), replace=True)],
            labels2[rng.choice(len(labels1), size=len(labels1), replace=True)],
        )
        for _ in range(CFG.n_bootstrap)
    ]
    lo, hi = np.percentile(kappas, [2.5, 97.5])
    return lo, hi


def main():
    iaa_path = Path(CFG.iaa_csv)
    if not iaa_path.exists():
        print(f"ERROR: {iaa_path} not found.")
        print("Create a CSV with columns: app_id, annotator1, annotator2, final_label")
        sys.exit(1)

    df = pd.read_csv(iaa_path)
    if not {"annotator1", "annotator2"}.issubset(df.columns):
        print("ERROR: CSV must have columns: annotator1, annotator2")
        sys.exit(1)

    a1 = df["annotator1"].values.astype(int)
    a2 = df["annotator2"].values.astype(int)
    n_apps = len(df)

    kappa = cohen_kappa_score(a1, a2)
    pct_agree = float((a1 == a2).mean()) * 100
    disagreements = int((a1 != a2).sum())
    ci_lo, ci_hi = bootstrap_kappa_ci(a1, a2)

    resolved_pos = resolved_neg = None
    if "final_label" in df.columns:
        dis_df = df[a1 != a2]
        resolved_pos = int((dis_df["final_label"] == 1).sum())
        resolved_neg = int((dis_df["final_label"] == 0).sum())

    print("=" * 55)
    print(f"Table 1: Inter-annotator agreement (N_ian = {n_apps})")
    print(f"Bootstrap 95% CI from n = {CFG.n_bootstrap} resamples")
    print("=" * 55)
    print(f"{'Metric':<30} {'Value'}")
    print("-" * 55)
    print(f"{'Percentage agreement':<30} {pct_agree:.1f}%")
    print(f"{'Cohen\'s κ':<30} {kappa:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"{'Disagreement rate':<30} {disagreements}/{n_apps}")
    if resolved_pos is not None:
        print(f"{'  resolved as positive':<30} {resolved_pos}")
        print(f"{'  resolved as negative':<30} {resolved_neg}")
    print("=" * 55)

    out_path = Path(CFG.runs_dir) / "cohen_kappa_iaa.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"n_apps={n_apps}\n")
        f.write(f"pct_agreement={pct_agree:.1f}\n")
        f.write(f"kappa={kappa:.3f}\n")
        f.write(f"kappa_ci_lo={ci_lo:.3f}\n")
        f.write(f"kappa_ci_hi={ci_hi:.3f}\n")
        f.write(f"disagreements={disagreements}\n")
        if resolved_pos is not None:
            f.write(f"resolved_positive={resolved_pos}\n")
            f.write(f"resolved_negative={resolved_neg}\n")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
