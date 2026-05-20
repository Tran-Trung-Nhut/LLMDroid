"""cohen_kappa_validation.py — Table 2: Code-level label validation (N=80 apps)."""
import sys
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG

# ── Input CSV format ───────────────────────────────────────────────────────────
# data/code_validation.csv  (path: CFG.code_validation_csv) with columns:
#   pkg_name               : string
#   listing_label          : int (0 or 1) — label used by LLMDroid
#   ai_discriminator_label : int (0 or 1) — output from AI Discriminator


def _stats(df: pd.DataFrame, ref_col: str, pred_col: str, name: str) -> dict:
    ref = df[ref_col].values.astype(int)
    pred = df[pred_col].values.astype(int)
    kappa = cohen_kappa_score(ref, pred)
    pct = float((ref == pred).mean()) * 100
    tn, fp, fn, tp = confusion_matrix(ref, pred, labels=[0, 1]).ravel()
    return dict(name=name, kappa=kappa, pct=pct,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def _print_block(df: pd.DataFrame, r: dict, col: str):
    pos_df = df[df["listing_label"] == 1]
    neg_df = df[df["listing_label"] == 0]
    pos_pos = int((pos_df[col] == 1).sum())
    pos_neg = int((pos_df[col] == 0).sum())
    neg_pos = int((neg_df[col] == 1).sum())
    neg_neg = int((neg_df[col] == 0).sum())

    print(f"\n  vs {r['name']}:")
    print(f"    {'':25} Positive  Negative")
    print(f"    {'Listing positive':<25} {pos_pos:>8}  {pos_neg:>8}")
    print(f"    {'Listing negative':<25} {neg_pos:>8}  {neg_neg:>8}")
    print(f"    {'Agreement':<25} {r['pct']:>7.1f}%")
    print(f"    {'Cohen kappa':<25} {r['kappa']:>8.3f}")


def main():
    val_path = Path(CFG.code_validation_csv)
    if not val_path.exists():
        print(f"ERROR: {val_path} not found.")
        print("Create a CSV with columns: pkg_name, listing_label, ai_discriminator_label")
        sys.exit(1)

    df = pd.read_csv(val_path)
    if not {"listing_label", "ai_discriminator_label"}.issubset(df.columns):
        print("ERROR: CSV must have columns: listing_label, ai_discriminator_label")
        sys.exit(1)

    n_apps = len(df)
    r = _stats(df, "listing_label", "ai_discriminator_label", "AI Discriminator")

    print("=" * 65)
    print(f"Table 2: Listing-label vs code-level references (N_code = {n_apps})")
    print("=" * 65)
    _print_block(df, r, "ai_discriminator_label")

    print("\n" + "=" * 65)
    print("Summary:")
    print(f"  {'':30} {'AI Disc.':>10}")
    print(f"  {'Agreement':<30} {r['pct']:>9.1f}%")
    print(f"  {'Cohen kappa':<30} {r['kappa']:>10.3f}")
    print("=" * 65)

    out_path = Path(CFG.runs_dir) / "cohen_kappa_validation.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"n_apps={n_apps}\n")
        f.write(f"ai_discriminator_kappa={r['kappa']:.3f}\n")
        f.write(f"ai_discriminator_pct_agreement={r['pct']:.1f}\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
