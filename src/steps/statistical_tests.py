"""
statistical_tests.py — McNemar's test + Bootstrap AUC CI.

Compares every fusion strategy against text_only baseline.
No retraining needed — uses saved predictions.csv files.

Output: runs/<run_name>/statistical_tests/
  results.json     — full test statistics
  summary.csv      — one row per comparison (for the paper table)
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_predictions(csv_path: Path):
    """Returns arrays (app_ids, y_true, y_prob, y_pred) sorted by app_id."""
    app_ids, y_true, y_prob = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            app_ids.append(row["app_id"])
            y_true.append(int(row["y_true"]))
            y_prob.append(float(row["y_prob"]))
    order = sorted(range(len(app_ids)), key=lambda i: app_ids[i])
    app_ids = [app_ids[i] for i in order]
    y_true = np.array([y_true[i] for i in order], dtype=int)
    y_prob = np.array([y_prob[i] for i in order], dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    return app_ids, y_true, y_prob, y_pred


# ── Statistical tests ─────────────────────────────────────────────────────────

def mcnemar_test(y_true: np.ndarray, pred_base: np.ndarray, pred_cmp: np.ndarray) -> dict:
    """McNemar's test with continuity correction.

    n01 = baseline wrong, comparison right  (improvement cases)
    n10 = baseline right, comparison wrong  (regression cases)
    """
    n01 = int(np.sum((pred_base != y_true) & (pred_cmp == y_true)))
    n10 = int(np.sum((pred_base == y_true) & (pred_cmp != y_true)))
    denom = n01 + n10
    if denom == 0:
        return {"n01": 0, "n10": 0, "chi2_stat": 0.0, "p_value": 1.0, "note": "no disagreements"}
    chi2_stat = float((abs(n01 - n10) - 1) ** 2 / denom)
    p_value = float(1.0 - chi2.cdf(chi2_stat, df=1))
    return {"n01": n01, "n10": n10, "chi2_stat": round(chi2_stat, 4), "p_value": round(p_value, 4)}


def bootstrap_auc(
    y_true: np.ndarray,
    prob_base: np.ndarray,
    prob_cmp: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for AUC difference (cmp - base) and one-sided p-value."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(
            roc_auc_score(y_true[idx], prob_cmp[idx])
            - roc_auc_score(y_true[idx], prob_base[idx])
        )
    diffs = np.array(diffs)
    auc_base = float(roc_auc_score(y_true, prob_base))
    auc_cmp = float(roc_auc_score(y_true, prob_cmp))
    observed_diff = auc_cmp - auc_base
    # One-sided p: fraction of bootstrap samples where diff <= 0 (null: cmp not better)
    p_one_sided = float(np.mean(diffs <= 0))
    return {
        "auc_base": round(auc_base, 4),
        "auc_cmp": round(auc_cmp, 4),
        "delta_auc": round(observed_diff, 4),
        "ci95_lower": round(float(np.percentile(diffs, 2.5)), 4),
        "ci95_upper": round(float(np.percentile(diffs, 97.5)), 4),
        "p_value_bootstrap": round(p_one_sided, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir = base_dir / "statistical_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Statistical Significance Tests")
    print("=" * 60)

    baseline_csv = base_dir / "text_only" / "predictions.csv"
    if not baseline_csv.exists():
        print(f"[error] Baseline predictions not found: {baseline_csv}")
        return

    base_ids, y_true_base, y_prob_base, y_pred_base = load_predictions(baseline_csv)
    print(f"  Baseline: text_only  AUC={roc_auc_score(y_true_base, y_prob_base):.4f}  n={len(y_true_base)}")

    comparisons = {
        "early_fusion": base_dir / "fusion" / "early_fusion" / "predictions.csv",
    }
    for strat in CFG.fusion_strategy:
        comparisons[f"late_{strat}"] = base_dir / "fusion" / f"late_fusion_{strat}" / "predictions.csv"

    results = {}
    for name, csv_path in comparisons.items():
        if not csv_path.exists():
            print(f"  [skip] {name}: predictions.csv not found")
            continue

        cmp_ids, y_true_cmp, y_prob_cmp, y_pred_cmp = load_predictions(csv_path)

        if base_ids != cmp_ids:
            print(f"  [warning] {name}: app_id order mismatch — skipping McNemar")
            mc = {"note": "app_id order mismatch"}
        else:
            mc = mcnemar_test(y_true_base, y_pred_base, y_pred_cmp)

        boot = bootstrap_auc(y_true_base, y_prob_base, y_prob_cmp)
        results[name] = {"mcnemar": mc, "bootstrap_auc": boot}

    write_json(out_dir / "results.json", results)

    # Write summary CSV for the paper table
    summary_rows = []
    for name, r in results.items():
        mc = r["mcnemar"]
        boot = r["bootstrap_auc"]
        summary_rows.append({
            "comparison": name,
            "auc_base": boot["auc_base"],
            "auc_cmp": boot["auc_cmp"],
            "delta_auc": boot["delta_auc"],
            "ci95_lower": boot["ci95_lower"],
            "ci95_upper": boot["ci95_upper"],
            "p_bootstrap": boot["p_value_bootstrap"],
            "mcnemar_n01": mc.get("n01", ""),
            "mcnemar_n10": mc.get("n10", ""),
            "mcnemar_p": mc.get("p_value", ""),
        })
    if summary_rows:
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    # Pretty print
    print(f"\n  {'Comparison':<20}  {'ΔAUC':>7}  {'95% CI':>18}  {'p(boot)':>8}  {'McNemar p':>10}")
    print("  " + "-" * 72)
    for name, r in results.items():
        mc = r["mcnemar"]
        boot = r["bootstrap_auc"]
        mc_p = f"{mc['p_value']:.4f}" if isinstance(mc.get("p_value"), float) else "N/A"
        sig_mc = "*" if isinstance(mc.get("p_value"), float) and mc["p_value"] < 0.05 else " "
        sig_boot = "*" if boot["p_value_bootstrap"] < 0.05 else " "
        print(
            f"  {name:<20}  {boot['delta_auc']:>+.4f}  "
            f"[{boot['ci95_lower']:>+.4f}, {boot['ci95_upper']:>+.4f}]  "
            f"{boot['p_value_bootstrap']:.4f}{sig_boot}  "
            f"{mc_p:>9}{sig_mc}"
        )
    print("\n  * p < 0.05")
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
