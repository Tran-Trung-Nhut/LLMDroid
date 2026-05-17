"""
calibration.py — Table 17: Brier score + ECE (raw and Platt-scaled).
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json


def load_pred_csv(path: Path):
    y_true, y_prob = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y_true.append(int(row["y_true"]))
            y_prob.append(float(row["y_prob"]))
    return np.array(y_true, dtype=int), np.array(y_prob, dtype=float)


def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))


def ece(y_true, y_prob, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    total_ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        total_ece += (mask.sum() / n) * abs(acc - conf)
    return round(total_ece, 4)


def platt_scale(y_true_val, y_prob_val, y_prob_test):
    clf = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    clf.fit(y_prob_val.reshape(-1, 1), y_true_val)
    return clf.predict_proba(y_prob_test.reshape(-1, 1))[:, 1]


def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir = base_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = {
        "Text-Only":    (base_dir / "text_only" / "predictions.csv",
                         base_dir / "text_only" / "validation_predictions.csv"),
        "Early Fusion": (base_dir / "fusion" / "early_fusion" / "predictions.csv",
                         base_dir / "fusion" / "early_fusion" / "validation_predictions.csv"),
        "Score-Max":    (base_dir / "fusion" / "late_fusion_score_max" / "predictions.csv",
                         base_dir / "fusion" / "late_fusion_score_max" / "validation_predictions.csv"),
        "Soft Voting":  (base_dir / "fusion" / "late_fusion_soft_voting" / "predictions.csv",
                         base_dir / "fusion" / "late_fusion_soft_voting" / "validation_predictions.csv"),
        "Stacking":     (base_dir / "fusion" / "late_fusion_stacking" / "predictions.csv",
                         base_dir / "fusion" / "late_fusion_stacking" / "validation_predictions.csv"),
    }

    results = {}
    print(f"\n{'Strategy':<20} {'Brier(raw)':>10} {'ECE(raw)':>9} {'Brier(Platt)':>13} {'ECE(Platt)':>11}")
    print("-" * 70)

    for name, (test_csv, val_csv) in strategies.items():
        if not test_csv.exists() or not val_csv.exists():
            continue
        y_test, p_test = load_pred_csv(test_csv)
        y_val, p_val   = load_pred_csv(val_csv)

        b_raw = brier_score(y_test, p_test)
        e_raw = ece(y_test, p_test)
        p_platt = platt_scale(y_val, p_val, p_test)
        b_platt = brier_score(y_test, p_platt)
        e_platt = ece(y_test, p_platt)

        results[name] = {
            "brier_raw": round(b_raw, 3),
            "ece_raw": e_raw,
            "brier_platt": round(b_platt, 3),
            "ece_platt": e_platt,
        }
        print(f"  {name:<20} {b_raw:>10.3f} {e_raw:>9.3f} {b_platt:>13.3f} {e_platt:>11.3f}")

    write_json(out_dir / "table17_calibration.json", results)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
