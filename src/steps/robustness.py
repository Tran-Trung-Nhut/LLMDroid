"""
robustness.py — Table 18: Soft Voting under missing modality conditions.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json


def load_fusion_preds(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "app_id":     row["app_id"],
                "y_true":     int(row["y_true"]),
                "y_prob":     float(row["y_prob"]),
                "text_prob":  float(row.get("text_prob", row["y_prob"])),
                "image_prob": float(row.get("image_prob", 0.0)),
            })
    return rows


def soft_vote(s_text, s_img, alpha=0.5, tau=0.5):
    y_prob = alpha * s_text + (1 - alpha) * s_img
    return y_prob, (y_prob >= tau).astype(int)


def compute_metrics(y_true, y_pred):
    return {
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 3),
    }


def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir = base_dir / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    sv_csv = base_dir / "fusion" / "late_fusion_soft_voting" / "predictions.csv"
    if not sv_csv.exists():
        print(f"[error] {sv_csv} not found")
        return

    rows = load_fusion_preds(sv_csv)
    y_true = np.array([r["y_true"] for r in rows])
    s_text = np.array([r["text_prob"] for r in rows])
    s_img  = np.array([r["image_prob"] for r in rows])
    alpha  = 0.5

    results = {}

    _, y_pred = soft_vote(s_text, s_img, alpha)
    results["Full listing (baseline)"] = compute_metrics(y_true, y_pred)
    results["Full listing (baseline)"]["delta_f1"] = 0.0
    base_f1 = results["Full listing (baseline)"]["f1"]

    _, y_pred_no_img = soft_vote(s_text, np.zeros_like(s_img), alpha)
    m = compute_metrics(y_true, y_pred_no_img)
    m["delta_f1"] = round(m["f1"] - base_f1, 3)
    results["Drop screenshots"] = m

    text_only_csv = base_dir / "text_only" / "predictions.csv"
    if text_only_csv.exists():
        with open(text_only_csv, newline="", encoding="utf-8") as f:
            text_only_rows = list(csv.DictReader(f))
        text_only_map = {r["app_id"]: float(r["y_prob"]) for r in text_only_rows}
        s_text_trunc = np.array([text_only_map.get(r["app_id"], r["text_prob"]) for r in rows])
        _, y_pred_trunc = soft_vote(s_text_trunc, np.zeros_like(s_img), alpha)
        m_trunc = compute_metrics(y_true, y_pred_trunc)
        m_trunc["delta_f1"] = round(m_trunc["f1"] - base_f1, 3)
        m_trunc["note"] = "Approximated — exact requires BGE re-encode on 50-char text"
        results["Truncate description to 50 chars"] = m_trunc

    _, y_pred_both = soft_vote(np.zeros_like(s_text), np.zeros_like(s_img), alpha)
    m_both = compute_metrics(y_true, y_pred_both)
    m_both["delta_f1"] = round(m_both["f1"] - base_f1, 3)
    results["Drop screenshots and truncate text"] = m_both

    write_json(out_dir / "table18_robustness.json", results)

    print("\nTable 18 (Robustness — Soft Voting):")
    print(f"  {'Condition':<40} {'Recall':>7} {'Precision':>10} {'F1':>6} {'ΔF1':>7}")
    print("  " + "-" * 75)
    for cond, m in results.items():
        print(f"  {cond:<40} {m['recall']:>7.3f} {m['precision']:>10.3f} "
              f"{m['f1']:>6.3f} {m['delta_f1']:>+7.3f}")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
