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
    import json
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir  = base_dir / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dir = base_dir / "independent_test"
    sv_csv   = test_dir / "predictions_soft_voting.csv"
    if not sv_csv.exists():
        print(f"[error] Run independent_test_eval.py first to generate {sv_csv}")
        return

    rows   = load_fusion_preds(sv_csv)
    y_true = np.array([r["y_true"]     for r in rows])
    s_text = np.array([r["text_prob"]  for r in rows])
    s_img  = np.array([r["image_prob"] for r in rows])

    sv_alpha_path = base_dir / "fusion" / "late_fusion_soft_voting" / "alpha_grid_search.json"
    alpha = 0.5
    if sv_alpha_path.exists():
        with open(sv_alpha_path) as f:
            alpha = json.load(f).get("mean_alpha", 0.5)

    results = {}

    _, y_pred = soft_vote(s_text, s_img, alpha)
    results["Full listing (baseline)"] = compute_metrics(y_true, y_pred)
    results["Full listing (baseline)"]["delta_f1"] = 0.0
    base_f1 = results["Full listing (baseline)"]["f1"]

    _, y_pred_no_img = soft_vote(s_text, np.zeros_like(s_img), alpha)
    m = compute_metrics(y_true, y_pred_no_img)
    m["delta_f1"] = round(m["f1"] - base_f1, 3)
    results["Drop screenshots"] = m

    # Condition 3: Truncate description to 50 chars (requires BGE re-encode)
    trunc_features_path = Path(CFG.features_test_trunc50_dir) / "text" / "features.npz"
    trunc_text_probs = None  # keep in scope for condition 4
    if trunc_features_path.exists():
        import joblib
        import lightgbm as lgb
        d = np.load(trunc_features_path, allow_pickle=True)
        trunc_feats  = np.concatenate([d["sbert"], d["keywords"], d["meta"]], axis=1)
        trunc_ids    = list(d["app_ids"])
        trunc_id2idx = {aid: i for i, aid in enumerate(trunc_ids)}
        test_ids     = [r["app_id"] for r in rows]
        fus_dir      = base_dir / "fusion" / "base_models_saved"
        trunc_text_probs = np.zeros(len(test_ids))
        for fold in range(CFG.n_folds):
            sel = joblib.load(fus_dir / f"text_selector_fold_{fold}.joblib")
            mdl = lgb.Booster(model_file=str(fus_dir / f"text_lgbm_fold_{fold}.txt"))
            aligned = np.array([trunc_feats[trunc_id2idx[aid]] if aid in trunc_id2idx
                                 else np.zeros(trunc_feats.shape[1]) for aid in test_ids])
            trunc_text_probs += mdl.predict(sel.transform(aligned))
        trunc_text_probs /= CFG.n_folds
        _, y_pred_trunc = soft_vote(trunc_text_probs, s_img, alpha)
        m_trunc = compute_metrics(y_true, y_pred_trunc)
        m_trunc["delta_f1"] = round(m_trunc["f1"] - base_f1, 3)
        results["Truncate description to 50 chars"] = m_trunc
    else:
        print(f"[skip] Truncate condition: {trunc_features_path} not found.")
        results["Truncate description to 50 chars"] = {
            "note": "requires re-encoding — run extract_text_features_trunc50.py first"
        }

    # Condition 4: Drop screenshots AND truncate text = trunc_text_probs + s_img=0
    if trunc_text_probs is not None:
        _, y_pred_both = soft_vote(trunc_text_probs, np.zeros_like(s_img), alpha)
        m_both = compute_metrics(y_true, y_pred_both)
    else:
        _, y_pred_both = soft_vote(np.zeros_like(s_text), np.zeros_like(s_img), alpha)
        m_both = compute_metrics(y_true, y_pred_both)
        m_both["note"] = "Approximated with s_text=0 — needs extract_text_features_trunc50.py"
    m_both["delta_f1"] = round(m_both["f1"] - base_f1, 3)
    results["Drop screenshots and truncate text"] = m_both

    write_json(out_dir / "table18_robustness.json", results)

    print(f"\nTable 18 (Robustness — Soft Voting, N=110 independent test set, alpha={alpha}):")
    print(f"  {'Condition':<40} {'Recall':>7} {'Precision':>10} {'F1':>6} {'ΔF1':>7}")
    print("  " + "-" * 75)
    for cond, m in results.items():
        if "note" in m and "recall" not in m:
            print(f"  {cond:<40} [SKIPPED — {m['note'][:35]}]")
            continue
        print(f"  {cond:<40} {m['recall']:>7.3f} {m['precision']:>10.3f} "
              f"{m['f1']:>6.3f} {m['delta_f1']:>+7.3f}")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
