"""
temporal_split.py — Table 15: Temporal generalization.

D_cut = 2025-06-01. Train on apps <= D_cut, test on apps > D_cut.
"""
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import CFG
from utils.io import read_jsonl, write_json
from utils.seed import set_seed
from utils.metrics import compute_binary_metrics
from steps.train_evaluate import (
    load_features, train_lgbm, predict_lgbm,
    fit_select_kbest, find_best_threshold_from_arrays,
)

D_CUT = datetime.strptime(CFG.temporal_d_cut, "%Y-%m-%d")


def parse_date(date_str):
    if not date_str:
        return None
    # Unix timestamp (integer)
    try:
        ts = int(date_str)
        return datetime.fromtimestamp(ts)
    except (ValueError, TypeError, OSError):
        pass
    # ISO string
    try:
        return datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    except ValueError:
        return None


def main():
    set_seed(CFG.seed)
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir = base_dir / "temporal_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(CFG.dataset_path)
    app_date = {r["app_id"]: parse_date(r.get("release_date")) for r in records}

    data = load_features()
    app_ids = data["app_ids"]
    labels = data["labels"]

    train_idx, test_idx = [], []
    skipped = 0
    for i, aid in enumerate(app_ids):
        dt = app_date.get(aid)
        if dt is None:
            skipped += 1
            train_idx.append(i)
            continue
        (train_idx if dt <= D_CUT else test_idx).append(i)

    print(f"Temporal split: train={len(train_idx)}, test={len(test_idx)}, skipped_date={skipped}")
    if len(test_idx) == 0:
        print("[error] No test apps found after D_cut. Check 'last_updated' field in apps.jsonl.")
        return

    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    y_train, y_test = labels[train_idx], labels[test_idx]

    rng = np.random.RandomState(CFG.seed)
    n_val = max(1, int(len(train_idx) * 0.2))
    val_rel = rng.choice(len(train_idx), n_val, replace=False)
    train_rel = np.setdiff1d(np.arange(len(train_idx)), val_rel)

    feature_results = {}
    for name, X_all in [("Text-Only", data["text_feats"]), ("Early Fusion", data["all_feats"])]:
        X_train, X_test_feats = X_all[train_idx], X_all[test_idx]
        _, X_tr_sel, X_val_sel, X_te_sel = fit_select_kbest(
            X_train[train_rel], y_train[train_rel],
            X_train[val_rel], X_test_feats, CFG.feature_selection_k,
        )
        model = train_lgbm(X_tr_sel, y_train[train_rel], X_val_sel, y_train[val_rel])
        y_prob = predict_lgbm(model, X_te_sel)
        m = compute_binary_metrics(y_test, y_prob, threshold=0.5)
        print(f"  {name}: F1={m['f1_pos']:.3f} ROC-AUC={m['roc_auc']:.3f}")
        feature_results[name] = m

    X_text_tr = data["text_feats"][train_idx]
    X_img_tr  = data["image_feats"][train_idx]
    X_text_te = data["text_feats"][test_idx]
    X_img_te  = data["image_feats"][test_idx]

    _, Xtt, Xtv, Xte_t = fit_select_kbest(X_text_tr[train_rel], y_train[train_rel],
                                           X_text_tr[val_rel], X_text_te, CFG.feature_selection_k)
    _, Xit, Xiv, Xte_i = fit_select_kbest(X_img_tr[train_rel], y_train[train_rel],
                                           X_img_tr[val_rel], X_img_te, CFG.feature_selection_k)
    m_text = train_lgbm(Xtt, y_train[train_rel], Xtv, y_train[val_rel])
    m_img  = train_lgbm(Xit, y_train[train_rel], Xiv, y_train[val_rel])
    p_text = predict_lgbm(m_text, Xte_t)
    p_img  = predict_lgbm(m_img, Xte_i)

    oof_text = np.zeros(len(train_idx))
    oof_img  = np.zeros(len(train_idx))
    inner_cv = StratifiedKFold(n_splits=CFG.stacking_inner_cv_folds, shuffle=True, random_state=CFG.seed)

    for in_tr, in_val in inner_cv.split(X_text_tr, y_train):
        _, Xtt_in, Xtv_in, _ = fit_select_kbest(X_text_tr[in_tr], y_train[in_tr],
                                                  X_text_tr[in_val], X_text_te, CFG.feature_selection_k)
        mdl_t_in = train_lgbm(Xtt_in, y_train[in_tr], Xtv_in, y_train[in_val])
        oof_text[in_val] = predict_lgbm(mdl_t_in, Xtv_in)

        _, Xit_in, Xiv_in, _ = fit_select_kbest(X_img_tr[in_tr], y_train[in_tr],
                                                  X_img_tr[in_val], X_img_te, CFG.feature_selection_k)
        mdl_i_in = train_lgbm(Xit_in, y_train[in_tr], Xiv_in, y_train[in_val])
        oof_img[in_val] = predict_lgbm(mdl_i_in, Xiv_in)

    scaler   = StandardScaler()
    meta_clf = LogisticRegression(C=CFG.meta_learner_C, solver="lbfgs",
                                   max_iter=CFG.meta_learner_max_iter, random_state=CFG.seed)
    meta_clf.fit(scaler.fit_transform(np.column_stack([oof_text, oof_img])), y_train)
    p_stack = meta_clf.predict_proba(scaler.transform(np.column_stack([p_text, p_img])))[:, 1]

    import json as _json
    sv_alpha_path = base_dir / "fusion" / "late_fusion_soft_voting" / "alpha_grid_search.json"
    sv_alpha = 0.5
    if sv_alpha_path.exists():
        with open(sv_alpha_path) as f:
            sv_alpha = _json.load(f).get("mean_alpha", 0.5)

    for strat_name, y_prob in [
        ("Score-Max",   np.maximum(p_text, p_img)),
        ("Soft Voting", sv_alpha * p_text + (1.0 - sv_alpha) * p_img),
        ("Stacking",    p_stack),
    ]:
        m = compute_binary_metrics(y_test, y_prob, threshold=0.5)
        print(f"  {strat_name}: F1={m['f1_pos']:.3f} ROC-AUC={m['roc_auc']:.3f}")
        feature_results[strat_name] = m

    random_f1 = {}
    cv_sources = {
        "Text-Only":    base_dir / "text_only" / "metrics_aggregated.json",
        "Early Fusion": base_dir / "fusion" / "early_fusion" / "metrics_aggregated.json",
        "Score-Max":    base_dir / "fusion" / "late_fusion_score_max" / "metrics_aggregated.json",
        "Soft Voting":  base_dir / "fusion" / "late_fusion_soft_voting" / "metrics_aggregated.json",
        "Stacking":     base_dir / "fusion" / "late_fusion_stacking" / "metrics_aggregated.json",
    }
    for name, path in cv_sources.items():
        if path.exists():
            with open(path) as f:
                agg = _json.load(f)
            random_f1[name] = round(agg.get("f1_pos_mean", 0.0), 3)

    table15 = {}
    for name, m in feature_results.items():
        temporal_f1 = round(m.get("f1_pos", 0.0), 3)
        rand_f1     = random_f1.get(name)
        delta       = round(temporal_f1 - rand_f1, 3) if rand_f1 is not None else None
        table15[name] = {
            "random_f1":        rand_f1,
            "temporal_f1":      temporal_f1,
            "delta":            delta,
            "roc_auc_temporal": round(m.get("roc_auc", 0.0), 3),
        }

    write_json(out_dir / "table15_temporal.json", table15)

    print(f"\nTable 15 (Temporal Generalization, D_cut=2025-06-01):")
    print(f"  {'Strategy':<16} {'Random F1':>10} {'Temporal F1':>12} {'Δ':>7} {'ROC-AUC(temp)':>14}")
    print("  " + "-" * 65)
    for name, r in table15.items():
        rf = f"{r['random_f1']:.3f}" if r["random_f1"] is not None else "N/A"
        d  = f"{r['delta']:+.3f}"    if r["delta"]      is not None else "N/A"
        print(f"  {name:<16} {rf:>10} {r['temporal_f1']:>12.3f} {d:>7} {r['roc_auc_temporal']:>14.3f}")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
