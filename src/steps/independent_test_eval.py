"""
independent_test_eval.py — Tables 12 and 20.

Evaluates the trained 5-fold ensemble on the independent test set (N=110).
Expects:
  data/features_test/text/features.npz
  data/features_test/image/features.npz
  runs/feature_fusion/fusion/base_models_saved/
"""
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import write_json
from utils.metrics import compute_binary_metrics


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    lo = float(beta_dist.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return round(lo, 3), round(hi, 3)


def load_features_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    return list(d["app_ids"]), d["labels"], d


def main():
    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir = base_dir / "independent_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_text_path = Path("data/features_test/text/features.npz")
    test_img_path  = Path("data/features_test/image/features.npz")
    if not test_text_path.exists():
        print(f"[error] Test features not found at {test_text_path}")
        print("  Extract features for the 110-app test set first, save to data/features_test/")
        return

    import joblib
    import lightgbm as lgb

    test_ids, test_labels, td = load_features_npz(test_text_path)
    _, _, imd = load_features_npz(test_img_path)

    text_feats  = np.concatenate([td["sbert"], td["keywords"], td["meta"]], axis=1)
    image_feats = np.concatenate([imd["clip_mean"], imd["clip_max"], imd["zeroshot"], imd["ocr"]], axis=1)

    models_dir = base_dir / "fusion" / "base_models_saved"
    text_probs = np.zeros(len(test_ids))
    img_probs  = np.zeros(len(test_ids))

    for fold in range(CFG.n_folds):
        sel_t = joblib.load(models_dir / f"text_selector_fold_{fold}.joblib")
        mdl_t = lgb.Booster(model_file=str(models_dir / f"text_lgbm_fold_{fold}.txt"))
        text_probs += mdl_t.predict(sel_t.transform(text_feats))

        sel_i = joblib.load(models_dir / f"img_selector_fold_{fold}.joblib")
        mdl_i = lgb.Booster(model_file=str(models_dir / f"img_lgbm_fold_{fold}.txt"))
        img_probs += mdl_i.predict(sel_i.transform(image_feats))

    text_probs /= CFG.n_folds
    img_probs  /= CFG.n_folds

    y_true = test_labels.astype(int)
    n_pos = int(y_true.sum())
    tau = 0.5

    strategies = {
        "Score-Max":   np.maximum(text_probs, img_probs),
        "Soft Voting": 0.5 * text_probs + 0.5 * img_probs,
    }

    results = {}
    print(f"\nIndependent Test Set: N={len(y_true)}, N_+={n_pos}")
    print(f"\n{'Strategy':<16} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'Recall CI':>16} {'F1':>6} {'AUC':>6}")
    print("-" * 70)

    for name, y_prob in strategies.items():
        m = compute_binary_metrics(y_true, y_prob, threshold=tau)
        tp = int(np.sum((y_prob >= tau) & (y_true == 1)))
        lo, hi = clopper_pearson_ci(tp, n_pos)
        results[name] = {**m, "recall_ci95": [lo, hi]}
        print(f"  {name:<14} {m['accuracy']:>6.3f} {m['precision_pos']:>6.3f} "
              f"{m['recall_pos']:>7.3f} [{lo:.3f},{hi:.3f}] "
              f"{m['f1_pos']:>6.3f} {m['roc_auc']:>6.3f}")

    write_json(out_dir / "table20_independent_test.json", results)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
