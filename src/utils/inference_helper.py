import numpy as np
from pathlib import Path
import joblib
import lightgbm as lgb
from typing import Tuple, Optional


def load_and_predict_base_models(
    X_text: np.ndarray, 
    X_image: np.ndarray, 
    base_models_dir: Path,
    fold: int
) -> Tuple[np.ndarray, np.ndarray]:
    # Text branch
    text_selector = joblib.load(base_models_dir / f"text_selector_fold_{fold}.joblib")
    X_text_sel = text_selector.transform(X_text)
    text_lgbm = lgb.Booster(model_file=str(base_models_dir / f"text_lgbm_fold_{fold}.txt"))
    text_prob = text_lgbm.predict(X_text_sel)
    
    # Image branch
    img_selector = joblib.load(base_models_dir / f"img_selector_fold_{fold}.joblib")
    X_img_sel = img_selector.transform(X_image)
    img_lgbm = lgb.Booster(model_file=str(base_models_dir / f"img_lgbm_fold_{fold}.txt"))
    img_prob = img_lgbm.predict(X_img_sel)
    
    return text_prob, img_prob


def predict_stacking_fusion(
    text_prob: np.ndarray,
    img_prob: np.ndarray,
    meta_models_dir: Path,
    fold: int
) -> np.ndarray:
    meta_clf = joblib.load(meta_models_dir / f"meta_clf_fold_{fold}.joblib")
    scaler = joblib.load(meta_models_dir / f"scaler_fold_{fold}.joblib")
    
    X_meta = np.column_stack([text_prob, img_prob])
    X_meta_scaled = scaler.transform(X_meta)
    final_prob = meta_clf.predict_proba(X_meta_scaled)[:, 1]
    
    return final_prob


def predict_soft_voting(text_prob: np.ndarray, img_prob: np.ndarray) -> np.ndarray:
    return (text_prob + img_prob) / 2.0


def predict_max_voting(text_prob: np.ndarray, img_prob: np.ndarray) -> np.ndarray:
    return np.maximum(text_prob, img_prob)


def predict_early_fusion(
    X_all: np.ndarray,
    early_fusion_dir: Path,
    fold: int
) -> np.ndarray:
    selector = joblib.load(early_fusion_dir / f"selector_fold_{fold}.joblib")
    X_sel = selector.transform(X_all)
    lgbm = lgb.Booster(model_file=str(early_fusion_dir / f"lgbm_fold_{fold}.txt"))
    prob = lgbm.predict(X_sel)
    return prob


def ensemble_across_folds(fold_probs: list) -> np.ndarray:
    return np.mean(fold_probs, axis=0)
