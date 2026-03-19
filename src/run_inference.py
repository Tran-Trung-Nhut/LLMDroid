import os
import sys
import numpy as np
from pathlib import Path
import json
import argparse
import shutil

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from utils.io import write_predictions_csv
from utils.metrics import compute_binary_metrics
from utils.inference_helper import (
    load_and_predict_base_models,
    predict_stacking_fusion,
    predict_soft_voting,
    predict_max_voting,
    predict_early_fusion,
    ensemble_across_folds
)

MODELS_DIR = Path(CFG.runs_dir) / CFG.run_name

def run_preprocessing_for_inference(raw_jsonl_path: str, output_jsonl_path: str):
    print("\n" + "=" * 60)
    print("STEP 1: Preprocessing")
    print("=" * 60)
    from steps import preprocessing
    import json
    
    rows = []
    with open(raw_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} apps")
    
    for row in rows:
        desc = preprocessing.clean_html(row.get("description", ""))
        desc = preprocessing.remove_low_signal(desc)
        short_desc = preprocessing.clean_html(row.get("short_description", ""))
        recent = preprocessing.clean_html(row.get("recent_changes_text", ""))
        
        parts = []
        if row.get("title"): parts.append(row["title"])
        if short_desc: parts.append(short_desc)
        if desc: parts.append(desc)
        if recent: parts.append(recent)
        row["text"] = " ".join(parts)
        
        img_paths = row.get("image_paths", [])
        if img_paths:
            row["image_paths"] = preprocessing.dedup_image_paths(img_paths)

        if row.get("label_binary") is None:
            row["label_binary"] = -1    
    
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✓ Saved to {output_jsonl_path}")

def run_ocr_for_inference(dataset_path: str):
    print("\n" + "=" * 60)
    print("STEP 2: OCR Extraction")
    print("=" * 60)
    from steps import run_ocr
    import json
    from tqdm import tqdm
    
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    
    total_images = sum(len(r.get("image_paths", [])) for r in rows)
    print(f"Processing {len(rows)} apps with {total_images} images")
    
    processed = 0
    pbar = tqdm(total=total_images, desc="OCR")
    for row in rows:
        if "ocr_by_image" not in row:
            row["ocr_by_image"] = {}
        for img_path in row.get("image_paths", []):
            if img_path not in row["ocr_by_image"]:
                row["ocr_by_image"][img_path] = run_ocr.run_ocr_on_image(img_path)
                processed += 1
            pbar.update(1)
    pbar.close()
    
    if processed > 0:
        with open(dataset_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"✓ Updated {dataset_path}")
    else:
        print("✓ Skipped (already done)")

def run_feature_extraction_for_inference(dataset_path: str, features_output_dir: str):
    features_dir = Path(features_output_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dataset_link = Path("data/apps_temp_inference.jsonl")
    temp_features_link = Path("data/features_temp_inference")
    
    try:
        if temp_dataset_link.exists():
            temp_dataset_link.unlink()
        temp_dataset_link.symlink_to(Path(dataset_path).resolve())
        
        if temp_features_link.exists():
            if temp_features_link.is_symlink():
                temp_features_link.unlink()
            else:
                shutil.rmtree(temp_features_link)
        temp_features_link.symlink_to(features_dir.resolve())
        
        from steps import extract_text_features, extract_image_features, extract_slm_features
        
        print("\n" + "=" * 60)
        print("STEP 3a: Text Features")
        print("=" * 60)
        
        original_dataset_path = CFG.dataset_path
        original_features_dir = CFG.features_dir
        object.__setattr__(CFG, 'dataset_path', dataset_path)
        object.__setattr__(CFG, 'features_dir', str(features_dir))
        
        extract_text_features.main()
        
        print("\n" + "=" * 60)
        print("STEP 3b: Image Features")
        print("=" * 60)
        extract_image_features.main()
        
        print("\n" + "=" * 60)
        print("STEP 3c: SLM Features")
        print("=" * 60)
        extract_slm_features.main()
        
        object.__setattr__(CFG, 'dataset_path', original_dataset_path)
        object.__setattr__(CFG, 'features_dir', original_features_dir)
        print(f"\n✓ Features extracted to {features_dir}")
        
    finally:
        if temp_dataset_link.exists() and temp_dataset_link.is_symlink():
            temp_dataset_link.unlink()
        if temp_features_link.exists() and temp_features_link.is_symlink():
            temp_features_link.unlink()

def load_test_features(features_dir: Path):
    print(f"Loading features from {features_dir}...")
    td = np.load(features_dir / "text" / "features.npz", allow_pickle=True)
    imd = np.load(features_dir / "image" / "features.npz", allow_pickle=True)
    slmd = np.load(features_dir / "slm" / "features.npz", allow_pickle=True)

    app_ids = list(td["app_ids"])
    labels = td["labels"]
    has_real_labels = len(np.unique(labels)) > 1
    
    text_feats = np.concatenate([td["sbert"], td["keywords"], td["meta"], slmd["slm_score"]], axis=1)
    image_feats = np.concatenate([imd["clip_mean"], imd["clip_max"], imd["zeroshot"], imd["ocr"]], axis=1)
    all_feats = np.concatenate([text_feats, image_feats], axis=1)
    
    assert list(td["app_ids"]) == list(imd["app_ids"])
    return app_ids, labels, text_feats, image_feats, all_feats, has_real_labels

def print_and_save_report(name, app_ids, y_true, y_prob, threshold, output_csv, has_real_labels=True):
    print(f"\n[{name}] Threshold = {threshold}")
    preds = []
    for i, aid in enumerate(app_ids):
        is_llm = int(y_prob[i] >= threshold)
        row = {"app_id": aid, "y_prob": float(y_prob[i]), "prediction_label": is_llm}
        if has_real_labels:
            row["y_true"] = int(y_true[i])
            row["correct"] = int(is_llm == int(y_true[i]))
        preds.append(row)
    
    write_predictions_csv(output_csv, preds)
    print(f"  Saved: {output_csv}")
    
    if has_real_labels:
        metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
        print(f"  Accuracy: {metrics['accuracy']:.3f}  Precision: {metrics['precision_pos']:.3f}  Recall: {metrics['recall_pos']:.3f}  F1: {metrics['f1_pos']:.3f}")
    else:
        print(f"  Predicted {sum([p['prediction_label'] for p in preds])} apps with LLM")

def ensemble_early_fusion(X_all, num_folds=None):
    if num_folds is None:
        num_folds = CFG.n_folds
    early_fusion_dir = MODELS_DIR / "fusion" / "early_fusion" / "saved_models"
    fold_probs = [predict_early_fusion(X_all, early_fusion_dir, fold) for fold in range(num_folds)]
    return ensemble_across_folds(fold_probs)

def ensemble_late_fusion_stacking(X_text, X_image, num_folds=None):
    if num_folds is None:
        num_folds = CFG.n_folds
    base_models_dir = MODELS_DIR / "fusion" / "base_models_saved"
    stacking_dir = MODELS_DIR / "fusion" / "late_fusion_stacking" / "saved_models"
    
    fold_probs = []
    for fold in range(num_folds):
        text_prob, img_prob = load_and_predict_base_models(X_text, X_image, base_models_dir, fold)
        final_prob = predict_stacking_fusion(text_prob, img_prob, stacking_dir, fold)
        fold_probs.append(final_prob)
    return ensemble_across_folds(fold_probs)

def ensemble_late_fusion_soft_voting(X_text, X_image, num_folds=None):
    if num_folds is None:
        num_folds = CFG.n_folds
    base_models_dir = MODELS_DIR / "fusion" / "base_models_saved"
    
    fold_probs = []
    for fold in range(num_folds):
        text_prob, img_prob = load_and_predict_base_models(X_text, X_image, base_models_dir, fold)
        final_prob = predict_soft_voting(text_prob, img_prob)
        fold_probs.append(final_prob)
    return ensemble_across_folds(fold_probs)

def ensemble_late_fusion_max_voting(X_text, X_image, num_folds=None):
    if num_folds is None:
        num_folds = CFG.n_folds
    base_models_dir = MODELS_DIR / "fusion" / "base_models_saved"
    
    fold_probs = []
    for fold in range(num_folds):
        text_prob, img_prob = load_and_predict_base_models(X_text, X_image, base_models_dir, fold)
        final_prob = predict_max_voting(text_prob, img_prob)
        fold_probs.append(final_prob)
    return ensemble_across_folds(fold_probs)

def get_optimal_threshold(model_dir: Path, default_threshold: float = None) -> float:
    if default_threshold is None:
        default_threshold = CFG.inference_default_threshold
    json_path = model_dir / "best_threshold_metrics.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return float(data.get("best_threshold", default_threshold))
    return default_threshold

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/run_inference.py --input data/features --output predictions
  python src/run_inference.py --input-raw data/apps_inference_raw.jsonl --output predictions""")
    
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--input", "-i", type=str, help=f"Path to features directory (default: {CFG.inference_features_dir})")
    input_group.add_argument("--input-raw", type=str, help=f"Path to raw JSONL (default: {CFG.raw_inference_dataset_path})")
    parser.add_argument("--output", "-o", type=str, default=CFG.inference_output_dir, help=f"Output directory (default: {CFG.inference_output_dir})")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR")
    args = parser.parse_args()
    
    if args.input_raw:
        raw_jsonl_path = args.input_raw
        if not Path(raw_jsonl_path).exists():
            print(f"[ERROR] File not found: {raw_jsonl_path}")
            return
        
        print("=" * 60)
        print("  FULL INFERENCE PIPELINE")
        print("=" * 60)
        
        preprocessed_path = CFG.inference_dataset_path
        features_dir = CFG.inference_features_dir
        
        if not args.skip_preprocessing:
            run_preprocessing_for_inference(raw_jsonl_path, preprocessed_path)
        else:
            print("\n[SKIP] Preprocessing")
        
        if not args.skip_ocr:
            run_ocr_for_inference(preprocessed_path)
        else:
            print("\n[SKIP] OCR")
        
        run_feature_extraction_for_inference(preprocessed_path, features_dir)
        TEST_FEATURES_DIR = Path(features_dir)
    else:
        TEST_FEATURES_DIR = Path(args.input) if args.input else Path(CFG.features_dir)
        print("=" * 60)
        print("  INFERENCE FROM PRE-EXTRACTED FEATURES")
        print("=" * 60)
    
    OUTPUT_DIR_CUSTOM = Path(args.output)
    OUTPUT_DIR_CUSTOM.mkdir(parents=True, exist_ok=True)
    
    if not TEST_FEATURES_DIR.exists():
        print(f"[ERROR] Features not found: {TEST_FEATURES_DIR}")
        return
    
    print("\n" + "=" * 60)
    print("  LOADING FEATURES & RUNNING INFERENCE")
    print("=" * 60)
    
    app_ids, labels, text_feats, image_feats, all_feats, has_real_labels = load_test_features(TEST_FEATURES_DIR)
    print(f"Loaded {len(app_ids)} apps")
    
    early_fusion_dir = MODELS_DIR / "fusion" / "early_fusion"
    stacking_dir = MODELS_DIR / "fusion" / "late_fusion_stacking"
    soft_voting_dir = MODELS_DIR / "fusion" / "late_fusion_soft_voting"
    max_voting_dir = MODELS_DIR / "fusion" / "late_fusion_max_voting"
    
    ef_opt_threshold = get_optimal_threshold(early_fusion_dir)
    stack_opt_threshold = get_optimal_threshold(stacking_dir)
    soft_opt_threshold = get_optimal_threshold(soft_voting_dir)
    max_opt_threshold = get_optimal_threshold(max_voting_dir)
    
    print("\n" + "=" * 60)
    print("  EARLY FUSION")
    print("=" * 60)
    ef_prob = ensemble_early_fusion(all_feats)
    print_and_save_report("EARLY FUSION", app_ids, labels, ef_prob, ef_opt_threshold,
                         OUTPUT_DIR_CUSTOM / "early_fusion_inference.csv", has_real_labels)
    
    print("\n" + "=" * 60)
    print("  STACKING")
    print("=" * 60)
    stack_prob = ensemble_late_fusion_stacking(text_feats, image_feats)
    print_and_save_report("STACKING", app_ids, labels, stack_prob, stack_opt_threshold,
                         OUTPUT_DIR_CUSTOM / "stacking_inference.csv", has_real_labels)
    
    print("\n" + "=" * 60)
    print("  SOFT VOTING")
    print("=" * 60)
    soft_prob = ensemble_late_fusion_soft_voting(text_feats, image_feats)
    print_and_save_report("SOFT VOTING", app_ids, labels, soft_prob, soft_opt_threshold,
                         OUTPUT_DIR_CUSTOM / "soft_voting_inference.csv", has_real_labels)
    
    print("\n" + "=" * 60)
    print("  MAX VOTING")
    print("=" * 60)
    max_prob = ensemble_late_fusion_max_voting(text_feats, image_feats)
    print_and_save_report("MAX VOTING", app_ids, labels, max_prob, max_opt_threshold,
                         OUTPUT_DIR_CUSTOM / "max_voting_inference.csv", has_real_labels)
    
    print("\n" + "=" * 60)
    print("  ✓ COMPLETED")
    print("=" * 60)
    print(f"\nResults: {OUTPUT_DIR_CUSTOM}")
    print(f"  - early_fusion_inference.csv")
    print(f"  - stacking_inference.csv")
    print(f"  - soft_voting_inference.csv")
    print(f"  - max_voting_inference.csv")

if __name__ == "__main__":
    main()
