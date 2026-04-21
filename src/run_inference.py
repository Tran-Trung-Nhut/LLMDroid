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


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")

def run_preprocessing_for_inference(raw_jsonl_path: str, output_jsonl_path: str):
    print_section("STEP 1: Preprocessing")
    from steps import preprocessing
    
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
    print_section("STEP 2: OCR Extraction")
    from steps import run_ocr
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
        
        print_section("STEP 3a: Text Features")
        
        original_dataset_path = CFG.dataset_path
        original_features_dir = CFG.features_dir
        object.__setattr__(CFG, 'dataset_path', dataset_path)
        object.__setattr__(CFG, 'features_dir', str(features_dir))
        
        extract_text_features.main()
        
        print_section("STEP 3b: Image Features")
        extract_image_features.main()
        
        print_section("STEP 3c: SLM Features")
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
        print(f"  Predicted {sum(p['prediction_label'] for p in preds)} apps with LLM")

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

def _is_within_dir(path: Path, root_dir: Path) -> bool:
    try:
        path.relative_to(root_dir)
        return True
    except ValueError:
        return False

def cleanup_inference_images(dataset_path: str, images_root: str) -> None:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        print(f"\n[SKIP] Cleanup images: dataset not found at {dataset_file}")
        return

    images_root_abs = (_PROJECT_ROOT / Path(images_root)).resolve()
    removed_files = 0
    missing_files = 0
    skipped_outside_root = 0
    candidate_files = set()

    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for image_path in row.get("image_paths", []):
                candidate_files.add(image_path)

    removed_parent_dirs = set()
    for image_path in candidate_files:
        img = Path(image_path)
        img_abs = img.resolve() if img.is_absolute() else (_PROJECT_ROOT / img).resolve()

        if not _is_within_dir(img_abs, images_root_abs):
            skipped_outside_root += 1
            continue

        if img_abs.exists() and img_abs.is_file():
            img_abs.unlink()
            removed_files += 1
            removed_parent_dirs.add(img_abs.parent)
        else:
            missing_files += 1

    # Remove now-empty app folders under images root.
    for parent in sorted(removed_parent_dirs, key=lambda p: len(p.parts), reverse=True):
        try:
            if parent.exists() and parent != images_root_abs and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            # Ignore non-empty or locked directories.
            pass

    print_section("IMAGE CLEANUP")
    print(f"Removed files: {removed_files}")
    if missing_files:
        print(f"Already missing: {missing_files}")
    if skipped_outside_root:
        print(f"Skipped outside {images_root_abs}: {skipped_outside_root}")
    print(f"Images root: {images_root_abs}")

def cleanup_inference_artifacts(preprocessed_path: str, features_dir: str, output_dir: Path) -> None:
    output_dir_abs = output_dir.resolve()

    artifact_paths = [
        Path(preprocessed_path),
        Path(features_dir),
        Path("data/apps_temp_inference.jsonl"),
        Path("data/features_temp_inference"),
    ]

    removed_files = 0
    removed_dirs = 0
    skipped_output_dir = 0

    for artifact in artifact_paths:
        artifact_abs = artifact.resolve() if artifact.is_absolute() else (_PROJECT_ROOT / artifact).resolve()

        if artifact_abs == output_dir_abs:
            skipped_output_dir += 1
            continue

        if artifact_abs.exists() and artifact_abs.is_file():
            artifact_abs.unlink()
            removed_files += 1
        elif artifact_abs.exists() and artifact_abs.is_dir():
            shutil.rmtree(artifact_abs)
            removed_dirs += 1

    print_section("ARTIFACT CLEANUP")
    print(f"Removed files: {removed_files}")
    print(f"Removed directories: {removed_dirs}")
    if skipped_output_dir:
        print(f"Skipped output directory: {output_dir_abs}")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/run_inference.py --input data/features --output predictions
  python src/run_inference.py --input-raw data/apps_inference_raw.jsonl --output predictions""")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", type=str, help="Path to pre-extracted features directory")
    input_group.add_argument("--input-raw", type=str, help="Path to raw JSONL")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR")
    parser.add_argument("--keep-images", action="store_true", help="Keep downloaded images after inference when using --input-raw")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep intermediate files generated during --input-raw inference")
    args = parser.parse_args()

    preprocessed_path = None
    features_dir = None
    
    if args.input_raw:
        raw_jsonl_path = args.input_raw
        if not Path(raw_jsonl_path).exists():
            print(f"[ERROR] File not found: {raw_jsonl_path}")
            return
        
        print_section("FULL INFERENCE PIPELINE")
        
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
        test_features_dir = Path(features_dir)
    else:
        test_features_dir = Path(args.input)
        print_section("INFERENCE FROM PRE-EXTRACTED FEATURES")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not test_features_dir.exists():
        print(f"[ERROR] Features not found: {test_features_dir}")
        return
    
    print_section("LOADING FEATURES & RUNNING INFERENCE")
    
    app_ids, labels, text_feats, image_feats, all_feats, has_real_labels = load_test_features(test_features_dir)
    print(f"Loaded {len(app_ids)} apps")
    
    early_fusion_dir = MODELS_DIR / "fusion" / "early_fusion"
    stacking_dir = MODELS_DIR / "fusion" / "late_fusion_stacking"
    soft_voting_dir = MODELS_DIR / "fusion" / "late_fusion_soft_voting"
    max_voting_dir = MODELS_DIR / "fusion" / "late_fusion_max_voting"
    
    ef_opt_threshold = get_optimal_threshold(early_fusion_dir)
    stack_opt_threshold = get_optimal_threshold(stacking_dir)
    soft_opt_threshold = get_optimal_threshold(soft_voting_dir)
    max_opt_threshold = get_optimal_threshold(max_voting_dir)
    
    print_section("EARLY FUSION")
    ef_prob = ensemble_early_fusion(all_feats)
    print_and_save_report("EARLY FUSION", app_ids, labels, ef_prob, ef_opt_threshold,
                         output_dir / "early_fusion_inference.csv", has_real_labels)
    
    print_section("STACKING")
    stack_prob = ensemble_late_fusion_stacking(text_feats, image_feats)
    print_and_save_report("STACKING", app_ids, labels, stack_prob, stack_opt_threshold,
                         output_dir / "stacking_inference.csv", has_real_labels)
    
    print_section("SOFT VOTING")
    soft_prob = ensemble_late_fusion_soft_voting(text_feats, image_feats)
    print_and_save_report("SOFT VOTING", app_ids, labels, soft_prob, soft_opt_threshold,
                         output_dir / "soft_voting_inference.csv", has_real_labels)
    
    print_section("MAX VOTING")
    max_prob = ensemble_late_fusion_max_voting(text_feats, image_feats)
    print_and_save_report("MAX VOTING", app_ids, labels, max_prob, max_opt_threshold,
                         output_dir / "max_voting_inference.csv", has_real_labels)
    
    print_section("COMPLETED")
    print(f"Results: {output_dir}")
    print(f"  - early_fusion_inference.csv")
    print(f"  - stacking_inference.csv")
    print(f"  - soft_voting_inference.csv")
    print(f"  - max_voting_inference.csv")

    if args.input_raw and not args.keep_images:
        cleanup_inference_images(CFG.inference_dataset_path, CFG.images_dir)
    elif args.input_raw and args.keep_images:
        print("\n[SKIP] Image cleanup because --keep-images was set")

    if args.input_raw and not args.keep_artifacts:
        cleanup_inference_artifacts(preprocessed_path, features_dir, output_dir)
    elif args.input_raw and args.keep_artifacts:
        print("\n[SKIP] Artifact cleanup because --keep-artifacts was set")

if __name__ == "__main__":
    main()
