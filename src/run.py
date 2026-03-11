"""
One-command runner.

Order:
1) preprocessing (if needed)
2) run_ocr (only if OCR data missing and --skip-ocr not set)
3) make_splits (if not exist)
4) train (5 folds)
5) infer multi-image (5 folds)

Run:
  python -m src.run                    # Full pipeline with OCR
  python -m src.run --demo             # Only fold 0 with OCR
  python -m src.run --skip-ocr         # Skip OCR, train with visual-only
  python -m src.run --demo --skip-ocr  # Fast test without OCR
"""
import argparse
import json
from pathlib import Path
from huggingface_hub import login
from src.config import CFG
import os

def _splits_exist() -> bool:
    return all(Path(CFG.splits_dir, f"fold_{i}.json").exists() for i in range(CFG.n_folds))

def _dataset_exists() -> bool:
    return Path(CFG.dataset_path).exists()

def _has_ocr_data() -> bool:
    if not _dataset_exists():
        return False
    
    try:
        with open(CFG.dataset_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line:
                return False
            row = json.loads(first_line)
            # Check if has ocr_by_image field with data
            ocr_map = row.get('ocr_by_image')
            return isinstance(ocr_map, dict) and len(ocr_map) > 0
    except:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="Run only fold 0 for quick test")
    ap.add_argument("--skip-ocr", action="store_true", help="Skip OCR step (train with visual-only)")
    args = ap.parse_args()

    demo_flag = "--demo" if args.demo else ""

    if CFG.hf_token:
        print("[INFO] Logging in to Hugging Face...")
        try:
            login(token=CFG.hf_token)
        except Exception as e:
            print(f"[WARNING] Could not login to HF: {e}")
            print("[INFO] Continuing anyway (model may already be cached)...")

    if not _dataset_exists():
        print("[INFO] Running preprocessing...")
        os.system("python -m src.preprocessing")

    if not args.skip_ocr:
        if not _has_ocr_data():
            print("[INFO] OCR data not found. Running OCR...")
            os.system("python -m src.run_ocr")
        else:
            print("[INFO] OCR data already exists, skipping OCR step")
    else:
        print("[INFO] --skip-ocr flag: Training with visual-only (no OCR)")

    if not _splits_exist():
        print("[INFO] Creating train/test splits...")
        os.system("python -m src.make_splits")

    print("[INFO] Training model...")
    os.system(f"python -m src.train_paligemma_lora_single_image {demo_flag}")
    
    print("[INFO] Running inference...")
    os.system(f"python -m src.infer_paligemma_multi_image {demo_flag}")

if __name__ == "__main__":
    main()