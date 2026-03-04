"""
One-command runner.

Order:
1) make_splits (only if not exist)
2) train (5 folds)
3) infer multi-image (5 folds)

Run:
  python -m src.run
  python -m src.run --demo  # only fold 0
"""
import argparse
from pathlib import Path
from huggingface_hub import login
from src.config import CFG
import os

def _splits_exist() -> bool:
    return all(Path(CFG.splits_dir, f"fold_{i}.json").exists() for i in range(CFG.n_folds))

def _dataset_exists() -> bool:
    return Path(CFG.dataset_path).exists()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="Run only fold 0 for quick test")
    args = ap.parse_args()

    demo_flag = "--demo" if args.demo else ""

    if CFG.hf_token:
        print("[INFO] Logging in to Hugging Face...")
        login(token=CFG.hf_token)

    if not _dataset_exists():
        os.system("python -m src.preprocessing")

    if not _splits_exist():
        os.system("python -m src.make_splits")

    os.system(f"python -m src.train_paligemma_lora_single_image {demo_flag}")
    os.system(f"python -m src.infer_paligemma_multi_image {demo_flag}")

if __name__ == "__main__":
    main()