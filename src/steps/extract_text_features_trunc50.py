"""
extract_text_features_trunc50.py — Re-encode test set text with first 50 chars only.
Output: data/features_test_trunc50/text/features.npz
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from utils.io import read_jsonl
from steps.extract_text_features import encode_texts, compute_keyword_features, compute_meta_features


def main():
    out_dir = Path("data/features_test_trunc50/text")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(CFG.raw_inference_dataset_path)

    label_map = {}
    label_csv = Path("data/inference_manual.csv")
    if label_csv.exists():
        with open(label_csv) as f:
            for r in csv.DictReader(f):
                label_map[r["pkg_name"]] = int(r["label"])

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_model)
    model = AutoModel.from_pretrained(CFG.text_model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    app_ids, labels_list = [], []
    sbert_list, kw_list, meta_list = [], [], []

    for r in rows:
        app_id    = r["app_id"]
        desc_trunc = (r.get("description") or "")[:50]
        title      = r.get("title", "")
        text       = f"{title} {desc_trunc}".strip()

        emb  = encode_texts([text], tokenizer, model, device, CFG.text_batch_size)[0]
        kw   = compute_keyword_features(text)
        meta = compute_meta_features(r)

        app_ids.append(app_id)
        labels_list.append(label_map.get(app_id, 0))
        sbert_list.append(emb)
        kw_list.append(kw)
        meta_list.append(meta)

    np.savez_compressed(
        out_dir / "features.npz",
        app_ids=np.array(app_ids),
        labels=np.array(labels_list, dtype=np.int32),
        sbert=np.stack(sbert_list),
        keywords=np.stack(kw_list),
        meta=np.stack(meta_list),
    )
    print(f"Saved truncated text features → {out_dir / 'features.npz'}")
    print(f"  sbert: {np.stack(sbert_list).shape}, N={len(app_ids)}")


if __name__ == "__main__":
    main()
