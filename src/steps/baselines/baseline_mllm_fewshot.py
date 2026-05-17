"""
baseline_mllm_fewshot.py — Few-shot MLLM baseline (GPT-4o, 6 in-context exemplars).
Paper Section 4.2, Table 12, Row 4.

Protocol (paper Appendix A):
- 6 exemplars per call: 3 positive + 3 negative, from corresponding training fold
- Exemplars include description + up to 4 screenshots (k-medoids)
- Ground-truth probability: 0.97 (positive) or 0.03 (negative)
- Evaluated on independent test set (N=110)

Requires: OPENAI_API_KEY in environment.
Output: runs/feature_fusion/independent_test/baseline_mllm_fewshot_gpt4o.json
"""
import base64
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from config import CFG
from utils.io import read_jsonl, write_json
from utils.metrics import compute_binary_metrics

SYSTEM_PROMPT = """You are an expert app-store reviewer specializing in AI capability detection.
You will be shown examples of apps labeled as LLM-integrated (probability 0.97) or not (probability 0.03),
then asked to score a new app. Reply only with a single floating-point probability in [0, 1]."""

APP_TEMPLATE = """App title: {title}
App description: {description}
Probability: {prob}"""

QUERY_TEMPLATE = """App title: {title}
App description: {description}
Probability:"""


def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def select_exemplars(train_records: list, label_map: dict, n_per_class: int = 3, seed: int = 42) -> list:
    rng = random.Random(seed)
    positives = [r for r in train_records if label_map.get(r["app_id"], -1) == 1]
    negatives = [r for r in train_records if label_map.get(r["app_id"], -1) == 0]
    selected  = rng.sample(positives, min(n_per_class, len(positives)))
    selected += rng.sample(negatives, min(n_per_class, len(negatives)))
    rng.shuffle(selected)
    return selected


def build_few_shot_messages(exemplars: list, query_record: dict, label_map: dict,
                             kmedoids_fn, max_images: int = 4) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in exemplars:
        label    = label_map.get(ex["app_id"], 0)
        prob_str = "0.97" if label == 1 else "0.03"
        img_paths = [p for p in ex.get("image_paths", []) if Path(p).exists()]
        selected  = kmedoids_fn(img_paths, k=max_images)

        content = [{"type": "text", "text": APP_TEMPLATE.format(
            title=ex.get("title", ""),
            description=(ex.get("description") or "")[:800],
            prob=prob_str,
        )}]
        for p in selected:
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{image_to_b64(p)}"}})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": prob_str})

    img_paths = [p for p in query_record.get("image_paths", []) if Path(p).exists()]
    selected  = kmedoids_fn(img_paths, k=max_images)
    content   = [{"type": "text", "text": QUERY_TEMPLATE.format(
        title=query_record.get("title", ""),
        description=(query_record.get("description") or "")[:800],
    )}]
    for p in selected:
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{image_to_b64(p)}"}})
    messages.append({"role": "user", "content": content})
    return messages


def main():
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("[error] Set OPENAI_API_KEY environment variable.")
        return

    import openai
    client = openai.OpenAI(api_key=openai_key)

    out_dir = Path(CFG.runs_dir) / CFG.run_name / "independent_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_records = read_jsonl(CFG.raw_inference_dataset_path)
    test_label_map = {}
    with open("data/inference_manual.csv") as f:
        for r in csv.DictReader(f):
            test_label_map[r["pkg_name"]] = int(r["label"])

    train_records    = read_jsonl(CFG.dataset_path)
    train_label_map  = {r["app_id"]: r.get("label_binary", r.get("label", -1)) for r in train_records}
    train_records    = [r for r in train_records if train_label_map.get(r["app_id"], -1) >= 0]

    splits = []
    for fold in range(CFG.n_folds):
        with open(f"data/splits/fold_{fold}.json") as f:
            splits.append(json.load(f))

    sys.path.insert(0, str(Path(__file__).parent))
    from baseline_mllm_zeroshot import kmedoids_select

    results, y_true_list, y_prob_list = {}, [], []

    for r in test_records:
        app_id = r["app_id"]
        y_true = test_label_map.get(app_id, -1)
        if y_true < 0:
            continue

        fold_train_ids  = set(splits[0]["train_ids"])
        fold_train_recs = [t for t in train_records if t["app_id"] in fold_train_ids]
        exemplars = select_exemplars(fold_train_recs, train_label_map,
                                     n_per_class=3, seed=hash(app_id) % 10000)
        messages  = build_few_shot_messages(exemplars, r, train_label_map,
                                            kmedoids_select, max_images=4)
        try:
            resp  = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=10,
                temperature=0.0,
            )
            text  = resp.choices[0].message.content.strip().split()[0].rstrip(".,:") if resp.choices else "0.5"
            y_prob = max(0.0, min(1.0, float(text)))
        except Exception as e:
            print(f"  [warn] {app_id}: {e}")
            y_prob = 0.5

        results[app_id] = {"y_true": y_true, "y_prob": y_prob}
        y_true_list.append(y_true)
        y_prob_list.append(y_prob)
        print(f"  {app_id}: {y_prob:.3f} (true={y_true})")

    m = compute_binary_metrics(np.array(y_true_list), np.array(y_prob_list), threshold=0.5)
    print(f"\nGPT-4o (6-shot): Acc={m['accuracy']:.3f} F1={m['f1_pos']:.3f} AUC={m['roc_auc']:.3f}")
    write_json(out_dir / "baseline_mllm_fewshot_gpt4o.json", {"metrics": m, "predictions": results})


if __name__ == "__main__":
    main()
