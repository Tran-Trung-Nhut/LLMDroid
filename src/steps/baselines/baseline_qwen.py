"""
baseline_qwen.py — Description-only LLM baseline (Qwen2.5-7B-Instruct, zero-shot).
Paper Section 4.2, Table 12, Row 1.

Protocol: prompt model with title + description, average 3 samples at temperature=0.3.
Output: runs/feature_fusion/independent_test/baseline_qwen.json
"""
import csv
import os
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

PROMPT_TEMPLATE = """You are an app-store reviewer. Decide whether the following app integrates a Large Language Model (LLM). An app integrates an LLM if it explicitly uses an LLM API, advertises generative text functionality (e.g., chatbot, free-form text generation, prompt-based Q&A), or names a specific LLM (e.g., ChatGPT, Claude, Gemini). Reply with a single floating-point probability in [0, 1] that the app integrates an LLM.

App title: {title}
App description: {description}

Probability:"""


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    out_dir = Path(CFG.runs_dir) / CFG.run_name / "independent_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(CFG.raw_inference_dataset_path)
    labels = {}
    with open(CFG.inference_manual_csv) as f:
        for r in csv.DictReader(f):
            labels[r["pkg_name"]] = int(r["label"])

    print("Loading Qwen2.5-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    results, y_true_list, y_prob_list = {}, [], []
    n_samples = 3

    for r in rows:
        app_id = r["app_id"]
        y_true = labels.get(app_id, -1)
        if y_true < 0:
            continue
        prompt = PROMPT_TEMPLATE.format(
            title=r.get("title", ""),
            description=(r.get("description") or "")[:1500],
        )
        scores = []
        for _ in range(n_samples):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10, temperature=0.3, do_sample=True)
            generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            try:
                score = float(generated.split()[0].rstrip(".,:"))
                score = max(0.0, min(1.0, score))
            except (ValueError, IndexError):
                score = 0.5
            scores.append(score)
        y_prob = float(np.mean(scores))
        results[app_id] = {"y_true": y_true, "y_prob": y_prob}
        y_true_list.append(y_true)
        y_prob_list.append(y_prob)

    m = compute_binary_metrics(np.array(y_true_list), np.array(y_prob_list), threshold=0.5)
    print(f"Qwen2.5-7B (desc only): Acc={m['accuracy']:.3f} F1={m['f1_pos']:.3f} AUC={m['roc_auc']:.3f}")
    write_json(out_dir / "baseline_qwen.json", {"metrics": m, "predictions": results})


if __name__ == "__main__":
    main()
