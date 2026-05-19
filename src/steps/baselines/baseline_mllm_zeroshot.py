"""
baseline_mllm_zeroshot.py — Zero-shot multimodal LLM baseline (GPT-4o-mini).
Paper Section 4.2, Table 12, Rows 2-3.

Protocol: k=4 representative screenshots via k-medoids on CLIP embeddings + description.
Requires: OPENAI_API_KEY in environment.
Output: runs/feature_fusion/independent_test/baseline_mllm_zeroshot_gpt4omini.json
"""
import base64
import os
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from config import CFG
from utils.io import read_jsonl, write_json, load_label_map
from utils.metrics import compute_binary_metrics

PROMPT = """You are an app-store reviewer. Decide whether the following Android app integrates a Large Language Model (LLM). An app integrates an LLM if it explicitly uses an LLM API, advertises generative text functionality (chatbot, free-form text generation, prompt-based Q&A), or names a specific LLM (ChatGPT, Claude, Gemini, etc.).

App title: {title}
App description: {description}

The screenshots are attached. Reply with a single floating-point probability in [0, 1] that the app integrates an LLM. Output only the number, nothing else."""


def kmedoids_select(image_paths: list, k: int = 4) -> list:
    if len(image_paths) <= k:
        return image_paths
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel

    processor = CLIPProcessor.from_pretrained(CFG.clip_model)
    clip      = CLIPModel.from_pretrained(CFG.clip_model, dtype=torch.float16)
    clip.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip.to(device)

    embeds = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            inp = processor(images=[img], return_tensors="pt").to(device)
            with torch.no_grad():
                e = clip.get_image_features(**inp)
            e = e / e.norm(dim=-1, keepdim=True)
            embeds.append(e.cpu().float().numpy()[0])
        except Exception:
            embeds.append(np.zeros(CFG.clip_embed_dim))

    embeds = np.array(embeds)
    rng = np.random.default_rng(42)
    medoid_idx = rng.choice(len(embeds), k, replace=False).tolist()
    for _ in range(100):
        dists = np.array([[np.linalg.norm(embeds[i] - embeds[m]) for m in medoid_idx]
                          for i in range(len(embeds))])
        clusters = [[] for _ in range(k)]
        for i, c in enumerate(dists.argmin(axis=1)):
            clusters[c].append(i)
        new_medoids = []
        for ci, members in enumerate(clusters):
            if not members:
                new_medoids.append(medoid_idx[ci])
                continue
            intra = np.array([[np.linalg.norm(embeds[a] - embeds[b]) for b in members] for a in members])
            new_medoids.append(members[intra.sum(axis=1).argmin()])
        if new_medoids == medoid_idx:
            break
        medoid_idx = new_medoids
    return [image_paths[i] for i in medoid_idx]


def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_openai(client, prompt: str, image_paths: list, model: str = "gpt-4o-mini") -> float:
    content = [{"type": "text", "text": prompt}]
    for p in image_paths:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_to_b64(p)}"}})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=10, temperature=0.0,
    )
    try:
        return max(0.0, min(1.0, float(resp.choices[0].message.content.strip().split()[0])))
    except Exception:
        return 0.5


def main():
    out_dir = Path(CFG.runs_dir) / CFG.run_name / "independent_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows   = read_jsonl(CFG.raw_inference_dataset_path)
    labels = load_label_map(CFG.inference_manual_csv)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("[error] Set OPENAI_API_KEY environment variable.")
        return

    import openai
    oai_client = openai.OpenAI(api_key=openai_key)
    providers = {
        "gpt_4o_mini": lambda prompt, imgs: call_openai(oai_client, prompt, imgs, model="gpt-4o-mini"),
    }

    for provider_name, call_fn in providers.items():
        results, y_true_list, y_prob_list, per_app_times = {}, [], [], []
        for r in rows:
            app_id = r["app_id"]
            y_true = labels.get(app_id, -1)
            if y_true < 0:
                continue
            img_paths = [p for p in r.get("image_paths", []) if Path(p).exists()]
            selected  = kmedoids_select(img_paths, k=4)
            prompt    = PROMPT.format(
                title=r.get("title", ""),
                description=(r.get("description") or "")[:1500],
            )
            t0 = time.perf_counter()
            y_prob = call_fn(prompt, selected)
            per_app_times.append(time.perf_counter() - t0)
            results[app_id] = {"y_true": y_true, "y_prob": y_prob}
            y_true_list.append(y_true)
            y_prob_list.append(y_prob)
            print(f"  [{provider_name}] {app_id}: {y_prob:.3f} (true={y_true})")

        latency = float(np.mean(per_app_times)) if per_app_times else 0.0
        m = compute_binary_metrics(np.array(y_true_list), np.array(y_prob_list), threshold=0.5)
        print(f"\n{provider_name} (zero-shot): Acc={m['accuracy']:.3f} F1={m['f1_pos']:.3f} AUC={m['roc_auc']:.3f}")
        print(f"Latency: {latency:.3f} s/app")
        write_json(out_dir / f"baseline_mllm_zeroshot_{provider_name}.json", {
            "metrics": m,
            "latency_s_per_app": round(latency, 4),
            "predictions": results,
        })


if __name__ == "__main__":
    main()
