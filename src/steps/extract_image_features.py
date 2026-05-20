"""extract_image_features.py — CLIP + zero-shot + OCR feature extraction."""
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from config import CFG
from steps.extract_text_features import compute_keyword_features
from utils.io import read_jsonl


def load_clip_model():
    processor = CLIPProcessor.from_pretrained(CFG.clip_model)
    model = CLIPModel.from_pretrained(CFG.clip_model, dtype=torch.float16)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device


def encode_images_clip(image_paths: list[str], processor, model, device,
                       batch_size: int = None) -> np.ndarray:
    if batch_size is None:
        batch_size = CFG.clip_batch_size
    all_embeds = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception:
                continue
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled_output = vision_outputs.pooler_output if hasattr(vision_outputs, "pooler_output") else vision_outputs[1]
            embeds = model.visual_projection(pooled_output)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(embeds.cpu().float().numpy())
    if all_embeds:
        return np.concatenate(all_embeds, axis=0)
    return np.zeros((0, CFG.clip_embed_dim), dtype=np.float32)


def encode_texts_clip(texts: list[str], processor, model, device) -> np.ndarray:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_outputs = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        pooled_output = text_outputs.pooler_output if hasattr(text_outputs, "pooler_output") else text_outputs[1]
        embeds = model.text_projection(pooled_output)
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds.cpu().float().numpy()


def compute_zeroshot_scores(image_embeds: np.ndarray,
                            pos_text_embeds: np.ndarray,
                            neg_text_embeds: np.ndarray) -> np.ndarray:
    if image_embeds.shape[0] == 0:
        return np.zeros(1, dtype=np.float32)
    pos_sims = image_embeds @ pos_text_embeds.T
    neg_sims = image_embeds @ neg_text_embeds.T
    s_chat = float(pos_sims.max()) - float(neg_sims.min())
    return np.array([s_chat], dtype=np.float32)


def compute_ocr_features(record: dict) -> np.ndarray:
    ocr_map = record.get("ocr_by_image", {})
    all_texts = list(ocr_map.values()) if ocr_map else []
    combined = " ".join(t for t in all_texts if t)
    n_images = max(len(record.get("image_paths", [])), 1)
    n_with_text = sum(1 for t in all_texts if t.strip())
    kw_feats = compute_keyword_features(combined)
    extra = np.array([float(len(combined)), float(n_with_text) / float(n_images)], dtype=np.float32)
    return np.concatenate([extra, kw_feats])


def main():
    rows = read_jsonl(CFG.dataset_path)
    out_dir = Path(CFG.features_dir) / "image"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP model: {CFG.clip_model} ...")
    processor, model, device = load_clip_model()

    pos_embeds = encode_texts_clip(list(CFG.clip_positive_prompts), processor, model, device)
    neg_embeds = encode_texts_clip(list(CFG.clip_negative_prompts), processor, model, device)

    app_ids, labels = [], []
    clip_mean_list, clip_max_list = [], []
    zs_list, ocr_list = [], []

    for r in tqdm(rows, desc="Image features"):
        app_ids.append(r["app_id"])
        labels.append(r["label_binary"])
        img_paths = r.get("image_paths", [])

        if img_paths:
            img_embeds = encode_images_clip(img_paths, processor, model, device, CFG.clip_batch_size)
        else:
            img_embeds = np.zeros((0, CFG.clip_embed_dim), dtype=np.float32)

        if img_embeds.shape[0] > 0:
            clip_mean_list.append(img_embeds.mean(axis=0))
            clip_max_list.append(img_embeds.max(axis=0))
        else:
            clip_mean_list.append(np.zeros(CFG.clip_embed_dim, dtype=np.float32))
            clip_max_list.append(np.zeros(CFG.clip_embed_dim, dtype=np.float32))

        zs_list.append(compute_zeroshot_scores(img_embeds, pos_embeds, neg_embeds))
        ocr_list.append(compute_ocr_features(r))

    clip_mean = np.stack(clip_mean_list)
    clip_max  = np.stack(clip_max_list)
    zeroshot  = np.stack(zs_list)
    ocr       = np.stack(ocr_list)

    np.savez_compressed(
        out_dir / "features.npz",
        app_ids=np.array(app_ids),
        labels=np.array(labels, dtype=np.int32),
        clip_mean=clip_mean,
        clip_max=clip_max,
        zeroshot=zeroshot,
        ocr=ocr,
    )
    print(f"Saved image features → {out_dir / 'features.npz'}")
    print(f"  clip_mean: {clip_mean.shape}")
    print(f"  clip_max:  {clip_max.shape}")
    print(f"  zeroshot:  {zeroshot.shape}")
    print(f"  ocr:       {ocr.shape}")

    del model, processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
