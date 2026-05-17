"""
baseline_e2e_transformer.py — End-to-end fine-tuned multimodal transformer.
Paper Appendix B, Table 12, Row 5.

Architecture:
  BGE-large-en-v1.5 (frozen, 1024-d) + CLIP ViT-L/14 mean-pooled (frozen, 768-d)
  → concat 1792-d → Linear(d_model=512) → 2 TransformerEncoder blocks (8 heads) → binary head
  AdamW (lr=1e-4, wd=1e-2), batch=16, max_epochs=50, early-stopping patience=10 on val F1
  5-fold CV (same splits as main pipeline)
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from config import CFG
from utils.io import write_json
from utils.metrics import compute_binary_metrics
from steps.train_evaluate import load_features, load_split


class MultimodalDataset(Dataset):
    def __init__(self, text_feats, image_feats, labels):
        self.text   = torch.tensor(text_feats, dtype=torch.float32)
        self.image  = torch.tensor(image_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.text[i], self.image[i], self.labels[i]


class MultimodalTransformer(nn.Module):
    def __init__(self, text_dim=1024, img_dim=768, d_model=512, nhead=8, n_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(text_dim + img_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, text, img):
        x = self.proj(torch.cat([text, img], dim=-1))
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return self.head(x).squeeze(-1)  # raw logits — sigmoid applied by BCEWithLogitsLoss


def train_one_fold(X_text_tr, X_img_tr, y_tr,
                   X_text_val, X_img_val, y_val,
                   X_text_te, X_img_te, seed: int, fold: int):
    torch.manual_seed(seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl = DataLoader(MultimodalDataset(X_text_tr, X_img_tr, y_tr), batch_size=16, shuffle=True)
    val_dl   = DataLoader(MultimodalDataset(X_text_val, X_img_val, y_val), batch_size=32, shuffle=False)

    model     = MultimodalTransformer(text_dim=X_text_tr.shape[1], img_dim=X_img_tr.shape[1]).to(device)
    n_pos     = float((y_tr == 1).sum())
    n_neg     = float((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    best_val_f1, best_state, patience_cnt = -1.0, None, 0
    for _ in range(50):
        model.train()
        for txt, img, lbl in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(txt.to(device), img.to(device)), lbl.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for txt, img, lbl in val_dl:
                preds.extend(torch.sigmoid(model(txt.to(device), img.to(device))).cpu().numpy())
                trues.extend(lbl.numpy())
        m = compute_binary_metrics(np.array(trues), np.array(preds), threshold=0.5)
        if m["f1_pos"] > best_val_f1:
            best_val_f1  = m["f1_pos"]
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 10:
                break

    model.load_state_dict(best_state)
    model.eval()
    te_dl = DataLoader(MultimodalDataset(X_text_te, X_img_te, np.zeros(len(X_text_te))), batch_size=32)
    test_preds = []
    with torch.no_grad():
        for txt, img, _ in te_dl:
            test_preds.extend(torch.sigmoid(model(txt.to(device), img.to(device))).cpu().numpy())
    return np.array(test_preds)


def main():
    from utils.seed import set_seed
    set_seed(CFG.seed)

    base_dir = Path(CFG.runs_dir) / CFG.run_name
    out_dir  = base_dir / "independent_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    data         = load_features()
    id2idx       = data["id2idx"]
    labels       = data["labels"]
    X_text_train = data["sbert"]
    X_img_train  = data["clip_mean"]

    test_text_path = Path("data/features_test/text/features.npz")
    test_img_path  = Path("data/features_test/image/features.npz")
    if not test_text_path.exists():
        print(f"[error] Test features not found: {test_text_path}")
        print("  Run feature extraction for the 110-app test set first.")
        return

    td          = np.load(test_text_path, allow_pickle=True)
    imd         = np.load(test_img_path,  allow_pickle=True)
    X_text_test = td["sbert"]
    X_img_test  = imd["clip_mean"]
    y_test      = td["labels"].astype(int)

    test_probs = np.zeros(len(y_test))

    for fold in range(CFG.n_folds):
        split     = load_split(fold)
        train_idx = [id2idx[a] for a in split["train_ids"] if a in id2idx]

        X_text_tr = X_text_train[train_idx]
        X_img_tr  = X_img_train[train_idx]
        y_tr      = labels[train_idx]

        n_val   = max(1, int(len(train_idx) * 0.2))
        rng     = np.random.RandomState(CFG.seed + fold)
        val_rel = rng.choice(len(train_idx), n_val, replace=False)
        tr_rel  = np.setdiff1d(np.arange(len(train_idx)), val_rel)

        preds = train_one_fold(
            X_text_tr[tr_rel], X_img_tr[tr_rel], y_tr[tr_rel],
            X_text_tr[val_rel], X_img_tr[val_rel], y_tr[val_rel],
            X_text_test, X_img_test,
            CFG.seed, fold,
        )
        test_probs += preds
        print(f"  Fold {fold}: trained, predicting on N=110 test set...")

    test_probs /= CFG.n_folds
    m = compute_binary_metrics(y_test, test_probs, threshold=0.5)
    print(f"\nE2E Transformer (5-fold ensemble → N=110 independent test):")
    print(f"  Acc={m['accuracy']:.3f}  F1={m['f1_pos']:.3f}  AUC={m['roc_auc']:.3f}")
    print(f"  (Paper Table 12 reference: F1=0.682, AUC=0.853)")

    write_json(out_dir / "baseline_e2e_transformer.json", {
        "metrics": m,
        "protocol": "5-fold ensemble on independent test set (N=110)",
        "architecture": "BGE(1024) + CLIP_mean(768) -> Linear(512) -> 2xTransformerEncoder -> binary head",
    })


if __name__ == "__main__":
    main()
