"""
ADIM 16 - BERT ile çok etiketli konu tespiti: eğitim.

Model: dbmdz/bert-base-turkish-cased (önceki adımlarla aynı taban) + 7 sigmoid çıkış (her konu ayrı evet/hayır;
bir yorumda birden çok konu olabilir). Kayıp: BCE, nadir konular için pos_weight = sqrt(neg/pos) (önceden sabit).
Eğitim: data/aspect_labels_step16 içindeki split=train yorumları. Val = Adım 15 val 100 + yeni val 200.
Test'e bu script HİÇ dokunmaz (evaluate_test.py, onaydan sonra bir kez).

PLAN.md'de önceden sabitlenenler: 3 tohum (0,1,2); epoch adayları 3/5/8/12 (aynı çalıştırmanın ara kayıtları,
sabit öğrenme oranı); her (tohum, epoch) için val olasılıkları val_probs.npz'ye, model model_s{tohum}_e{epoch}/'a.

Kullanım: python train.py            -> 3 tohum x (3,5,8) epoch, val olasılıkları + modeller
"""

import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import HERE, ROOT, norm  # step15_aspect'i sys.path'e ekler
from baseline import ASPECTS, turkish_lower  # noqa: E402

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
SEEDS = (0, 1, 2)
EPOCHS = (3, 5, 8, 12)   # 12: val görülmeden eklendi (kaba modelin eğitim kaybı 5 epochta hâlâ 0.57 idi)
LR, BATCH, MAX_LEN = 3e-5, 16, 128
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NEW_LABELS = os.path.join(ROOT, "data", "aspect_labels_step16", "aspect_labels16.csv")
OLD_LABELS = os.path.join(ROOT, "data", "aspect_labels", "aspect_labels.csv")


def targets(df):
    return (df[ASPECTS] != "").to_numpy(dtype=np.float32)


def load_data(max_round=None):
    """(eğitim, val) — val = eski val 100 + yeni val 200; `origin` sütunu hangisi olduğunu söyler."""
    new = pd.read_csv(NEW_LABELS, keep_default_na=False)
    old = pd.read_csv(OLD_LABELS, keep_default_na=False)
    train = new[new["split"] == "train"]
    if max_round is not None:
        train = train[train["round"] <= max_round]
    val = pd.concat([old[old["split"] == "val"].assign(origin="eski"),
                     new[new["split"] == "val"].assign(origin="yeni")], ignore_index=True)
    test_norm = set(old.loc[old["split"] == "test", "text"].map(norm))
    assert not set(train["text"].map(norm)) & set(val["text"].map(norm)), "eğitim-val sızıntısı"
    assert not set(train["text"].map(norm)) & test_norm, "eğitim-test sızıntısı"
    assert not set(val["text"].map(norm)) & test_norm, "val-test sızıntısı"
    return train.reset_index(drop=True), val


def predict(model, tokenizer, texts, batch=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tokenizer([turkish_lower(t) for t in texts[i:i + batch]], padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            out.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
    return np.concatenate(out)


def train_model(train, seed, n_epochs, on_epoch_end=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(ASPECTS), problem_type="multi_label_classification").to(DEVICE)
    texts, y = train["text"].tolist(), targets(train)
    pos = y.sum(0)
    pos_weight = torch.tensor(np.sqrt((len(y) - pos) / np.maximum(pos, 1)), dtype=torch.float32, device=DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    for epoch in range(1, n_epochs + 1):
        model.train()
        order, total = rng.permutation(len(texts)), 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            enc = tokenizer([turkish_lower(texts[j]) for j in idx], padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            loss = loss_fn(model(**enc).logits, torch.tensor(y[idx], device=DEVICE))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        print(f"  tohum {seed} epoch {epoch}: eğitim kaybı {total / len(order):.4f}", flush=True)
        if on_epoch_end:
            on_epoch_end(epoch, model, tokenizer)
    return model, tokenizer


if __name__ == "__main__":
    train, val = load_data()
    print(f"Eğitim {len(train)} | val {len(val)} ({val['origin'].value_counts().to_dict()}) | cihaz {DEVICE}")
    print("Eğitimde konu başına örnek:", dict(zip(ASPECTS, targets(train).sum(0).astype(int))))
    probs = {}
    for seed in SEEDS:
        def save(epoch, model, tokenizer, seed=seed):
            if epoch in EPOCHS:
                probs[f"s{seed}_e{epoch}"] = predict(model, tokenizer, val["text"].tolist())
                out = os.path.join(HERE, f"model_s{seed}_e{epoch}")
                model.save_pretrained(out)
                tokenizer.save_pretrained(out)
        train_model(train, seed, max(EPOCHS), save)
    np.savez(os.path.join(HERE, "val_probs.npz"), ids=val["id"].to_numpy(), **probs)
    print("val olasılıkları -> val_probs.npz ; modeller -> model_s*_e*/")
