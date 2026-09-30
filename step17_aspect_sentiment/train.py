"""
ADIM 17 - Konuya koşullu duygu modeli: eğitim.

Girdi: [CLS] <konu ifadesi> [SEP] <yorum> [SEP]  ->  pozitif / negatif (nötr KAPALI; PLAN.md madde 1).
Eğitim: Adım 16'nın 600 eğitim yorumundaki (konu, duygu) çiftleri; altın nötr çiftler eğitime girmez,
karışık çiftler baskın duygularıyla girer. Val: eski val 100 + yeni val 200 (Adım 16 ile aynı bölme).
Test'lere bu script HİÇ dokunmaz.

Önceden sabit (PLAN.md): başlangıç adayları bert (dbmdz/bert-base-turkish-cased, 2 çıkış) ve v2b
(step14_three_class/model_v2b, 3'lü başlık korunur; kayıp ve karar sadece negatif/pozitif logit'lerinde);
3 tohum; lr 3e-5 sabit, batch 16, wd 0.01; epoch adayları 2/4/6/8/12/16 (aynı çalıştırmanın ara kayıtları).
Çıktı: val_probs.npz (her başlangıç/tohum/epoch için P(pozitif)), modeller model_<başlangıç>_s<tohum>_e<epoch>/.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step16_topic_bert"))
from train import ASPECTS, DEVICE, load_data  # noqa: E402  (Adım 16: veri + sızıntı assert'leri)
from baseline import turkish_lower  # noqa: E402

INITS = {"bert": ("dbmdz/bert-base-turkish-cased", (0, 1)),                      # (model, (negatif, pozitif) logit sütunu)
         "v2b": (os.path.join(ROOT, "step14_three_class", "model_v2b"), (0, 2))}  # id2label: 0 negatif, 1 nötr, 2 pozitif
SEEDS = (0, 1, 2)
EPOCHS = (2, 4, 6, 8, 12, 16)
LR, BATCH, MAX_LEN = 3e-5, 16, 160
PHRASE = {"kargo": "kargo ve teslimat", "fiyat": "fiyat", "kalite": "kalite", "performans": "performans ve özellikler",
          "boyut": "boyut", "gorunum": "görünüm", "satici": "satıcı ve hizmet"}


def pairs(df):
    """Yorumlardan (konu, yorum, duygu) çiftleri; `row` = yorumun df içindeki sırası."""
    rows = [(n, r["id"], a, r["text"], r[a], int(a in str(r["karisik_konular"]).split(";")))
            for n, (_, r) in enumerate(df.iterrows()) for a in ASPECTS if r[a]]
    return pd.DataFrame(rows, columns=["row", "id", "konu", "text", "duygu", "karisik"])


def encode(tokenizer, konular, texts):
    return tokenizer([PHRASE[k] for k in konular], [turkish_lower(t) for t in texts], padding=True,
                     truncation="only_second", max_length=MAX_LEN, return_tensors="pt").to(DEVICE)


def predict(model, tokenizer, cols, konular, texts, batch=64):
    """P(pozitif), sadece negatif/pozitif logit'leri arasında."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            lg = model(**encode(tokenizer, konular[i:i + batch], texts[i:i + batch])).logits[:, list(cols)]
            out.append(torch.softmax(lg, 1)[:, 1].float().cpu().numpy())
    return np.concatenate(out)


def train_model(init, train_pairs, seed, on_epoch_end):
    name, cols = INITS[init]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tokenizer = AutoTokenizer.from_pretrained(name)
    kw = {"num_labels": 2} if init == "bert" else {}
    model = AutoModelForSequenceClassification.from_pretrained(name, **kw).to(DEVICE)
    konular, texts = train_pairs["konu"].tolist(), train_pairs["text"].tolist()
    y = (train_pairs["duygu"] == "pozitif").to_numpy().astype(np.int64)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    for epoch in range(1, max(EPOCHS) + 1):
        model.train()
        order, total = rng.permutation(len(texts)), 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            lg = model(**encode(tokenizer, [konular[j] for j in idx], [texts[j] for j in idx])).logits[:, list(cols)]
            loss = torch.nn.functional.cross_entropy(lg, torch.tensor(y[idx], device=DEVICE))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        print(f"  {init} tohum {seed} epoch {epoch}: eğitim kaybı {total / len(order):.4f}", flush=True)
        if epoch in EPOCHS:
            on_epoch_end(epoch, model, tokenizer, cols)


if __name__ == "__main__":
    train, val = load_data()
    tp, vp = pairs(train), pairs(val)
    tp = tp[tp["duygu"] != "nötr"].reset_index(drop=True)
    print(f"Eğitim çifti {len(tp)} ({tp['duygu'].value_counts().to_dict()}, karışık {tp['karisik'].sum()}) | "
          f"val çifti {len(vp)} ({vp['duygu'].value_counts().to_dict()}) | cihaz {DEVICE}")
    probs = {}
    for init in INITS:
        for seed in SEEDS:
            def save(epoch, model, tokenizer, cols, init=init, seed=seed):
                probs[f"{init}_s{seed}_e{epoch}"] = predict(model, tokenizer, cols, vp["konu"].tolist(), vp["text"].tolist())
                out = os.path.join(HERE, f"model_{init}_s{seed}_e{epoch}")
                model.save_pretrained(out)
                tokenizer.save_pretrained(out)
            train_model(init, tp, seed, save)
            np.savez(os.path.join(HERE, "val_probs.npz"), ids=vp["id"].to_numpy(), konu=vp["konu"].to_numpy(), **probs)
    print("val olasılıkları -> val_probs.npz ; modeller -> model_*_s*_e*/")
