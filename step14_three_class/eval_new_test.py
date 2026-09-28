"""
Yapılacaklar #1 - V2b'yi hiç görülmemiş yeni test setinde ölç.

data/neutral_labels/batch_06.csv (prepare_new_test.py ile seçildi, HİÇBİR eğitim/val/test/
candidate setinde yok, hem eğitimde hem de val'i genişletirken hiç görülmedi) -> ilk kez burada
bakılıyor. Kıyas: model C + %95 eşik (mevcut app). V2b bias'ı (+3.00) Adım 14 Deney 3b'de val'de
seçilmişti, burada YENİDEN seçilmiyor (test setine bakarak ayar yapmak = sızıntı).
"""

import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

C_MODEL_DIR = "../step12_confident_learning/model"
V2B_MODEL_DIR = "model_v2b"
C_THRESHOLD = 0.95
V2B_BIAS = 3.00
LABELS = ["negatif", "nötr", "pozitif"]
label_map = {name: i for i, name in enumerate(LABELS)}
HERE = os.path.dirname(os.path.abspath(__file__))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def turkish_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


def make_batches(texts, batch_size=64):
    for start in range(0, len(texts), batch_size):
        b = texts[start:start + batch_size]
        yield [turkish_lower(t) for t in b]


@torch.no_grad()
def get_logits(model, tokenizer, texts):
    model.eval()
    out = []
    for chunk in make_batches(texts):
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out)


def decide_v2b(logits, bias):
    lg = logits.copy()
    lg[:, label_map["nötr"]] += bias
    return np.array(LABELS)[lg.argmax(1)]


def scores(true, pred):
    out = {}
    for c in LABELS:
        tp = ((pred == c) & (true == c)).sum()
        p, r = tp / max((pred == c).sum(), 1), tp / max((true == c).sum(), 1)
        out[c] = (p, r, 2 * p * r / max(p + r, 1e-9))
    return (pred == true).mean(), np.mean([f for _, _, f in out.values()]), out


test = pd.read_csv("../data/neutral_labels/batch_06.csv")
print(f"Yeni test seti: {len(test)} yorum | {test['label'].value_counts().to_dict()}")
true = test["label"].values
texts = test["text"].tolist()

# --- Kıyas: C + %95 eşik (mevcut app mantığı) ---
c_tok = AutoTokenizer.from_pretrained(C_MODEL_DIR)
c_model = AutoModelForSequenceClassification.from_pretrained(C_MODEL_DIR).to(device)
c_logits = get_logits(c_model, c_tok, texts)
c_probs = torch.softmax(torch.tensor(c_logits), 1).numpy()
baseline_pred = np.where(c_probs.max(1) < C_THRESHOLD, "nötr",
                          np.where(c_probs.argmax(1) == 1, "pozitif", "negatif"))
del c_model
torch.mps.empty_cache() if device.type == "mps" else None

# --- V2b + bias +3.00 (val'de önceden seçilmiş, burada sabit) ---
v2b_tok = AutoTokenizer.from_pretrained(V2B_MODEL_DIR)
v2b_model = AutoModelForSequenceClassification.from_pretrained(V2B_MODEL_DIR).to(device)
v2b_logits = get_logits(v2b_model, v2b_tok, texts)
v2b_pred = decide_v2b(v2b_logits, V2B_BIAS)
del v2b_model

print(f"\n=== YENİ, HİÇ GÖRÜLMEMİŞ TEST ({len(test)} kısa yorum: "
      f"{(true == 'pozitif').sum()} poz / {(true == 'nötr').sum()} nötr / {(true == 'negatif').sum()} neg) ===")
print(f"{'':28s} {'doğr.':>6s} {'mF1':>6s} | {'nötr P/R':>10s} | {'neg P/R':>10s} | {'poz P/R':>10s}")
for name, pred in [("Kıyas: C + %95 eşik", baseline_pred), (f"V2b, bias {V2B_BIAS:+.2f}", v2b_pred)]:
    acc, mf1, sc = scores(true, pred)
    pr = " | ".join(f"{sc[c][0]:.2f}/{sc[c][1]:.2f}".rjust(10) for c in ["nötr", "negatif", "pozitif"])
    print(f"{name:28s} {acc:6.3f} {mf1:6.3f} | {pr}")

print("\nV2b karışıklık matrisi:")
print(pd.crosstab(pd.Series(true, name="doğrusu"), pd.Series(v2b_pred, name="tahmin"))
      .reindex(index=LABELS, columns=LABELS, fill_value=0).to_string())

print("\n--- V2b hataları ---")
for text, t, p in zip(texts, true, v2b_pred):
    if t != p:
        print(f"  doğrusu={t:8s} tahmin={p:8s} | {text}")
