"""
ADIM 14 - 3 sınıflı model: pozitif / NÖTR / negatif.

Deney 1'de Vikipedi "nötr"leriyle model üslup kısayolu öğrendi (gerçek nötr yorumlarda 0.114).
Bu yüzden nötr sınıfı için ÜRÜN YORUMU nötrleri elle etiketlendi (Claude, active learning ile
seçilmiş 1000 kısa yorum -> data/neutral_labels/batch_0X.csv: 642 poz, 319 nötr, 39 neg).

Eğitim seti:
  - Adım 12'nin temizlenmiş pozitif/negatif seti (train_c: confident learning sonrası)
    (elle etiketlenen yorumlarla çakışanlar çıkarılır -> onların yerine elle etiketler geçer)
  - elle etiketlenmiş 1000 yorumun hepsi (nötr + kısa pozitif/negatif)
  - Nötrler NEUTRAL_OVERSAMPLE kez tekrarlanır: 319'a karşı ~16.000 poz/neg varken model
    "nötr" demeyi neredeyse hiç öğrenmezdi (sınıf dengesizliği)

Kısayol riski: nötrlerin hepsi kısa (<=10 kelime) -> model "kısa = nötr" öğrenebilir.
Önlem: elle etiketlenmiş 681 kısa POZİTİF/NEGATİF de eğitimde. Kontrol: kısayol testi.

Kıyas noktası: mevcut uygulama (Adım 12 modeli C + %95 "emin değilim" eşiği) — "emin değilim"
dediğini "nötr" sayarak aynı testte ölçülür. 3 sınıflı model ancak bundan iyiyse değer.

Ayarlar (tekrar sayısı, epoch) test sonuçlarına bakmadan ÖNCEDEN sabitlendi.
Çalıştırma: python train.py   (~20-30 dk)
"""

import gc
import glob
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
C_MODEL_DIR = "../step12_confident_learning/model"
HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(HERE, "model")
N_TRAIN = 16000
NEUTRAL_OVERSAMPLE = 8
EPOCHS = 2
C_THRESHOLD = 0.95
LABELS = ["negatif", "nötr", "pozitif"]
label_map = {name: i for i, name in enumerate(LABELS)}
NEG_PATTERN = re.compile(r"\b(?:fena|kötü|berbat|kalitesiz)\w*\s+(?:değil|degil|sayılmaz|sayilmaz)")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Cihaz: {device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def turkish_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


def release_memory():
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


# --- Adım 12'nin temizlenmiş eğitim setini (train_c) yeniden kur ---
pool = pd.read_csv("../data/real/train_pool.csv")
pool["has_pattern"] = pool["text"].str.contains(NEG_PATTERN)
pattern_rows = pool[pool["has_pattern"]]
pattern_ratio = (pattern_rows["label"] == "pozitif").mean()
holdout_df = pattern_rows.sample(200, random_state=42)
pool = pool.drop(holdout_df.index)
pos = pool[pool["label"] == "pozitif"].sample(frac=1, random_state=42).iloc[500:]
neg = pool[pool["label"] == "negatif"].sample(frac=1, random_state=42).iloc[500:]
train_a = pd.concat([pos.iloc[:N_TRAIN // 2], neg.iloc[:N_TRAIN // 2]])
n_neg_pat = (train_a["has_pattern"] & (train_a["label"] == "negatif")).sum()
n_pos_pat = (train_a["has_pattern"] & (train_a["label"] == "pozitif")).sum()
target_pos = int(round(n_neg_pat * pattern_ratio / (1 - pattern_ratio)))
extra_pool = pos.iloc[N_TRAIN // 2:]
extra = extra_pool[extra_pool["has_pattern"]].head(max(0, target_pos - n_pos_pat))
train_b = pd.concat([train_a, extra]).reset_index(drop=True)
flagged = set(pd.read_csv("../step12_confident_learning/flagged.csv")["text"])
train_c = train_b[~train_b["text"].isin(flagged)][["text", "label"]]

# --- Elle etiketlenmiş 1000 yorum ---
manual = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../data/neutral_labels/batch_*.csv"))])
train_c = train_c[~train_c["text"].isin(set(manual["text"]))]
neutral = manual[manual["label"] == "nötr"][["text", "label"]]
manual_sent = manual[manual["label"] != "nötr"][["text", "label"]]
train_df = pd.concat([train_c, manual_sent] + [neutral] * NEUTRAL_OVERSAMPLE).reset_index(drop=True)
print(f"Eğitim seti: {len(train_df)} satır | {train_df['label'].value_counts().to_dict()}")
print(f"  (nötr: {len(neutral)} benzersiz yorum x {NEUTRAL_OVERSAMPLE})")

# --- Test setleri ---
short = pd.read_csv("../data/short_clean_test.csv").rename(columns={"clean_label": "label"})
real_test = pd.read_csv("../data/real/test.csv")
neg_test = pd.read_csv("../data/negation_test.csv")
hard = pd.read_csv("../data/hard_test.csv")
for name, df in [("kısa temiz", short), ("gerçek test", real_test)]:
    overlap = set(df["text"]) & set(train_df["text"])
    assert not overlap, f"{name} eğitim setiyle çakışıyor: {list(overlap)[:3]}"


def make_batches(texts, labels=None, batch_size=16, shuffle=False):
    idx = np.arange(len(texts))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        b = idx[start:start + batch_size]
        enc = tokenizer([turkish_lower(texts[i]) for i in b], padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        y = None if labels is None else torch.tensor(labels[b], dtype=torch.long, device=device)
        yield enc, y


@torch.no_grad()
def predict_probs(model, texts):
    model.eval()
    return np.concatenate([torch.softmax(model(**enc).logits, 1).cpu().numpy()
                           for enc, _ in make_batches(texts, batch_size=64)])


# --- Eğitim ---
torch.manual_seed(42)
np.random.seed(42)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
texts, labels = train_df["text"].tolist(), train_df["label"].map(label_map).values
start = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total, n = 0.0, 0
    for enc, y in make_batches(texts, labels, shuffle=True):
        optimizer.zero_grad()
        loss = model(**enc, labels=y).loss
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    print(f"epoch {epoch} | loss {total / n:.4f}")
print(f"süre: {(time.time() - start) / 60:.1f} dk")
model.config.id2label = dict(enumerate(LABELS))
model.config.label2id = label_map
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model kaydedildi: {SAVE_DIR}")


def predict_3class(texts):
    return np.array(LABELS)[predict_probs(model, texts).argmax(1)]


# --- Kıyas: model C + %95 eşik ("emin değilim" = nötr) ---
c_model = AutoModelForSequenceClassification.from_pretrained(C_MODEL_DIR).to(device)


def predict_baseline(texts):
    p = predict_probs(c_model, texts)
    return np.where(p.max(1) < C_THRESHOLD, "nötr", np.where(p.argmax(1) == 1, "pozitif", "negatif"))


def confusion(true, pred):
    return pd.crosstab(pd.Series(true, name="doğrusu"), pd.Series(pred, name="tahmin"),
                       dropna=False).reindex(index=LABELS, columns=LABELS, fill_value=0)


def class_scores(true, pred):
    out = {}
    for c in LABELS:
        tp = ((pred == c) & (true == c)).sum()
        prec = tp / max((pred == c).sum(), 1)
        rec = tp / max((true == c).sum(), 1)
        out[c] = (prec, rec)
    return out


print("\n=== 1) KISA TEMİZ TEST (elle etiketli, 3 sınıf: 97 poz / 35 nötr / 68 neg) ===")
true = short["label"].values
results = {}
for name, fn in [("Kıyas: C + %95 eşik", predict_baseline), ("3 sınıflı model", predict_3class)]:
    pred = fn(short["text"].tolist())
    results[name] = pred
    sc = class_scores(true, pred)
    macro_f1 = np.mean([2 * p * r / max(p + r, 1e-9) for p, r in sc.values()])
    print(f"\n{name}: doğruluk {(pred == true).mean():.3f} | macro-F1 {macro_f1:.3f}")
    for c, (p, r) in sc.items():
        print(f"  {c:8s} kesinlik {p:.2f} | yakalama {r:.2f}")
    print(confusion(true, pred).to_string())

print("\n=== 2) Diğer testler (2 sınıflı etiketler) ===")
for name, df in [("gerçek test (gürültülü)", real_test), ("negasyon: elle", neg_test), ("sentetik zor", hard)]:
    for mname, fn in [("kıyas", predict_baseline), ("3 sınıf", predict_3class)]:
        pred = fn(df["text"].tolist())
        answered = pred != "nötr"
        acc = (pred[answered] == df["label"].values[answered]).mean()
        print(f"  {name:24s} [{mname:7s}] nötr/emin değil: %{100 * (~answered).mean():4.1f} | "
              f"cevap verilenlerde doğruluk {acc:.3f}")

print("\n=== 3) Kısayol testi: kısa ama duygulu cümleler nötr olmamalı ===")
probe = ["çok sağlam", "harika", "berbat", "bayıldım", "iade ettim", "tavsiye etmem", "süper ürün",
         "fena değil", "idare eder", "kargo 2 günde geldi.", "saygılarımla.", "bunda da aynı bedeni alın.",
         "ürün dün elime ulaştı.", "telefonun rengi siyah.",
         "Ürün 2019 yılında piyasaya sürülmüştür.",
         "Cihaz yüksek performansı ile kullanıcılar tarafından beğenilmiştir."]
p3 = predict_probs(model, probe)
pb = predict_baseline(probe)
for text, p, b in zip(probe, p3, pb):
    print(f"  3 sınıf: {LABELS[p.argmax()]:8s} %{100 * p.max():3.0f} | kıyas: {b:8s} | {text}")

print("\n--- 3 sınıflı modelin kısa temiz testteki hataları ---")
pred = results["3 sınıflı model"]
for text, t, p in zip(short["text"], true, pred):
    if t != p:
        print(f"  doğrusu={t:8s} tahmin={p:8s} | {text}")
