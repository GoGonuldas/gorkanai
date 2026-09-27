"""
ADIM 12: Confident learning — eğitim verisindeki hatalı etiketleri bulup temizlemek.

Adım 11'de gördük: veri setinde kısa "negatif" yorumların ~%19'u aslında pozitif
("kargo çok hızlıydı teşekkürler" -> negatif). Bu hatalar test setinde olduğu
gibi EĞİTİM setinde de var ve model onlardan da öğreniyor.

Fikir (Northcutt ve ark., "Confident Learning", 2021):
  1) Her eğitim örneği için, o örneği HİÇ GÖRMEMİŞ bir modelin tahminini al.
     (K-fold: seti K parçaya böl, K model eğit; her model kendi dışındaki parçayı tahmin eder.)
     Aynı modelle tahmin etseydik, hatalı etiketi zaten ezberlemiş olurdu.
  2) Her sınıf için bir güven eşiği hesapla: "etiketi pozitif olan örneklerde
     modelin ortalama pozitif olasılığı". Bu eşik sınıfa göre değişir, çünkü
     model bazı sınıflarda genel olarak daha emindir.
  3) Etiketi i olan bir örnekte model DİĞER sınıf j için o sınıfın eşiğini
     aşıyorsa -> "muhtemelen yanlış etiketli" diye işaretle.
  4) İşaretlenenleri çıkar, modeli yeniden eğit (model C).

Karşılaştırma: B = Adım 10 modeli (aynı eğitim seti, temizlenmemiş).
Ölçümler:
  - data/short_clean_test.csv  : ELLE etiketlenmiş kısa yorumlar (asıl ölçü!)
  - data/real/test.csv         : gerçek test — AMA etiketleri gürültülü; temizlenmiş
                                 model burada "düşük" görünebilir çünkü hatalı etiketleri
                                 artık taklit etmiyor
  - negasyon holdout, data/negation_test.csv, data/hard_test.csv

Çalıştırma: python train.py   (~30 dk; K-fold tahminleri oof_probs.npy'ye kaydedilir,
                                tekrar çalıştırınca o kısım atlanır)
"""

import gc
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
B_MODEL_DIR = "../step10_negation/model"
HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(HERE, "model")
OOF_PATH = os.path.join(HERE, "oof_probs.npy")
N_TRAIN = 16000
K_FOLDS = 3
EPOCHS = 2
MAX_LENGTH = 128
THRESHOLD = 0.80   # app'teki "emin değilim" eşiği (Adım 11)

NEG_PATTERN = re.compile(r"\b(?:fena|kötü|berbat|kalitesiz)\w*\s+(?:değil|degil|sayılmaz|sayilmaz)")
label_map = {"negatif": 0, "pozitif": 1}
inv_label_map = {v: k for k, v in label_map.items()}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Cihaz: {device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def turkish_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


# --- Adım 10'daki B eğitim setini birebir yeniden kur ---
pool = pd.read_csv("../data/real/train_pool.csv")
pool["has_pattern"] = pool["text"].str.contains(NEG_PATTERN)
pattern_rows = pool[pool["has_pattern"]]
pattern_ratio = (pattern_rows["label"] == "pozitif").mean()
holdout_df = pattern_rows.sample(200, random_state=42)
pool = pool.drop(holdout_df.index)
pos = pool[pool["label"] == "pozitif"].sample(frac=1, random_state=42)
neg = pool[pool["label"] == "negatif"].sample(frac=1, random_state=42)
pos, neg = pos.iloc[500:], neg.iloc[500:]
train_a = pd.concat([pos.iloc[:N_TRAIN // 2], neg.iloc[:N_TRAIN // 2]])
n_neg_pat = (train_a["has_pattern"] & (train_a["label"] == "negatif")).sum()
n_pos_pat = (train_a["has_pattern"] & (train_a["label"] == "pozitif")).sum()
target_pos = int(round(n_neg_pat * pattern_ratio / (1 - pattern_ratio)))
extra_pool = pos.iloc[N_TRAIN // 2:]
extra = extra_pool[extra_pool["has_pattern"]].head(max(0, target_pos - n_pos_pat))
train_b = pd.concat([train_a, extra]).reset_index(drop=True)
print(f"B eğitim seti: {len(train_b)} örnek")

real_test_df = pd.read_csv("../data/real/test.csv")
short_clean = pd.read_csv("../data/short_clean_test.csv")
short_clean = short_clean[short_clean["clean_label"] != "nötr"].rename(columns={"clean_label": "label"})
neg_test_df = pd.read_csv("../data/negation_test.csv")
hard_df = pd.read_csv("../data/hard_test.csv")


def make_batches(texts, labels=None, batch_size=16, shuffle=False):
    idx = np.arange(len(texts))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        b = idx[start:start + batch_size]
        enc = tokenizer([turkish_lower(texts[i]) for i in b], padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        y = None if labels is None else torch.tensor(labels[b], dtype=torch.long, device=device)
        yield enc, y


@torch.no_grad()
def predict_probs(model, texts):
    model.eval()
    out = [torch.softmax(model(**enc).logits, dim=1).cpu().numpy()
           for enc, _ in make_batches(texts, batch_size=64)]
    return np.concatenate(out)


def train_model(df, name):
    torch.manual_seed(42)
    np.random.seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    texts, labels = df["text"].tolist(), df["label"].map(label_map).values
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
        print(f"  [{name}] epoch {epoch} | loss {total / n:.4f}")
    print(f"  [{name}] süre: {(time.time() - start) / 60:.1f} dk")
    return model


def release_memory():
    # Not: fonksiyon içinde `del model` yazmak dışarıdaki değişkeni silmez — model bellekte kalır.
    # Bu yüzden çağıran yerde önce `del model`, sonra bu fonksiyon.
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


# --- 1) K-fold ile "görmediği örnek" tahminleri (out-of-fold) ---
if os.path.exists(OOF_PATH):
    oof = np.load(OOF_PATH)
    print(f"\nK-fold tahminleri diskten yüklendi: {OOF_PATH}")
else:
    print(f"\n=== 1) {K_FOLDS}-fold çapraz tahmin ===")
    rng = np.random.RandomState(0)
    fold_of = rng.randint(0, K_FOLDS, size=len(train_b))
    oof = np.zeros((len(train_b), 2))
    for k in range(K_FOLDS):
        model = train_model(train_b[fold_of != k], f"fold {k + 1}/{K_FOLDS}")
        oof[fold_of == k] = predict_probs(model, train_b[fold_of == k]["text"].tolist())
        del model
        release_memory()
    np.save(OOF_PATH, oof)

# --- 2) Confident learning: sınıf başına eşik, şüpheli etiketleri işaretle ---
given = train_b["label"].map(label_map).values
thresholds = np.array([oof[given == j, j].mean() for j in range(2)])
print(f"\n=== 2) Sınıf eşikleri: negatif {thresholds[0]:.3f} | pozitif {thresholds[1]:.3f} ===")
other = 1 - given
suspect = oof[np.arange(len(given)), other] >= thresholds[other]

train_b["n_words"] = train_b["text"].str.split().str.len()
train_b["p_other"] = oof[np.arange(len(given)), other]
flagged = train_b[suspect].sort_values("p_other", ascending=False)
flagged[["text", "label", "p_other"]].to_csv(os.path.join(HERE, "flagged.csv"), index=False)

print(f"Şüpheli etiket: {suspect.sum()} / {len(train_b)} (%{100 * suspect.mean():.1f})")
for name, j in label_map.items():
    m = given == j
    print(f"  '{name}' etiketlilerin %{100 * suspect[m].mean():.1f}'i şüpheli")
short = train_b["n_words"] <= 5
print(f"  kısa (<=5 kelime): %{100 * suspect[short].mean():.1f} | uzun: %{100 * suspect[~short].mean():.1f}")
print("\nEn şüpheli 25 örnek (modelin DİĞER sınıfa verdiği olasılıkla):")
for r in flagged.head(25).itertuples():
    t = r.text if len(r.text) <= 100 else r.text[:97] + "..."
    print(f"  etiket={r.label:8s} diğer sınıf=%{100 * r.p_other:3.0f} | {t}")

# --- 3) Temizlenmiş setle yeniden eğit (C) ---
train_c = train_b[~suspect].drop(columns=["n_words", "p_other"])
print(f"\n=== 3) Model C: {len(train_c)} örnek ({suspect.sum()} çıkarıldı) ===")
print(train_c["label"].value_counts().to_string())
model_c = train_model(train_c, "C")
model_c.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model kaydedildi: {SAVE_DIR}")


# --- 4) B ve C'yi karşılaştır ---
def evaluate(model):
    res = {}
    for name, df in [("gerçek test (gürültülü)", real_test_df), ("KISA TEMİZ (elle)", short_clean),
                     ("negasyon holdout", holdout_df), ("elle negasyon", neg_test_df), ("sentetik zor", hard_df)]:
        p = predict_probs(model, df["text"].tolist())
        ok = p.argmax(1) == df["label"].map(label_map).values
        res[name] = ok.mean()
        if name == "KISA TEMİZ (elle)":
            hi = p.max(1) >= THRESHOLD
            res["kısa temiz: kapsam@%80"] = hi.mean()
            res["kısa temiz: doğruluk@%80"] = ok[hi].mean()
        if name == "gerçek test (gürültülü)":
            hi = p.max(1) >= THRESHOLD
            res["gerçek: kapsam@%80"] = hi.mean()
    return res


print("\n=== 4) Karşılaştırma ===")
res_c = evaluate(model_c)
del model_c
release_memory()
model_b = AutoModelForSequenceClassification.from_pretrained(B_MODEL_DIR).to(device)
res_b = evaluate(model_b)
del model_b
release_memory()
print(f"{'':28s} {'B (Adım 10)':>12s} {'C (temiz)':>10s} {'fark':>7s}")
for key in res_b:
    print(f"{key:28s} {res_b[key]:12.3f} {res_c[key]:10.3f} {res_c[key] - res_b[key]:+7.3f}")
