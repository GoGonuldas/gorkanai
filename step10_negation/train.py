"""
ADIM 10: Çift olumsuzlamayı düzeltmek ("hiç fena değildi" -> pozitif).

Teşhis: Sorun modelde değil, bizim örneklememizdeydi. Havuzda "fena değil"
geçen yorumların %71'i pozitif. Ama dengeli set kurarken negatiflerin
yarısından fazlasını, pozitiflerin ise sadece %4'ünü aldık. Sonuç: eğitim
setimizde "fena değil" yorumlarının sadece %12'si pozitifti. Model gördüğünü
öğrendi: "fena değil = negatif".

Düzeltme: Eğitim setinde bu kalıbı içeren yorumların pozitif oranını
havuzdaki (gerçek dünyadaki) orana geri getiriyoruz — havuzdan kalıbı
içeren pozitif yorumlar ekleyerek. Uydurma (sentetik) cümle yok, hepsi gerçek.

Aynı script iki modeli yan yana eğitir ve karşılaştırır:
  A) önceki gibi dengeli örnekleme (kıyas noktası)
  B) A + kalıp oranı düzeltilmiş

Değerlendirme (hiçbiri eğitimde yok):
  - data/real/test.csv       : genel doğruluk (düzeltme bunu BOZMAMALI)
  - negasyon holdout         : havuzdan ayrılan 200 gerçek "fena/kötü değil" yorumu
  - data/negation_test.csv   : elle yazılmış 25 cümle — çift olumsuzlama (pozitif)
                               + basit olumsuzlama (negatif, "aşırı düzeltme" kontrolü)
  - data/hard_test.csv       : eski sentetik zor test

Kullanım:
    python train.py                 # 4000 örnekle hızlı deney (~8 dk)
    python train.py 16000 --save    # büyük model, B'yi step10_negation/model/'e kaydet
"""

import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4000
SAVE = "--save" in sys.argv
EPOCHS = 2
MAX_LENGTH = 128
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# "fena değil", "kötü değildi", "berbat sayılmaz", "kalitesiz değil" ...
NEG_PATTERN = re.compile(r"\b(?:fena|kötü|berbat|kalitesiz)\w*\s+(?:değil|degil|sayılmaz|sayilmaz)")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Cihaz: {device} | eğitim boyutu: {N_TRAIN}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label_map = {"negatif": 0, "pozitif": 1}


def turkish_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


# --- Veri ---
pool = pd.read_csv("../data/real/train_pool.csv")
pool["has_pattern"] = pool["text"].str.contains(NEG_PATTERN)
real_test_df = pd.read_csv("../data/real/test.csv")
neg_test_df = pd.read_csv("../data/negation_test.csv")
hard_df = pd.read_csv("../data/hard_test.csv")

# 1) Kalıp içeren 200 gerçek yorumu test için ayır (doğal oranıyla)
pattern_rows = pool[pool["has_pattern"]]
pattern_ratio = (pattern_rows["label"] == "pozitif").mean()
holdout_df = pattern_rows.sample(200, random_state=42)
pool = pool.drop(holdout_df.index)
print(f"Kalıp içeren yorum: {len(pattern_rows)} | havuzdaki pozitif oranı: %{100 * pattern_ratio:.0f}")

# 2) Part 2'deki gibi val + dengeli eğitim seti (A)
pos = pool[pool["label"] == "pozitif"].sample(frac=1, random_state=42)
neg = pool[pool["label"] == "negatif"].sample(frac=1, random_state=42)
val_df = pd.concat([pos.iloc[:500], neg.iloc[:500]])
pos, neg = pos.iloc[500:], neg.iloc[500:]
train_a = pd.concat([pos.iloc[:N_TRAIN // 2], neg.iloc[:N_TRAIN // 2]])

# 3) B: kalıp içeren yorumların pozitif oranını havuz oranına geri getir
n_neg_pat = (train_a["has_pattern"] & (train_a["label"] == "negatif")).sum()
n_pos_pat = (train_a["has_pattern"] & (train_a["label"] == "pozitif")).sum()
target_pos = int(round(n_neg_pat * pattern_ratio / (1 - pattern_ratio)))
extra_pool = pos.iloc[N_TRAIN // 2:]
extra = extra_pool[extra_pool["has_pattern"]].head(max(0, target_pos - n_pos_pat))
train_b = pd.concat([train_a, extra])


def pattern_stats(df):
    p = df[df["has_pattern"]]
    return f"{len(p)} kalıplı yorum, pozitif %{100 * (p['label'] == 'pozitif').mean():.0f}"


print(f"A: {len(train_a)} örnek | {pattern_stats(train_a)}")
print(f"B: {len(train_b)} örnek (+{len(extra)} gerçek pozitif) | {pattern_stats(train_b)}")


def make_batches(df, batch_size=16, shuffle=True):
    idx = np.arange(len(df))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        texts = [turkish_lower(t) for t in df["text"].values[batch_idx]]
        labels = df["label"].map(label_map).values[batch_idx]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        yield encoded, torch.tensor(labels, dtype=torch.long, device=device)


@torch.no_grad()
def predict(model, df):
    model.eval()
    preds = []
    for encoded, _ in make_batches(df, batch_size=32, shuffle=False):
        preds.extend(model(**encoded).logits.argmax(dim=1).cpu().tolist())
    return np.array(preds)


def accuracy(model, df):
    preds = predict(model, df)
    return (preds == df["label"].map(label_map).values).mean(), preds


def train(train_df, name):
    torch.manual_seed(42)
    np.random.seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    print(f"\n=== Model {name}: {len(train_df)} örnek, {EPOCHS} epoch ===")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for encoded, labels in make_batches(train_df):
            optimizer.zero_grad()
            outputs = model(**encoded, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
            n_batches += 1
        val_acc, _ = accuracy(model, val_df)
        print(f"epoch {epoch} | ortalama loss: {total_loss / n_batches:.4f} | val doğruluk: {val_acc:.2f}")
    print(f"süre: {(time.time() - start) / 60:.1f} dk")
    return model


def report(model, name):
    real_acc, _ = accuracy(model, real_test_df)
    hold_acc, _ = accuracy(model, holdout_df)
    neg_acc, neg_preds = accuracy(model, neg_test_df)
    hard_acc, _ = accuracy(model, hard_df)
    true = neg_test_df["label"].map(label_map).values
    by_type = {t: (neg_preds[m] == true[m]).mean()
               for t in ["cift_olumsuz", "basit_olumsuz"]
               for m in [(neg_test_df["type"] == t).values]}
    return {"model": name, "gerçek test": real_acc, "negasyon holdout": hold_acc,
            "elle: çift olumsuz": by_type["cift_olumsuz"], "elle: basit olumsuz": by_type["basit_olumsuz"],
            "sentetik zor": hard_acc, "_neg_preds": neg_preds}


results = []
for name, df in [("A (eski örnekleme)", train_a), ("B (oran düzeltilmiş)", train_b)]:
    model = train(df, name)
    results.append(report(model, name))
    if SAVE and name.startswith("B"):
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"Model kaydedildi: {SAVE_DIR}")
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

print("\n=== ÖZET ===")
cols = [c for c in results[0] if not c.startswith("_")]
print(pd.DataFrame(results)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("(Negasyon holdout'ta %100 beklenmez: 'fena değil' gerçekten iki anlamlı, ~%29'u negatif etiketli)")

print("\n--- Elle yazılmış negasyon testi: A -> B ---")
inv = {v: k for k, v in label_map.items()}
for i, (text, label, _) in enumerate(neg_test_df.itertuples(index=False)):
    a, b = inv[results[0]["_neg_preds"][i]], inv[results[1]["_neg_preds"][i]]
    mark = lambda p: "OK" if p == label else "X "
    print(f"A:{mark(a)} B:{mark(b)} gerçek={label:8s} | {text}")
