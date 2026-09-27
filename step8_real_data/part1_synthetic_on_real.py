"""
ADIM 8 - Kısım 1: Sentetik veride eğitilen model gerçek dünyada ne yapar?

Adım 7'de BERT'i 330 sentetik (şablonla üretilmiş) cümlede fine-tune ettik
ve zor testte 0.88 aldık. Ama o cümleler hep kısa, düzgün yazılmış ve bizim
seçtiğimiz kelimelerle kuruluydu.

Burada AYNI modeli AYNI şekilde eğitiyoruz, ama bu kez gerçek insanların
yazdığı ürün yorumlarında (data/real/test.csv, 500 pozitif + 500 negatif)
test ediyoruz. Soru: 0.88 gerçekten "Türkçe duygu anlamayı" mı gösteriyordu,
yoksa sadece "bizim sentetik cümlelerimizi" mi?

Not: Test seti dengeli olduğu için doğruluğun yanında her sınıfın
doğruluğunu ayrı ayrı yazdırıyoruz — biri %100, diğeri %0 ise toplam %50
görünür ama model aslında hep aynı şeyi söylüyor demektir.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.manual_seed(42)
np.random.seed(42)
MODEL_NAME = "dbmdz/bert-base-turkish-cased"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Cihaz: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

label_map = {"negatif": 0, "pozitif": 1}
inv_label_map = {v: k for k, v in label_map.items()}

# Eğitim: Adım 7'deki sentetik veri (birebir aynı bölme)
full_train_df = pd.read_csv("../data/reviews.csv")
train_df, val_df = train_test_split(
    full_train_df, test_size=0.15, random_state=42, stratify=full_train_df["label"]
)
hard_df = pd.read_csv("../data/hard_test.csv")
real_test_df = pd.read_csv("../data/real/test.csv")


def make_batches(df, batch_size=16, shuffle=True, max_length=32):
    idx = np.arange(len(df))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        texts = df["text"].values[batch_idx].tolist()
        labels = df["label"].map(label_map).values[batch_idx]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        yield encoded, labels


@torch.no_grad()
def evaluate(df, max_length=32):
    model.eval()
    all_preds = []
    for encoded, _ in make_batches(df, batch_size=32, shuffle=False, max_length=max_length):
        all_preds.extend(model(**encoded).logits.argmax(dim=1).cpu().tolist())
    preds = np.array(all_preds)
    true = df["label"].map(label_map).values
    return (preds == true).mean(), preds


def per_class_report(df, preds):
    true = df["label"].map(label_map).values
    for name, idx in label_map.items():
        mask = true == idx
        print(f"  {name:8s}: {(preds[mask] == idx).mean():.2f}  ({mask.sum()} örnek)")


# Fine-tuning (Adım 7 ile aynı ayarlar)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
print("\n--- Sentetik veride fine-tuning (Adım 7 tekrarı) ---")
for epoch in range(1, 5):
    model.train()
    total_loss, n_batches = 0.0, 0
    for encoded, labels in make_batches(train_df):
        optimizer.zero_grad()
        outputs = model(**encoded, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
        n_batches += 1
    val_acc, _ = evaluate(val_df)
    print(f"epoch {epoch} | ortalama loss: {total_loss / n_batches:.4f} | val doğruluk: {val_acc:.2f}")

hard_acc, _ = evaluate(hard_df)
print(f"\nZor test (sentetik) doğruluğu: {hard_acc:.2f}   (Adım 7'de: 0.88)")

# Gerçek yorumlar daha uzun -> 128 token'a kadar okuyalım
real_acc, real_preds = evaluate(real_test_df, max_length=128)
print(f"\nGERÇEK yorum testi doğruluğu: {real_acc:.2f}   (yazı tura: 0.50)")
per_class_report(real_test_df, real_preds)

print("\n--- Yanlış tahminlerden örnekler ---")
wrong = real_test_df[real_preds != real_test_df["label"].map(label_map).values]
for text, true_label in wrong.head(12).itertuples(index=False):
    short = text if len(text) <= 110 else text[:107] + "..."
    print(f"X  gerçek={true_label:8s} | {short}")
