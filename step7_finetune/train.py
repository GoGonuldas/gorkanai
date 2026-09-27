"""
ADIM 7: Hazır bir Türkçe BERT modelini fine-tune etmek.

Adım 1-6'da hep sıfırdan öğrendik: kendi küçük veri setimizden kelime
vektörleri, kendi küçük ağımızı. Sorun hep aynıydı: veri o kadar az ki
model dilin genel kurallarını (olumsuzlama, eş anlamlılık) öğrenemiyordu.

Burada farklı bir şey yapıyoruz: "dbmdz/bert-base-turkish-cased" adlı,
Türkçe metinlerin MİLYARLARCA kelimesinde önceden eğitilmiş bir modeli
indiriyoruz. Bu model zaten "sıkıcı" ile "kötü"nün yakın, "değil"in
anlamı tersine çevirdiğini görmüş durumda. Biz sadece son katmanını
kendi pozitif/negatif etiketimize göre AYARLIYORUZ (fine-tuning).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.manual_seed(42)
MODEL_NAME = "dbmdz/bert-base-turkish-cased"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Cihaz: {device}")

# 1) Önceden eğitilmiş tokenizer ve modeli indir
print(f"\n'{MODEL_NAME}' indiriliyor (ilk çalıştırmada biraz sürer)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

label_map = {"negatif": 0, "pozitif": 1}
inv_label_map = {v: k for k, v in label_map.items()}

# 2) Veriyi yükle ve train/val olarak böl (hard_test hâlâ hiç görülmeyecek)
full_train_df = pd.read_csv("../data/reviews.csv")
hard_df = pd.read_csv("../data/hard_test.csv")
train_df, val_df = train_test_split(
    full_train_df, test_size=0.15, random_state=42, stratify=full_train_df["label"]
)


def make_batches(df, batch_size=16, shuffle=True):
    idx = np.arange(len(df))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        texts = df["text"].values[batch_idx].tolist()
        labels = df["label"].map(label_map).values[batch_idx]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        yield encoded, labels


@torch.no_grad()
def evaluate(df):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    for encoded, labels in make_batches(df, batch_size=32, shuffle=False):
        outputs = model(**encoded)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_preds.extend(preds.cpu().tolist())
    return correct / total, all_preds


# 3) Fine-tuning: küçük öğrenme oranı (2e-5) -> zaten iyi olan ağırlıkları hafifçe ayarlıyoruz
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
epochs = 4

print("\n--- Fine-tuning ---")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for encoded, labels in make_batches(train_df, batch_size=16, shuffle=True):
        optimizer.zero_grad()
        outputs = model(**encoded, labels=labels)   # model kendi loss'unu hesaplıyor
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
        n_batches += 1

    val_acc, _ = evaluate(val_df)
    print(f"epoch {epoch} | ortalama loss: {total_loss / n_batches:.4f} | val doğruluk: {val_acc:.2f}")

# 4) Zor test setinde değerlendir
hard_acc, hard_preds = evaluate(hard_df)
print(f"\nZor test doğruluğu: {hard_acc:.2f}")
print("Önceki adımlar -> TF-IDF+bigram: 0.50 | Embedding ortalaması: 0.69 | LSTM: 0.69 | Attention: 0.69\n")

print("--- Zor test tahminleri ---")
for text, true_label, pred_idx in zip(hard_df["text"], hard_df["label"], hard_preds):
    pred_label = inv_label_map[pred_idx]
    isaret = "OK " if true_label == pred_label else "X  "
    print(f"{isaret} gerçek={true_label:8s} tahmin={pred_label:8s} | {text}")
