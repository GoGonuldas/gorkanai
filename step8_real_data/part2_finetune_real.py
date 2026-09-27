"""
ADIM 8 - Kısım 2: BERT'i GERÇEK yorumlarla fine-tune etmek.

Kısım 1'de sentetik veride eğitilen model gerçek yorumlarda 0.58'de kaldı
(domain shift). Şimdi modeli doğrudan gerçek yorumlarla eğitiyoruz.

Soru sadece "işe yarıyor mu?" değil, "NE KADAR veri gerekiyor?". Bu yüzden
aynı deneyi 3 farklı veri miktarıyla tekrarlıyoruz: 1000, 4000, 16000 yorum
(her biri yarı pozitif, yarı negatif). Her seferinde modeli sıfırdan
(önceden eğitilmiş BERT'ten) başlatıyoruz ki karşılaştırma adil olsun.

Neden dengeli örnekliyoruz? Havuz %94 pozitif. Olduğu gibi verirsek model
"hep pozitif de" kısayolunu öğrenir; negatifleri hiç tanımaz.

Değerlendirme:
  - data/real/test.csv  : 1000 gerçek yorum (500/500), hiç görülmedi
  - data/hard_test.csv  : bizim 16 sentetik zor cümlemiz (olumsuzlama vs.)
  - val (havuzdan ayrı 1000 yorum): sadece eğitim sırasında izlemek için

En büyük veriyle eğitilen model step8_real_data/model/ altına kaydedilir
(ileride uygulama/API yaparsak kullanmak için).
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
TRAIN_SIZES = [1000, 4000, 16000]   # toplam örnek (yarısı pozitif, yarısı negatif)
EPOCHS = 2
MAX_LENGTH = 128                    # gerçek yorumlar uzun: ~128 token çoğunu kapsıyor
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Cihaz: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label_map = {"negatif": 0, "pozitif": 1}

# --- Veri ---
pool = pd.read_csv("../data/real/train_pool.csv")
real_test_df = pd.read_csv("../data/real/test.csv")
hard_df = pd.read_csv("../data/hard_test.csv")

# Pozitif ve negatifleri bir kez karıştır; önce val'ı ayır, kalanlardan
# ilk N/2'yi al. Böylece 1000'lik set 4000'liğin, o da 16000'liğin alt kümesi
# olur -> sadece "veri miktarı" değişiyor, başka bir şey değil.
pos = pool[pool["label"] == "pozitif"].sample(frac=1, random_state=42)
neg = pool[pool["label"] == "negatif"].sample(frac=1, random_state=42)
val_df = pd.concat([pos.iloc[:500], neg.iloc[:500]])
pos, neg = pos.iloc[500:], neg.iloc[500:]
print(f"Havuzda kullanılabilir negatif yorum: {len(neg)} (en fazla {2 * len(neg)} dengeli örnek)")


def make_batches(df, batch_size=16, shuffle=True):
    idx = np.arange(len(df))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        texts = df["text"].values[batch_idx].tolist()
        labels = df["label"].map(label_map).values[batch_idx]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        yield encoded, torch.tensor(labels, dtype=torch.long, device=device)


@torch.no_grad()
def evaluate(model, df):
    model.eval()
    preds = []
    for encoded, _ in make_batches(df, batch_size=32, shuffle=False):
        preds.extend(model(**encoded).logits.argmax(dim=1).cpu().tolist())
    preds = np.array(preds)
    true = df["label"].map(label_map).values
    per_class = {name: (preds[true == i] == i).mean() for name, i in label_map.items()}
    return (preds == true).mean(), per_class, preds


results = []
for n in TRAIN_SIZES:
    torch.manual_seed(42)
    np.random.seed(42)
    train_df = pd.concat([pos.iloc[:n // 2], neg.iloc[:n // 2]])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    print(f"\n=== {n} gerçek yorumla fine-tuning ({EPOCHS} epoch) ===")
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
        val_acc, _, _ = evaluate(model, val_df)
        print(f"epoch {epoch} | ortalama loss: {total_loss / n_batches:.4f} | val doğruluk: {val_acc:.2f}")

    real_acc, real_pc, real_preds = evaluate(model, real_test_df)
    hard_acc, _, _ = evaluate(model, hard_df)
    minutes = (time.time() - start) / 60
    print(f"GERÇEK test: {real_acc:.3f} (negatif {real_pc['negatif']:.2f} / pozitif {real_pc['pozitif']:.2f})"
          f" | sentetik zor test: {hard_acc:.2f} | süre: {minutes:.1f} dk")
    results.append({"n": n, "real": real_acc, "neg": real_pc["negatif"], "pos": real_pc["pozitif"],
                    "hard": hard_acc, "dk": minutes})

    if n == TRAIN_SIZES[-1]:
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"Model kaydedildi: {SAVE_DIR}")

        print("\n--- En büyük modelin yanlışlarından örnekler ---")
        wrong = real_test_df[real_preds != real_test_df["label"].map(label_map).values]
        for text, true_label in wrong.head(12).itertuples(index=False):
            short = text if len(text) <= 110 else text[:107] + "..."
            print(f"X  gerçek={true_label:8s} | {short}")

    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()

print("\n=== ÖZET ===")
print("Kısım 1 (330 sentetik cümle) -> gerçek test: 0.58 | sentetik zor test: 0.88")
for r in results:
    print(f"{r['n']:>6} gerçek yorum -> gerçek test: {r['real']:.3f} "
          f"(neg {r['neg']:.2f} / poz {r['pos']:.2f}) | sentetik zor test: {r['hard']:.2f}")
