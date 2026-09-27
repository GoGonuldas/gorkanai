"""
ADIM 11 - Kısım 2: "Emin değilim" eşiğini seçmek.

Eşiği gerçek test setine bakarak seçersek test setini "ayar" için kullanmış
oluruz (sızıntı) ve test sonucu fazla iyimser çıkar. Bu yüzden:
  1) Eşiği VAL setinde seç (Adım 10'daki val: havuzdan, eğitimde kullanılmadı)
  2) Seçilen eşiği gerçek test setinde SADECE doğrula

Kural: cevap verdiğimiz yorumlarda doğruluk en az HEDEF olsun; bunu sağlayan
en düşük eşiği (= en çok yoruma cevap veren) seç.
"""

import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "../step10_negation/model"
TARGET_ACC = 0.93
NEG_PATTERN = re.compile(r"\b(?:fena|kötü|berbat|kalitesiz)\w*\s+(?:değil|degil|sayılmaz|sayilmaz)")
label_map = {"negatif": 0, "pozitif": 1}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()


def turkish_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


@torch.no_grad()
def conf_and_correct(df, batch_size=32):
    probs = []
    texts = df["text"].tolist()
    for i in range(0, len(texts), batch_size):
        batch = [turkish_lower(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        probs.append(torch.softmax(model(**enc).logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    return probs.max(axis=1), probs.argmax(axis=1) == df["label"].map(label_map).values


# Adım 10'daki val setini birebir yeniden oluştur (holdout çıkarıldıktan sonra ilk 500+500)
pool = pd.read_csv("../data/real/train_pool.csv")
holdout = pool[pool["text"].str.contains(NEG_PATTERN)].sample(200, random_state=42)
pool = pool.drop(holdout.index)
pos = pool[pool["label"] == "pozitif"].sample(frac=1, random_state=42)
neg = pool[pool["label"] == "negatif"].sample(frac=1, random_state=42)
val_df = pd.concat([pos.iloc[:500], neg.iloc[:500]])
test_df = pd.read_csv("../data/real/test.csv")

val_conf, val_ok = conf_and_correct(val_df)
test_conf, test_ok = conf_and_correct(test_df)

print(f"Hedef: cevap verilenlerde doğruluk >= {TARGET_ACC}\n")
print(f"{'eşik':>6} | {'val kapsam':>10} | {'val doğruluk':>12}")
chosen = None
for t in np.arange(0.50, 0.96, 0.05):
    m = val_conf >= t
    print(f"  %{100 * t:3.0f} | {100 * m.mean():9.1f}% | {val_ok[m].mean():12.3f}")
    if chosen is None and val_ok[m].mean() >= TARGET_ACC:
        chosen = round(float(t), 2)

print(f"\nSeçilen eşik (val'e göre): %{100 * chosen:.0f}")
m = test_conf >= chosen
print(f"Gerçek testte doğrulama: kapsam %{100 * m.mean():.1f} | cevap verilenlerde doğruluk {test_ok[m].mean():.3f}"
      f" | 'emin değilim' denen {(~m).sum()} yorumda doğruluk {test_ok[~m].mean():.3f}")
print(f"(eşiksiz gerçek test doğruluğu: {test_ok.mean():.3f})")
