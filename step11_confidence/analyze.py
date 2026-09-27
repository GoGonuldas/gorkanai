"""
ADIM 11 - Kısım 1: Modelin güveni "dürüst" mü? (Kalibrasyon analizi)

Kısa cümlelerde güven düşük ("Kumaşı kötü değil." -> %55 pozitif). Bu bir
hata mı, yoksa model gerçekten emin olmaması gereken yerde emin değil mi?

İyi kalibre edilmiş bir model: %90 güvenle söylediklerinin ~%90'ında haklıdır,
%60 güvenle söylediklerinin ~%60'ında. Güven gerçek başarıdan DÜŞÜKSE model
"gereğinden çekingen" (underconfident) — düzeltilebilir. Eşitse, düşük güven
bir hata değil, dürüstlüktür.

Burada ölçtüklerimiz (hepsi eğitimde hiç görülmemiş veriler):
  1) Güven aralıklarına göre gerçek doğruluk (gerçek test, 1000 yorum)
  2) Aynı analiz, kısa (<=5 kelime) ve uzun yorumlar için ayrı
  3) Negasyon holdout'u (Adım 10'daki 200 gerçek "fena/kötü değil" yorumu)
"""

import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "../step10_negation/model"
NEG_PATTERN = re.compile(r"\b(?:fena|kötü|berbat|kalitesiz)\w*\s+(?:değil|degil|sayılmaz|sayilmaz)")
label_map = {"negatif": 0, "pozitif": 1}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()


def turkish_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


@torch.no_grad()
def predict_probs(texts, batch_size=32):
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = [turkish_lower(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        probs.append(torch.softmax(model(**enc).logits, dim=1).cpu().numpy())
    return np.concatenate(probs)


def calibration_table(df, title):
    probs = predict_probs(df["text"].tolist())
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == df["label"].map(label_map).values
    print(f"\n--- {title} ({len(df)} örnek, doğruluk {correct.mean():.3f}) ---")
    print(f"{'güven aralığı':>15} | {'örnek':>5} | {'ort. güven':>10} | {'gerçek doğruluk':>15}")
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.001]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        print(f"  %{100 * lo:3.0f} - %{min(100 * hi, 100):3.0f}   | {m.sum():5d} | {conf[m].mean():10.3f} | {correct[m].mean():15.3f}")
    # ECE: güven ile doğruluk arasındaki ortalama fark (0 = mükemmel kalibrasyon)
    ece = sum((((conf >= lo) & (conf < hi)).sum() / len(conf)) *
              abs(conf[(conf >= lo) & (conf < hi)].mean() - correct[(conf >= lo) & (conf < hi)].mean())
              for lo, hi in zip(bins[:-1], bins[1:]) if ((conf >= lo) & (conf < hi)).any())
    print(f"  Kalibrasyon hatası (ECE): {ece:.3f}   | ortalama güven: {conf.mean():.3f}")
    return conf, correct


real_test = pd.read_csv("../data/real/test.csv")
n_words = real_test["text"].str.split().str.len()

calibration_table(real_test, "Gerçek test — tümü")
calibration_table(real_test[n_words <= 5], "Gerçek test — KISA yorumlar (<=5 kelime)")
calibration_table(real_test[n_words > 5], "Gerçek test — uzun yorumlar (>5 kelime)")

# Adım 10'daki negasyon holdout'unu aynı şekilde yeniden oluştur
pool = pd.read_csv("../data/real/train_pool.csv")
holdout = pool[pool["text"].str.contains(NEG_PATTERN)].sample(200, random_state=42)
calibration_table(holdout, "Negasyon holdout (gerçek 'fena/kötü değil' yorumları)")
