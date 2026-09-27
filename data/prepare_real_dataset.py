"""
Gerçek Türkçe ürün yorumlarını (fthbrmnby/turkish_product_reviews, ~235 bin)
indirip eğitim/test için hazırlar.

Önemli gözlem: veri çok DENGESİZ (~%94 pozitif). Her şeye "pozitif" diyen
bir model bile %94 doğruluk alır. Bu yüzden test setini DENGELİ kuruyoruz
(500 pozitif + 500 negatif) — böylece %50 = yazı tura, anlamlı bir ölçü olur.

Çıktılar (data/real/):
  test.csv        1000 örnek, dengeli, HİÇ eğitimde kullanılmayacak
  train_pool.csv  geri kalan her şey (eğitim için buradan örnek çekeceğiz)
"""

import os

import pandas as pd
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "real")
os.makedirs(OUT_DIR, exist_ok=True)

df = load_dataset("fthbrmnby/turkish_product_reviews")["train"].to_pandas()
df = df.rename(columns={"sentence": "text"})
df["label"] = df["sentiment"].map({0: "negatif", 1: "pozitif"})
df = df[["text", "label"]]

# Temizlik: boş yorumları ve birebir tekrar edenleri at
# (tekrarlar kalırsa aynı cümle hem eğitimde hem testte olabilir -> sızıntı)
df["text"] = df["text"].str.strip()
df = df[df["text"].str.len() > 0]
df = df.drop_duplicates(subset="text")
print(f"Temizlik sonrası: {len(df)} yorum")
print(df["label"].value_counts(), "\n")

# Dengeli test seti
test_df = pd.concat([
    df[df["label"] == "pozitif"].sample(500, random_state=42),
    df[df["label"] == "negatif"].sample(500, random_state=42),
]).sample(frac=1, random_state=42)

train_pool = df.drop(test_df.index)

test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)
train_pool.to_csv(os.path.join(OUT_DIR, "train_pool.csv"), index=False)
print(f"test.csv: {len(test_df)} | train_pool.csv: {len(train_pool)}")
