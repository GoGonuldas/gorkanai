"""
ADIM 1: Bag-of-Words + Lojistik Regresyon ile duygu analizi.

ML akışı 5 adımdan oluşur: veri yükle -> özellik çıkar -> böl -> eğit -> değerlendir.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1) Veriyi yükle
df = pd.read_csv("../data/reviews.csv")
print(f"Toplam örnek sayısı: {len(df)}")
print(df["label"].value_counts(), "\n")

# 2) Metni sayılara çevir: "Bag of Words" -> her kelimenin cümlede kaç kez geçtiğini sayan bir tablo
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]
print(f"Kelime dağarcığı (vocabulary) boyutu: {len(vectorizer.vocabulary_)}\n")

# 3) 50 örnek çok küçük olduğu için tek bir train/test bölmesi yerine
#    5-katlı çapraz doğrulama kullanıyoruz: veriyi 5 parçaya bölüp her seferinde
#    4 parçayla eğitip 1 parçayla test ediyoruz, sonra ortalamasını alıyoruz.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("--- Regularization gücü (C) karşılaştırması ---")
print("Düşük C = daha güçlü regularization = ezberlemeye karşı daha dirençli\n")
for C in [10.0, 1.0, 0.1, 0.01]:
    model = LogisticRegression(max_iter=1000, C=C)
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"C={C:<5} ortalama doğruluk: {scores.mean():.2f}  (foldlar: {[round(s, 2) for s in scores]})")

# En güçlü regularization'ı seçip son modeli tüm veriyle eğitelim
best_C = 0.1
model = LogisticRegression(max_iter=1000, C=best_C)
model.fit(X, y)

# Kendi cümlelerimizle deneyelim
ornekler = [
    "Bu ürün tam bir hayal kırıklığıydı",
    "Kesinlikle harika bir deneyimdi, çok mutluyum",
    "Fena değildi ama daha iyisini beklerdim",
]
ornek_vec = vectorizer.transform(ornekler)
tahminler = model.predict(ornek_vec)
print("\n--- Yeni cümle tahminleri ---")
for cumle, tahmin in zip(ornekler, tahminler):
    print(f"{tahmin:8s} -> {cumle}")
