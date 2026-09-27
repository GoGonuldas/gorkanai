"""
ADIM 2: TF-IDF, n-gram'lar ve model karşılaştırması.

Adım 1'in sonunda gördüğümüz sorun: Bag-of-Words kelime sırasını bilmiyor,
bu yüzden "iyi değil" (olumsuz) ile "değil, iyi" gibi cümleleri ayıramıyor.
Burada iki şeyi karşılaştırıyoruz:
  1) CountVectorizer  vs  TfidfVectorizer   (kelime ağırlıklandırma yöntemi)
  2) unigram (tek kelime)  vs  bigram (iki kelimelik öbekler)

Ölçüm için iki farklı test kullanıyoruz:
  - Çapraz doğrulama (cross-validation): eğitim verisinin kendi içinden
  - Zor test seti (hard_test.csv): modelin HİÇ görmediği, farklı üslupla
    yazılmış cümleler -> gerçek genelleme yeteneğini gösterir
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("../data/reviews.csv")
hard_df = pd.read_csv("../data/hard_test.csv")
print(f"Eğitim örnek sayısı: {len(train_df)}")
print(f"Zor test örnek sayısı: {len(hard_df)}\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configs = {
    "CountVectorizer (unigram)": CountVectorizer(ngram_range=(1, 1)),
    "TfidfVectorizer (unigram)": TfidfVectorizer(ngram_range=(1, 1)),
    "TfidfVectorizer (unigram+bigram)": TfidfVectorizer(ngram_range=(1, 2)),
}

print(f"{'Yöntem':<35} {'CV doğruluk':>12} {'Zor test doğruluk':>20}")
print("-" * 70)

results = {}
for name, vectorizer in configs.items():
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    model = LogisticRegression(max_iter=1000)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)

    model.fit(X_train, y_train)
    X_hard = vectorizer.transform(hard_df["text"])
    hard_pred = model.predict(X_hard)
    hard_acc = accuracy_score(hard_df["label"], hard_pred)

    results[name] = (vectorizer, model, hard_pred)
    print(f"{name:<35} {cv_scores.mean():>11.2f}  {hard_acc:>19.2f}")

# En iyi yöntemle (bigram) zor test setindeki tahminleri tek tek görelim
print("\n--- TfidfVectorizer (unigram+bigram) ile zor test tahminleri ---")
_, _, hard_pred = results["TfidfVectorizer (unigram+bigram)"]
for text, true, pred in zip(hard_df["text"], hard_df["label"], hard_pred):
    isaret = "OK " if true == pred else "X  "
    print(f"{isaret} gerçek={true:8s} tahmin={pred:8s} | {text}")
