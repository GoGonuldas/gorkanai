"""
ADIM 4: Kelime temsilleri (word embeddings).

Adım 2-3'te gördüğümüz kör nokta: TF-IDF/BoW her kelimeyi birbirinden
bağımsız bir sütun olarak görüyor. "kötü" ile "berbat" onun için
"elma" ile "masa" kadar alakasız -> hiç görülmemiş kelimede sıfır sinyal.

Word2Vec bunu şöyle çözer: her kelimeyi düşük boyutlu (burada 50 boyutlu)
YOĞUN bir vektörle temsil eder ve bu vektörleri "benzer bağlamda geçen
kelimeler benzer vektörlere sahip olsun" prensibiyle öğrenir. Etiket
(pozitif/negatif) GÖRMEZ -> tamamen ham metinden (unlabeled_corpus.txt)
öğrenir. Sonra bu vektörleri, etiketli veri (reviews.csv) üzerinde
eğitilen basit bir sınıflandırıcıya girdi olarak veririz.
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score


def tokenize(text):
    return text.lower().replace(",", "").replace(".", "").split()


# 1) Word2Vec'i ETİKETSİZ korpüsle eğit
with open("../data/unlabeled_corpus.txt", encoding="utf-8") as f:
    unlabeled_sentences = [tokenize(line) for line in f if line.strip()]

w2v = Word2Vec(
    sentences=unlabeled_sentences,
    vector_size=50,
    window=4,
    min_count=1,
    sg=1,          # skip-gram: küçük veri ve nadir kelimeler için CBOW'dan genelde daha iyi
    epochs=30,
    seed=42,
)
print(f"Word2Vec kelime dağarcığı: {len(w2v.wv)} kelime\n")

# 2) Sağlık kontrolü: model gerçekten anlamca yakın kelimeleri yakın mı koydu?
print("--- 'kötü' kelimesine en yakın 5 kelime ---")
for word, sim in w2v.wv.most_similar("kötü", topn=5):
    print(f"  {word:<20} benzerlik: {sim:.3f}")

print("\n--- 'harika' kelimesine en yakın 5 kelime ---")
for word, sim in w2v.wv.most_similar("harika", topn=5):
    print(f"  {word:<20} benzerlik: {sim:.3f}")


# 3) Bir cümleyi vektöre çevir: cümledeki kelime vektörlerinin ortalamasını al
def sentence_vector(text):
    tokens = tokenize(text)
    vectors = [w2v.wv[t] for t in tokens if t in w2v.wv]
    if not vectors:
        return np.zeros(w2v.vector_size)
    return np.mean(vectors, axis=0)


# 4) Etiketli veriyi yükle ve embedding tabanlı özelliklere çevir
train_df = pd.read_csv("../data/reviews.csv")
hard_df = pd.read_csv("../data/hard_test.csv")

X_train = np.array([sentence_vector(t) for t in train_df["text"]])
X_hard = np.array([sentence_vector(t) for t in hard_df["text"]])
y_train = train_df["label"]
y_hard = hard_df["label"]

# 5) Basit bir sınıflandırıcı eğit (Adım 2 ile karşılaştırmak için yine LogisticRegression)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
print(f"\nÇapraz doğrulama (eğitim verisi): {cv_scores.mean():.2f}")

model.fit(X_train, y_train)
hard_pred = model.predict(X_hard)
hard_acc = accuracy_score(y_hard, hard_pred)
print(f"Zor test doğruluğu: {hard_acc:.2f}")
print("Önceki adımlar -> TF-IDF+bigram: 0.50 | NN (reg.siz): 0.44 | NN (reg.li): 0.31\n")

print("--- Zor test tahminleri ---")
for text, true, pred in zip(hard_df["text"], y_hard, hard_pred):
    isaret = "OK " if true == pred else "X  "
    print(f"{isaret} gerçek={true:8s} tahmin={pred:8s} | {text}")
