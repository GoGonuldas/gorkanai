"""
ADIM 5: LSTM ile sıralı (sequential) modelleme.

Adım 4'teki embedding ortalaması kelime SIRASINI tamamen kaybediyordu:
"iyi değil" ve "değil iyi" ortalamada birebir aynı vektörü verir.

LSTM cümleyi kelime kelime, SOLDAN SAĞA okur ve her adımda bir "hafıza"
(hidden state) günceller. Bu sayede "değil" kelimesi geldiğinde, ondan
önce gördüğü "iyi" kelimesinin anlamını tersine çevirebilme İMKANI doğar
(garantisi değil - bunu veriden öğrenmesi gerekiyor).

Adım 4'te eğittiğimiz Word2Vec vektörlerini burada embedding katmanının
başlangıç ağırlıkları olarak kullanıyoruz (bir tür transfer learning).
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

torch.manual_seed(42)


def tokenize(text):
    return text.lower().replace(",", "").replace(".", "").split()


# 1) Adım 4'teki gibi Word2Vec'i etiketsiz korpüsle eğit
with open("../data/unlabeled_corpus.txt", encoding="utf-8") as f:
    unlabeled_sentences = [tokenize(line) for line in f if line.strip()]

w2v = Word2Vec(
    sentences=unlabeled_sentences, vector_size=50, window=4,
    min_count=1, sg=1, epochs=30, seed=42,
)

# 2) Kelime -> indeks sözlüğü kur. 0: <pad>, 1: <unk>, 2+: word2vec kelimeleri
PAD_IDX, UNK_IDX = 0, 1
vocab = ["<pad>", "<unk>"] + list(w2v.wv.index_to_key)
word2idx = {w: i for i, w in enumerate(vocab)}
embed_dim = w2v.vector_size

embedding_matrix = np.zeros((len(vocab), embed_dim), dtype=np.float32)
embedding_matrix[UNK_IDX] = np.mean(w2v.wv.vectors, axis=0)  # bilinmeyen kelimeler için ortalama vektör
for word, idx in word2idx.items():
    if word in w2v.wv:
        embedding_matrix[idx] = w2v.wv[word]

MAX_LEN = 12


def encode(text):
    tokens = tokenize(text)[:MAX_LEN]
    ids = [word2idx.get(t, UNK_IDX) for t in tokens]
    length = len(ids)
    ids = ids + [PAD_IDX] * (MAX_LEN - length)  # sona <pad> ile doldur
    return ids, length


# 3) Etiketli veriyi yükle, train/val olarak böl, hard_test'i saklı tut
full_train_df = pd.read_csv("../data/reviews.csv")
hard_df = pd.read_csv("../data/hard_test.csv")
train_df, val_df = train_test_split(
    full_train_df, test_size=0.15, random_state=42, stratify=full_train_df["label"]
)

label_map = {"negatif": 0, "pozitif": 1}


def to_tensors(df):
    encoded = [encode(t) for t in df["text"]]
    ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
    lengths = torch.tensor([e[1] for e in encoded], dtype=torch.long)
    labels = torch.tensor(df["label"].map(label_map).values, dtype=torch.long)
    return ids, lengths, labels


X_train, len_train, y_train = to_tensors(train_df)
X_val, len_val, y_val = to_tensors(val_df)
X_hard, len_hard, y_hard = to_tensors(hard_df)


# 4) Model: Embedding -> LSTM -> (son gerçek kelimenin hidden state'i) -> Linear
class SentimentLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=32, dropout=0.3):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix), freeze=False, padding_idx=PAD_IDX
        )
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x, lengths):
        embedded = self.embedding(x)                    # (batch, seq_len, embed_dim)
        outputs, _ = self.lstm(embedded)                 # (batch, seq_len, hidden_dim)
        # Her cümle için son GERÇEK kelimenin (pad değil) hidden state'ini al.
        # LSTM soldan sağa çalıştığı için pad'ler sondaysa önceki adımları etkilemez.
        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(x.size(0))
        last_hidden = outputs[batch_idx, last_idx]        # (batch, hidden_dim)
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)


model = SentimentLSTM(embedding_matrix)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)

# 5) Eğitim + early stopping (Adım 3'teki yöntemle aynı)
print("--- Eğitim ---")
epochs, patience = 200, 20
best_val_loss, best_state, no_improve = float("inf"), None, 0

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train, len_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val, len_val)
        val_loss = loss_fn(val_outputs, y_val).item()
        val_acc = (val_outputs.argmax(dim=1) == y_val).float().mean().item()

    if val_loss < best_val_loss:
        best_val_loss, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
    else:
        no_improve += 1

    if epoch % 20 == 0 or epoch == 1:
        train_acc = (outputs.argmax(dim=1) == y_train).float().mean().item()
        print(f"epoch {epoch:>4} | train acc: {train_acc:.2f} | val loss: {val_loss:.4f} acc: {val_acc:.2f}")

    if no_improve >= patience:
        print(f"\nEarly stopping (epoch {epoch}).")
        break

model.load_state_dict(best_state)

# 6) Zor test setinde değerlendir
model.eval()
with torch.no_grad():
    hard_outputs = model(X_hard, len_hard)
    hard_pred = hard_outputs.argmax(dim=1)
    hard_acc = (hard_pred == y_hard).float().mean().item()

print(f"\nZor test doğruluğu: {hard_acc:.2f}")
print("Önceki adımlar -> TF-IDF+bigram: 0.50 | NN (avg embed): 0.69\n")

inv_label_map = {v: k for k, v in label_map.items()}
print("--- Zor test tahminleri ---")
for text, true_idx, pred_idx in zip(hard_df["text"], y_hard.tolist(), hard_pred.tolist()):
    isaret = "OK " if true_idx == pred_idx else "X  "
    print(f"{isaret} gerçek={inv_label_map[true_idx]:8s} tahmin={inv_label_map[pred_idx]:8s} | {text}")
