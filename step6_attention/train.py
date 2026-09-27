"""
ADIM 6: Attention mekanizması.

Adım 5'in kör noktası: LSTM tüm cümleyi TEK bir "son hidden state"e
sıkıştırıyordu. İki cümlecikli örnekler ("kötü değildi, güzeldi") bu
sıkışmada zarar görüyordu -> ilk cümlecikteki bilgi son adıma kadar
zayıflayarak taşınıyor.

Attention'ın fikri basit: son hidden state'e güvenmek yerine, LSTM'in
HER adımdaki hidden state'ine bakıp "bu cümle için hangi kelimeler
önemli?" sorusunu öğrenilebilir bir ağırlıklandırmayla cevaplıyoruz.
Sonuç: tüm kelimelerin AĞIRLIKLI ORTALAMASI (attention weights ile).

Not: Bu, gerçek bir Transformer'ın kullandığı self-attention'ın
basitleştirilmiş bir öncülü (additive/Bahdanau attention). Transformer'lar
LSTM'i tamamen çıkarıp sadece attention kullanır - buna Adım 7'de
(hazır bir Transformer'ı fine-tune ederek) dolaylı olarak değineceğiz.
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


with open("../data/unlabeled_corpus.txt", encoding="utf-8") as f:
    unlabeled_sentences = [tokenize(line) for line in f if line.strip()]

w2v = Word2Vec(
    sentences=unlabeled_sentences, vector_size=50, window=4,
    min_count=1, sg=1, epochs=30, seed=42,
)

PAD_IDX, UNK_IDX = 0, 1
vocab = ["<pad>", "<unk>"] + list(w2v.wv.index_to_key)
word2idx = {w: i for i, w in enumerate(vocab)}
embed_dim = w2v.vector_size

embedding_matrix = np.zeros((len(vocab), embed_dim), dtype=np.float32)
embedding_matrix[UNK_IDX] = np.mean(w2v.wv.vectors, axis=0)
for word, idx in word2idx.items():
    if word in w2v.wv:
        embedding_matrix[idx] = w2v.wv[word]

MAX_LEN = 12


def encode(text):
    tokens = tokenize(text)[:MAX_LEN]
    ids = [word2idx.get(t, UNK_IDX) for t in tokens]
    length = len(ids)
    ids = ids + [PAD_IDX] * (MAX_LEN - length)
    return ids, length


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


def make_mask(lengths, max_len=MAX_LEN):
    positions = torch.arange(max_len).unsqueeze(0)          # (1, max_len)
    return positions < lengths.unsqueeze(1)                  # (batch, max_len) bool


class AdditiveAttention(nn.Module):
    """score_t = v^T tanh(W h_t)  ->  softmax  ->  ağırlıklı ortalama"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, outputs, mask):
        scores = self.v(torch.tanh(self.attn(outputs))).squeeze(-1)   # (batch, seq_len)
        scores = scores.masked_fill(~mask, float("-inf"))              # pad'leri sistem dışına at
        weights = torch.softmax(scores, dim=1)                         # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)  # (batch, hidden_dim)
        return context, weights


class SentimentAttention(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix), freeze=False, padding_idx=PAD_IDX
        )
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, batch_first=True)
        self.attention = AdditiveAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x, lengths):
        mask = make_mask(lengths)
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        context, weights = self.attention(outputs, mask)
        context = self.dropout(context)
        return self.fc(context), weights


model = SentimentAttention(embedding_matrix)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)

print("--- Eğitim ---")
epochs, patience = 200, 20
best_val_loss, best_state, no_improve = float("inf"), None, 0

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    outputs, _ = model(X_train, len_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs, _ = model(X_val, len_val)
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

model.eval()
with torch.no_grad():
    hard_outputs, hard_weights = model(X_hard, len_hard)
    hard_pred = hard_outputs.argmax(dim=1)
    hard_acc = (hard_pred == y_hard).float().mean().item()

print(f"\nZor test doğruluğu: {hard_acc:.2f}")
print("Önceki adımlar -> TF-IDF+bigram: 0.50 | Embedding ortalaması: 0.69 | LSTM (son hidden state): 0.69\n")

inv_label_map = {v: k for k, v in label_map.items()}
print("--- Zor test tahminleri ---")
for text, true_idx, pred_idx in zip(hard_df["text"], y_hard.tolist(), hard_pred.tolist()):
    isaret = "OK " if true_idx == pred_idx else "X  "
    print(f"{isaret} gerçek={inv_label_map[true_idx]:8s} tahmin={inv_label_map[pred_idx]:8s} | {text}")

# İki cümlecikli, Adım 5'te bozulan örneklerde attention hangi kelimeye bakıyor?
print("\n--- Attention ağırlıkları (Adım 5'te yanlış olan örnekler) ---")
ilgi_cekenler = ["Bu restoran kötü değildi, güzeldi.", "Bu kitap gerçekten sıkıcı değildi, çok akıcıydı."]
for cumle in ilgi_cekenler:
    ids, length = encode(cumle)
    x = torch.tensor([ids])
    l = torch.tensor([length])
    with torch.no_grad():
        _, w = model(x, l)
    tokens = tokenize(cumle)[:length]
    print(f"\n{cumle}")
    for tok, weight in zip(tokens, w[0][:length].tolist()):
        print(f"  {tok:<12} {'#' * int(weight * 50)} {weight:.2f}")
