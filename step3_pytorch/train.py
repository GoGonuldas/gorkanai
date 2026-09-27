"""
ADIM 3: PyTorch ile ilk sinir ağın.

Adım 2'deki TF-IDF özelliklerini aynen kullanıyoruz, ama sınıflandırıcıyı
scikit-learn'den PyTorch'a taşıyoruz. Amaç scikit-learn'ün perde arkasında
otomatik yaptığı şeyi elle görmek:

  tensor -> forward pass -> loss -> backward pass (autograd) -> optimizer.step()

Ağ mimarisi: girdi (TF-IDF vektörü) -> gizli katman (ReLU) -> çıkış (2 sınıf)
"""

import copy

import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

# 1) Veriyi yükle. Eğitim verisini kendi içinde train/val olarak bölüyoruz:
#    val seti "early stopping" için kullanılacak, hard_test HİÇ eğitim/tuning
#    sürecine karışmayacak -> gerçek genelleme testi olarak saklı kalacak.
full_train_df = pd.read_csv("../data/reviews.csv")
hard_df = pd.read_csv("../data/hard_test.csv")
train_df, val_df = train_test_split(
    full_train_df, test_size=0.15, random_state=42, stratify=full_train_df["label"]
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_np = vectorizer.fit_transform(train_df["text"]).toarray()
X_val_np = vectorizer.transform(val_df["text"]).toarray()
X_hard_np = vectorizer.transform(hard_df["text"]).toarray()

label_map = {"negatif": 0, "pozitif": 1}
y_train_np = train_df["label"].map(label_map).values
y_val_np = val_df["label"].map(label_map).values
y_hard_np = hard_df["label"].map(label_map).values

# 2) NumPy dizilerini PyTorch tensor'larına çevir
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)
X_val = torch.tensor(X_val_np, dtype=torch.float32)
y_val = torch.tensor(y_val_np, dtype=torch.long)
X_hard = torch.tensor(X_hard_np, dtype=torch.float32)
y_hard = torch.tensor(y_hard_np, dtype=torch.long)

input_dim = X_train.shape[1]
print(f"Girdi boyutu (TF-IDF özellik sayısı): {input_dim}")


# 3) Modeli tanımla: girdi -> gizli katman (ReLU + Dropout) -> 2 sınıf çıkışı
#    Dropout: eğitim sırasında nöronların bir kısmını rastgele "kapatarak"
#    modelin belirli nöronlara aşırı bağımlı olup ezberlemesini engelliyor.
class SentimentNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = SentimentNet(input_dim)
loss_fn = nn.CrossEntropyLoss()
# weight_decay: L2 regularization -> büyük ağırlıkları cezalandırır, ezberlemeyi zorlaştırır
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-2)

# 4) Eğitim döngüsü + early stopping: val loss iyileşmeyi durdurunca eğitimi kes
#    ve en iyi val performansına sahip ağırlıkları geri yükle.
print("\n--- Eğitim ---")
epochs = 300
patience = 20
best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = loss_fn(val_outputs, y_val).item()
        val_acc = (val_outputs.argmax(dim=1) == y_val).float().mean().item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch % 20 == 0 or epoch == 1:
        train_acc = (outputs.argmax(dim=1) == y_train).float().mean().item()
        print(f"epoch {epoch:>4} | train loss: {loss.item():.4f} acc: {train_acc:.2f} "
              f"| val loss: {val_loss:.4f} acc: {val_acc:.2f}")

    if epochs_without_improvement >= patience:
        print(f"\nEarly stopping: {patience} epoch boyunca val loss iyileşmedi (epoch {epoch}).")
        break

model.load_state_dict(best_state)

# 5) Zor test setinde değerlendir (gradyan hesaplamaya gerek yok)
model.eval()
with torch.no_grad():
    hard_outputs = model(X_hard)
    hard_pred = hard_outputs.argmax(dim=1)
    hard_acc = (hard_pred == y_hard).float().mean().item()

print(f"\nZor test doğruluğu: {hard_acc:.2f}  (Adım 2'deki en iyi sonuç: 0.50, regularization'sız NN: 0.44)")

inv_label_map = {v: k for k, v in label_map.items()}
print("\n--- Zor test tahminleri ---")
for text, true_idx, pred_idx in zip(hard_df["text"], y_hard.tolist(), hard_pred.tolist()):
    isaret = "OK " if true_idx == pred_idx else "X  "
    print(f"{isaret} gerçek={inv_label_map[true_idx]:8s} tahmin={inv_label_map[pred_idx]:8s} | {text}")
