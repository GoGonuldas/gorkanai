"""
ADIM 14 - Deney 1: Vikipedi cümleleriyle "nötr" sınıfı öğretilebilir mi?

Hazır 3 sınıflı veri seti (winvoker/turkish-sentiment-analysis-dataset) bulduk, ama
nötr örneklerinin %99.7'si Vikipedi cümlesi ("Okul bir dönem başka amaçlar için
kullanılmıştır ."). Pozitif/negatifler ise yorum.

Tehlike — KISAYOL ÖĞRENME (shortcut learning): model "duygu yok = nötr" yerine
"ansiklopedi üslubu = nötr" öğrenebilir. Eğitim setinde ikisi hep birlikte
olduğu için model hangisini öğrendiğini "söylemez"; ancak doğru testte görürüz.

Önlem: Vikipedi cümlelerindeki bariz BİÇİM ipuçlarını siliyoruz (büyük harf -> zaten
turkish_lower ile gidiyor; " ." -> "."). Ama İÇERİK/ÜSLUP farkı (tarih, coğrafya,
edilgen çatı) kalıyor.

Eğitim: 3000 pozitif + 3000 negatif (bizim gerçek havuzdan) + 3000 Vikipedi nötr, 2 epoch.
Asıl test: data/short_clean_test.csv içindeki 35 ELLE etiketlenmiş gerçek nötr yorum
("kargo 2 günde geldi.", "bunda da aynı bedeni alın.").
"""

import re

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
N_PER_CLASS = 3000
EPOCHS = 2
LABELS = ["negatif", "nötr", "pozitif"]
label_map = {name: i for i, name in enumerate(LABELS)}
NEG_PATTERN = re.compile(r"\b(?:fena|kötü|berbat|kalitesiz)\w*\s+(?:değil|degil|sayılmaz|sayilmaz)")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def turkish_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


def normalize_wiki(text):
    # "kullanılmıştır ." -> "kullanılmıştır."  (Vikipedi'deki tokenize edilmiş boşlukları kaldır)
    return re.sub(r"\s+([.,;:!?])", r"\1", text.strip())


# --- Veri ---
short = pd.read_csv("../data/short_clean_test.csv")
exclude = set(pd.read_csv("../data/real/test.csv")["text"]) | set(short["text"])

pool = pd.read_csv("../data/real/train_pool.csv")
pool = pool[~pool["text"].str.contains(NEG_PATTERN) & ~pool["text"].isin(exclude)]
pos = pool[pool["label"] == "pozitif"].sample(N_PER_CLASS, random_state=0)
neg = pool[pool["label"] == "negatif"].sample(N_PER_CLASS, random_state=0)

wiki = load_dataset("winvoker/turkish-sentiment-analysis-dataset")
wiki = pd.concat([wiki[s].to_pandas() for s in wiki], ignore_index=True)
wiki = wiki[wiki["dataset"] == "wiki"].sample(N_PER_CLASS + 500, random_state=0)
wiki["text"] = wiki["text"].map(normalize_wiki)
wiki_train, wiki_test = wiki.iloc[:N_PER_CLASS], wiki.iloc[N_PER_CLASS:]

train_df = pd.concat([pos, neg, wiki_train.assign(label="nötr")[["text", "label"]]])
print(f"Eğitim: {train_df['label'].value_counts().to_dict()}")


def make_batches(texts, labels=None, batch_size=16, shuffle=False):
    idx = np.arange(len(texts))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        b = idx[start:start + batch_size]
        enc = tokenizer([turkish_lower(texts[i]) for i in b], padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        y = None if labels is None else torch.tensor(labels[b], dtype=torch.long, device=device)
        yield enc, y


@torch.no_grad()
def predict(texts):
    model.eval()
    out = [torch.softmax(model(**enc).logits, 1).cpu().numpy() for enc, _ in make_batches(texts, batch_size=64)]
    return np.concatenate(out)


torch.manual_seed(42)
np.random.seed(42)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
texts, labels = train_df["text"].tolist(), train_df["label"].map(label_map).values
for epoch in range(1, EPOCHS + 1):
    model.train()
    total, n = 0.0, 0
    for enc, y in make_batches(texts, labels, shuffle=True):
        optimizer.zero_grad()
        loss = model(**enc, labels=y).loss
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    print(f"epoch {epoch} | loss {total / n:.4f}")


def report(name, df, label_col):
    p = predict(df["text"].tolist())
    pred = np.array(LABELS)[p.argmax(1)]
    acc = (pred == df[label_col].values).mean()
    dist = pd.Series(pred).value_counts().reindex(LABELS, fill_value=0).to_dict()
    print(f"  {name:42s} doğruluk {acc:.3f} | tahmin dağılımı {dist}")
    return pred, p


print("\n=== Sonuçlar ===")
report("Vikipedi nötr (görülmemiş, 500)", wiki_test.assign(label="nötr"), "label")
neutral_pred, neutral_p = report("GERÇEK nötr yorumlar (elle, 35)", short[short["clean_label"] == "nötr"], "clean_label")
report("gerçek pozitif/negatif kısa yorumlar (elle, 165)", short[short["clean_label"] != "nötr"], "clean_label")

print("\n--- Gerçek nötr yorumlarda tahminler ---")
for text, pred, p in zip(short[short["clean_label"] == "nötr"]["text"], neutral_pred, neutral_p):
    print(f"  {'OK' if pred == 'nötr' else 'X '} {pred:8s} %{100 * p.max():3.0f} | {text}")

print("\n--- Kısayol testi: aynı anlam, farklı üslup ---")
probe = ["Ürün 2019 yılında piyasaya sürülmüştür.",           # ansiklopedik, nötr
         "ürün dün elime ulaştı.",                             # yorum dili, nötr
         "Cihaz yüksek performansı ile kullanıcılar tarafından beğenilmiştir.",  # ansiklopedik, pozitif
         "cihaz çok hızlı, bayıldım.",                         # yorum dili, pozitif
         "Saygılarımla.", "kargo 2 günde geldi."]
for text, p in zip(probe, predict(probe)):
    print(f"  {LABELS[p.argmax()]:8s} %{100 * p.max():3.0f} | {text}")
