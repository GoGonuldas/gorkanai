"""
ADIM 14 - Etiketleme adaylarını seçmek (active learning).

Deney 1 gösterdi ki nötr sınıfı için ÜRÜN YORUMU olan nötr örnekler lazım.
Havuzda bol var (kısa yorumların ~%17'si) ama etiketsiz — elle etiketlemek gerekiyor.

Etiketleme emeği pahalı, o yüzden adayları akıllıca seçiyoruz:
  - %50 "belirsiz": Adım 12 modelinin (C) kararsız kaldığı kısa yorumlar
    (nötrlerin burada yoğunlaşmasını bekliyoruz — active learning / uncertainty sampling)
  - %50 rastgele kısa yorum: modelin YANLIŞLIKLA emin olduğu nötrleri kaçırmamak için
    ("kargo 2 günde geldi." -> %99 pozitif!)

Dışarıda tutulanlar: gerçek test, kısa temiz test, negasyon holdout, val seti.

Çıktı: step14_three_class/candidates/batch_XX.csv (250'lik gruplar, id + text)
"""

import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "../step12_confident_learning/model"
N_CANDIDATES = 1000
BATCH = 250
MAX_WORDS = 10
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidates")
NEG_PATTERN = re.compile(r"\b(?:fena|kötü|berbat|kalitesiz)\w*\s+(?:değil|degil|sayılmaz|sayilmaz)")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Dışarıda tutulacaklar (Adım 10-13 ile aynı kurulum) ---
pool = pd.read_csv("../data/real/train_pool.csv")
holdout = pool[pool["text"].str.contains(NEG_PATTERN)].sample(200, random_state=42)
rest = pool.drop(holdout.index)
pos = rest[rest["label"] == "pozitif"].sample(frac=1, random_state=42)
neg = rest[rest["label"] == "negatif"].sample(frac=1, random_state=42)
val_idx = set(pos.index[:500]) | set(neg.index[:500])
exclude_text = set(pd.read_csv("../data/real/test.csv")["text"]) | set(pd.read_csv("../data/short_clean_test.csv")["text"])

cand = pool.drop(list(holdout.index) + list(val_idx))
cand = cand[~cand["text"].isin(exclude_text)]
cand = cand[cand["text"].str.split().str.len() <= MAX_WORDS].drop_duplicates("text")
print(f"Kısa (<= {MAX_WORDS} kelime) aday havuzu: {len(cand)}")

# --- Model C'nin güveni (sadece 20k'lık rastgele bir alt kümede, hız için) ---
sub = cand.sample(20000, random_state=1)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
low = lambda t: t.replace("I", "ı").replace("İ", "i").lower()
conf = []
with torch.no_grad():
    texts = sub["text"].tolist()
    for i in range(0, len(texts), 128):
        enc = tokenizer([low(t) for t in texts[i:i + 128]], padding=True, truncation=True,
                        max_length=64, return_tensors="pt").to(device)
        conf.append(torch.softmax(model(**enc).logits, 1).max(1).values.cpu().numpy())
sub["conf"] = np.concatenate(conf)

uncertain = sub[sub["conf"] < 0.90].sample(N_CANDIDATES // 2, random_state=2)
random_part = sub.drop(uncertain.index).sample(N_CANDIDATES // 2, random_state=3)
chosen = pd.concat([uncertain.assign(source="belirsiz"), random_part.assign(source="rastgele")])
chosen = chosen.sample(frac=1, random_state=4).reset_index().rename(columns={"index": "pool_idx"})
chosen["id"] = range(len(chosen))
print(f"Model C'nin %90'dan az emin olduğu kısa yorum oranı: %{100 * (sub['conf'] < 0.90).mean():.1f}")

chosen.to_csv(os.path.join(OUT_DIR, "all_candidates.csv"), index=False)
for b in range(0, len(chosen), BATCH):
    chosen.iloc[b:b + BATCH][["id", "text"]].to_csv(os.path.join(OUT_DIR, f"batch_{b // BATCH + 1:02d}.csv"), index=False)
print(f"{len(chosen)} aday -> {OUT_DIR}/batch_XX.csv")
