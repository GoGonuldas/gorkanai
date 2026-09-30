"""
ADIM 17 (1/?) - Yeni, hiç görülmemiş test seti: 200 yorum (id 6000-6199).

Neden: Adım 15 testinin 200 yorumu Adım 16.5'te hata hata okundu -> orada ölçmek iyimser olur (PLAN.md, madde 3).
Tarif Adım 15/16 ile aynı: 8-40 kelime, 100 "pozitif" + 100 "negatif" havuz etiketli, hiçbir adımda görülmemiş.
Adım 16'nın 800 yorumu da dışarıda. Sızıntı assert'leri: step16 common.check_no_leak (ham + normalleştirilmiş).

Çıktı: test17_set.csv (id, text, pool_label) + test17_batch_01..04.csv (sadece id + text, karışık sıra)
       + human_blind_20.csv (Görkan için: havuz etiketine göre tabakalı 10 + 10, sabit tohum; boş `etiket` sütunu).
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "step16_topic_bert"))

from common import LABEL_SET, candidate_pool, check_no_leak, norm  # noqa: E402

step16 = pd.read_csv(LABEL_SET)
cand, exclude, exclude_norm = candidate_pool(extra_exclude=set(step16["text"]))
cand = cand[~cand["text"].map(norm).isin(set(step16["text"].map(norm)))]
print(f"Aday havuzu: {cand['label'].value_counts().to_dict()}")

chosen = pd.concat([cand[cand["label"] == "pozitif"].sample(100, random_state=171),
                    cand[cand["label"] == "negatif"].sample(100, random_state=172)])
chosen = chosen.sample(frac=1, random_state=173).reset_index(drop=True)
chosen["id"] = range(6000, 6000 + len(chosen))
chosen = chosen.rename(columns={"label": "pool_label"})[["id", "text", "pool_label"]]

assert len(chosen) == 200
check_no_leak(chosen.assign(split="test"), exclude, exclude_norm)
assert not set(chosen["text"].map(norm)) & set(step16["text"].map(norm)), "Adım 16 eğitim/val ile çakışma"

chosen.to_csv(os.path.join(HERE, "test17_set.csv"), index=False)
for b in range(0, 200, 50):
    chosen.iloc[b:b + 50][["id", "text"]].to_csv(os.path.join(HERE, f"test17_batch_{b // 50 + 1:02d}.csv"), index=False)

blind = pd.concat([chosen[chosen["pool_label"] == lab].sample(10, random_state=174) for lab in ("pozitif", "negatif")])
blind = blind.sample(frac=1, random_state=175)[["id", "text"]].assign(etiket="")
blind.insert(0, "no", range(1, 21))
blind.to_csv(os.path.join(HERE, "human_blind_20.csv"), index=False)
print(chosen["pool_label"].value_counts().to_dict(), "-> test17_set.csv, test17_batch_01..04.csv, human_blind_20.csv | sızıntı kontrolleri geçti")
