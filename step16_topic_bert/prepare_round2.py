"""
ADIM 16 - Etiketlenecek yorumları seçmek, tur 2: eğitim 300 = 150 rastgele + 150 "kararsız" (active learning).

Tur 1'in 300 eğitim yorumuyla kaba bir konu modeli eğitilir (tohum 0, 5 epoch — val'e bakılmadan sabit).
Aday havuzunun rastgele 20000'lik alt kümesinde her yorum için u = min_k |p_k - 0.5| hesaplanır (7 konu);
u en küçük 150 yorum (yarı "pozitif" yarı "negatif" havuz etiketli) = modelin en kararsız olduğu yorumlar.
Rastgele 150, aynı alt kümenin kalanından. Kaba model val/test metinlerini hiç görmez (aday havuzu onlardan ayrık).

Çıktı: label_set16.csv'ye id 5500-5799 eklenir (round=2) + batch_11..16.csv (sadece id + text, karışık).
"""

import os

import numpy as np
import pandas as pd

from common import HERE, LABEL_SET, candidate_pool, check_no_leak, norm
from train import ASPECTS, load_data, predict, train_model

N_SUB, ROUGH_EPOCHS = 20000, 5

prev = pd.read_csv(LABEL_SET)
assert set(prev["round"]) == {1}, "tur 2 zaten eklenmiş"
cand, exclude, exclude_norm = candidate_pool(extra_exclude=set(prev["text"]))
cand = cand[~cand["text"].map(norm).isin(set(prev["text"].map(norm)))]
# negatif havuz etiketliler kıt: hepsi alt kümeye girer, kalanı rastgele pozitif
neg = cand[cand["label"] == "negatif"]
sub = pd.concat([neg, cand[cand["label"] == "pozitif"].sample(N_SUB - len(neg), random_state=60)])
print(f"Aday alt kümesi: {sub['label'].value_counts().to_dict()}")

train, val = load_data(max_round=1)
assert len(train) == 300
assert not set(sub["text"].map(norm)) & (set(train["text"].map(norm)) | set(val["text"].map(norm)))
model, tokenizer = train_model(train, seed=0, n_epochs=ROUGH_EPOCHS)
p = predict(model, tokenizer, sub["text"].tolist())
sub = sub.assign(u=np.abs(p - 0.5).min(1), en_kararsiz_konu=[ASPECTS[k] for k in np.abs(p - 0.5).argmin(1)])
np.save(os.path.join(HERE, "round2_pool_probs.npy"), p)

parts = []
for label, seed in (("pozitif", 61), ("negatif", 62)):
    s = sub[sub["label"] == label]
    unc = s.nsmallest(75, "u").assign(source="kararsiz")
    rnd = s.drop(unc.index).sample(75, random_state=seed).assign(source="rastgele")
    parts += [unc, rnd]
    print(f"{label}: kararsız 75'in u aralığı {unc['u'].min():.3f}-{unc['u'].max():.3f} "
          f"(alt kümede medyan u {s['u'].median():.3f})")
chosen = pd.concat(parts).sample(frac=1, random_state=63).reset_index(drop=True)
print("Kararsız 150'de en kararsız konu:", chosen[chosen["source"] == "kararsiz"]["en_kararsiz_konu"].value_counts().to_dict())
chosen["id"] = range(5500, 5500 + len(chosen))
chosen["split"], chosen["round"] = "train", 2
chosen = chosen.rename(columns={"label": "pool_label"})[list(prev.columns)]

both = pd.concat([prev, chosen], ignore_index=True)
assert len(chosen) == 300 and len(both) == 800
check_no_leak(both, exclude - set(prev["text"]), exclude_norm - set(prev["text"].map(norm)))
both.to_csv(LABEL_SET, index=False)
for b in range(0, len(chosen), 50):
    chosen.iloc[b:b + 50][["id", "text"]].to_csv(os.path.join(HERE, f"batch_{11 + b // 50:02d}.csv"), index=False)
print(chosen.groupby(["source", "pool_label"]).size().to_string())
print("300 yorum -> label_set16.csv (id 5500-5799) + batch_11..16.csv | sızıntı kontrolleri geçti")
