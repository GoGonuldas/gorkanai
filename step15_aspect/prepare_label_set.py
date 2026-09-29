"""
ADIM 15.2 (1/2) - Etiketlenecek 300 yorumu seçmek ve val/test bölmesini ETİKETLEMEDEN ÖNCE sabitlemek.

15.1'deki keşif setinden (explore_300.csv) ve daha önce görülmüş her şeyden ayrık (`unseen_pool`).
Aynı kurgu: 8-40 kelime, 150 "pozitif" + 150 "negatif" etiketli havuz yorumu.
Bölme etiketlerden önce yapılıyor ki "zor örnekleri teste koyma" gibi bir seçim etkisi olmasın;
havuz etiketine göre tabakalı: val 50p+50n, test 100p+100n.

Çıktı: step15_aspect/label_set.csv (id 4000-4299, split) + etiketleme için batch_1..6.csv (50'şer).
"""

import os

import pandas as pd

from prepare_explore import HERE, MAX_WORDS, MIN_WORDS, unseen_pool

explore = pd.read_csv(os.path.join(HERE, "explore_300.csv"))
unseen, exclude = unseen_pool(extra_exclude=explore["text"])
n_words = unseen["text"].str.split().str.len()
cand = unseen[(n_words >= MIN_WORDS) & (n_words <= MAX_WORDS)]

pos = cand[cand["label"] == "pozitif"].sample(150, random_state=25)
neg = cand[cand["label"] == "negatif"].sample(150, random_state=26)
pos = pos.assign(split=["val"] * 50 + ["test"] * 100)
neg = neg.assign(split=["val"] * 50 + ["test"] * 100)
chosen = pd.concat([pos, neg]).sample(frac=1, random_state=27).reset_index(drop=True)
chosen["id"] = range(4000, 4000 + len(chosen))
chosen = chosen.rename(columns={"label": "pool_label"})[["id", "text", "pool_label", "split"]]

assert not set(chosen["text"]) & exclude, "daha önce görülmüş metin var"
assert not set(chosen["text"]) & set(explore["text"]), "15.1 keşif setiyle çakışma var"
assert chosen["text"].is_unique

chosen.to_csv(os.path.join(HERE, "label_set.csv"), index=False)
for b in range(0, len(chosen), 50):
    chosen.iloc[b:b + 50][["id", "text"]].to_csv(os.path.join(HERE, f"batch_{b // 50 + 1}.csv"), index=False)
print(chosen.groupby(["split", "pool_label"]).size().to_string())
print(f"{len(chosen)} yorum -> label_set.csv + batch_1..{len(chosen) // 50}.csv")
