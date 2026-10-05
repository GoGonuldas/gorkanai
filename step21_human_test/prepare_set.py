"""
ADIM 21 (1) - İnsan testi seti: 100 hiç görülmemiş yorum (id 9500-9599), 70 pozitif + 30 negatif havuz etiketli.

Tarif Adım 15-19 ile aynı (8-40 kelime). Dışarıda: Adım 16 etiket seti, Adım 17 testi, Adım 18 kalibrasyon + kasa testi,
Adım 19 seçimi (select19.csv). Sızıntı: check_no_leak + her setle normalleştirilmiş metin çakışması.
Çıktı: human_set.csv (id, text, pool_label) + human_gorkan_01..04.csv (25'er; no, id, text, boş `etiket`; karışık sıra)
       + human_batch.csv (id + text; taze oturum için) + log_prepare_set.txt
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step16_topic_bert"))

from common import LABEL_SET, candidate_pool, check_no_leak, norm  # noqa: E402

S18 = os.path.join(ROOT, "step18_big_test")
seen_sets = {
    "Adım 16 etiket seti": pd.read_csv(LABEL_SET),
    "Adım 17 test": pd.read_csv(os.path.join(ROOT, "step17_aspect_sentiment", "test17_set.csv")),
    "Adım 18 kalibrasyon": pd.read_csv(os.path.join(S18, "calib_set.csv")),
    "Adım 18 kasa testi": pd.read_csv(os.path.join(S18, "test18_set.csv")),
    "Adım 19 seçimi": pd.read_csv(os.path.join(ROOT, "step19_topic_v2", "select19.csv")),
}
seen = set().union(*(set(df["text"]) for df in seen_sets.values()))
cand, exclude, exclude_norm = candidate_pool(extra_exclude=seen)
cand = cand[~cand["text"].map(norm).isin({norm(t) for t in seen})]

chosen = pd.concat([cand[cand["label"] == "pozitif"].sample(70, random_state=211),
                    cand[cand["label"] == "negatif"].sample(30, random_state=212)])
chosen = chosen.sample(frac=1, random_state=213).reset_index(drop=True)
chosen["id"] = range(9500, 9600)
chosen = chosen.rename(columns={"label": "pool_label"})[["id", "text", "pool_label"]]

check_no_leak(chosen.assign(split="test"), exclude, exclude_norm)
for name, df in seen_sets.items():
    assert not set(chosen["text"].map(norm)) & set(df["text"].map(norm)), f"{name} ile çakışma"

chosen.to_csv(os.path.join(HERE, "human_set.csv"), index=False)
chosen[["id", "text"]].to_csv(os.path.join(HERE, "human_batch.csv"), index=False)
g = chosen.sample(frac=1, random_state=214)[["id", "text"]].assign(etiket="").reset_index(drop=True)
g.insert(0, "no", range(1, 101))
for b in range(4):
    g.iloc[b * 25:(b + 1) * 25].to_csv(os.path.join(HERE, f"human_gorkan_{b + 1:02d}.csv"), index=False)

rest_neg = int((cand["label"] == "negatif").sum()) - 30
out = [f"ADIM 21 — insan testi seçimi. Aday havuzu: {cand['label'].value_counts().to_dict()}",
       f"seçilen 100 (id 9500-9599) {chosen.pool_label.value_counts().to_dict()}; kalan negatif havuz {rest_neg}",
       "sızıntı kontrolleri geçti (check_no_leak + Adım 16/17/18/19 setleri)"]
open(os.path.join(HERE, "log_prepare_set.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
