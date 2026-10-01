"""
ADIM 18 (1/?) - Kalibrasyon 100 (id 7000-7099) + ana test 500 (id 8000-8499), hiç görülmemiş yorumlardan.

Tarif Adım 15/16/17 ile aynı: 8-40 kelime, havuz etiketine göre yarı "pozitif" yarı "negatif", sabit tohumlar.
Dışarıda: Adım 16'nın 800 yorumu ve Adım 17'nin test17_set.csv'si (step16 common'ın zaten dışladıklarına ek olarak).
Sızıntı assert'leri: step16 common.check_no_leak (ham + normalleştirilmiş); kalibrasyon ile ana test de çakışmaz.

Çıktı: calib_set.csv (id, text, pool_label) + calib_gorkan_01..04.csv (25'er; no, id, text, boş `etiket`; karışık sıra)
       test18_set.csv (id, text, pool_label) + test18_batch_01..10.csv (50'şer; id + text).
KASA: ana testin metinleri bu script tarafından yazdırılmaz; sadece sayılar.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step16_topic_bert"))

from common import LABEL_SET, candidate_pool, check_no_leak, norm  # noqa: E402

step16 = pd.read_csv(LABEL_SET)
test17 = pd.read_csv(os.path.join(ROOT, "step17_aspect_sentiment", "test17_set.csv"))
seen = set(step16["text"]) | set(test17["text"])
cand, exclude, exclude_norm = candidate_pool(extra_exclude=seen)
cand = cand[~cand["text"].map(norm).isin({norm(t) for t in seen})]
print(f"Aday havuzu: {cand['label'].value_counts().to_dict()}")


def pick(pool, n_per, seeds, start):
    df = pd.concat([pool[pool["label"] == lab].sample(n_per, random_state=s) for lab, s in zip(("pozitif", "negatif"), seeds)])
    df = df.sample(frac=1, random_state=seeds[2]).reset_index(drop=True)
    df["id"] = range(start, start + len(df))
    return df.rename(columns={"label": "pool_label"})[["id", "text", "pool_label"]]


calib = pick(cand, 50, (181, 182, 183), 7000)
rest = cand[~cand["text"].map(norm).isin(set(calib["text"].map(norm)))]
test = pick(rest, 250, (184, 185, 186), 8000)

assert len(calib) == 100 and len(test) == 500
assert (calib["pool_label"].value_counts() == 50).all() and (test["pool_label"].value_counts() == 250).all()
both = pd.concat([calib, test], ignore_index=True)
check_no_leak(both.assign(split="test"), exclude, exclude_norm)
for name, df in (("Adım 16 etiket seti", step16), ("Adım 17 test", test17)):
    assert not set(both["text"].map(norm)) & set(df["text"].map(norm)), f"{name} ile çakışma"
assert not set(calib["text"].map(norm)) & set(test["text"].map(norm)), "kalibrasyon ile ana test çakışıyor"
assert both["id"].is_unique

calib.to_csv(os.path.join(HERE, "calib_set.csv"), index=False)
g = calib.sample(frac=1, random_state=187)[["id", "text"]].assign(etiket="").reset_index(drop=True)
g.insert(0, "no", range(1, 101))
for b in range(4):
    g.iloc[b * 25:(b + 1) * 25].to_csv(os.path.join(HERE, f"calib_gorkan_{b + 1:02d}.csv"), index=False)
test.to_csv(os.path.join(HERE, "test18_set.csv"), index=False)
for b in range(10):
    test.iloc[b * 50:(b + 1) * 50][["id", "text"]].to_csv(os.path.join(HERE, f"test18_batch_{b + 1:02d}.csv"), index=False)

out = [f"ADIM 18 — set seçimi. Aday havuzu: {cand['label'].value_counts().to_dict()}",
       f"kalibrasyon: {len(calib)} (id {calib.id.min()}-{calib.id.max()}) {calib.pool_label.value_counts().to_dict()} -> calib_set.csv, calib_gorkan_01..04.csv",
       f"ana test   : {len(test)} (id {test.id.min()}-{test.id.max()}) {test.pool_label.value_counts().to_dict()} -> test18_set.csv, test18_batch_01..10.csv",
       f"kalan negatif havuz: {int((rest['label'] == 'negatif').sum()) - 250}",
       "sızıntı kontrolleri geçti (check_no_leak + Adım 16 + Adım 17 test + kalibrasyon/test çakışması)"]
open(os.path.join(HERE, "log_prepare_sets.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
