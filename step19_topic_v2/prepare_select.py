"""
ADIM 19 (1/?) - Konu başına kotalı hedefli etiketleme, parça 1: 270 yorum (id 9000-9269).

PLAN §1b: S 120, G 60, B 60 (Adım 15'in DONDURULMUŞ V2 anahtar kelime listesiyle hedefli) + 30 rastgele.
Örtük S 30 (id 9270-9299) ayrı: prepare_implicit_s.py, Mac mini'de (Adım 16 modelleri orada).
Negatif havuz neredeyse bitti (302) -> her grupta %20 negatif (satıcı kelimeli negatif sadece 16 -> 16).

Dışarıda: Adım 16'nın 800'ü, Adım 17 testi, Adım 18 kalibrasyon 100 ve ana test 500 (kasa).
Etiketleme dosyalarında (batch_XX.csv) sadece id + text, karışık sıra. Kaynak sadece select19.csv'de.
KASA: bu script hiçbir metni yazdırmaz; sadece sayılar.

Çıktı: select19.csv (id, text, pool_label, source) + batch_01..05.csv (54'er) + log_prepare_select.txt
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step16_topic_bert"))

from common import LABEL_SET, candidate_pool, check_no_leak, norm  # noqa: E402  (step15_aspect'i path'e ekler)
from baseline import detect  # noqa: E402

S18 = os.path.join(ROOT, "step18_big_test")
seen_sets = {
    "Adım 16 etiket seti": pd.read_csv(LABEL_SET),
    "Adım 17 test": pd.read_csv(os.path.join(ROOT, "step17_aspect_sentiment", "test17_set.csv")),
    "Adım 18 kalibrasyon": pd.read_csv(os.path.join(S18, "calib_set.csv")),
    "Adım 18 ana test (kasa)": pd.read_csv(os.path.join(S18, "test18_set.csv")),
}
seen = set().union(*(set(df["text"]) for df in seen_sets.values()))
cand, exclude, exclude_norm = candidate_pool(extra_exclude=seen)
cand = cand[~cand["text"].map(norm).isin({norm(t) for t in seen})]
print(f"Aday havuzu: {cand['label'].value_counts().to_dict()}")

TARGETS = [("hedefli:satici", 120), ("hedefli:gorunum", 60), ("hedefli:boyut", 60), ("rastgele", 30)]
NEG_SHARE = 0.2


def take(df, n, seed):
    n_neg = min(round(n * NEG_SHARE), int((df["label"] == "negatif").sum()))   # satıcıda yetmiyor (16) -> pozitifle tamamla
    neg = df[df["label"] == "negatif"].sample(n_neg, random_state=seed)
    pos = df[df["label"] == "pozitif"].sample(n - n_neg, random_state=seed + 1)
    return pd.concat([pos, neg])


found = cand["text"].map(detect)
rest, parts = cand, []
for i, (source, n) in enumerate(TARGETS):
    if source == "rastgele":
        pool = rest
    else:
        topic = source.split(":")[1]
        pool = rest[found.reindex(rest.index).map(lambda s: topic in s)]
    part = take(pool, n, 190 + 2 * i).assign(source=source)
    parts.append(part)
    rest = rest.drop(part.index)

chosen = pd.concat(parts).sample(frac=1, random_state=199).reset_index(drop=True)
chosen["id"] = range(9000, 9000 + len(chosen))
chosen = chosen.rename(columns={"label": "pool_label"})[["id", "text", "pool_label", "source"]]

assert len(chosen) == 270
check_no_leak(chosen.assign(split="train"), exclude, exclude_norm)
for name, df in seen_sets.items():
    assert not set(chosen["text"].map(norm)) & set(df["text"].map(norm)), f"{name} ile çakışma"

chosen.to_csv(os.path.join(HERE, "select19.csv"), index=False)
for b in range(5):
    chosen.iloc[b * 54:(b + 1) * 54][["id", "text"]].to_csv(os.path.join(HERE, f"batch_{b + 1:02d}.csv"), index=False)

out = [f"ADIM 19 — seçim parça 1. Aday havuzu: {cand['label'].value_counts().to_dict()}",
       chosen.groupby(["source", "pool_label"]).size().to_string(),
       f"toplam {len(chosen)} (id {chosen.id.min()}-{chosen.id.max()}), negatif {int((chosen.pool_label == 'negatif').sum())}",
       f"kalan negatif havuz: {int((rest['label'] == 'negatif').sum())}",
       "sızıntı kontrolleri geçti (check_no_leak + Adım 16 + Adım 17 test + Adım 18 kalibrasyon + Adım 18 kasa testi)"]
open(os.path.join(HERE, "log_prepare_select.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
