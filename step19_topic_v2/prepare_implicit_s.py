"""
ADIM 19 (1b/?) - Seçim parça 2: örtük satıcı 30 (id 9270-9299). MAC MİNİ'DE çalışır (Adım 16 modelleri orada).

PLAN §1b: V2 satıcı anahtar kelimesi TUTMAYAN ama Adım 16 modelinin (dondurulmuş, 3 tohum ortalaması) S olasılığı
0.15-0.60 olan yorumlar. Konu başına kotalı kararsızlık (Adım 16 dersi: global kararsızlık nadir konuyu getirmiyor).
Aday: parça 1 (select19.csv) ile aynı dışlamalar + parça 1'in 270'i; pozitif havuzun rastgele 20000'i + bütün
kalan negatifler. Bant içinden rastgele 24 pozitif + 6 negatif.

KASA: hiçbir metin yazdırılmaz; sadece sayılar. Çıktı: select19.csv'ye eklenir + batch_06.csv + log_prepare_implicit_s.txt
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
S16 = os.path.join(ROOT, "step16_topic_bert")
sys.path.insert(0, S16)

from common import LABEL_SET, candidate_pool, check_no_leak, norm  # noqa: E402
from baseline import ASPECTS, detect  # noqa: E402
from train import DEVICE, predict  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

LOW, HIGH, N_POS, N_NEG = 0.15, 0.60, 24, 6
S18 = os.path.join(ROOT, "step18_big_test")
part1 = pd.read_csv(os.path.join(HERE, "select19.csv"))
assert len(part1) == 270 and part1["id"].max() == 9269, "parça 1 eksik ya da parça 2 zaten eklenmiş"
seen_sets = {
    "Adım 16 etiket seti": pd.read_csv(LABEL_SET),
    "Adım 17 test": pd.read_csv(os.path.join(ROOT, "step17_aspect_sentiment", "test17_set.csv")),
    "Adım 18 kalibrasyon": pd.read_csv(os.path.join(S18, "calib_set.csv")),
    "Adım 18 ana test (kasa)": pd.read_csv(os.path.join(S18, "test18_set.csv")),
    "Adım 19 parça 1": part1,
}
seen = set().union(*(set(df["text"]) for df in seen_sets.values()))
cand, exclude, exclude_norm = candidate_pool(extra_exclude=seen)
cand = cand[~cand["text"].map(norm).isin({norm(t) for t in seen})]
cand = cand[~cand["text"].map(lambda t: "satici" in detect(t))]
sub = pd.concat([cand[cand["label"] == "pozitif"].sample(20000, random_state=200), cand[cand["label"] == "negatif"]])

cfg = json.load(open(os.path.join(S16, "frozen_config.json")))
probs = []
for d in cfg["model_dirs"]:
    path = os.path.join(ROOT, d)
    tok, model = AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
    probs.append(predict(model, tok, sub["text"].tolist()))
p_s = np.mean(probs, axis=0)[:, ASPECTS.index("satici")]
sub = sub.assign(p_s=p_s)
band = sub[(sub["p_s"] >= LOW) & (sub["p_s"] <= HIGH)]

chosen = pd.concat([band[band["label"] == "pozitif"].sample(N_POS, random_state=201),
                    band[band["label"] == "negatif"].sample(min(N_NEG, int((band["label"] == "negatif").sum())), random_state=202)])
chosen = chosen.sample(frac=1, random_state=203).reset_index(drop=True)
chosen["id"] = range(9270, 9270 + len(chosen))
chosen = chosen.rename(columns={"label": "pool_label"}).assign(source="kararsiz:satici_ortuk")[["id", "text", "pool_label", "source"]]

check_no_leak(chosen.assign(split="train"), exclude, exclude_norm)
for name, df in seen_sets.items():
    assert not set(chosen["text"].map(norm)) & set(df["text"].map(norm)), f"{name} ile çakışma"

full = pd.concat([part1, chosen], ignore_index=True)
assert full["id"].is_unique and full["text"].map(norm).is_unique
full.to_csv(os.path.join(HERE, "select19.csv"), index=False)
chosen[["id", "text"]].to_csv(os.path.join(HERE, "batch_06.csv"), index=False)

out = [f"ADIM 19 — seçim parça 2 (örtük S). Alt küme: {sub['label'].value_counts().to_dict()} (satıcı kelimesi tutmayan)",
       f"S olasılığı: medyan {np.median(p_s):.3f}, >=0.15: {int((p_s >= LOW).sum())}, bant [{LOW}, {HIGH}]: {len(band)} {band['label'].value_counts().to_dict()}",
       f"seçilen {len(chosen)} (id {chosen.id.min()}-{chosen.id.max()}) {chosen.pool_label.value_counts().to_dict()}",
       f"select19.csv toplam {len(full)}, negatif {int((full.pool_label == 'negatif').sum())}",
       "sızıntı kontrolleri geçti"]
open(os.path.join(HERE, "log_prepare_implicit_s.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
