"""
ADIM 19 (3) - Eğitim seti: Adım 16 eğitimi 600 + okunmuş setler 500 (PLAN §1a) + yeni 300 (§1b, taze oturum).

origin: adim16_train (600, Adım 16 oturumu) | adim15_test (200, Adım 15 oturumu) | adim17_test (200, gold_1e) |
        adim18_calib (100, calib_fresh = taze oturum) | adim19_new (300, taze oturum, labels19_batch_01..06)
Ablasyon A = origin != adim19_new; B = hepsi. Val (Adım 16 val 300) AYNEN kalır ve burada hiçbir satıra girmez.
Sızıntı: val ile ve Adım 18 kasa testiyle (500) normalleştirilmiş metin çakışması assert'le yasak.
Çıktı: data/aspect_labels_step19/train19.csv + log_save_labels.txt (sadece sayılar)
"""

import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step16_topic_bert"))

from common import norm  # noqa: E402  (step15_aspect'i path'e ekler)
from save_aspect_labels import ASPECTS, parse  # noqa: E402
from train import NEW_LABELS, OLD_LABELS, load_data  # noqa: E402

OUT = os.path.join(ROOT, "data", "aspect_labels_step19", "train19.csv")
NAMES = list(ASPECTS.values())


def with_raw(meta, labels, origin, source=None):
    df = meta[["id", "text", "pool_label"]].merge(labels[["id", "raw"]], on="id", validate="1:1")
    assert len(df) == len(meta) == len(labels), origin
    return df.assign(origin=origin, source=source if source is not None else origin)


def read(path):
    return pd.read_csv(path, keep_default_na=False)


s16 = read(NEW_LABELS)
old = read(OLD_LABELS)
s17 = os.path.join(ROOT, "step17_aspect_sentiment")
s18 = os.path.join(ROOT, "step18_big_test")
sel = pd.read_csv(os.path.join(HERE, "select19.csv"))
new = pd.concat([read(f) for f in sorted(glob.glob(os.path.join(HERE, "labels19_batch_*.csv")))], ignore_index=True)
assert len(sel) == 300 and len(new) == 300 and sorted(new["id"]) == sorted(sel["id"]), "yeni 300 eksik"

parts = [
    s16[s16["split"] == "train"][["id", "text", "pool_label", "raw", "source"]].assign(origin="adim16_train"),
    with_raw(old[old["split"] == "test"], old[old["split"] == "test"], "adim15_test"),
    with_raw(read(os.path.join(s17, "test17_set.csv")), read(os.path.join(ROOT, "data", "aspect_labels_step17", "gold_1e.csv")), "adim17_test"),
    with_raw(read(os.path.join(s18, "calib_set.csv")), read(os.path.join(ROOT, "data", "aspect_labels_step18", "calib_fresh.csv")), "adim18_calib"),
    sel.merge(new, on="id", validate="1:1").assign(origin="adim19_new"),
]
df = pd.concat(parts, ignore_index=True)
parsed = df["raw"].map(parse)   # biçim hatası varsa assert burada patlar
for name in NAMES:
    df[name] = parsed.map(lambda p: p[0].get(name, ""))
df["karisik_konular"] = parsed.map(lambda p: ";".join(p[1]))

_, val = load_data()
test18 = pd.read_csv(os.path.join(s18, "test18_set.csv"))
tn = set(df["text"].map(norm))
assert df["id"].is_unique and df["text"].map(norm).is_unique, "eğitimde tekrar"
assert not tn & set(val["text"].map(norm)), "eğitim-val sızıntısı"
assert not tn & set(test18["text"].map(norm)), "eğitim-KASA testi sızıntısı"
df.to_csv(OUT, index=False)

y = df[NAMES] != ""
out = ["ADIM 19 — eğitim seti (train19.csv). % yorumda konu var / ort. konu:",
       f"{'origin':<14}{'n':>5}" + "".join(f"{k[:6]:>8}" for k in NAMES) + f"{'ort':>6}{'konusuz':>9}"]
for o, g in list(df.groupby("origin", sort=False)) + [("TOPLAM A", df[df.origin != "adim19_new"]), ("TOPLAM B", df)]:
    yy = g[NAMES] != ""
    out.append(f"{o:<14}{len(g):>5}" + "".join(f"{100 * yy[k].mean():>8.0f}" for k in NAMES)
               + f"{yy.sum(axis=1).mean():>6.2f}{int((yy.sum(axis=1) == 0).sum()):>9}")
for name, g in (("A", df[df.origin != "adim19_new"]), ("B", df)):
    out.append(f"Eğitim {name} konu başına örnek: " + str({k: int((g[k] != '').sum()) for k in NAMES}))
nw = df[df.origin == "adim19_new"]
out.append("Yeni 300, kaynağa göre % yorumda S / G / B:")
for s, g in nw.groupby("source"):
    out.append(f"  {s:<24} n={len(g):>3}  S {100 * (g.satici != '').mean():>3.0f}  G {100 * (g.gorunum != '').mean():>3.0f}  B {100 * (g.boyut != '').mean():>3.0f}")
out.append("sızıntı kontrolleri geçti (val 300, Adım 18 kasa testi 500)")
open(os.path.join(HERE, "log_save_labels.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
