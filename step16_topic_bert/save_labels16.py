"""
ADIM 16 - Elle verilen konu-duygu etiketlerini kaydeder (Adım 15 save_aspect_labels.py ile aynı format ve kurallar).

Kullanım: python save_labels16.py <batch_no> <grup1> <grup2> ...   (her grup "/" ile ayrılmış TAM 10 etiket)
Çıktı: data/aspect_labels_step16/batch_NN.csv + birleşim aspect_labels16.csv (split/source label_set16.csv'den).
"""

import glob
import os
import sys

import pandas as pd

from common import HERE, LABEL_SET, ROOT  # step15_aspect'i sys.path'e ekler
from save_aspect_labels import ASPECTS, parse  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "aspect_labels_step16")


def merge_all():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "batch_*.csv")))
    labels = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files])
    meta = pd.read_csv(LABEL_SET)
    df = meta.merge(labels[["id", "raw"]], on="id", validate="one_to_one")
    assert len(df) == len(labels), "etiketlenen id'ler label_set16 ile eşleşmiyor"
    parsed = df["raw"].map(parse)
    for name in ASPECTS.values():
        df[name] = parsed.map(lambda p: p[0].get(name, ""))
    df["karisik_konular"] = parsed.map(lambda p: ";".join(p[1]))
    df["karisik"] = (df["karisik_konular"] != "").astype(int)
    df.to_csv(os.path.join(OUT_DIR, "aspect_labels16.csv"), index=False)
    return df


if __name__ == "__main__":
    batch_no = int(sys.argv[1])
    groups = [g.split("/") for g in sys.argv[2:]]
    for i, g in enumerate(groups[:-1]):
        assert len(g) == 10, f"{i}. grup ({i * 10}-{i * 10 + 9}) 10 değil: {len(g)} -> {g}"
    tokens = [t for g in groups for t in g]
    for t in tokens:
        parse(t)
    batch = pd.read_csv(os.path.join(HERE, f"batch_{batch_no:02d}.csv"))
    assert len(tokens) == len(batch), f"{len(tokens)} etiket, {len(batch)} yorum"
    batch["raw"] = tokens
    os.makedirs(OUT_DIR, exist_ok=True)
    batch.to_csv(os.path.join(OUT_DIR, f"batch_{batch_no:02d}.csv"), index=False)
    df = merge_all()
    print(f"batch {batch_no} kaydedildi. Toplam etiketli: {len(df)}")
    print({a: int((df[a] != "").sum()) for a in ASPECTS.values()}, "| konusuz:", int((df["raw"] == "0").sum()))
