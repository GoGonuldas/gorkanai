"""
ADIM 21 (2) - Görkan'ın 100 kör etiketini kaydet: human_gorkan_01..04.csv -> data/aspect_labels_step21/human_gorkan.csv (id, raw).

Numbers'tan kaydedilen dosyalarda etiket tırnaksızdı (`Qp,Pp` fazladan sütun olarak okunuyor) ve bazı satırlarda konular
boşlukla ayrılmıştı (`Qp Fp`). Okuma: csv modülüyle no, id, text; kalan bütün sütunlar etiket; boşluk/virgül -> virgül.
Etiketin İÇERİĞİ değiştirilmez; her etiket save_aspect_labels.parse ile doğrulanır. Orijinal dosyalar olduğu gibi kalır.
"""

import csv
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step15_aspect"))
from save_aspect_labels import parse  # noqa: E402

rows, changed = [], 0
for b in range(1, 5):
    with open(os.path.join(HERE, f"human_gorkan_{b:02d}.csv"), encoding="utf-8") as f:
        reader = csv.reader(f)
        assert next(reader) == ["no", "id", "text", "etiket"]
        for no, id_, _text, *lab in reader:
            raw = ",".join(x.strip() for x in lab if x.strip())
            label = ",".join(t for t in re.split(r"[,\s]+", raw) if t)
            parse(label)                      # geçersizse assert
            changed += label != raw
            rows.append((int(no), int(id_), label))

df = pd.DataFrame(rows, columns=["no", "id", "raw"]).sort_values("id")
ids = pd.read_csv(os.path.join(HERE, "human_set.csv"))["id"]
assert sorted(df["no"]) == list(range(1, 101)) and sorted(df["id"]) == sorted(ids), "eksik ya da fazla satır"
df[["id", "raw"]].to_csv(os.path.join(ROOT, "data", "aspect_labels_step21", "human_gorkan.csv"), index=False)
n_pairs = df["raw"].map(lambda r: len(parse(r)[0])).sum()
print(f"Görkan: 100 yorum, {n_pairs} (konu, duygu) çifti, konusuz {int((df['raw'] == '0').sum())}; "
      f"boşluk->virgül düzeltilen satır {changed} -> data/aspect_labels_step21/human_gorkan.csv")
