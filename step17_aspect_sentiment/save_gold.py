"""
ADIM 17 - Yeni test (id 6000-6199) için bir etiketleyicinin kör etiketlerini kaydeder.

Kullanım: python save_gold.py <kim> <batch_no> <grup1> ...   ("/" ile ayrılmış TAM 10 etiket; LABEL_RULES.md biçimi)
Çıktı: data/aspect_labels_step17/gold_<kim>.csv (id, raw) — 4 parti bitince 200 satır.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "step15_aspect"))
from save_aspect_labels import parse  # noqa: E402

OUT_DIR = os.path.join(HERE, "..", "data", "aspect_labels_step17")

who, batch_no = sys.argv[1], int(sys.argv[2])
groups = [g.split("/") for g in sys.argv[3:]]
assert all(len(g) == 10 for g in groups), [len(g) for g in groups]
tokens = [t for g in groups for t in g]
for t in tokens:
    parse(t)
batch = pd.read_csv(os.path.join(HERE, f"test17_batch_{batch_no:02d}.csv"))
assert len(tokens) == len(batch), f"{len(tokens)} etiket, {len(batch)} yorum"
os.makedirs(OUT_DIR, exist_ok=True)
path = os.path.join(OUT_DIR, f"gold_{who}.csv")
new = pd.DataFrame({"id": batch["id"], "raw": tokens})
if os.path.exists(path):
    old = pd.read_csv(path, keep_default_na=False)
    new = pd.concat([old[~old["id"].isin(new["id"])], new]).sort_values("id")
new.to_csv(path, index=False)
print(f"{who} batch {batch_no} kaydedildi; toplam {len(new)} / 200")
