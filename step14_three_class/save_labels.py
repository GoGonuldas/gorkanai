"""
Elle verilen etiketleri kaydeder: python save_labels.py <batch_no> <10'luk gruplar...>
Her grup tam 10 harf olmalı (p=pozitif, n=negatif, x=nötr); son grup kısa olabilir.
Grup uzunlukları kontrol edilir ki bir etiket kayarsa sonraki tüm etiketler kaymasın.
Çıktı: data/neutral_labels/batch_XX.csv (id, text, label, source)
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
batch_no = int(sys.argv[1])
groups = sys.argv[2:]
for i, g in enumerate(groups[:-1]):
    assert len(g) == 10, f"{i}. grup ({i * 10}-{i * 10 + 9}) 10 değil: {g!r}"
labels = "".join(groups)
assert set(labels) <= set("pnx"), f"geçersiz harf: {set(labels) - set('pnx')}"

batch = pd.read_csv(os.path.join(HERE, "candidates", f"batch_{batch_no:02d}.csv"))
assert len(labels) == len(batch), f"{len(labels)} etiket, {len(batch)} yorum"
source = pd.read_csv(os.path.join(HERE, "candidates", "all_candidates.csv")).set_index("id")["source"]

batch["label"] = [{"p": "pozitif", "n": "negatif", "x": "nötr"}[c] for c in labels]
batch["source"] = batch["id"].map(source)
out_dir = os.path.join(HERE, "..", "data", "neutral_labels")
os.makedirs(out_dir, exist_ok=True)
batch.to_csv(os.path.join(out_dir, f"batch_{batch_no:02d}.csv"), index=False)

print(batch["label"].value_counts().to_string())
print("\nKaynağa göre nötr oranı:")
print(batch.groupby("source")["label"].apply(lambda s: f"%{100 * (s == 'nötr').mean():.0f}").to_string())
