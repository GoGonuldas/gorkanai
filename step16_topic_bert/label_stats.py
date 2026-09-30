"""ADIM 16 - Etiket dağılımı: kaynak (rastgele / hedefli / kararsız) ve bölme bazında konu oranları."""

import os

import pandas as pd

from common import ROOT
from save_aspect_labels import ASPECTS  # noqa: E402

df = pd.read_csv(os.path.join(ROOT, "data", "aspect_labels_step16", "aspect_labels16.csv"), keep_default_na=False)
old = pd.read_csv(os.path.join(ROOT, "data", "aspect_labels", "aspect_labels.csv"), keep_default_na=False)
names = list(ASPECTS.values())


def row(name, d):
    has = (d[names] != "")
    print(f"{name:<26}{len(d):>5}" + "".join(f"{100 * has[a].mean():>8.0f}" for a in names)
          + f"{has.sum(axis=1).mean():>8.2f}{100 * (has.sum(axis=1) == 0).mean():>8.0f}")


print(f"{'grup (% yorumda konu var)':<26}{'n':>5}" + "".join(f"{a[:6]:>8}" for a in names) + f"{'ort.':>8}{'konusuz':>8}")
row("Adım 15 val (eski altın)", old[old["split"] == "val"])
row("Adım 15 test (altın)", old[old["split"] == "test"])
row("yeni val (rastgele)", df[df["split"] == "val"])
train = df[df["split"] == "train"]
for r in sorted(train["round"].unique()):
    t = train[train["round"] == r]
    row(f"tur {r} rastgele", t[t["source"] == "rastgele"])
    for s in sorted(set(t["source"]) - {"rastgele"}):
        row(f"tur {r} {s}", t[t["source"] == s])
row("EĞİTİM toplam", train)
print("\nEğitimde konu başına örnek:", {a: int((train[a] != "").sum()) for a in names})
print("Val (eski+yeni) konu başına örnek:",
      {a: int((df.loc[df['split'] == 'val', a] != '').sum() + (old.loc[old['split'] == 'val', a] != '').sum()) for a in names})
