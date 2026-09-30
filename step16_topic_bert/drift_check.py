"""
ADIM 16 - Etiketleyici kayması kontrolü (yeni etiketlemeye başlamadan ÖNCE).

Adım 15'in 300 altın etiketini başka bir Claude oturumu verdi. Bu oturum aynı kuralları (save_aspect_labels.py
docstring'i) aynı şekilde uyguluyor mu? Mevcut VAL'den (test'e dokunulmuyor) 30 yorum, altın etiketlere
bakılmadan yeniden etiketlenir ve altınla karşılaştırılır. Kıyas: Claude-Görkan çift F1 0.667.
(id 4001 dışarıda: altın etiketi dosya başlığına bakarken görüldü.)

Kullanım:
  python drift_check.py prepare            -> drift_30.csv (id, text) — altın etiket YOK
  python drift_check.py save <grup> ...    -> drift_30_labels.csv ("/" ile ayrılmış 10'luk gruplar)
  python drift_check.py                    -> uyum tablosu + farklar
  (ikinci tur için her komutun sonuna "b")
"""

import os
import sys

import pandas as pd

from common import HERE, ROOT  # step15_aspect'i sys.path'e ekler
from agreement import kappa  # noqa: E402
from save_aspect_labels import ASPECTS, parse  # noqa: E402

# İkinci tur (kalibrasyon kararından sonra): komut sonuna "b" eklenir -> drift_30b*.csv, ilk 30 dışarıda.
ROUND_B = sys.argv[-1] == "b"
if ROUND_B:
    sys.argv.pop()
TEXTS = os.path.join(HERE, "drift_30b.csv" if ROUND_B else "drift_30.csv")
LABELS = os.path.join(HERE, "drift_30b_labels.csv" if ROUND_B else "drift_30_labels.csv")

if len(sys.argv) > 1 and sys.argv[1] == "prepare":
    ls = pd.read_csv(os.path.join(ROOT, "step15_aspect", "label_set.csv"))
    val = ls[(ls["split"] == "val") & (ls["id"] != 4001)]
    seed = 160
    if ROUND_B:
        val = val[~val["id"].isin(pd.read_csv(os.path.join(HERE, "drift_30.csv"))["id"])]
        seed = 170
    pick = pd.concat([val[val["pool_label"] == lab].sample(15, random_state=seed) for lab in ("pozitif", "negatif")])
    pick = pick.sample(frac=1, random_state=seed + 1)
    assert (pick["split"] == "val").all()
    pick[["id", "text"]].to_csv(TEXTS, index=False)
    print(f"{len(pick)} val yorumu -> {os.path.basename(TEXTS)}")
elif len(sys.argv) > 1 and sys.argv[1] == "save":
    groups = [g.split("/") for g in sys.argv[2:]]
    assert all(len(g) == 10 for g in groups), [len(g) for g in groups]
    tokens = [t for g in groups for t in g]
    for t in tokens:
        parse(t)
    df = pd.read_csv(TEXTS)
    assert len(tokens) == len(df)
    df["raw_new"] = tokens
    df.to_csv(LABELS, index=False)
    print(f"{len(df)} etiket -> {os.path.basename(LABELS)}")
else:
    new = pd.read_csv(LABELS, keep_default_na=False)
    gold = pd.read_csv(os.path.join(ROOT, "data", "aspect_labels", "aspect_labels.csv"), keep_default_na=False)
    df = new.merge(gold[["id", "raw"]], on="id")
    assert len(df) == 30
    c = [parse(r)[0] for r in df["raw"]]       # altın (Adım 15 oturumu)
    g = [parse(r)[0] for r in df["raw_new"]]   # bu oturum
    print(f"{'konu':<12}{'n':>4}{'kappa':>8}")
    for a in ASPECTS.values():
        ca, ga = [a in x for x in c], [a in x for x in g]
        print(f"{a:<12}{sum(x or y for x, y in zip(ca, ga)):>4}{kappa(ca, ga):>8.2f}")
    cp = {(i, a, s) for i, x in enumerate(c) for a, s in x.items()}
    gp = {(i, a, s) for i, x in enumerate(g) for a, s in x.items()}
    ca_, ga_ = {(i, a) for i, a, _ in cp}, {(i, a) for i, a, _ in gp}
    f1 = lambda A, B: 2 * len(A & B) / max(len(A) + len(B), 1)
    print(f"\nKonu F1: {f1(ca_, ga_):.3f} ({len(ca_)} altın / {len(ga_)} yeni / {len(ca_ & ga_)} ortak)")
    print(f"Çift F1: {f1(cp, gp):.3f}   (kıyas: Claude-Görkan 0.667)")
    print("\nFARKLAR (id | altın | yeni | metin):")
    for r, x, y in zip(df.itertuples(), c, g):
        if x != y:
            print(f"{r.id} | {r.raw:<14} | {r.raw_new:<14} | {r.text[:200]}")
