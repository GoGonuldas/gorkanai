"""
ADIM 19 (2) - Etiketleyici kayması kontrolü (PLAN §1c): taze oturum, kalibrasyon 100'den 30 yorumu (drift_30.csv)
eski etiketlerine bakmadan yeniden etiketledi (drift_30_fresh.csv). Kıyas aynı 30'da: fresh'in Adım 18 etiketleri
(kendi kendine tutarlılık), macmini, Görkan. Eşik (önceden): fresh(yeni)–fresh(Adım 18) çift F1 >= 0.85; değilse DUR.
Kullanım: python drift_check.py -> log_drift_check.txt
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step15_aspect"))
from agreement import kappa  # noqa: E402
from save_aspect_labels import ASPECTS, parse  # noqa: E402

NAMES = list(ASPECTS.values())
THRESHOLD = 0.85
ids = pd.read_csv(os.path.join(HERE, "drift_30.csv"))["id"].tolist()


def load(path):
    df = pd.read_csv(path, keep_default_na=False)
    df = df[df["id"].isin(ids)]
    assert sorted(df["id"]) == sorted(ids), f"{path}: id'ler drift_30 ile aynı değil"
    return {r.id: parse(r.raw)[0] for r in df.itertuples()}


def compare(a, b, name_a, name_b, out):
    A = {(i, k, v) for i in ids for k, v in a[i].items()}
    B = {(i, k, v) for i in ids for k, v in b[i].items()}
    At, Bt = {(i, k) for i, k, _ in A}, {(i, k) for i, k, _ in B}
    f1 = lambda x, y: 2 * len(x & y) / max(len(x) + len(y), 1)
    both = At & Bt
    same = sum(a[i][k] == b[i][k] for i, k in both)
    only = lambda X, Y: pd.Series([k for _, k in X - Y], dtype=object).value_counts().to_dict()
    kap = " ".join(f"{k[:4]} {kappa([k in a[i] for i in ids], [k in b[i] for i in ids]):.2f}" for k in NAMES)
    out += [f"\n--- {name_a} vs {name_b} ({len(ids)} yorum) ---",
            f"  çift sayısı: {len(A)} / {len(B)} (oran {len(B) / max(len(A), 1):.2f})",
            f"  konu F1 {f1(At, Bt):.3f} | çift F1 {f1(A, B):.3f} | ortak konuda duygu uyumu {same}/{len(both)}",
            f"  kappa: {kap}",
            f"  sadece {name_a}: {only(At, Bt)} | sadece {name_b}: {only(Bt, At)}"]
    return f1(A, B)


G18 = os.path.join(ROOT, "data", "aspect_labels_step18")
new = load(os.path.join(HERE, "drift_30_fresh.csv"))
out = ["ADIM 19 — kayma kontrolü: taze oturumun 30 yeni kör etiketi (fresh19) vs aynı 30'un önceki etiketleri."]
r = compare(load(os.path.join(G18, "calib_fresh.csv")), new, "fresh18", "fresh19", out)
compare(load(os.path.join(G18, "calib_macmini.csv")), new, "macmini", "fresh19", out)
compare(load(os.path.join(G18, "calib_gorkan.csv")), new, "görkan", "fresh19", out)
out.append(f"\nKARAR (önceden yazılı eşik {THRESHOLD}): fresh18–fresh19 çift F1 {r:.3f} → "
           f"{'GEÇTİ, ana etiketlemeye geçilebilir' if r >= THRESHOLD else 'GEÇMEDİ — DUR, Görkan ile konuş'}")
open(os.path.join(HERE, "log_drift_check.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
