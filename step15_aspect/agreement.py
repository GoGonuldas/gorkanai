"""
ADIM 15 - Etiketleyiciler arası uyum: Claude (altın etiketler) vs Görkan (kör, 20 yorum).

Görkan sadece metinleri ve etiketleme kurallarını gördü (Claude'un etiketleri, karisik sütunu, model yok);
15.2'de etiketlerini gördüğü 30 yorum (review_sample.csv) bu 20'nin dışında tutuldu.

Kullanım:
  python agreement.py save "<1: Fp,Qp>" "<2: 0>" ...   -> data/aspect_labels/human_blind_20.csv
  python agreement.py                                  -> uyum tablosu + anlaşmazlık listesi
Ölçüler:
  - Konu bazında bahsedilme uyumu: Cohen's kappa (20 yorumda konu var/yok), n = iki etiketleyiciden
    en az birinin işaretlediği yorum sayısı.
  - Ortak konularda duygu uyumu (ikisinin de işaretlediği konularda duygu aynı mı).
  - Uçtan uca (konu, duygu) çift F1: Claude'u referans alarak Görkan'ın F1'i (simetrik; referans
    değişse de F1 aynı) -> modelin uçtan uca F1'i için "insan tavanı".
"""

import os
import re
import sys

import numpy as np
import pandas as pd

from save_aspect_labels import ASPECTS, parse

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(HERE, "..", "data", "aspect_labels")
HUMAN = os.path.join(LABEL_DIR, "human_blind_20.csv")


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def save(args):
    ids = pd.read_csv(os.path.join(HERE, "blind_20_ids.csv"))
    labels = {}
    for arg in args:
        m = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", arg)
        assert m, f"format '<no>: <etiket>' olmalı: {arg!r}"
        no, raw = int(m.group(1)), m.group(2).replace(" ", "")
        parse(raw)
        assert no not in labels, f"{no} iki kez verildi"
        labels[no] = raw
    missing = sorted(set(ids["no"]) - set(labels))
    assert not missing, f"eksik numaralar: {missing}"
    ids["raw_gorkan"] = ids["no"].map(labels)
    ids.to_csv(HUMAN, index=False)
    print(f"{len(ids)} etiket -> {HUMAN}")


def report():
    human = pd.read_csv(HUMAN, keep_default_na=False)
    gold = pd.read_csv(os.path.join(LABEL_DIR, "aspect_labels.csv"), keep_default_na=False)
    df = human.merge(gold[["id", "text", "raw"]], on="id")
    c = [parse(r)[0] for r in df["raw"]]          # Claude
    g = [parse(r)[0] for r in df["raw_gorkan"]]   # Görkan

    print(f"{'konu':<12}{'n':>4}{'kappa':>8}{'ortak':>7}{'duygu uyumu':>13}")
    for a in ASPECTS.values():
        ca, ga = [a in x for x in c], [a in x for x in g]
        both = [(x[a], y[a]) for x, y in zip(c, g) if a in x and a in y]
        n = sum(x or y for x, y in zip(ca, ga))
        agree = f"{sum(p == q for p, q in both)}/{len(both)}" if both else "-"
        print(f"{a:<12}{n:>4}{kappa(ca, ga):>8.2f}{len(both):>7}{agree:>13}")

    cp = {(i, a, s) for i, x in enumerate(c) for a, s in x.items()}
    gp = {(i, a, s) for i, x in enumerate(g) for a, s in x.items()}
    ca_only = {(i, a) for i, a, _ in cp}
    ga_only = {(i, a) for i, a, _ in gp}
    f1 = lambda A, B: 2 * len(A & B) / max(len(A) + len(B), 1)
    both_asp = ca_only & ga_only
    sent_ok = sum(c[i][a] == g[i][a] for i, a in both_asp)
    print(f"\nKonu tespiti F1 (insan-insan): {f1(ca_only, ga_only):.3f}  "
          f"({len(ca_only)} Claude / {len(ga_only)} Görkan / {len(both_asp)} ortak)")
    print(f"Ortak konularda duygu uyumu: {sent_ok}/{len(both_asp)} = {sent_ok / max(len(both_asp), 1):.3f}")
    print(f"Uçtan uca (konu, duygu) çift F1 — İNSAN TAVANI: {f1(cp, gp):.3f}")

    print("\nANLAŞMAZLIKLAR (no | id | Claude | Görkan | metin):")
    for r, x, y in zip(df.itertuples(), c, g):
        if x != y:
            print(f"{r.no:>2} | {r.id} | {r.raw:<14} | {r.raw_gorkan:<14} | {r.text[:160]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        save(sys.argv[2:])
    else:
        report()
