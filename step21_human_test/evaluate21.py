"""
ADIM 21 (4) - İnsan testi raporu (PLAN §3; ölçüm kuralları ve beklenti etiketler görülmeden yazıldı).

Etiketler (data/aspect_labels_step21/): human_gorkan.csv (Görkan), human_fresh.csv (taze Claude oturumu),
human_model.csv (uygulama hattı). Hepsi kör ve bağımsız; uzlaştırma yok.
ANA: çift F1 (model–Görkan) − (fresh–Görkan), yorum bazında eşleştirilmiş bootstrap (2000, tohum 21).
Kullanım: python evaluate21.py -> log_evaluate.txt (sadece toplu sayılar; yorum bazlı okuma rapordan SONRA, ayrı onayla)
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step15_aspect"))
from agreement import kappa  # noqa: E402
from save_aspect_labels import ASPECTS, parse  # noqa: E402

NAMES = list(ASPECTS.values())
LAB = os.path.join(ROOT, "data", "aspect_labels_step21")
meta = pd.read_csv(os.path.join(HERE, "human_set.csv"))
ids = meta["id"].tolist()
pool = dict(zip(meta["id"], meta["pool_label"]))

EXPECT = ["BEKLENTİ (PLAN §4, etiketlerden ÖNCE yazıldı):",
          "  fresh–Görkan çift F1 0.60-0.68 (kalibrasyonda 0.64); model–Görkan 0.55-0.65",
          "  fark (model − fresh) −0.08 ile 0; 0'ı dışlama ~yarı yarıya",
          "  ortak konuda duygu uyumu: fresh ~0.87, model ~0.85; model–fresh çift F1 ~0.72-0.80",
          "  konu kapsamı: model ve fresh Görkan'dan ~%25-40 fazla konu"]


def load(who):
    df = pd.read_csv(os.path.join(LAB, f"human_{who}.csv"), keep_default_na=False)
    assert sorted(df["id"]) == sorted(ids), f"{who}: id'ler eksik/fazla"
    return {r.id: parse(r.raw)[0] for r in df.itertuples()}


def sets(lab, idx, drop_neutral=False):
    P = {(i, k, v) for i in idx for k, v in lab[i].items() if not (drop_neutral and v == "nötr")}
    return P, {(i, k) for i, k, _ in P}


def pair_f1(a, b, idx, drop_neutral=False):
    A, _ = sets(a, idx, drop_neutral)
    B, _ = sets(b, idx, drop_neutral)
    return 2 * len(A & B) / max(len(A) + len(B), 1)


def describe(name_a, a, name_b, b, idx, out):
    A, At = sets(a, idx)
    B, Bt = sets(b, idx)
    both = At & Bt
    same = sum(a[i][k] == b[i][k] for i, k in both)
    out.append(f"\n--- {name_a} vs {name_b} ({len(idx)} yorum) ---")
    out.append(f"  çift: {name_a} {len(A)}, {name_b} {len(B)} (oran {len(A) / max(len(B), 1):.2f})")
    out.append(f"  çift F1 {pair_f1(a, b, idx):.3f} | nötr hariç çift F1 {pair_f1(a, b, idx, True):.3f} | "
               f"konu F1 {2 * len(both) / max(len(At) + len(Bt), 1):.3f}")
    out.append(f"  {name_b}'nin konularını bulma (recall) {len(both) / max(len(Bt), 1):.3f} | "
               f"{name_a}'nin konuları {name_b}'de var (precision) {len(both) / max(len(At), 1):.3f}")
    out.append(f"  ortak konuda duygu uyumu {same}/{len(both)} = {same / max(len(both), 1):.3f}")
    out.append("  konu başına (sayı " + name_a + "/" + name_b + ", kappa): " + ", ".join(
        f"{k[:4]} {sum(k in a[i] for i in idx)}/{sum(k in b[i] for i in idx)} "
        f"{kappa([k in a[i] for i in idx], [k in b[i] for i in idx]):.2f}" for k in NAMES))
    only = lambda X, Y: pd.Series([k for _, k in X - Y], dtype=object).value_counts().to_dict()
    out.append(f"  sadece {name_a}: {only(At, Bt)} | sadece {name_b}: {only(Bt, At)}")


def main():
    g, f, m = load("gorkan"), load("fresh"), load("model")
    out = ["ADIM 21 — İNSAN TESTİ: 100 yorum (id 9500-9599; 70 pozitif + 30 negatif havuz), Görkan / taze Claude / model.",
           "Model = uygulama hattı (Adım 19 konu v2 + Adım 17 duygu, 3 tohum); konu başına nötr üretmez.",
           "Görkan'ın nötr çifti: " + str(sum(v == "nötr" for d in g.values() for v in d.values())) +
           " | fresh'in nötr çifti: " + str(sum(v == "nötr" for d in f.values() for v in d.values())), ""] + EXPECT

    describe("model", m, "görkan", g, ids, out)
    describe("fresh", f, "görkan", g, ids, out)
    describe("model", m, "fresh", f, ids, out)

    rng = np.random.default_rng(21)
    arr = np.array(ids)
    stats = lambda s: (pair_f1(m, g, s) - pair_f1(f, g, s), pair_f1(m, g, s, True) - pair_f1(f, g, s, True))
    point = np.array(stats(ids))
    boot = np.array([stats(list(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(2000)])
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    out.append("\nANA — çift F1 farkı (model–Görkan) − (fresh–Görkan), bootstrap 2000 / tohum 21:")
    for j, lab in enumerate(("tüm çiftler", "nötr hariç (adil kıyas)")):
        verdict = ("0'ı içeriyor: model, Görkan'a bir Claude etiketleyici kadar yakın (fark ayırt edilemiyor)"
                   if lo[j] <= 0 <= hi[j] else ("0'ı DIŞLIYOR: model Claude etiketleyiciden GERİDE" if hi[j] < 0
                                               else "0'ı DIŞLIYOR: model Claude etiketleyiciden İLERİDE"))
        out.append(f"  {lab:<24}{point[j]:+.3f}  [{lo[j]:+.3f}, {hi[j]:+.3f}]  {verdict}")

    out.append("\nHavuza göre çift F1 (model–Görkan / fresh–Görkan):")
    for p in ("pozitif", "negatif"):
        s = [i for i in ids if pool[i] == p]
        out.append(f"  {p} ({len(s)}): {pair_f1(m, g, s):.3f} / {pair_f1(f, g, s):.3f}")

    open(os.path.join(HERE, "log_evaluate.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
