"""
ADIM 18 - Kalibrasyon 100 (id 7000-7099) etiketleyici uyumu: taze oturum (fresh), Mac mini oturumu (macmini), Görkan.
Üç dosya da kör ve bağımsız etiketlendi; anlaşmazlıklar UZLAŞTIRILMAZ, dosyalar olduğu gibi kalır.

Ölçüler (Adım 17 agreement.py ile aynı): konu F1, (konu, duygu) çift F1, ortak konularda duygu uyumu, konu başına kappa
ve duygu uyumu; birebir aynı yorum, sadece birinin verdiği konular, Q/P takası. Ek: Görkan'ın iki Claude'dan da ayrıştığı
ama iki Claude'un kendi arasında anlaştığı yorumlar (kural konuşmasının gündemi).
Kullanım: python agreement.py -> log_agreement.txt
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "step15_aspect"))
from agreement import kappa  # noqa: E402
from save_aspect_labels import ASPECTS, parse  # noqa: E402

GOLD_DIR = os.path.join(HERE, "..", "data", "aspect_labels_step18")
NAMES = list(ASPECTS.values())
EXPECT = {("fresh", "macmini"): (0.88, 0.92), ("görkan", "fresh"): (0.60, 0.70), ("görkan", "macmini"): (0.60, 0.70)}


def load(who):
    df = pd.read_csv(os.path.join(GOLD_DIR, f"calib_{who}.csv"), keep_default_na=False)
    assert len(df) == 100 and df["id"].is_unique and sorted(df["id"]) == list(range(7000, 7100))
    return {r.id: parse(r.raw)[0] for r in df.itertuples()}, dict(zip(df["id"], df["raw"]))


def compare(a, b, ids, name_a, name_b, out):
    A = {(i, k, v) for i in ids for k, v in a[i].items()}
    B = {(i, k, v) for i in ids for k, v in b[i].items()}
    At, Bt = {(i, k) for i, k, _ in A}, {(i, k) for i, k, _ in B}
    f1 = lambda x, y: 2 * len(x & y) / max(len(x) + len(y), 1)
    both = At & Bt
    same = sum(a[i][k] == b[i][k] for i, k in both)
    out.append(f"\n--- {name_a} vs {name_b} ({len(ids)} yorum) ---")
    out.append(f"  çift sayısı: {name_a} {len(A)}, {name_b} {len(B)} (oran {len(B) / max(len(A), 1):.2f})")
    out.append(f"  konu F1 {f1(At, Bt):.3f} | (konu, duygu) çift F1 {f1(A, B):.3f} | ortak konu {len(both)}, "
               f"duygu uyumu {same}/{len(both)} = {same / max(len(both), 1):.3f}")
    out.append(f"  {'konu':<12}{name_a:>9}{name_b:>9}{'ortak':>7}{'kappa':>8}{'duygu uyumu':>13}")
    kap = {}
    for k in NAMES:
        xa, xb = [k in a[i] for i in ids], [k in b[i] for i in ids]
        bt = [(a[i][k], b[i][k]) for i in ids if k in a[i] and k in b[i]]
        kap[k] = kappa(xa, xb)
        out.append(f"  {k:<12}{sum(xa):>9}{sum(xb):>9}{len(bt):>7}{kap[k]:>8.2f}"
                   f"{(str(sum(p == q for p, q in bt)) + '/' + str(len(bt))):>13}")
    diff = [i for i in ids if a[i] != b[i]]
    only = lambda X, Y: pd.Series([k for _, k in X - Y], dtype=object).value_counts().to_dict()
    out.append(f"  birebir aynı etiketlenen yorum: {len(ids) - len(diff)}/{len(ids)}")
    out.append(f"  sadece {name_a}'nin verdiği konular: {only(At, Bt)}")
    out.append(f"  sadece {name_b}'nin verdiği konular: {only(Bt, At)}")
    swap = sum(1 for i in ids if ("kalite" in a[i]) != ("kalite" in b[i]) and ("performans" in a[i]) != ("performans" in b[i])
               and ("kalite" in a[i]) != ("performans" in a[i]))
    out.append(f"  Q/P takası (biri Q, diğeri P demiş) olan yorum: {swap}")
    return {"pair_f1": f1(A, B), "sent": same / max(len(both), 1), "kappa": kap}


out = ["ADIM 18 — kalibrasyon 100 (id 7000-7099) etiketleyici uyumu: fresh (taze Claude oturumu), macmini (Mac mini oturumu),",
       "görkan. Kıyas: Adım 17 Claude–Claude çift F1 0.903; Adım 17 kör 20'de Görkan–Claude çift F1 0.567 / 0.615."]
gf, rf = load("fresh")
gm, rm = load("macmini")
gg, rg = load("gorkan")
ids = list(range(7000, 7100))
res = {("fresh", "macmini"): compare(gf, gm, ids, "fresh", "macmini", out),
       ("görkan", "fresh"): compare(gg, gf, ids, "görkan", "fresh", out),
       ("görkan", "macmini"): compare(gg, gm, ids, "görkan", "macmini", out)}

out.append("\nBEKLENTİYLE KIYAS (PLAN.md madde 4, önceden):")
for (x, y), r in res.items():
    lo, hi = EXPECT[(x, y)]
    where = "aralıkta" if lo <= r["pair_f1"] <= hi else ("ALTINDA" if r["pair_f1"] < lo else "ÜSTÜNDE")
    out.append(f"  {x}–{y}: çift F1 {r['pair_f1']:.3f} (beklenti {lo:.2f}-{hi:.2f} → {where}); ortak konuda duygu uyumu "
               f"{r['sent']:.3f}{' (beklenti ~0.85)' if x == 'görkan' else ''}; kappa kalite {r['kappa']['kalite']:.2f}, "
               f"performans {r['kappa']['performans']:.2f}{' (beklenti Q/P < 0.4)' if x == 'görkan' else ''}")

texts = pd.read_csv(os.path.join(HERE, "calib_set.csv")).set_index("id")["text"]
agenda = [i for i in ids if gf[i] == gm[i] and gg[i] != gf[i]]
out.append(f"\nGÜNDEM — Görkan iki Claude'dan da ayrışıyor, iki Claude kendi arasında birebir aynı: {len(agenda)} yorum")
out.append("  (id | görkan | fresh = macmini | metin)")
for i in agenda:
    out.append(f"  {i} | {rg[i]:<14} | {rf[i]:<14} | {texts[i]}")
out.append(f"\nİKİ CLAUDE BİRBİRİNDEN AYRIŞIYOR: {sum(1 for i in ids if gf[i] != gm[i])} yorum; "
           f"üçü birebir aynı: {sum(1 for i in ids if gf[i] == gm[i] == gg[i])}")
out.append("\nTÜM ETİKETLER (id | görkan | fresh | macmini):")
for i in ids:
    out.append(f"  {i} | {rg[i]:<14} | {rf[i]:<16} | {rm[i]}")
open(os.path.join(HERE, "log_agreement.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
