"""
ADIM 17 - Yeni testte iki altın arasındaki uyum: gorkanai-1e (ANA altın) vs Mac mini oturumu (eğitim etiketlerini veren).
Görkan'ın kör 20'si dolduysa üçlü uyum da eklenir. Anlaşmazlıklar UZLAŞTIRILMAZ; iki dosya olduğu gibi kalır.

Ölçüler: konu F1, (konu, duygu) çift F1, konu başına kappa, ortak konularda duygu uyumu ((i) ölçüsünün pratik tavanı).
Kullanım: python agreement.py -> log_agreement.txt
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "step15_aspect"))
from agreement import kappa  # noqa: E402
from save_aspect_labels import ASPECTS, parse  # noqa: E402

GOLD_DIR = os.path.join(HERE, "..", "data", "aspect_labels_step17")
NAMES = list(ASPECTS.values())


def load(who):
    df = pd.read_csv(os.path.join(GOLD_DIR, f"gold_{who}.csv"), keep_default_na=False)
    assert len(df) == 200 and df["id"].is_unique
    return {r.id: parse(r.raw)[0] for r in df.itertuples()}, dict(zip(df["id"], df["raw"]))


def compare(a, b, ids, name_a, name_b, out):
    A = {(i, k, v) for i in ids for k, v in a[i].items()}
    B = {(i, k, v) for i in ids for k, v in b[i].items()}
    At, Bt = {(i, k) for i, k, _ in A}, {(i, k) for i, k, _ in B}
    f1 = lambda x, y: 2 * len(x & y) / max(len(x) + len(y), 1)
    both = At & Bt
    same = sum(a[i][k] == b[i][k] for i, k in both)
    out.append(f"--- {name_a} vs {name_b} ({len(ids)} yorum) ---")
    out.append(f"  çift sayısı: {name_a} {len(A)}, {name_b} {len(B)} (oran {len(B) / max(len(A), 1):.2f})")
    out.append(f"  konu F1 {f1(At, Bt):.3f} | (konu, duygu) çift F1 {f1(A, B):.3f} | ortak konu {len(both)}, "
               f"duygu uyumu {same}/{len(both)} = {same / max(len(both), 1):.3f}")
    out.append(f"  {'konu':<12}{name_a:>9}{name_b:>9}{'ortak':>7}{'kappa':>8}{'duygu uyumu':>13}")
    for k in NAMES:
        xa, xb = [k in a[i] for i in ids], [k in b[i] for i in ids]
        bt = [(a[i][k], b[i][k]) for i in ids if k in a[i] and k in b[i]]
        out.append(f"  {k:<12}{sum(xa):>9}{sum(xb):>9}{len(bt):>7}{kappa(xa, xb):>8.2f}"
                   f"{(str(sum(p == q for p, q in bt)) + '/' + str(len(bt))):>13}")
    return At, Bt


out = ["ADIM 17 — yeni test (id 6000-6199) etiketleyici uyumu. Kıyas: Adım 15 Claude-Görkan çift F1 0.667;",
       "Adım 16 kayma kontrolleri (bu oturum vs Adım 15 altını, 30'ar yorum) çift F1 0.867 / 0.897.\n"]
g1e, raw1e = load("1e")
gmm, rawmm = load("macmini")
assert set(g1e) == set(gmm)
ids = sorted(g1e)
At, Bt = compare(g1e, gmm, ids, "1e", "macmini", out)

texts = pd.read_csv(os.path.join(HERE, "test17_set.csv")).set_index("id")["text"]
diff = [i for i in ids if g1e[i] != gmm[i]]
out.append(f"\n  birebir aynı etiketlenen yorum: {len(ids) - len(diff)}/{len(ids)}")
only = lambda X, Y: pd.Series([k for _, k in X - Y]).value_counts().to_dict()
out.append(f"  sadece 1e'nin verdiği konular: {only(At, Bt)}")
out.append(f"  sadece macmini'nin verdiği konular: {only(Bt, At)}")
swap = sum(1 for i in ids if ("kalite" in g1e[i]) != ("kalite" in gmm[i]) and ("performans" in g1e[i]) != ("performans" in gmm[i])
           and ("kalite" in g1e[i]) != ("performans" in g1e[i]))
out.append(f"  Q/P takası (biri Q, diğeri P demiş) olan yorum: {swap}")

human_path = os.path.join(HERE, "human_blind_20.csv")
human = pd.read_csv(human_path, keep_default_na=False)
if (human["etiket"].str.strip() != "").all():
    gh = {r.id: parse(r.etiket.replace(" ", ""))[0] for r in human.itertuples()}
    hid = sorted(gh)
    out.append("\nGÖRKAN'IN KÖR 20'Sİ (n=20, sadece fikir verir)")
    compare(gh, g1e, hid, "görkan", "1e", out)
    compare(gh, gmm, hid, "görkan", "macmini", out)
    compare(g1e, gmm, hid, "1e", "macmini", out)
else:
    out.append("\n(Görkan'ın kör 20'si henüz doldurulmadı.)")

out.append("\nFARKLAR (id | 1e | macmini | metin):")
for i in diff:
    out.append(f"  {i} | {raw1e[i]:<16} | {rawmm[i]:<16} | {texts[i][:150]}")
open(os.path.join(HERE, "log_agreement.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out[:40]))
