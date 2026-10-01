"""
ADIM 18 - Ana test 500 (id 8000-8499): iki altının (fresh = ana, macmini = ikinci) uyumu. KASA KURALI: sadece toplu
sayılar; yorum bazlı liste, id ya da metin yazılmaz, ayrışan yorumlara bakılmaz. Dosyaların sha256'sı frozen_test.json ile
karşılaştırılır.
Kullanım: python test_agreement.py -> log_test_agreement.txt
"""

import hashlib
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step15_aspect"))
from agreement import kappa  # noqa: E402
from save_aspect_labels import ASPECTS, parse  # noqa: E402

NAMES = list(ASPECTS.values())
cfg = json.load(open(os.path.join(HERE, "frozen_test.json")))


def load(key):
    path = os.path.join(ROOT, cfg[key]["file"])
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == cfg[key]["sha256"], f"{key} değişmiş"
    df = pd.read_csv(path, keep_default_na=False)
    assert len(df) == 500 and sorted(df["id"]) == list(range(8000, 8500))
    return {r.id: parse(r.raw) for r in df.itertuples()}


f, m = load("gold_main"), load("gold_second")
ids = list(range(8000, 8500))
A = {(i, k, v) for i in ids for k, v in f[i][0].items()}
B = {(i, k, v) for i in ids for k, v in m[i][0].items()}
At, Bt = {(i, k) for i, k, _ in A}, {(i, k) for i, k, _ in B}
f1 = lambda x, y: 2 * len(x & y) / max(len(x) + len(y), 1)
both = At & Bt
same = sum(f[i][0][k] == m[i][0][k] for i, k in both)
mixed = lambda g: sum(len(g[i][1]) for i in ids)
neutral = lambda S: sum(v == "nötr" for _, _, v in S)
out = ["ADIM 18 — ana test 500 (id 8000-8499) iki altının uyumu. KASA: sadece toplu sayılar.",
       f"ana altın fresh ({cfg['gold_main']['sha256'][:12]}…), ikinci macmini ({cfg['gold_second']['sha256'][:12]}…); "
       f"kurallar LABEL_RULES.md ({cfg['label_rules']['sha256'][:12]}…)",
       "Kıyas: kalibrasyon 100'de fresh–macmini çift F1 0.935, duygu uyumu 0.989; Adım 17 testinde 1e–macmini 0.903 / 0.967.\n",
       f"çift sayısı: fresh {len(A)} (nötr {neutral(A)}, karışık {mixed(f)}), macmini {len(B)} (nötr {neutral(B)}, karışık {mixed(m)}); "
       f"oran {len(B) / len(A):.2f}",
       f"konu yok (0) yorum: fresh {sum(not f[i][0] for i in ids)}, macmini {sum(not m[i][0] for i in ids)}",
       f"konu F1 {f1(At, Bt):.3f} | (konu, duygu) çift F1 {f1(A, B):.3f} | ortak konu {len(both)}, "
       f"duygu uyumu {same}/{len(both)} = {same / len(both):.3f}",
       f"birebir aynı etiketlenen yorum: {sum(f[i][0] == m[i][0] for i in ids)}/500",
       f"\n{'konu':<12}{'fresh':>8}{'macmini':>9}{'ortak':>7}{'kappa':>8}{'duygu uyumu':>14}"]
for k in NAMES:
    xa, xb = [k in f[i][0] for i in ids], [k in m[i][0] for i in ids]
    bt = [(f[i][0][k], m[i][0][k]) for i in ids if k in f[i][0] and k in m[i][0]]
    out.append(f"{k:<12}{sum(xa):>8}{sum(xb):>9}{len(bt):>7}{kappa(xa, xb):>8.2f}"
               f"{(str(sum(p == q for p, q in bt)) + '/' + str(len(bt))):>14}")
only = lambda X, Y: pd.Series([k for _, k in X - Y], dtype=object).value_counts().to_dict()
out.append(f"\nsadece fresh'in verdiği konular (sayı): {only(At, Bt)}")
out.append(f"sadece macmini'nin verdiği konular (sayı): {only(Bt, At)}")
swap = sum(1 for i in ids if ("kalite" in f[i][0]) != ("kalite" in m[i][0]) and ("performans" in f[i][0]) != ("performans" in m[i][0])
           and ("kalite" in f[i][0]) != ("performans" in f[i][0]))
out.append(f"Q/P takası olan yorum sayısı: {swap}")
open(os.path.join(HERE, "log_test_agreement.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
