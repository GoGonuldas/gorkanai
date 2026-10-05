"""
ADIM 22 - Kabul edilebilir etiket ölçüsü (PLAN §2, kararlardan önce yazıldı).
A (kabul kümesi) = auto_accept.csv (üçünün ortak çiftleri) + judge_gorkan.csv'de k olan adaylar.
R (zorunlu küme) = Görkan'ın kendi çiftlerinden A'da kalanlar.
Her etiketleyici: kabul precision = |X ∩ A| / |X|; zorunlu recall = |X ∩ R| / |R|; kabul F1.
ANA: kabul F1 model − taze Claude, eşleştirilmiş bootstrap (2000, tohum 22). Kullanım: python evaluate22.py -> log_evaluate.txt
"""

import csv
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step15_aspect"))
from save_aspect_labels import ASPECTS, SENT, parse  # noqa: E402

L = os.path.join(ROOT, "data", "aspect_labels_step21")
rd = lambda w: {r.id: set(parse(r.raw)[0].items()) for r in pd.read_csv(os.path.join(L, f"human_{w}.csv"), keep_default_na=False).itertuples()}
lab = {"model": rd("model"), "claude": rd("fresh"), "görkan": rd("gorkan")}
ids = sorted(lab["görkan"])

A = {i: set() for i in ids}
for r in pd.read_csv(os.path.join(HERE, "auto_accept.csv")).itertuples():
    A[r.id].add((r.konu, r.duygu))
wrong = {i: set() for i in ids}
with open(os.path.join(HERE, "judge_gorkan.csv"), encoding="utf-8") as f:
    reader = csv.reader(f)
    assert next(reader) == ["no", "id", "text", "aday", "karar"]
    n = 0
    for no, id_, _t, aday, karar in reader:
        code = re.match(r"^([KFQPBGS])([pnx]) =", aday)
        pair = (ASPECTS[code.group(1)], SENT[code.group(2)])
        k = karar.strip().lower()
        assert k in ("k", "y"), f"satır {no}: karar '{karar}'"
        (A if k == "k" else wrong)[int(id_)].add(pair)
        n += 1
assert n == 142
R = {i: lab["görkan"][i] & A[i] for i in ids}
# her etiketleyicinin her çifti ya A'da ya 'wrong'da olmalı (eksik karar yok)
for who, d in lab.items():
    for i in ids:
        assert d[i] <= A[i] | wrong[i], f"{who} {i}: kararsız çift"


def scores(d, idx):
    tp_a = sum(len(d[i] & A[i]) for i in idx)
    npred = sum(len(d[i]) for i in idx)
    tp_r, nr = sum(len(d[i] & R[i]) for i in idx), sum(len(R[i]) for i in idx)
    p, r = tp_a / max(npred, 1), tp_r / max(nr, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


out = ["ADIM 22 — KABUL EDİLEBİLİR ETİKET ÖLÇÜSÜ: 100 yorum (9500-9599), Görkan'ın 142 k/y kararı (kaynak gizli) + 85 otomatik kabul.",
       f"A = {sum(map(len, A.values()))} kabul çifti; y = {sum(map(len, wrong.values()))}; R (Görkan'ın onayladığı kendi çiftleri) = "
       f"{sum(map(len, R.values()))} / {sum(len(v) for v in lab['görkan'].values())}",
       "", "BEKLENTİ (PLAN §3, kararlardan önce): model kabul F1 0.80-0.88; Claude 0.82-0.90; fark −0.05 ile +0.02;",
       "  modelin y'leri en çok performans/kalite; Görkan kendi çiftlerinin 3-6'sını eler.", "",
       f"{'etiketleyici':<12}{'çift':>6}{'kabul P':>9}{'zorunlu R':>11}{'kabul F1':>10}{'y çift':>8}   (Adım 21 katı çift F1)"]
strict = {"model": 0.636, "claude": 0.664, "görkan": 1.0}
for who, d in lab.items():
    p, r, f1 = scores(d, ids)
    ny = sum(len(d[i] & wrong[i]) for i in ids)
    out.append(f"{who:<12}{sum(map(len, d.values())):>6}{p:>9.3f}{r:>11.3f}{f1:>10.3f}{ny:>8}   ({strict[who]:.3f})")

out.append("\nY (yanlış) çiftlerin konuya göre dağılımı — kimde:")
for who in lab:
    c = Counter(k for i in ids for k, _ in lab[who][i] & wrong[i])
    out.append(f"  {who:<8}" + ", ".join(f"{k} {v}" for k, v in c.most_common()))
out.append("Y çiftlerde tür (konu yanlış = o konu kabul kümesinde hiç yok; duygu yanlış = konu kabul, duygu değil):")
for who in lab:
    t = Counter("duygu yanlış" if any(k == kk for kk, _ in A[i]) else "konu yanlış"
                for i in ids for k, _ in lab[who][i] & wrong[i])
    out.append(f"  {who:<8}{dict(t)}")
out.append("Zorunlu (R) çiftlerden kaçırılanlar, konuya göre:")
for who in ("model", "claude"):
    c = Counter(k for i in ids for k, _ in R[i] - lab[who][i])
    out.append(f"  {who:<8}" + ", ".join(f"{k} {v}" for k, v in c.most_common()))

rng = np.random.default_rng(22)
arr = np.array(ids)
diff = lambda s: scores(lab["model"], s)[2] - scores(lab["claude"], s)[2]
boot = np.array([diff(list(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(2000)])
lo, hi = np.quantile(boot, [0.025, 0.975])
pt = diff(ids)
out.append(f"\nANA — kabul F1 model − Claude: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] → "
           + ("0'ı içeriyor: ayırt edilemiyor" if lo <= 0 <= hi else ("0'ı DIŞLIYOR: model geride" if hi < 0 else "0'ı DIŞLIYOR: model ileride")))
open(os.path.join(HERE, "log_evaluate.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
