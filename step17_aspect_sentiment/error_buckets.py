"""
ADIM 17.5 - Konuya koşullu duygu modelinin test hata kovaları. EĞİTİM YOK, AYAR YOK, test sayıları yeniden
hesaplanmıyor: sadece dondurulmuş tahminlerin (test17_probs.npz; v2b, epoch 8, tohum 0/1/2 ortalaması) analizi.
Eski hat (Adım 15: bölme A + V2b, nötr kapalı) deterministik; aynı fonksiyonla yeniden üretilir (log_test ile aynı).

Buradan model/ayar/etiket değişikliği TÜRETİLMEZ; çıkan her fikir "test görüldü -> yeni test gerekir" notuyla yazılır.
Test altınlarına dokunulmaz; şüpheli bulunanlar sadece listelenir. Bu adımdan sonra yeni test (6000-6199) "okunmuş" sayılır.

Kapsam (ana: yeni test, gorkanai-1e altını, altın konular verilmiş, (i) ölçüsü):
  1) YENİ modelin yanlış bildiği tüm çiftler, elle kovalı
  2) iki hattın ayrıştığı çiftler: YENİ doğru / eski doğru, aynı kovalarla
  3) örtük çiftlerde eski doğru / YENİ yanlış (test + val; val sadece desenin tekrarı için)
  4) zıt duygulu yorumlarda YENİ'nin yanlışları
  5) Görkan'ın kör 20'si: modelin Görkan'la ayrıştığı çiftler, aynı çiftte 1e ne demiş

Elle sınıflandırma: test17_errors_manual.csv (tur, konu, id, ..., kova, aciklama, text). Kovalar:
  altın nötr | karışık | başka konunun duygusu | örtük/dolaylı duygu | olumsuzlama/ironi | altın şüpheli | diğer model hatası
Kullanım: python error_buckets.py --liste  (elle sınıflandırma için tam liste)
          python error_buckets.py          -> log_error_buckets.txt
"""

import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

import evaluate as E
from train import ASPECTS, HERE, load_data, pairs
from baseline import LABELS, PATTERNS, clause_logits, split_clauses, turkish_lower  # noqa: E402
from save_aspect_labels import parse  # noqa: E402

MANUAL = os.path.join(HERE, "test17_errors_manual.csv")
SHOW = "--liste" in sys.argv
KOVALAR = ["altın nötr", "karışık", "başka konunun duygusu", "örtük/dolaylı duygu", "olumsuzlama/ironi",
           "altın şüpheli", "diğer model hatası"]
cfg = E.json.load(open(E.FROZEN))


def baseline_scores(texts):
    """Adım 15 hattı: (duygu, P(pozitif) nötr kapalı, konunun kelimesi bir cümlecikte geçiyor mu) — E.baseline_matrix ile aynı karar."""
    cache = clause_logits({c for t in texts for c in split_clauses(t, E.SPLIT)} | {turkish_lower(t) for t in texts})
    pos, neg = LABELS.index("pozitif"), LABELS.index("negatif")
    P, X = np.zeros((len(texts), len(ASPECTS))), np.zeros((len(texts), len(ASPECTS)), bool)
    for i, t in enumerate(texts):
        clauses = split_clauses(t, E.SPLIT)
        for j, a in enumerate(ASPECTS):
            cl = [c for c in clauses if PATTERNS[a].search(c)]
            X[i, j] = bool(cl)
            lg = np.array([cache[c] for c in (cl or [turkish_lower(t)])], dtype=float)[:, [neg, pos]]
            pr = np.exp(lg - lg.max(1, keepdims=True))
            pr /= pr.sum(1, keepdims=True)
            s = pr.sum(0)
            P[i, j] = s[1] / s.sum()
    return np.where(P > 0.5, "pozitif", "negatif"), P, X


# --- yeni test, 1e altını ---
texts = pd.read_csv(os.path.join(HERE, "test17_set.csv"))[["id", "text"]]
npz = np.load(os.path.join(HERE, "test17_probs.npz"))
assert (npz["ids"] == texts["id"].to_numpy()).all()
p_new = npz["sent_pos"]
new = np.where(p_new >= 0.5, "pozitif", "negatif")
base, p_base, explicit = baseline_scores(texts["text"].tolist())
base_chk, exp_chk = E.baseline_matrix(texts["text"].tolist())
assert (base_chk == base).all() and (exp_chk == explicit).all(), "eski hat yeniden üretilemedi"
d1, dm = E.gold_df("1e", texts), E.gold_df("macmini", texts)
g, mixed = E.gold_matrix(d1)
gm = E.gold_matrix(dm)[0]
has = g != ""
opp = np.array([{"pozitif", "negatif"} <= set(r[r != ""]) for r in g])
ok_n, ok_b = (new == g) & has, (base == g) & has
out = [f"ADIM 17.5 — duygu modeli test hata kovaları (dondurulmuş: {cfg['init']}, epoch {cfg['epoch']}, tohumlar {cfg['seeds']}).",
       "Eğitim/ayar yok, test sayıları yeniden hesaplanmadı. Bu adımdan sonra yeni test (6000-6199) OKUNMUŞ sayılır.\n",
       f"Yeni test, 1e altını: {int(has.sum())} çift | YENİ doğru {int(ok_n.sum())} ({ok_n.sum() / has.sum():.3f}), "
       f"eski doğru {int(ok_b.sum())} ({ok_b.sum() / has.sum():.3f}) — log_test.txt ile aynı olmalı (0.887 / 0.868)"]
assert round(ok_n.sum() / has.sum(), 3) == 0.887 and round(ok_b.sum() / has.sum(), 3) == 0.868


def raw_of(df, i):
    return ",".join(f"{a}:{df.at[i, a]}" for a in ASPECTS if df.at[i, a]) + (f" karışık={df.at[i, 'karisik_konular']}" if df.at[i, "karisik_konular"] else "")


rows = []
for i in range(len(texts)):
    for j, a in enumerate(ASPECTS):
        if not has[i, j] or (ok_n[i, j] and ok_b[i, j]):
            continue
        tur = "ikisi yanlış" if not ok_n[i, j] and not ok_b[i, j] else "YENİ yanlış, eski doğru" if not ok_n[i, j] else "YENİ doğru, eski yanlış"
        rows.append(("test " + tur, a, int(texts.at[i, "id"]), round(float(p_new[i, j]), 2), round(float(p_base[i, j]), 2),
                     g[i, j], int(not explicit[i, j]), int(opp[i]), raw_of(d1, i), dm.at[i, "raw"], texts.at[i, "text"]))

# --- val: örtük çiftlerde eski doğru / YENİ yanlış (sadece desen) ---
_, val = load_data()
val = val.reset_index(drop=True)
vp = pairs(val)
vz = np.load(os.path.join(HERE, "val_probs.npz"), allow_pickle=True)
assert (vz["ids"] == vp["id"].to_numpy()).all() and (vz["konu"] == vp["konu"].to_numpy()).all()
vp["p_new"] = np.mean([vz[f"{cfg['init']}_s{s}_e{cfg['epoch']}"] for s in cfg["seeds"]], axis=0)
vb, vpb, vx = baseline_scores(val["text"].tolist())
vg = E.gold_matrix(val)[0]
vopp = np.array([{"pozitif", "negatif"} <= set(r[r != ""]) for r in vg])
vp["j"] = vp["konu"].map(ASPECTS.index)
vp["base"] = vb[vp["row"], vp["j"]]
vp["p_base"] = vpb[vp["row"], vp["j"]]
vp["ortuk"] = ~vx[vp["row"], vp["j"]]
vp["opp"] = vopp[vp["row"]]
vp["new"] = np.where(vp["p_new"] >= 0.5, "pozitif", "negatif")
vok_n, vok_b = vp["new"] == vp["duygu"], vp["base"] == vp["duygu"]
for r in vp[vp.ortuk & vok_b & ~vok_n].itertuples():
    rows.append(("val örtük YENİ yanlış, eski doğru", r.konu, int(r.id), round(float(r.p_new), 2), round(float(r.p_base), 2),
                 r.duygu, 1, int(r.opp), raw_of(val, r.row), "", r.text))

cols = ["tur", "konu", "id", "p_yeni", "p_eski", "altin", "ortuk_mu", "zit_duygulu_mu", "altin_konular", "altin_macmini", "text"]
errors = pd.DataFrame(rows, columns=cols)
if os.path.exists(MANUAL):
    man = pd.read_csv(MANUAL, keep_default_na=False)
    errors = errors.merge(man[["tur", "konu", "id", "kova", "aciklama"]], on=["tur", "konu", "id"], how="left").fillna("")
else:
    errors["kova"], errors["aciklama"] = "", ""
if SHOW:
    for r in errors.itertuples():
        print(f"[{r.tur}] {r.konu} {r.id} altın={r.altin} p_yeni={r.p_yeni} p_eski={r.p_eski} örtük={r.ortuk_mu} "
              f"zıt={r.zit_duygulu_mu} | altın {r.altin_konular} | mm {r.altin_macmini} | kova={r.kova}\n    {r.text}")
    sys.exit()
errors[["tur", "konu", "id", "p_yeni", "p_eski", "altin", "ortuk_mu", "zit_duygulu_mu", "kova", "aciklama",
        "altin_konular", "altin_macmini", "text"]].to_csv(MANUAL, index=False)
bad = set(errors.kova) - set(KOVALAR)
assert not bad, f"bilinmeyen/boş kova: {bad}"

T = errors[errors.tur.str.startswith("test")]
new_wrong = T[T.tur.isin(["test ikisi yanlış", "test YENİ yanlış, eski doğru"])]
fixed, broke = T[T.tur == "test YENİ doğru, eski yanlış"], T[T.tur == "test YENİ yanlış, eski doğru"]
assert len(new_wrong) == int((has & ~ok_n).sum())


def table(title, groups):
    out.append(f"\n{title}")
    out.append(f"  {'kova':<26}" + "".join(f"{k:>{max(len(k) + 2, 10)}}" for k in groups))
    for kv in KOVALAR:
        out.append(f"  {kv:<26}" + "".join(f"{int((d.kova == kv).sum()):>{max(len(k) + 2, 10)}}" for k, d in groups.items()))
    out.append(f"  {'TOPLAM':<26}" + "".join(f"{len(d):>{max(len(k) + 2, 10)}}" for k, d in groups.items()))


# 1) + 2)
table("1-2) KOVALAR — yeni test, 1e altını (kova, YENİ'nin hatası için; 'YENİ doğru' sütununda eski hattın hatası için)",
      {"YENİ yanlış (tümü)": new_wrong, "ikisi yanlış": T[T.tur == "test ikisi yanlış"],
       "YENİ doğru/eski yanlış": fixed, "eski doğru/YENİ yanlış": broke})
out.append("  okuma: 'YENİ doğru/eski yanlış' = yeni modelin düzelttiği; 'eski doğru/YENİ yanlış' = yeni modelin bozduğu.")
out.append("  YENİ'nin yanlışları konu başına: " + ", ".join(f"{a} {c}" for a, c in new_wrong.konu.value_counts().items()))
out.append("  YENİ'nin yanlışları, gerçek duygu: " + ", ".join(f"{a} {c}" for a, c in new_wrong.altin.value_counts().items())
           + f" | YENİ pozitif dedi: {int((new_wrong.p_yeni >= 0.5).sum())}, negatif dedi: {int((new_wrong.p_yeni < 0.5).sum())}")
conf = new_wrong[(new_wrong.p_yeni >= 0.9) | (new_wrong.p_yeni <= 0.1)]
out.append(f"  YENİ'nin emin olduğu (p≥0.9 ya da ≤0.1) yanlışlar: {len(conf)} / {len(new_wrong)}")
sus = new_wrong[new_wrong.kova == "altın şüpheli"]
mm_diff = sum(gm[texts.index[texts.id == r.id][0], ASPECTS.index(r.konu)] not in ("", r.altin) for r in new_wrong.itertuples())
mm_none = sum(gm[texts.index[texts.id == r.id][0], ASPECTS.index(r.konu)] == "" for r in new_wrong.itertuples())
out.append(f"  YENİ'nin yanlışlarında macmini altını: aynı konuya farklı duygu {mm_diff}, o konuyu hiç vermemiş {mm_none}")

# 3) örtük
out.append("\n3) ÖRTÜK ÇİFTLER (konunun V2 kelimesi hiçbir cümlecikte yok)")
imp = has & ~explicit
out.append(f"  test: {int(imp.sum())} örtük çift; eski {(ok_b & imp).sum() / imp.sum():.3f} → YENİ {(ok_n & imp).sum() / imp.sum():.3f} | "
           f"eski doğru/YENİ yanlış {int((ok_b & ~ok_n & imp).sum())}, YENİ doğru/eski yanlış {int((ok_n & ~ok_b & imp).sum())}")
vimp = vp.ortuk
out.append(f"  val : {int(vimp.sum())} örtük çift; eski {vok_b[vimp].mean():.3f} → YENİ {vok_n[vimp].mean():.3f} | "
           f"eski doğru/YENİ yanlış {int((vok_b & ~vok_n & vimp).sum())}, YENİ doğru/eski yanlış {int((vok_n & ~vok_b & vimp).sum())}")
vbroke = errors[errors.tur == "val örtük YENİ yanlış, eski doğru"]
tbroke = broke[broke.ortuk_mu == 1]
table("  örtük, eski doğru / YENİ yanlış — kovalar", {"test": tbroke, "val": vbroke})
for name, d in (("test", tbroke), ("val", vbroke)):
    out.append(f"  {name} listesi:")
    for r in d.itertuples():
        out.append(f"    {r.id} {r.konu}:{r.altin} p_yeni {r.p_yeni:.2f} p_eski {r.p_eski:.2f} [{r.kova}] {r.aciklama} | {r.text[:100]}")
for name, d in (("test", tbroke), ("val", vbroke)):
    out.append(f"  {name}: YENİ'nin bu çiftlerde verdiği yön — pozitif {int((d.p_yeni >= 0.5).sum())}, negatif {int((d.p_yeni < 0.5).sum())}; "
               f"zıt duygulu yorumdan {int(d.zit_duygulu_mu.sum())} / {len(d)}")

# 4) zıt duygulu
zw = new_wrong[new_wrong.zit_duygulu_mu == 1]
zopp = has & opp[:, None]
out.append(f"\n4) ZIT DUYGULU YORUMLAR ({int(opp.sum())} yorum, {int(zopp.sum())} çift): eski {(ok_b & zopp).sum() / zopp.sum():.3f} → "
           f"YENİ {(ok_n & zopp).sum() / zopp.sum():.3f}; YENİ'nin yanlışı {len(zw)} -> "
           + ", ".join(f"{k} {c}" for k, c in zw.kova.value_counts().items()))
out.append(f"  zıt duygulu çiftlerde düzelttiği {int(((fixed.zit_duygulu_mu == 1)).sum())}, bozduğu {int((broke.zit_duygulu_mu == 1).sum())}")

# 5) Görkan'ın kör 20'si
human = pd.read_csv(os.path.join(HERE, "human_blind_20.csv"), keep_default_na=False)
out.append("\n5) GÖRKAN'IN KÖR 20'Sİ (n=20, SADECE FİKİR) — Görkan'ın verdiği her (konu, duygu) çifti: YENİ / eski / 1e")
n_pairs = agree = agree_1e = n_1e = 0
for r in human.itertuples():
    i = texts.index[texts.id == r.id][0]
    hg, hmix = parse(",".join(t for t in re.split(r"[,\s]+", r.etiket.strip()) if t))
    for a, s in hg.items():
        j = ASPECTS.index(a)
        n_pairs += 1
        agree += new[i, j] == s
        if g[i, j]:
            n_1e += 1
            agree_1e += g[i, j] == s
        if new[i, j] != s:
            out.append(f"  {r.id} {a}: Görkan {s}{' (karışık)' if a in hmix else ''} | YENİ {new[i, j]} ({p_new[i, j]:.2f}) | "
                       f"eski {base[i, j]} | 1e {g[i, j] or '— (konu yok)'} | {texts.at[i, 'text'][:110]}")
out.append(f"  YENİ–Görkan duygu uyumu: {agree}/{n_pairs} ({agree / n_pairs:.3f}); 1e–Görkan (1e'nin de konu verdiği çiftlerde): "
           f"{agree_1e}/{n_1e} ({agree_1e / max(n_1e, 1):.3f})")
out.append(f"  karşılaştırma: YENİ–1e (aynı 20 yorum, 1e konuları): "
           + (lambda m: f"{int((ok_n & m).sum())}/{int((has & m).sum())} ({(ok_n & m).sum() / (has & m).sum():.3f})")(
               np.isin(texts["id"].to_numpy(), human["id"].to_numpy())[:, None] & has)
           + f"; YENİ–1e tüm test {ok_n.sum() / has.sum():.3f}")

# altın şüpheli listesi
out.append(f"\nALTIN ŞÜPHELİ listesi (test altınına DOKUNULMADI; {len(sus)} çift):")
for r in sus.itertuples():
    out.append(f"  {r.id} {r.konu}:{r.altin} (macmini: {r.altin_macmini}) | {r.aciklama} | {r.text[:100]}")
out.append("\nNOT: buradan çıkan her fikir 'test görüldü -> yeni test gerekir'. Yeni test (6000-6199) artık okunmuş sayılır.")

open(os.path.join(HERE, "log_error_buckets.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
