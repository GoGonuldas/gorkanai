"""
ADIM 16.5 - BERT konu modelinin test hata kovaları. EĞİTİM YOK, AYAR YOK, test sayıları yeniden hesaplanmıyor:
sadece dondurulmuş tahminlerin (test_probs.npz, 3 tohum ortalaması, eşik 0.60) hata analizi.

Buradan model/eşik/etiket değişikliği TÜRETİLMEZ; çıkan her fikir "test görüldü -> yeni test gerekir" notuyla yazılır.
Test altınına dokunulmaz; şüpheli bulunanlar sadece listelenir.

Elle sınıflandırma: test_bert_errors_manual.csv (tur, konu, id, kova). Dosya yoksa şablonu yazılır.
  kalite FP kovaları : altın şüpheli | cömert Q | model hatası
  FN kovaları        : örtük | açık ifade (kelime var, model kaçırdı) | altın şüpheli
  satıcı/diğer FP    : altın şüpheli | model hatası
Kullanım: python error_buckets.py  -> log_error_buckets.txt
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

import evaluate as E
from train import ASPECTS, HERE, OLD_LABELS, targets
from baseline import PATTERNS, split_clauses, turkish_lower  # noqa: E402
from save_aspect_labels import parse  # noqa: E402

MANUAL = os.path.join(HERE, "test_bert_errors_manual.csv")
STEP15_FN = os.path.join(HERE, "..", "step15_aspect", "test_fn_manual.csv")
SHOW = "--liste" in sys.argv   # elle sınıflandırma için tam metin listesi

cfg = json.load(open(E.FROZEN))
old = pd.read_csv(OLD_LABELS, keep_default_na=False)
test = old[old["split"] == "test"].reset_index(drop=True)
npz = np.load(os.path.join(HERE, "test_probs.npz"))
assert (npz["ids"] == test["id"].to_numpy()).all()
p = np.mean([npz[f"s{s}"] for s in cfg["seeds"]], axis=0)
y, kw, pred = targets(test), E.keyword_matrix(test["text"]), (p >= cfg["threshold"]).astype(float)
cache = E.clause_cache(test)
out = [f"ADIM 16.5 — BERT test hata kovaları (dondurulmuş tahminler: eşik {cfg['threshold']}, tohumlar {cfg['seeds']})\n"]


def sentiment_errors(pr):
    """Konu doğru, duygu yanlış: (konu, id, tür). tür: yedek kural (konunun kelimesi hiçbir cümlecikte yok ->
    duygu tüm yorumdan) | bölme hatası (cümlecikte zıt duygulu başka altın konu) | V2b hatası."""
    errs = []
    for i, ((pd_, used), (_, r)) in enumerate(zip(E.e2e_preds(test, pr, cache), test.iterrows())):
        g = {a: r[a] for a in ASPECTS if r[a]}
        clauses = split_clauses(r["text"], E.SPLIT)
        for a in pd_:
            if a in g and g[a] != pd_[a]:
                fallback = not any(PATTERNS[a].search(c) for c in clauses)
                conflict = [b for b in g if b != a and g[b] != g[a]
                            and (fallback or any(PATTERNS[b].search(c) for c in used[a]))]
                kind = ("yedek kural" if fallback and len(clauses) > 1 else
                        "bölme hatası" if conflict else "V2b hatası")
                errs.append((a, r["id"], kind, g[a], pd_[a], int(r["karisik"])))
    return errs


# --- 1. Sayım ---
out.append("1) SAYIM (test) — anahtar kelime V2 vs BERT")
out.append(f"{'':<18}{'konu FP':>9}{'konu FN':>9}{'duygu hatası':>14}{'  (yedek kural / bölme / V2b)'}")
sent = {}
for name, pr in (("anahtar kelime", kw), ("BERT", pred)):
    sent[name] = sentiment_errors(pr)
    c = Counter(k for _, _, k, *_ in sent[name])
    out.append(f"{name:<18}{int(((1 - y) * pr).sum()):>9}{int((y * (1 - pr)).sum()):>9}{len(sent[name]):>14}"
               f"  ({c['yedek kural']} / {c['bölme hatası']} / {c['V2b hatası']})")
out.append(f"{'konu':<12}" + "".join(f"{n:>22}" for n in ("FP kelime → BERT", "FN kelime → BERT", "duygu h. kelime → BERT")))
for k, a in enumerate(ASPECTS):
    out.append(f"{a:<12}{f'{int(((1 - y[:, k]) * kw[:, k]).sum())} → {int(((1 - y[:, k]) * pred[:, k]).sum())}':>22}"
               f"{f'{int((y[:, k] * (1 - kw[:, k])).sum())} → {int((y[:, k] * (1 - pred[:, k])).sum())}':>22}"
               f"{str(sum(x[0] == a for x in sent['anahtar kelime'])) + ' → ' + str(sum(x[0] == a for x in sent['BERT'])):>22}")

# --- 1b. Geçiş tablosu: anahtar kelimenin FN'leri ---
idx = {i: n for n, i in enumerate(test["id"])}
s15 = pd.read_csv(STEP15_FN, keep_default_na=False)
s15["bert"] = [bool(pred[idx[r.id], ASPECTS.index(r.konu)]) for r in s15.itertuples()]
out.append("\n1b) GEÇİŞ — anahtar kelimenin 96 FN'si (Adım 15.4 elle kovaları): BERT kaçını düzeltti")
for name, m in [("örtük", s15.kova == "örtük"), ("kelime eksik (toplam)", s15.kova == "kelime eksik"),
                ("  yeni kelime", s15.alt == "kelime"), ("  yazım varyantı", s15.alt == "yazım"), ("  ek/yumuşama", s15.alt == "ek")]:
    out.append(f"  {name:<24}{int(s15[m].bert.sum()):>3} / {int(m.sum()):<3} düzeldi ({s15[m].bert.mean():.0%}), {int((~s15[m].bert).sum())} kaldı")
fp_fixed = int(((1 - y) * kw * (1 - pred)).sum())
out.append(f"  anahtar kelimenin 67 FP'sinden BERT'te kalmayan: {fp_fixed}; BERT'in YENİ FP'si (kelime yok, BERT var): "
           f"{int(((1 - y) * (1 - kw) * pred).sum())}; ortak FP: {int(((1 - y) * kw * pred).sum())}")
new_fn = [(a, test.at[i, "id"]) for i in range(len(test)) for k, a in enumerate(ASPECTS) if y[i, k] and kw[i, k] and not pred[i, k]]
out.append(f"  BERT'in YENİ FN'si (kelime tutuyordu, BERT kaçırdı): {len(new_fn)} / 263 -> "
           + ", ".join(f"{a} {c}" for a, c in Counter(a for a, _ in new_fn).most_common()))

# --- Elle sınıflandırılacak hata listesi ---
rows = []
for i in range(len(test)):
    for k, a in enumerate(ASPECTS):
        if pred[i, k] and not y[i, k]:
            rows.append(("FP", a, test.at[i, "id"], round(float(p[i, k]), 2), int(kw[i, k])))
        elif y[i, k] and not pred[i, k]:
            rows.append(("FN", a, test.at[i, "id"], round(float(p[i, k]), 2), int(kw[i, k])))
errors = pd.DataFrame(rows, columns=["tur", "konu", "id", "olasilik", "kelime_tutuyor"])
errors = errors.merge(test[["id", "raw", "text"]], on="id")
if os.path.exists(MANUAL):
    man = pd.read_csv(MANUAL, keep_default_na=False)
    errors = errors.merge(man[["tur", "konu", "id", "kova", "aciklama"]], on=["tur", "konu", "id"], how="left").fillna("")
else:
    errors["kova"], errors["aciklama"] = "", ""
if SHOW:
    for r in errors.itertuples():
        print(f"{r.tur} {r.konu} {r.id} p={r.olasilik} kw={r.kelime_tutuyor} altın={r.raw} | {r.text}")
    sys.exit()
errors[["tur", "konu", "id", "olasilik", "kelime_tutuyor", "kova", "aciklama", "raw", "text"]].to_csv(MANUAL, index=False)

# --- 2-4. Elle kovalar ---
out.append("\n2-4) ELLE KOVALAR (test_bert_errors_manual.csv) — BERT'in tüm konu hataları")
out.append(f"{'konu':<12}{'tür':<4}{'n':>4}  kovalar")
for a in ASPECTS:
    for t in ("FP", "FN"):
        e = errors[(errors.konu == a) & (errors.tur == t)]
        if len(e):
            out.append(f"{a:<12}{t:<4}{len(e):>4}  " + ", ".join(f"{k or 'SINIFLANMADI'} {c}" for k, c in e.kova.value_counts().items()))
for t in ("FP", "FN"):
    e = errors[errors.tur == t]
    out.append(f"TOPLAM {t}: {len(e)} -> " + ", ".join(f"{k or 'SINIFLANMADI'} {c}" for k, c in e.kova.value_counts().items()))
sus = errors[errors.kova == "altın şüpheli"]
out.append(f"\nALTIN ŞÜPHELİ listesi (test altınına DOKUNULMADI; {len(sus)} çift):")
for r in sus.itertuples():
    out.append(f"  {r.id} [{r.tur} {r.konu}] altın={r.raw} | {r.aciklama} | {r.text[:110]}")

# --- 5. Duygu tarafı ---
out.append("\n5) DUYGU HATALARI (konu doğru, duygu yanlış) — BERT konularıyla")
b = sent["BERT"]
c = Counter(k for _, _, k, *_ in b)
n_pairs = int((y * pred).sum())
fb_total = sum(1 for i, (_, r) in enumerate(test.iterrows()) for k, a in enumerate(ASPECTS)
               if y[i, k] and pred[i, k] and not any(PATTERNS[a].search(cl) for cl in split_clauses(r["text"], E.SPLIT)))
fb_err = sum(1 for a, i, kind, *_ in b
             if not any(PATTERNS[a].search(cl) for cl in split_clauses(test.loc[test.id == i, "text"].iloc[0], E.SPLIT)))
out.append(f"  doğru bulunan konu: {n_pairs}; duygu hatası {len(b)} ({len(b) / n_pairs:.1%})")
out.append(f"  tür: yedek kural (çok cümlecikli yorumda duygu TÜM yorumdan) {c['yedek kural']}, bölme hatası {c['bölme hatası']}, V2b hatası {c['V2b hatası']}")
out.append(f"  konunun kelimesi hiçbir cümlecikte geçmeyen (örtük) doğru konular: {fb_total}; bunlarda duygu hatası {fb_err} "
           f"({fb_err / max(fb_total, 1):.1%}) | kelimesi geçenler: {n_pairs - fb_total}, hata {len(b) - fb_err} "
           f"({(len(b) - fb_err) / max(n_pairs - fb_total, 1):.1%})")
out.append(f"  karışık (altında karisik=1) yorumlardaki hata: {sum(x[5] for x in b)}; altın nötr olup kaçan: {sum(x[3] == 'nötr' for x in b)}")

# --- 6. Elmayla elma ---
human = pd.read_csv(os.path.join(os.path.dirname(OLD_LABELS), "human_blind_20.csv"), keep_default_na=False)
human = human[human["split"] == "test"]
ii = [idx[i] for i in human["id"]]
sub = test.iloc[ii].reset_index(drop=True)
out.append("\n6) ELMAYLA ELMA — testteki 10 kör yorum (n=10, SADECE NİTEL): Görkan | altın | anahtar kelime | BERT")
fmt = lambda d: ",".join(f"{a[:3]}:{v[:3]}" for a, v in sorted(d.items())) or "-"
mk = [q for q, _ in E.e2e_preds(sub, kw[ii], cache)]
mb = [q for q, _ in E.e2e_preds(sub, pred[ii], cache)]
for r, g, k_, b_ in zip(sub.itertuples(), [parse(x)[0] for x in human["raw_gorkan"]], mk, mb):
    gold_ = {a: getattr(r, a) for a in ASPECTS if getattr(r, a)}
    out.append(f"  {r.id} G[{fmt(g)}] A[{fmt(gold_)}] K[{fmt(k_)}] B[{fmt(b_)}] | {r.text[:90]}")

open(os.path.join(HERE, "log_error_buckets.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
