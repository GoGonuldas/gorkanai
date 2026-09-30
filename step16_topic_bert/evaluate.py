"""
ADIM 16 - Değerlendirme: BERT konu modeli vs Adım 15 anahtar kelime çizgisi (V2, dondurulmuş).

  python evaluate.py val    -> log_val.txt + frozen_config.json (epoch, global eşik, tohumlar)
  python evaluate.py test   -> log_test.txt — BİR KEZ, frozen_config.json ile; log varsa çalışmaz.

PLAN.md'de önceden sabitlenen kurallar:
  - epoch: tohum ortalaması val micro-F1 (eşik 0.5) en yüksek olan aday;
  - teste giren model: o epoch'ta 3 tohumun olasılık ortalaması;
  - ANA SONUÇ: tek global eşik (ortalama olasılıklarda val micro-F1, ızgara 0.05-0.95);
    konu başına eşik sadece "iyimser" satır;
  - örtük ölçü: altın (yorum, konu) çiftleri "V2 anahtar kelimesi tutuyor / tutmuyor" diye ayrılır, recall ayrı;
  - uçtan uca: BERT konuları + Adım 15'in dondurulmuş duygu hattı (bölme A, nötr kapalı). Konunun anahtar kelimesi
    hiçbir cümlecikte geçmiyorsa (örtük konu) Adım 15'teki aynı yedek kural: tüm yorum tek cümlecik.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from train import ASPECTS, EPOCHS, HERE, OLD_LABELS, SEEDS, load_data, predict as bert_predict, targets
from baseline import PATTERNS, clause_logits, decide, detect, metrics, split_clauses, turkish_lower  # noqa: E402

GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
FROZEN = os.path.join(HERE, "frozen_config.json")
SPLIT, MODE = "A", "b"   # Adım 15 chosen_config.json


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def det(y, pred):
    """Konu tespiti: konu başına (n, P, R, F1) + micro P/R/F1 + macro F1."""
    tp, fp, fn = (y * pred).sum(0), ((1 - y) * pred).sum(0), (y * (1 - pred)).sum(0)
    rows = {a: (int(tp[k] + fn[k]), tp[k] / max(tp[k] + fp[k], 1), tp[k] / max(tp[k] + fn[k], 1), f1(tp[k], fp[k], fn[k]))
            for k, a in enumerate(ASPECTS)}
    P, R = tp.sum() / max(tp.sum() + fp.sum(), 1), tp.sum() / max(tp.sum() + fn.sum(), 1)
    return rows, {"P": P, "R": R, "micro": f1(tp.sum(), fp.sum(), fn.sum()), "macro": np.mean([r[3] for r in rows.values()])}


def best_threshold(y, p):
    scores = [det(y, (p >= t).astype(float))[1]["micro"] for t in GRID]
    return float(GRID[int(np.argmax(scores))]), max(scores)


def per_topic_thresholds(y, p):
    return np.array([GRID[int(np.argmax([f1(*_counts(y[:, k], p[:, k] >= t)) for t in GRID]))] for k in range(len(ASPECTS))])


def _counts(y, pred):
    return (y * pred).sum(), ((1 - y) * pred).sum(), (y * (1 - pred)).sum()


def keyword_matrix(texts):
    found = [detect(t) for t in texts]
    return np.array([[float(a in f) for a in ASPECTS] for f in found])


def implicit_recall(y, kw, pred):
    """Altın çiftlerde recall: anahtar kelimenin tuttuğu ve tutmadığı alt kümelerde ayrı."""
    hit, miss = (y * kw).astype(bool), (y * (1 - kw)).astype(bool)
    return (pred[hit].mean() if hit.sum() else float("nan"), int(hit.sum()),
            pred[miss].mean() if miss.sum() else float("nan"), int(miss.sum()))


def end_to_end(df, pred, cache):
    """BERT (veya anahtar kelime) konuları + Adım 15 duygu hattı -> baseline.metrics özeti."""
    preds = []
    for text, row in zip(df["text"], pred):
        clauses = split_clauses(text, SPLIT)
        out, used = {}, {}
        for k, a in enumerate(ASPECTS):
            if row[k]:
                cl = [c for c in clauses if PATTERNS[a].search(c)] or [turkish_lower(text)]
                out[a], used[a] = decide([cache[c] for c in cl], MODE), cl
        preds.append((out, used))
    return metrics(df.reset_index(drop=True), preds)[1]


def clause_cache(df):
    return clause_logits({c for t in df["text"] for c in split_clauses(t, SPLIT)} | {turkish_lower(t) for t in df["text"]})


def topic_table(y, preds, names):
    lines = [f"{'konu':<12}{'n':>5}" + "".join(f"{n + ' P/R/F1':>24}" for n in names)]
    all_rows = [det(y, p)[0] for p in preds]
    for a in ASPECTS:
        lines.append(f"{a:<12}{all_rows[0][a][0]:>5}" + "".join(
            f"{f'{r[a][1]:.2f}/{r[a][2]:.2f}/{r[a][3]:.2f}':>24}" for r in all_rows))
    return lines


def summary_line(name, y, pred, kw, df=None, cache=None):
    _, s = det(y, pred)
    rh, nh, rm, nm = implicit_recall(y, kw, pred)
    line = (f"{name:<34}{s['P']:>7.3f}{s['R']:>7.3f}{s['micro']:>8.3f}{s['macro']:>8.3f}"
            f"{rh:>10.3f}{rm:>10.3f}")
    if df is not None:
        e = end_to_end(df, pred, cache)
        line += f"{e['uc_F1_micro']:>10.3f}{e['uc_F1_macro']:>8.3f}"
    return line


HEADER = (f"{'':<34}{'P':>7}{'R':>7}{'F1mic':>8}{'F1mac':>8}{'R(kelime+)':>10}{'R(kelime-)':>10}"
          f"{'uçtan uca':>10}{'macro':>8}")


def run_val():
    train, val = load_data()
    y, kw = targets(val), keyword_matrix(val["text"])
    npz = np.load(os.path.join(HERE, "val_probs.npz"))
    assert (npz["ids"] == val["id"].to_numpy()).all()
    out = [f"ADIM 16 — VAL ({len(val)} yorum: eski {sum(val.origin == 'eski')} + yeni {sum(val.origin == 'yeni')}), "
           f"eğitim {len(train)} yorum. Test'e dokunulmadı.\n",
           "1) EPOCH SEÇİMİ — val micro-F1, eşik 0.5 (tohum başına, ortalama, min-max, 3 tohum ortalaması olasılık)"]
    means = {}
    for e in EPOCHS:
        per = [det(y, (npz[f"s{s}_e{e}"] >= 0.5).astype(float))[1]["micro"] for s in SEEDS]
        ens = np.mean([npz[f"s{s}_e{e}"] for s in SEEDS], axis=0)
        means[e] = np.mean(per)
        out.append(f"  epoch {e:>2}: " + " ".join(f"{v:.3f}" for v in per) + f" | ort. {np.mean(per):.3f} "
                   f"(min-max {min(per):.3f}-{max(per):.3f}) | ortalama olasılık {det(y, (ens >= 0.5).astype(float))[1]['micro']:.3f}")
    epoch = max(EPOCHS, key=lambda e: means[e])
    p = np.mean([npz[f"s{s}_e{epoch}"] for s in SEEDS], axis=0)
    thr, _ = best_threshold(y, p)
    out.append(f"  SEÇİLEN epoch = {epoch}" + ("  (ızgaranın üst sınırı!)" if epoch == max(EPOCHS) else ""))
    out.append("\n2) GLOBAL EŞİK (3 tohum ortalaması olasılık, val micro-F1): "
               + " ".join(f"{t:.2f}:{det(y, (p >= t).astype(float))[1]['micro']:.3f}" for t in GRID))
    out.append(f"  SEÇİLEN eşik = {thr:.2f}")
    for name, m in (("eski", val.origin == "eski"), ("yeni", val.origin == "yeni")):
        t, s = best_threshold(y[m.to_numpy()], p[m.to_numpy()])
        out.append(f"  tanı: sadece {name} val'de seçilseydi eşik {t:.2f} (F1 {s:.3f}); "
                   f"seçilen {thr:.2f} ile o kümede F1 {det(y[m.to_numpy()], (p[m.to_numpy()] >= thr).astype(float))[1]['micro']:.3f}")

    cache = clause_cache(val)
    pred = (p >= thr).astype(float)
    out += ["\n3) ANA TABLO (val 300)", HEADER,
            summary_line("Anahtar kelime V2 (Adım 15)", y, kw, kw, val, cache),
            summary_line(f"BERT 3 tohum ort., eşik {thr:.2f} (ANA)", y, pred, kw, val, cache)]
    per_seed = [det(y, (npz[f"s{s}_e{epoch}"] >= thr).astype(float))[1] for s in SEEDS]
    out.append(f"  tek tohumlar (aynı eşik): micro " + " ".join(f"{s['micro']:.3f}" for s in per_seed)
               + f" -> ort. {np.mean([s['micro'] for s in per_seed]):.3f}, min-max "
               f"{min(s['micro'] for s in per_seed):.3f}-{max(s['micro'] for s in per_seed):.3f} | macro "
               + " ".join(f"{s['macro']:.3f}" for s in per_seed))
    tthr = per_topic_thresholds(y, p)
    out.append(summary_line("BERT konu başına eşik (İYİMSER)", y, (p >= tthr).astype(float), kw, val, cache))
    out.append("  konu başına eşikler: " + ", ".join(f"{a} {t:.2f}" for a, t in zip(ASPECTS, tthr)))
    union = np.maximum(pred, kw)
    out.append(summary_line("BERT VEYA anahtar kelime (bilgi)", y, union, kw, val, cache))
    out.append("  R(kelime+) / R(kelime-): altın (yorum, konu) çiftlerinde recall; V2 anahtar kelimesinin tuttuğu / tutmadığı "
               f"alt küme (n = {implicit_recall(y, kw, pred)[1]} / {implicit_recall(y, kw, pred)[3]}).")

    out += ["\n4) KONU BAŞINA (val 300)"] + topic_table(y, [kw, pred], ["anahtar kelime", "BERT"])
    out.append("  örtük (kelime-) recall, konu başına BERT: " + ", ".join(
        f"{a} {int((pred[:, k] * y[:, k] * (1 - kw[:, k])).sum())}/{int((y[:, k] * (1 - kw[:, k])).sum())}"
        for k, a in enumerate(ASPECTS)))

    out.append("\n5) ESKİ VAL 100 (Adım 15 oturumunun altını) vs YENİ VAL 200 (bu oturumun etiketleri)")
    for name, m in (("eski 100", (val.origin == "eski").to_numpy()), ("yeni 200", (val.origin == "yeni").to_numpy())):
        sub = val[m]
        out += [f"--- {name} ---", HEADER,
                summary_line("Anahtar kelime V2", y[m], kw[m], kw[m], sub, cache),
                summary_line("BERT (ANA)", y[m], pred[m], kw[m], sub, cache)]
        out += topic_table(y[m], [kw[m], pred[m]], ["anahtar kelime", "BERT"])

    json.dump({"epoch": int(epoch), "threshold": thr, "seeds": list(SEEDS),
               "per_topic_thresholds_optimistic": dict(zip(ASPECTS, map(float, tthr)))},
              open(FROZEN, "w"), indent=1)
    out.append(f"\nDONDURULAN -> frozen_config.json: epoch {epoch}, global eşik {thr:.2f}, tohumlar {list(SEEDS)} (olasılık ortalaması)")
    open(os.path.join(HERE, "log_val.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


def run_test():
    log = os.path.join(HERE, "log_test.txt")
    assert not os.path.exists(log), "test zaten ölçüldü (log_test.txt var) — ikinci ölçüm yok"
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from train import DEVICE
    cfg = json.load(open(FROZEN))
    old = pd.read_csv(OLD_LABELS, keep_default_na=False)
    test = old[old["split"] == "test"].reset_index(drop=True)
    assert len(test) == 200
    probs = []
    for s in cfg["seeds"]:
        d = os.path.join(HERE, f"model_s{s}_e{cfg['epoch']}")
        tok = AutoTokenizer.from_pretrained(d)
        model = AutoModelForSequenceClassification.from_pretrained(d).to(DEVICE)
        probs.append(bert_predict(model, tok, test["text"].tolist()))
    np.savez(os.path.join(HERE, "test_probs.npz"), ids=test["id"].to_numpy(), **{f"s{s}": q for s, q in zip(cfg["seeds"], probs)})
    p = np.mean(probs, axis=0)
    y, kw, cache = targets(test), keyword_matrix(test["text"]), clause_cache(test)
    pred = (p >= cfg["threshold"]).astype(float)
    tthr = np.array([cfg["per_topic_thresholds_optimistic"][a] for a in ASPECTS])
    out = [f"ADIM 16 — TEST (200 yorum), BİR KEZ. DONDURULAN: {json.dumps({k: cfg[k] for k in ('epoch', 'threshold', 'seeds')})}",
           "Kıyas (Adım 15.4): konu F1 micro/macro 0.763/0.722, uçtan uca 0.653/0.616.\n", HEADER,
           summary_line("Anahtar kelime V2 (Adım 15.4)", y, kw, kw, test, cache),
           summary_line(f"BERT 3 tohum ort., eşik {cfg['threshold']:.2f} (ANA)", y, pred, kw, test, cache)]
    per_seed = [det(y, (q >= cfg["threshold"]).astype(float))[1] for q in probs]
    out.append("  tek tohumlar (aynı eşik): micro " + " ".join(f"{s['micro']:.3f}" for s in per_seed)
               + " | macro " + " ".join(f"{s['macro']:.3f}" for s in per_seed))
    out.append(summary_line("BERT konu başına eşik (İYİMSER)", y, (p >= tthr).astype(float), kw, test, cache))
    out.append(summary_line("BERT VEYA anahtar kelime (bilgi)", y, np.maximum(pred, kw), kw, test, cache))
    rh, nh, rm, nm = implicit_recall(y, kw, pred)
    out.append(f"  R(kelime+) n={nh}, R(kelime-) n={nm} (anahtar kelime çizgisi kelime-'de tanım gereği 0).")
    out += ["\nKONU BAŞINA (test)"] + topic_table(y, [kw, pred], ["anahtar kelime", "BERT"])
    out.append("  örtük (kelime-) recall, konu başına BERT: " + ", ".join(
        f"{a} {int((pred[:, k] * y[:, k] * (1 - kw[:, k])).sum())}/{int((y[:, k] * (1 - kw[:, k])).sum())}"
        for k, a in enumerate(ASPECTS)))
    open(log, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    {"val": run_val, "test": run_test}[sys.argv[1]]()
