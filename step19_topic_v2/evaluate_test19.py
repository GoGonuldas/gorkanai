"""
ADIM 19 (4) - Kasa testi 500'ün SON hakkı (PLAN 18 §3 madde 4; PLAN 19 §4): konu modeli v2 (B) vs Adım 16 BERT. BİR KEZ.
MAC MİNİ'DE çalışır (Adım 16, 17 ve 19 modelleri orada).

  python evaluate_test19.py --calib  -> log_calib_dryrun.txt  (KURU ÇALIŞTIRMA: kalibrasyon 100; okunmuş ve artık
                                         EĞİTİMDE olan küme -> sayılar anlamsız, sadece kod kontrolü)
  python evaluate_test19.py test     -> log_test.txt + test19_probs.npz  (BİR KEZ; log varsa durur)

Önceden sabit (PLAN 19 §4 + val sonrası güncellenmiş beklenti, test görülmeden):
  ANA İDDİA: konu F1 macro, v2 − Adım 16, ana altın (fresh). Kazandı = %95 aralık 0'ı dışlar.
  İkincil: konu F1 micro (koruma: aralığın alt sınırı > −0.01), S/G/B F1, uçtan uca çift F1 micro
  (duygu: Adım 17 modeli, dondurulmuş, iki hatta aynı). İkincil altın: macmini. Bootstrap 2000, tohum 16, yorum bazında.
Adım 16 olasılıkları canlı hesaplanır ve Adım 18'in kaydettiği test18_probs.npz ile eşitliği assert edilir (test modunda).
KASA KURALI: yorum bazlı hiçbir çıktı yok; sadece toplu sayılar. Olasılıklar npz'ye, incelenmez. Sonra test EMEKLİ.
"""

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
S18 = os.path.join(ROOT, "step18_big_test")
_spec = importlib.util.spec_from_file_location("evaluate18", os.path.join(S18, "evaluate.py"))
E18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E18)
E17 = E18.E
ASPECTS, DEVICE, turkish_lower = E18.ASPECTS, E18.DEVICE, E18.turkish_lower
FROZEN19 = os.path.join(HERE, "frozen_config.json")
sha = lambda p, n=64: hashlib.sha256(open(p, "rb").read()).hexdigest()[:n]

EXPECT = ["BEKLENTİ (ölçümden ÖNCE yazıldı; PLAN 19 §4, val sonrası güncellendi 2026-10-02):",
          "  ANA: konu F1 macro 0.741 → +0.01 ile +0.03; 0'ı dışlama olasılığı ~%30-40",
          "  satıcı F1 0.17 → 0.20-0.40 (n≈38, gürültülü); micro +0.00 ile +0.02 (koruma: alt sınır > −0.01)",
          "  uçtan uca micro +0.00 ile +0.015, 0'ı dışlamayabilir",
          "  (ilk plan beklentisi: macro +3 ile +6, S 0.40-0.60 — val bunu desteklemedi)"]


def topic_probs(model_dirs, texts):
    probs = []
    for d in model_dirs:
        path = os.path.join(ROOT, d)
        tok, model = AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
        model.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = tok([turkish_lower(t) for t in texts[i:i + 64]], padding=True, truncation=True,
                          max_length=128, return_tensors="pt").to(DEVICE)
                chunks.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
        probs.append(np.concatenate(chunks))
        del model
    return np.mean(probs, axis=0)


def scores(pred, gold, sent_ok, idx=slice(None)):
    """konu F1 micro, macro, konu başına F1 (7), uçtan uca micro — yorum alt kümesi idx üzerinde."""
    p, g, ok = pred[idx], gold[idx], sent_ok[idx]
    tp = (p & g).sum(0)
    per = 2 * tp / np.maximum(p.sum(0) + g.sum(0), 1)
    micro = 2 * tp.sum() / max(p.sum() + g.sum(), 1)
    e2e = 2 * (p & ok).sum() / max(p.sum() + g.sum(), 1)
    return np.concatenate([[micro, per.mean()], per, [e2e]])


LABELS = ["konu F1 micro", "konu F1 macro (ANA)"] + [f"F1 {a}" for a in ASPECTS] + ["uçtan uca micro"]


def compare(name, df, pred_old, pred_new, sent, out):
    g = E17.gold_matrix(df)[0]
    gold = g != ""
    ok = gold & (sent == g)          # duygu doğru (Adım 17 modeli, iki hatta aynı)
    a, b = scores(pred_old, gold, ok), scores(pred_new, gold, ok)
    rng = np.random.default_rng(16)
    d = np.array([scores(pred_new, gold, ok, i) - scores(pred_old, gold, ok, i)
                  for i in (rng.integers(0, len(df), len(df)) for _ in range(2000))])
    lo, hi = np.quantile(d, [0.025, 0.975], axis=0)
    out.append(f"\n=== {name}: {len(df)} yorum, {int(gold.sum())} altın konu; konu başına n "
               + str(dict(zip(ASPECTS, gold.sum(0).astype(int).tolist()))) + " ===")
    for k_, pr in (("Adım 16", pred_old), ("v2 B", pred_new)):
        out.append(f"  {k_:<8} P {(pr & gold).sum() / max(pr.sum(), 1):.3f} R {(pr & gold).sum() / gold.sum():.3f} "
                   f"| konu başına P/R: " + ", ".join(f"{x[:4]} {(pr & gold)[:, j].sum() / max(pr[:, j].sum(), 1):.2f}/"
                                                      f"{(pr & gold)[:, j].sum() / max(gold[:, j].sum(), 1):.2f}" for j, x in enumerate(ASPECTS)))
    out.append(f"  {'ölçü':<22}{'Adım 16':>9}{'v2 B':>9}{'fark':>9}   %95 aralık")
    res = {}
    for j, lab in enumerate(LABELS):
        verdict = "0'ı DIŞLIYOR" if lo[j] > 0 or hi[j] < 0 else "0'ı içeriyor"
        out.append(f"  {lab:<22}{a[j]:>9.3f}{b[j]:>9.3f}{b[j] - a[j]:>+9.3f}   [{lo[j]:+.3f}, {hi[j]:+.3f}] {verdict}")
        res[lab] = (b[j] - a[j], lo[j], hi[j])
    return res


def run(mode):
    E18.check_frozen()
    f19 = json.load(open(FROZEN19))
    assert f19["config"] == "B" and not f19["or_keyword_topics"]
    model_sha = [sha(os.path.join(ROOT, d, "model.safetensors"), 16) for d in f19["model_dirs"]]
    s16 = json.load(open(os.path.join(ROOT, "step16_topic_bert", "frozen_config.json")))
    cfg17 = json.load(open(E17.FROZEN))
    if mode == "calib":
        texts = pd.read_csv(os.path.join(S18, "calib_set.csv"))[["id", "text"]]
        main_path, second_path = (os.path.join(E18.GOLD_DIR, f"calib_{w}.csv") for w in ("fresh", "macmini"))
        log, npz = os.path.join(HERE, "log_calib_dryrun.txt"), None
        title = "KURU ÇALIŞTIRMA — kalibrasyon 100 (EĞİTİMDE olan küme; sayılar ANLAMSIZ, sadece kod kontrolü)"
    else:
        log = os.path.join(HERE, "log_test.txt")
        assert not os.path.exists(log), "test zaten ölçüldü (log_test.txt var) — ikinci ölçüm yok"
        texts = pd.read_csv(os.path.join(ROOT, E18.FROZEN["test_set"]["file"]))[["id", "text"]]
        assert len(texts) == 500
        main_path, second_path = (os.path.join(ROOT, E18.FROZEN[k]["file"]) for k in ("gold_main", "gold_second"))
        npz, title = os.path.join(HERE, "test19_probs.npz"), "TEST, BİR KEZ — kasa testi 500 (id 8000-8499), SON HAK"
    t = texts["text"].tolist()
    out = [f"ADIM 19 — {title}",
           f"v2 B: {json.dumps(f19, ensure_ascii=False)} | model sha256[:16] {model_sha}",
           f"Adım 16: epoch {s16['epoch']}, eşik {s16['threshold']}; duygu: Adım 17 {json.dumps({k: cfg17[k] for k in ('init', 'epoch', 'seeds')})}",
           "Adım 18 frozen_test.json sha256'ları doğrulandı. KASA: yorum bazlı çıktı yok.", ""] + EXPECT

    p16 = topic_probs([os.path.join("step16_topic_bert", f"model_s{s}_e{s16['epoch']}") for s in s16["seeds"]], t)
    p19 = topic_probs(f19["model_dirs"], t)
    sent, p_sent = E17.model_matrix(cfg17, t)
    if mode == "test":
        old = np.load(os.path.join(S18, "test18_probs.npz"))
        assert (old["ids"] == texts["id"].to_numpy()).all()
        assert np.allclose(old["topic"], p16, atol=1e-3) and np.allclose(old["sent_pos"], p_sent, atol=1e-3), \
            "Adım 16/17 olasılıkları Adım 18'dekinden farklı — ortam değişmiş"
        np.savez(npz, ids=texts["id"].to_numpy(), topic_v2=p19)
    pred16, pred19 = p16 >= s16["threshold"], p19 >= f19["threshold"]

    res = compare("ANA — fresh altını", E18.gold_df(main_path, texts), pred16, pred19, sent, out)
    compare("İKİNCİL — macmini altını", E18.gold_df(second_path, texts), pred16, pred19, sent, out)
    pt, lo, hi = res["konu F1 macro (ANA)"]
    mpt, mlo, _ = res["konu F1 micro"]
    out.append(f"\nÖZET (ana iddia): konu F1 macro v2 − Adım 16 {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] → "
               + ("0'ı DIŞLIYOR: kazandı" if lo > 0 else ("0'ı DIŞLIYOR: KÖTÜLEŞTİ" if hi < 0 else "0'ı içeriyor: fark gürültüden ayrılamıyor")))
    out.append(f"Koruma: micro {mpt:+.3f}, alt sınır {mlo:+.3f} → {'tuttu' if mlo > -0.01 else 'TUTMADI'}")
    open(log, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    run("calib" if "--calib" in sys.argv else {"test": "test"}[sys.argv[1]])
