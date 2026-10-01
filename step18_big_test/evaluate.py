"""
ADIM 18 - Ana test 500 (id 8000-8499) üzerinde ölçüm 1-3 (PLAN.md madde 3), BİR KEZ. ANA altın: fresh; ikinci: macmini.

  python evaluate.py --calib  -> log_calib_dryrun.txt  (KURU ÇALIŞTIRMA: kalibrasyon 100, zaten okunmuş küme; kod kontrolü)
  python evaluate.py test     -> log_test.txt + test18_probs.npz  (BİR KEZ; log varsa durur)

Ölçümler (Adım 17 evaluate.py'nin report() fonksiyonu aynen; bootstrap 2000 / tohum 17, konu bootstrap tohum 16):
  1. (i) altın konularla duygu doğruluğu micro: Adım 17 duygu modeli (YENİ) vs Adım 15 hattı (eski hat). ANA İDDİA.
  2. Konu tespiti: Adım 16 BERT (eşik 0.60, 3 tohum ortalaması) vs anahtar kelime V2.
  3. (ii) uçtan uca (konu, duygu) çift F1: Adım 16 konuları + YENİ vs Adım 16 konuları + eski hat.
  İkincil: macmini altını, iki altının ortak çiftleri. Ana iddiaya terfi yok.
KASA KURALI: yorum bazlı hiçbir çıktı (id, metin, satır/hata listesi) yazılmaz; olasılıklar npz'ye kaydedilir, incelenmez.
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
S17 = os.path.join(ROOT, "step17_aspect_sentiment")
sys.path.insert(0, S17)
_spec = importlib.util.spec_from_file_location("evaluate17", os.path.join(S17, "evaluate.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)
from save_aspect_labels import parse  # noqa: E402

ASPECTS, DEVICE, turkish_lower = E.ASPECTS, E.DEVICE, E.turkish_lower
GOLD_DIR = os.path.join(ROOT, "data", "aspect_labels_step18")
FROZEN = json.load(open(os.path.join(HERE, "frozen_test.json")))
sha = lambda p, n=64: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()[:n]

EXPECT = ["BEKLENTİ (PLAN.md madde 4, ölçümden ÖNCE yazıldı):",
          "  ölçüm 1: fark +1 ile +3 puan; aralık ±0.022 → 0'ı dışlama olasılığı yaklaşık yarı yarıya",
          "  ölçüm 2: BERT − kelime +0.06 ile +0.09, aralık 0'ı dışlar",
          "  ölçüm 3: +1 ile +2.5 puan, 0'ı dışlamayabilir"]


def check_frozen():
    """frozen_test.json'daki bütün sha256'lar diskteki dosyalarla aynı olmalı."""
    for key in ("test_set", "gold_main", "gold_second", "label_rules"):
        assert sha(FROZEN[key]["file"]) == FROZEN[key]["sha256"], f"{key} değişmiş"
    fm = FROZEN["frozen_models"]
    for key in ("step16_topic", "step17_sentiment"):
        assert sha(fm[key]["config"]) == fm[key]["config_sha256"], f"{key} config değişmiş"
    s16, s17 = fm["step16_topic"], fm["step17_sentiment"]
    for s, h in zip(s16["seeds"], s16["model_sha256_16"]):
        assert sha(f"step16_topic_bert/model_s{s}_e{s16['epoch']}/model.safetensors", 16) == h, "Adım 16 modeli değişmiş"
    for s, h in zip(s17["seeds"], s17["model_sha256_16"]):
        assert sha(f"step17_aspect_sentiment/model_{s17['init']}_s{s}_e{s17['epoch']}/model.safetensors", 16) == h, "Adım 17 modeli değişmiş"


def gold_df(path, texts):
    g = pd.read_csv(path, keep_default_na=False)
    df = texts.merge(g, on="id", validate="one_to_one")
    assert (df["id"] == texts["id"]).all()
    parsed = df["raw"].map(parse)
    for a in ASPECTS:
        df[a] = parsed.map(lambda q: q[0].get(a, ""))
    df["karisik_konular"] = parsed.map(lambda q: ";".join(q[1]))
    return df


def topic_probs(texts):
    """Adım 16 BERT konu modeli: 3 tohum olasılık ortalaması (Adım 17 run_test ile aynı)."""
    s16 = json.load(open(os.path.join(ROOT, "step16_topic_bert", "frozen_config.json")))
    probs = []
    for s in s16["seeds"]:
        d = os.path.join(ROOT, "step16_topic_bert", f"model_s{s}_e{s16['epoch']}")
        tok, model = AutoTokenizer.from_pretrained(d), AutoModelForSequenceClassification.from_pretrained(d).to(DEVICE)
        model.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = tok([turkish_lower(t) for t in texts[i:i + 64]], padding=True, truncation=True,
                          max_length=128, return_tensors="pt").to(DEVICE)
                chunks.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
        probs.append(np.concatenate(chunks))
        del model
    return np.mean(probs, axis=0), s16["threshold"]


def run(mode):
    check_frozen()
    if mode == "calib":
        texts = pd.read_csv(os.path.join(HERE, "calib_set.csv"))[["id", "text"]]
        main_path, second_path = os.path.join(GOLD_DIR, "calib_fresh.csv"), os.path.join(GOLD_DIR, "calib_macmini.csv")
        log, npz, title = os.path.join(HERE, "log_calib_dryrun.txt"), None, "KURU ÇALIŞTIRMA — kalibrasyon 100 (okunmuş küme; sadece kod kontrolü)"
    else:
        log = os.path.join(HERE, "log_test.txt")
        assert not os.path.exists(log), "test zaten ölçüldü (log_test.txt var) — ikinci ölçüm yok"
        texts = pd.read_csv(os.path.join(ROOT, FROZEN["test_set"]["file"]))[["id", "text"]]
        assert len(texts) == 500
        main_path, second_path = os.path.join(ROOT, FROZEN["gold_main"]["file"]), os.path.join(ROOT, FROZEN["gold_second"]["file"])
        npz, title = os.path.join(HERE, "test18_probs.npz"), "TEST, BİR KEZ — ana test 500 (id 8000-8499)"
    cfg17 = json.load(open(E.FROZEN))
    out = [f"ADIM 18 — {title}",
           f"donmuş: Adım 16 konu (epoch 12, eşik 0.60, tohum 0/1/2), Adım 17 duygu {json.dumps({k: cfg17[k] for k in ('init', 'epoch', 'seeds')})}; "
           f"Adım 15 hattı (bölme A + V2b, nötr kapalı). frozen_test.json sha256'ları doğrulandı.",
           "KASA: yorum bazlı çıktı yok; sadece toplu sayılar.", ""] + EXPECT

    topic_p, thr = topic_probs(texts["text"].tolist())
    new, p_new = E.model_matrix(cfg17, texts["text"].tolist())
    base, explicit = E.baseline_matrix(texts["text"].tolist())
    if npz:
        np.savez(npz, ids=texts["id"].to_numpy(), topic=topic_p, sent_pos=p_new)
    topic_pred = topic_p >= thr
    dmain, dsec = gold_df(main_path, texts), gold_df(second_path, texts)

    res = E.report("ANA — fresh altını (ölçüm 1 = (i) micro ANA İDDİA; ölçüm 2 = KONU TESPİTİ; ölçüm 3 = (ii))",
                   dmain, topic_pred, base, new, explicit, out)
    E.report("İKİNCİL — macmini altını", dsec, topic_pred, base, new, explicit, out)
    g1, g2 = E.gold_matrix(dmain)[0], E.gold_matrix(dsec)[0]
    E.report("İKİNCİL — iki altının ORTAK çiftleri (aynı konu + aynı duygu)", dmain, topic_pred, base, new, explicit, out,
             mask_pairs=(g1 != "") & (g1 == g2))

    pt, lo, hi = res["(i) micro doğruluk"]
    out.append(f"\nÖZET (ana iddia): (i) micro YENİ − eski hat {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] → "
               + ("0'ı DIŞLIYOR: kazandı" if lo > 0 or hi < 0 else "0'ı içeriyor: fark yok (gürültüden ayrılamıyor)"))
    open(log, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    run("calib" if "--calib" in sys.argv else {"test": "test"}[sys.argv[1]])
