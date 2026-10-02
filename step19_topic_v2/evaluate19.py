"""
ADIM 19 (3) - Val değerlendirmesi ve dondurma. Sadece Adım 16 val 300; hiçbir teste dokunmaz.

  python evaluate19.py val  -> log_val.txt + frozen_config.json

Önceden sabit kurallar (PLAN §2, Ekler 2026-10-02; val sayıları görülmeden yazıldı):
  - Her yapılandırma (A, B) için epoch: 3 tohum ortalaması val micro-F1 (eşik 0.5) en yüksek aday (5/8/12/16).
  - Teste giren model: B (tam veri), seçilen epoch'ta 3 tohumun olasılık ortalaması. A sadece ablasyon satırı.
  - Tek global eşik: ortalama olasılıklarda val micro-F1 (ızgara 0.05-0.95).
  - VEYA adayı: B + V2 anahtar kelimesi sadece S/G/B için. Teste o girer ancak val macro'yu B'ye göre artırıyorsa VE
    S, G, B precision'larının her biri >= 0.70 ise.
  - Kıyas: Adım 16 BERT (dondurulmuş, epoch 12, eşik 0.60, 3 tohum), anahtar kelime V2 — aynı val'de.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
S16 = os.path.join(ROOT, "step16_topic_bert")
sys.path.insert(0, S16)

from evaluate import best_threshold, det, keyword_matrix, per_topic_thresholds, summary_line, topic_table  # noqa: E402
from train import ASPECTS, SEEDS, load_data, targets  # noqa: E402

EPOCHS = (5, 8, 12, 16)
OR_TOPICS = ("satici", "gorunum", "boyut")
FROZEN = os.path.join(HERE, "frozen_config.json")


def seed_avg(npz, epoch):
    return np.mean([npz[f"s{s}_e{epoch}"] for s in SEEDS], axis=0)


def run_val():
    _, val = load_data()
    y, kw = targets(val), keyword_matrix(val["text"].tolist())
    out = [f"ADIM 19 — val 300 (Adım 16 val: eski 100 + yeni 200). Konu başına n: "
           + str(dict(zip(ASPECTS, y.sum(0).astype(int).tolist())))]

    chosen = {}
    for cfg in ("A", "B"):
        npz = np.load(os.path.join(HERE, f"val_probs_{cfg}.npz"))
        assert (npz["ids"] == val["id"].to_numpy()).all()
        out.append(f"\n[{cfg}] epoch seçimi (eşik 0.5, micro-F1: 3 tohum ort. (min-max) | ortalama olasılıkla):")
        best = None
        for e in EPOCHS:
            singles = [det(y, (npz[f"s{s}_e{e}"] >= 0.5).astype(float))[1]["micro"] for s in SEEDS]
            avg = det(y, (seed_avg(npz, e) >= 0.5).astype(float))[1]["micro"]
            out.append(f"  epoch {e:>2}: {np.mean(singles):.3f} ({min(singles):.3f}-{max(singles):.3f}) | {avg:.3f}")
            if best is None or np.mean(singles) > best[1]:
                best = (e, np.mean(singles))
        p = seed_avg(npz, best[0])
        t, f = best_threshold(y, p)
        curve = " ".join(f"{g:.2f}:{det(y, (p >= g).astype(float))[1]['micro']:.3f}" for g in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
        out.append(f"  seçilen epoch {best[0]}, global eşik {t:.2f} (micro {f:.3f}); eşik eğrisi {curve}")
        chosen[cfg] = (best[0], t, p)

    s16cfg = json.load(open(os.path.join(S16, "frozen_config.json")))
    p16 = seed_avg(np.load(os.path.join(S16, "val_probs.npz")), s16cfg["epoch"])
    preds = {"Anahtar kelime V2": kw, "Adım 16 BERT (eşik 0.60)": (p16 >= s16cfg["threshold"]).astype(float)}
    for cfg in ("A", "B"):
        e, t, p = chosen[cfg]
        preds[f"v2 {cfg} (e{e}, eşik {t:.2f})"] = (p >= t).astype(float)
    eB, tB, pB = chosen["B"]
    predB = (pB >= tB).astype(float)
    orB = predB.copy()
    for a in OR_TOPICS:
        k = ASPECTS.index(a)
        orB[:, k] = np.maximum(orB[:, k], kw[:, k])
    preds["v2 B VEYA kelime (S/G/B)"] = orB
    pt = per_topic_thresholds(y, pB)
    preds["v2 B konu başına eşik (iyimser)"] = (pB >= pt).astype(float)

    out.append(f"\n{'model':<34}{'P':>7}{'R':>7}{'micro':>8}{'macro':>8}{'R kel+':>10}{'R kel−':>10}")
    for name, pr in preds.items():
        out.append(summary_line(name, y, pr, kw))
    names = list(preds)
    out.append("\nKonu başına P/R/F1:")
    out += topic_table(y, [preds[n] for n in names[:5]], ["kelime", "A16", "v2A", "v2B", "B|kel"])
    for origin in ("eski", "yeni"):
        m = (val["origin"] == origin).to_numpy()
        out.append(f"  val {origin} ({m.sum()}): micro A16 {det(y[m], preds[names[1]][m])[1]['micro']:.3f} → v2B "
                   f"{det(y[m], predB[m])[1]['micro']:.3f}; macro {det(y[m], preds[names[1]][m])[1]['macro']:.3f} → "
                   f"{det(y[m], predB[m])[1]['macro']:.3f}")

    rowsB, sB = det(y, predB)
    rowsO, sO = det(y, orB)
    use_or = sO["macro"] > sB["macro"] and all(rowsO[a][1] >= 0.70 for a in OR_TOPICS)
    out.append(f"\nVEYA kuralı: macro {sB['macro']:.3f} → {sO['macro']:.3f}; precision S/G/B "
               + " ".join(f"{rowsO[a][1]:.2f}" for a in OR_TOPICS) + f" → {'VEYA TESTE GİRER' if use_or else 'B tek başına teste girer'}")

    frozen = {"config": "B", "epoch": int(eB), "threshold": tB, "seeds": list(SEEDS),
              "or_keyword_topics": list(OR_TOPICS) if use_or else [],
              "model_dirs": [f"step19_topic_v2/model_B_s{s}_e{eB}" for s in SEEDS],
              "ablation_A": {"epoch": int(chosen["A"][0]), "threshold": chosen["A"][1]}}
    json.dump(frozen, open(FROZEN, "w"), indent=1, ensure_ascii=False)
    out.append(f"\nDONDURULDU -> frozen_config.json: {json.dumps(frozen, ensure_ascii=False)}")
    open(os.path.join(HERE, "log_val.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    assert sys.argv[1:] == ["val"], "kullanım: python evaluate19.py val (test ayrı onayla, ayrı script)"
    run_val()
