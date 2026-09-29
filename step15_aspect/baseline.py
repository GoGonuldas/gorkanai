"""
ADIM 15.3 - Temel çizgi (eğitim yok): anahtar kelime ile konu tespiti + cümleciğe bölme + V2b.

Yöntem:
  1. Konu tespiti: yorumda konunun anahtar kelimelerinden biri geçiyor mu? (tüm yorum üzerinde)
  2. Yorumu cümleciklere böl; her konu için o konunun kelimesinin geçtiği cümlecik(ler)i al.
  3. Her cümleciğe V2b uygula; konu birden fazla cümlecikte geçiyorsa olasılıklar toplanır.

Kurallar (sızıntıya karşı):
  - Anahtar kelime listeleri SADECE 15.1 keşif setinden (explore_300.csv) + genel Türkçe bilgisinden türetildi,
    val'e bakılarak kelime eklenmedi. Konu başına ayrı eşik/kural yok (val'de görünüm 7, satıcı 6 örnek).
  - Val sadece iki genel karar için: bölme kuralı (A/B/yok) ve nötr bias seçeneği (a/b).
    V2b'nin +3.00 nötr bias'ı tüm yorum için seçilmişti (yorumların ~%11'i nötr); konu içi nötr ise %3.
      (a) bias 0, 3 sınıf (nötr çıkabilir)   (b) sadece poz/neg logit'leri arasında argmax (nötr hiç çıkmaz)
  - Test sonuçları 15.4'te (evaluate.py) SADECE burada val'de seçilen ayarla ölçülür.

Kullanım: python baseline.py          -> tüm cümleciklerin V2b logit'lerini cache'ler, val'de 3x2 ayar tablosu
Çıktı: clause_logits.npz (cache), log_baseline_val.txt, chosen_config.json
"""

import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = ["negatif", "nötr", "pozitif"]
ASPECTS = ["kargo", "fiyat", "kalite", "performans", "boyut", "gorunum", "satici"]
MODEL_DIR = os.path.join(HERE, "..", "step14_three_class", "model_v2b")
CACHE = os.path.join(HERE, "clause_logits.npz")


def turkish_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


W = r"(?<!\w)"   # kelime başı: "kargo" -> "kargoya", "kargosu" da eşleşir (Türkçe ekler)
KEYWORDS = {
    "kargo": [r"kargo", r"teslim", r"gönderi", r"gönderil", r"sevkiyat", r"paketlen", r"kolilen", r"ambalaj",
              r"elime", r"elimize", r"elimde", r"eli̇me", r"kapıma", r"kapımda", r"ulaştı", r"ulaşti", r"ertesi gün",
              r"\d+ ?günde", r"\d+ gün (?:içinde|sonra)", r"hızlı gel", r"gecik", r"kurye", r"yarın kapında"],
    "fiyat": [r"fiyat", r"fıyat", r"fi̇yat", r"para(?:sı|sına|ya|nın|nızı|yı|sını)?(?!\w)", r"ucuz", r"pahalı",
              r"indirim", r"kampanya", r"hesaplı", r"f/p", r"\d+ ?tl(?!\w)", r"taksit", r"paranın hakkı"],
    "kalite": [r"kalite", r"kali̇te", r"malzeme", r"kumaş", r"sağlam", r"dayanık", r"plastik", r"dikiş", r"işçilik",
               r"kırıl", r"kırık", r"bozul", r"yırt", r"pasla", r"çatla", r"sunta", r"arıza"],
    "performans": [r"çalış", r"şarj", r"sarj", r"pil(?!\w)", r"pili", r"batarya", r"ses(?:i|li|siz)?(?!\w)",
                   r"performans", r"ısın", r"çekim", r"çekiş", r"ekran", r"görüntü", r"kamera", r"kullanış",
                   r"kullanım", r"işe yar", r"iş gör", r"işimi gör", r"fayda", r"etki", r"koku", r"kalıcı", r"tadı",
                   r"lezzet", r"temizl", r"kurulum", r"montaj", r"pratik", r"fonksiyon", r"özellik"],
    "boyut": [r"beden", r"kalıp", r"boyut", r"ebat", r"ölçü", r"büyük", r"küçük", r"dar(?!\w)", r"geniş", r"hafif"],
    "gorunum": [r"renk", r"görün", r"resim", r"fotoğraf", r"tasarım", r"şık", r"desen", r"görsel", r"dekoratif",
                r"güzel duruyor", r"kaba duruyor", r"parlak", r"mat(?!\w)"],
    "satici": [r"satıcı", r"mağaza", r"firma", r"iletişim", r"müşteri hizmet", r"iade", r"değişim", r"servis",
               r"garanti", r"eksik", r"kılavuz", r"klavuz"],
}
PATTERNS = {a: re.compile(W + "(?:" + "|".join(ws) + ")") for a, ws in KEYWORDS.items()}

CONTRAST = r"(?<!\w)(?:ama|fakat|ancak|lakin|yalnız|yalniz|ne var ki|oysa)(?!\w)"
SPLITTERS = {
    "A": re.compile(r"[.!?;]+|" + CONTRAST),              # cümle sonu + zıtlık bağlacı
    "B": re.compile(r"[.!?;,]+|" + CONTRAST),             # A + virgül
    "yok": None,                                          # bölme yok: tüm yorum tek cümlecik
}
MODES = {"a": "bias 0, 3 sınıf", "b": "sadece poz/neg argmax"}


def split_clauses(text, variant):
    text = turkish_lower(text)
    if SPLITTERS[variant] is None:
        return [text]
    parts = [p.strip() for p in SPLITTERS[variant].split(text)]
    return [p for p in parts if len(p) > 1] or [text]


def detect(text):
    t = turkish_lower(text)
    return {a for a, pat in PATTERNS.items() if pat.search(t)}


def clause_logits(texts):
    """V2b logit'leri, metin -> (3,) cache'li."""
    cache = dict(np.load(CACHE, allow_pickle=True)["d"].item()) if os.path.exists(CACHE) else {}
    todo = sorted({t for t in texts if t not in cache})
    if todo:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
        with torch.no_grad():
            for i in range(0, len(todo), 64):
                chunk = todo[i:i + 64]   # zaten küçük harfli
                enc = tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                for t, lg in zip(chunk, model(**enc).logits.float().cpu().numpy()):
                    cache[t] = lg
        np.savez(CACHE, d=np.array(cache, dtype=object))
        print(f"V2b: {len(todo)} yeni cümlecik işlendi (cache: {len(cache)})")
    return cache


def decide(logits, mode):
    """Bir konunun cümleciklerinin logit'lerinden tek duygu."""
    lg = np.array(logits, dtype=float)
    if mode == "b":
        lg[:, 1] = -np.inf                                   # nötrü tamamen kapat
    probs = np.exp(lg - lg.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    return LABELS[int(probs.sum(0).argmax())]               # birden çok cümlecik: olasılıkları topla


def predict(df, variant, mode, cache):
    """Her yorum için {konu: duygu} ve hangi cümlecik(ler)in kullanıldığı."""
    out = []
    for text in df["text"]:
        clauses = split_clauses(text, variant)
        pred, used = {}, {}
        for a in detect(text):
            cl = [c for c in clauses if PATTERNS[a].search(c)] or [turkish_lower(text)]
            pred[a] = decide([cache[c] for c in cl], mode)
            used[a] = cl
        out.append((pred, used))
    return out


def gold(df):
    return [{a: r[a] for a in ASPECTS if r[a]} for _, r in df.iterrows()]


def metrics(df, preds):
    """Konu tespiti P/R/F1 (micro + macro), konu-duygu doğruluğu (tespit edilen doğru konularda)."""
    golds = gold(df)
    rows = {}
    for a in ASPECTS:
        tp = sum(a in g and a in p for g, (p, _) in zip(golds, preds))
        fp = sum(a not in g and a in p for g, (p, _) in zip(golds, preds))
        fn = sum(a in g and a not in p for g, (p, _) in zip(golds, preds))
        hit = [(g[a] == p[a], k) for g, (p, _), k in zip(golds, preds, df["karisik"]) if a in g and a in p]
        rows[a] = {"n": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                   "P": tp / max(tp + fp, 1), "R": tp / max(tp + fn, 1),
                   "duygu_ok": sum(h for h, _ in hit), "duygu_n": len(hit),
                   "duygu_ok_k0": sum(h for h, k in hit if k == 0), "duygu_n_k0": sum(k == 0 for _, k in hit),
                   "uc": sum(g[a] == p[a] for g, (p, _) in zip(golds, preds) if a in g and a in p)}
    for r in rows.values():
        r["F1"] = 2 * r["P"] * r["R"] / max(r["P"] + r["R"], 1e-9)
        r["duygu_acc"] = r["duygu_ok"] / max(r["duygu_n"], 1)
        r["duygu_acc_k0"] = r["duygu_ok_k0"] / max(r["duygu_n_k0"], 1)
        # uçtan uca: (konu, duygu) çifti doğru mu — F1
        pp, rr = r["uc"] / max(r["tp"] + r["fp"], 1), r["uc"] / max(r["n"], 1)
        r["uc_F1"] = 2 * pp * rr / max(pp + rr, 1e-9)
    s = lambda k: sum(r[k] for r in rows.values())
    P, R = s("tp") / max(s("tp") + s("fp"), 1), s("tp") / max(s("n"), 1)
    ucP, ucR = s("uc") / max(s("tp") + s("fp"), 1), s("uc") / max(s("n"), 1)
    summary = {
        "tespit_F1_micro": 2 * P * R / max(P + R, 1e-9), "tespit_P_micro": P, "tespit_R_micro": R,
        "tespit_F1_macro": np.mean([r["F1"] for r in rows.values()]),
        "duygu_acc_micro": s("duygu_ok") / max(s("duygu_n"), 1),
        "duygu_acc_macro": np.mean([r["duygu_acc"] for r in rows.values() if r["duygu_n"]]),
        "duygu_acc_micro_k0": s("duygu_ok_k0") / max(s("duygu_n_k0"), 1),
        "uc_F1_micro": 2 * ucP * ucR / max(ucP + ucR, 1e-9),
        "uc_F1_macro": np.mean([r["uc_F1"] for r in rows.values()]),
        "tahmin_notr": sum(v == "nötr" for p, _ in preds for v in p.values()),
        "altin_notr": sum(v == "nötr" for g in golds for v in g.values()),
    }
    return rows, summary


def load_labels():
    return pd.read_csv(os.path.join(HERE, "..", "data", "aspect_labels", "aspect_labels.csv"), keep_default_na=False)


if __name__ == "__main__":
    labels = load_labels()
    # Test cümleciklerinin logit'leri de burada cache'lenir (sadece hesaplanır, SONUÇLARINA bakılmaz).
    all_clauses = {c for t in labels["text"] for v in SPLITTERS for c in split_clauses(t, v)}
    all_clauses |= {turkish_lower(t) for t in labels["text"]}
    cache = clause_logits(all_clauses)

    val = labels[labels["split"] == "val"].reset_index(drop=True)
    lines = [f"VAL ({len(val)} yorum) — 3 bölme x 2 nötr seçeneği. Konu tespiti bölmeden bağımsız:"]
    rows, _ = metrics(val, predict(val, "A", "a", cache))
    lines.append("  " + " | ".join(f"{a} P{r['P']:.2f}/R{r['R']:.2f} (n={r['n']})" for a, r in rows.items()))
    lines.append(f"{'bölme':<6}{'nötr':<24}{'duygu_micro':>12}{'duygu_macro':>12}{'duygu_k0':>10}"
                 f"{'uç_F1_micro':>12}{'uç_F1_macro':>12}{'tah.nötr':>9}")
    results = []
    for v in SPLITTERS:
        for m in MODES:
            _, sm = metrics(val, predict(val, v, m, cache))
            results.append((sm["duygu_acc_micro"], sm["duygu_acc_macro"], v, m))
            lines.append(f"{v:<6}{m + ' (' + MODES[m] + ')':<24}{sm['duygu_acc_micro']:>12.3f}"
                         f"{sm['duygu_acc_macro']:>12.3f}{sm['duygu_acc_micro_k0']:>10.3f}{sm['uc_F1_micro']:>12.3f}"
                         f"{sm['uc_F1_macro']:>12.3f}{sm['tahmin_notr']:>9d}")
    lines.append(f"(altın nötr çift sayısı val'de: {sm['altin_notr']})")
    # Seçim: duygu doğruluğu micro, sonra macro; hâlâ eşitse DAHA BASİT ayar (daha az bölme: yok < A < B;
    # nötr seçeneğinde a < b sırası yok, ölçüt karar verir). Eşitlik = val bu farkı ölçemiyor demek.
    simplicity = {"yok": 0, "A": 1, "B": 2}
    best = max(results, key=lambda r: (round(r[0], 6), round(r[1], 6), -simplicity[r[2]]))
    ties = [r for r in results if round(r[0], 6) == round(best[0], 6) and round(r[1], 6) == round(best[1], 6)]
    lines.append(f"SEÇİLEN (val duygu doğruluğu micro, eşitlikte macro, sonra basit olan): bölme={best[2]}, "
                 f"nötr={best[3]} ({MODES[best[3]]})" + (f" — val'de eşit: {[(r[2], r[3]) for r in ties]}"
                                                         if len(ties) > 1 else ""))
    va = predict(val, "A", best[3], cache)
    vb = predict(val, "B", best[3], cache)
    n_diff = sum(pa != pb for (pa, _), (pb, _) in zip(va, vb))
    lines.append(f"A ile B'nin val'de farklı tahmin verdiği yorum sayısı: {n_diff}")
    print("\n".join(lines))
    with open(os.path.join(HERE, "log_baseline_val.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "chosen_config.json"), "w") as f:
        json.dump({"split": best[2], "mode": best[3]}, f)
