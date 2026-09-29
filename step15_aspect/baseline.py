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
# V1: ESKİ Q/P tanımıyla yazılmış liste (15.3 ilk sürüm) — sadece kıyas için duruyor.
KEYWORDS_V1 = {
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

# V2: YENİ Q/P tanımı (2026-09-29). Yine SADECE explore_300'den türetildi, val'e bakılarak kelime eklenmedi.
#   Q = genel yargı + genel işe yarama/arıza/pil-şarj + malzeme/işçilik
#   P = adı konan özellik/teknik ölçü
# Kalıp tavsiye ("tavsiye ederim", "alın") bilerek listede YOK (kural: tek başına Q değil).
GENERIC_PRODUCT = r"(?:ürün|urun|cihaz|makine|makina|alet|telefon)"
KEYWORDS = {
    "kargo": KEYWORDS_V1["kargo"] + [r"tedarik", r"kargola", r"geç teslim"],                  # explore: 3256, 3263, 3260
    "fiyat": KEYWORDS_V1["fiyat"] + [r"paraya", r"fırsat"],                                   # explore: 3277, 3270
    "kalite": KEYWORDS_V1["kalite"] + [
        # genel yargı (explore: 3013, 3054, 3117, 3254, 3257, 3275, 3285, 3290 ...)
        r"mükemmel", r"harika", r"muhteşem", r"süper", r"berbat", r"rezalet", r"hayal kırıklığı",
        r"memnun", r"beğen", r"pişman", r"başarılı", r"başarısız", r"vasat", r"sıradan",
        r"(?:güzel|iyi|kötü|harika|süper) bir " + GENERIC_PRODUCT,
        GENERIC_PRODUCT + r" (?:çok |gayet |gerçekten )?(?:güzel|iyi|kötü|harika|süper|mükemmel|berbat)",
        # genel işe yarama / arıza / pil-şarj (V1'de P'deydi -> Q'ya taşındı)
        r"işe yar", r"iş gör", r"işimi gör", r"işinizi gör", r"fayda", r"etki", r"çalışm[ıa]", r"çalışıyor",
        r"şarj", r"sarj", r"pil(?!\w)", r"pili", r"batarya", r"severek", r"bayıl"],
    "performans": [r"ses(?:i|li|siz)?(?!\w)", r"gürültü", r"performans", r"ısın", r"çekim", r"çekiş", r"emiş",
                   r"ekran", r"görüntü", r"kamera", r"hızı", r"uyum", r"kullanış", r"kullanım", r"kurulum",
                   r"montaj", r"pratik", r"ergonomi", r"fonksiyon", r"özellik", r"koku", r"kalıcı", r"tadı",
                   r"lezzet", r"konfor", r"rahat(?!lıkla)", r"ayar", r"menü", r"ışık", r"lümen", r"temizl"],
    "boyut": KEYWORDS_V1["boyut"] + [r"ağır(?!\w|lık)", r"ağırlı", r"kapasite", r"\d+ ?(?:ml|lt|litre)(?!\w)"],
    "gorunum": KEYWORDS_V1["gorunum"] + [r"duruyor", r"zarif"],                                 # explore: 3258, 3290
    "satici": KEYWORDS_V1["satici"] + [r"yanında gel", r"içinde yok", r"promosyon"],           # explore: 3288, 3298
}


def compile_patterns(keywords):
    return {a: re.compile(W + "(?:" + "|".join(ws) + ")") for a, ws in keywords.items()}


PATTERNS_V1 = compile_patterns(KEYWORDS_V1)
PATTERNS = compile_patterns(KEYWORDS)

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


def detect(text, patterns=None):
    patterns = patterns or PATTERNS
    t = turkish_lower(text)
    return {a for a, pat in patterns.items() if pat.search(t)}


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


def predict(df, variant, mode, cache, patterns=None):
    """Her yorum için {konu: duygu} ve hangi cümlecik(ler)in kullanıldığı."""
    patterns = patterns or PATTERNS
    out = []
    for text in df["text"]:
        clauses = split_clauses(text, variant)
        pred, used = {}, {}
        for a in detect(text, patterns):
            cl = [c for c in clauses if patterns[a].search(c)] or [turkish_lower(text)]
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


HUMAN_CEILING = 0.667   # insan-insan uçtan uca çift F1 (20 kör yorum, yeni kurallar, karar öncesi) — İYİMSER


def select_config(val, cache, patterns, title):
    """Val'de 3 bölme x 2 nötr seçeneği; seçim: duygu micro, sonra macro, sonra basit olan."""
    lines = [f"--- {title} ---"]
    rows, _ = metrics(val, predict(val, "A", "b", cache, patterns))
    lines.append("  konu P/R: " + " | ".join(f"{a} {r['P']:.2f}/{r['R']:.2f} (n={r['n']})" for a, r in rows.items()))
    lines.append(f"  {'bölme':<6}{'nötr':<24}{'duygu_micro':>12}{'duygu_macro':>12}{'duygu_k0':>10}"
                 f"{'uç_F1_micro':>12}{'uç_F1_macro':>12}{'tah.nötr':>9}")
    results = []
    for v in SPLITTERS:
        for m in MODES:
            _, sm = metrics(val, predict(val, v, m, cache, patterns))
            results.append((sm["duygu_acc_micro"], sm["duygu_acc_macro"], v, m, sm))
            lines.append(f"  {v:<6}{m + ' (' + MODES[m] + ')':<24}{sm['duygu_acc_micro']:>12.3f}"
                         f"{sm['duygu_acc_macro']:>12.3f}{sm['duygu_acc_micro_k0']:>10.3f}{sm['uc_F1_micro']:>12.3f}"
                         f"{sm['uc_F1_macro']:>12.3f}{sm['tahmin_notr']:>9d}")
    # Eşitlik = val bu farkı ölçemiyor -> daha basit ayar (daha az bölme: yok < A < B).
    simplicity = {"yok": 0, "A": 1, "B": 2}
    key = lambda r: (round(r[0], 6), round(r[1], 6))
    best = max(results, key=lambda r: key(r) + (-simplicity[r[2]],))
    ties = [(r[2], r[3]) for r in results if key(r) == key(best)]
    lines.append(f"  SEÇİLEN: bölme={best[2]}, nötr={best[3]} ({MODES[best[3]]})"
                 + (f" — val'de eşit: {ties}" if len(ties) > 1 else ""))
    return lines, best, rows


def list_diff():
    lines = ["Anahtar kelime listesi farkı (V1 -> V2):"]
    for a in ASPECTS:
        old, new = set(KEYWORDS_V1[a]), set(KEYWORDS[a])
        if old != new:
            lines.append(f"  {a}: + {sorted(new - old)}" + (f"  - {sorted(old - new)}" if old - new else ""))
    return lines


def old_labels():
    """Q/P tanım değişikliğinden ÖNCEKİ altın etiketler (commit 4a5b8d6)."""
    import io
    import subprocess
    csv = subprocess.run(["git", "show", "4a5b8d6:data/aspect_labels/aspect_labels.csv"], capture_output=True,
                         text=True, check=True, cwd=HERE).stdout
    return pd.read_csv(io.StringIO(csv), keep_default_na=False)


if __name__ == "__main__":
    labels = load_labels()
    # Test cümleciklerinin logit'leri de burada cache'lenir (sadece hesaplanır, SONUÇLARINA bakılmaz).
    all_clauses = {c for t in labels["text"] for v in SPLITTERS for c in split_clauses(t, v)}
    all_clauses |= {turkish_lower(t) for t in labels["text"]}
    cache = clause_logits(all_clauses)

    val = labels[labels["split"] == "val"].reset_index(drop=True)
    old_val = old_labels().query("split == 'val'").reset_index(drop=True)
    setups = [("eski kurallar + eski liste (V1)", old_val, PATTERNS_V1),
              ("yeni kurallar + eski liste (V1)", val, PATTERNS_V1),
              ("yeni kurallar + yeni liste (V2)", val, PATTERNS)]
    lines, summary = [f"VAL ({len(val)} yorum)"] + list_diff(), []
    for title, df, pats in setups:
        l, best, _ = select_config(df, cache, pats, title)
        lines += l
        summary.append((title, best))
    lines.append(f"\nÖZET (her satır kendi val seçimiyle) — insan tavanı (çift F1): {HUMAN_CEILING} (iyimser)")
    lines.append(f"{'kurulum':<34}{'ayar':<8}{'duygu mic/mac':>15}{'k0':>7}{'uç F1 mic/mac':>15}{'tavana oran':>13}")
    for title, b in summary:
        sm = b[4]
        lines.append(f"{title:<34}{b[2] + '/' + b[3]:<8}{sm['duygu_acc_micro']:>8.3f}/{sm['duygu_acc_macro']:.3f}"
                     f"{sm['duygu_acc_micro_k0']:>7.3f}{sm['uc_F1_micro']:>8.3f}/{sm['uc_F1_macro']:.3f}"
                     f"{sm['uc_F1_micro'] / HUMAN_CEILING:>13.2f}")
    print("\n".join(lines))
    with open(os.path.join(HERE, "log_baseline_val.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    final = summary[-1][1]
    with open(os.path.join(HERE, "chosen_config.json"), "w") as f:
        json.dump({"split": final[2], "mode": final[3], "keywords": "V2"}, f)
