"""
ADIM 15.4 - Test ölçümü (200 yorum), BİR KEZ. Eğitim yok, ayar yok.

Dondurulan: KEYWORDS (V2) + chosen_config.json (bölme A, nötr b) — hash'leri log'un başına yazılır.
Sonuçları gördükten sonra yapılan her değişiklik ayrı bir "test sonrası (iyimser)" satırı olur.

Bölümler:
  1. Ana tablo: test + val yan yana; konu başına P/R, n (n<30 -> "gürültülü"), duygu micro/macro,
     karisik=0 duygu, uçtan uca F1 micro/macro.
  2. Karşılaştırma (test, yeni altın): (a) V2 + bölme yok, (b) V1 (eski liste) + A/b.
  3. Elmayla elma: Görkan'ın kör 20'si (KARAR ÖNCESİ etiketler): model-Claude, model-Görkan, Claude-Görkan
     çift F1; tümü / 18 kör / sadece test'teki 10. n çok küçük -> sadece fikir verir.
  4. Hata kovaları (test):
     - konu FP -> "yanlış eşleşme" (otomatik: kelime eşleşti, konu altında yok)
     - konu FN -> "örtük" / "kelime eksik": ELLE sınıflandırıldı (test_fn_manual.csv). Ölçüt:
         kelime eksik = konunun açık bir kelimesi/ifadesi var ama listede yok ("aktarım hızı" gibi);
         örtük = konu hiç adlandırılmıyor, çıkarım gerekiyor ("2 saatte doldu" -> P, "bir kere düştü" -> Q).
     - duygu hatası (konu doğru, duygu yanlış) -> otomatik sezgisel:
         bölme hatası = kullanılan cümlecik(ler)de altın duygusu FARKLI olan başka bir altın konu da geçiyor
                        (yani cümlecik iki zıt görüşü birlikte taşıyor);
         V2b hatası   = cümlecik tek görüş taşıyor ama V2b yanlış.
Kullanım: python evaluate.py            -> log_test.txt (+ test_fn_manual.csv yoksa şablonunu yazar)
"""

import hashlib
import json
import os

import pandas as pd

from baseline import (ASPECTS, KEYWORDS, PATTERNS, PATTERNS_V1, SPLITTERS, clause_logits, gold,
                      load_labels, metrics, predict, split_clauses, turkish_lower)
from save_aspect_labels import parse

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "log_test.txt")
FN_MANUAL = os.path.join(HERE, "test_fn_manual.csv")
CODE = {"K": "kargo", "F": "fiyat", "Q": "kalite", "P": "performans", "B": "boyut", "G": "gorunum", "S": "satici"}


def pair_f1(a, b):
    """a, b: liste of {konu: duygu}; çift (yorum, konu, duygu) F1."""
    A = {(i, k, v) for i, d in enumerate(a) for k, v in d.items()}
    B = {(i, k, v) for i, d in enumerate(b) for k, v in d.items()}
    return 2 * len(A & B) / max(len(A) + len(B), 1), len(A), len(B)


def main():
    cfg = json.load(open(os.path.join(HERE, "chosen_config.json")))
    kw_hash = hashlib.sha256(json.dumps(KEYWORDS, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    out = [f"DONDURULAN: config={cfg} | KEYWORDS(V2) sha256[:16]={kw_hash}",
           "Test BİR KEZ ölçüldü; aşağıdaki her şey bu dondurulmuş ayarla.\n"]

    labels = load_labels()
    cache = clause_logits({c for t in labels["text"] for v in SPLITTERS for c in split_clauses(t, v)}
                          | {turkish_lower(t) for t in labels["text"]})
    split, mode = cfg["split"], cfg["mode"]
    val = labels[labels.split == "val"].reset_index(drop=True)
    test = labels[labels.split == "test"].reset_index(drop=True)

    # --- 1. Ana tablo ---
    vr, vs = metrics(val, predict(val, split, mode, cache))
    tp_pred = predict(test, split, mode, cache)
    tr, ts = metrics(test, tp_pred)
    out.append("1) ANA TABLO — konu tespiti (test | val)")
    out.append(f"{'konu':<12}{'n_test':>7}{'P':>7}{'R':>7}{'F1':>7}{'duygu':>9}   |{'n_val':>6}{'P':>7}{'R':>7}  not")
    for a in ASPECTS:
        t, v = tr[a], vr[a]
        out.append(f"{a:<12}{t['n']:>7}{t['P']:>7.2f}{t['R']:>7.2f}{t['F1']:>7.2f}"
                   f"{t['duygu_ok']:>5}/{t['duygu_n']:<3}  |{v['n']:>6}{v['P']:>7.2f}{v['R']:>7.2f}"
                   f"  {'gürültülü (n<30)' if t['n'] < 30 else ''}")
    out.append(f"\n{'':<24}{'test':>10}{'val':>10}")
    for k, name in [("tespit_P_micro", "konu P micro"), ("tespit_R_micro", "konu R micro"),
                    ("tespit_F1_micro", "konu F1 micro"), ("tespit_F1_macro", "konu F1 macro"),
                    ("duygu_acc_micro", "duygu micro"), ("duygu_acc_macro", "duygu macro"),
                    ("duygu_acc_micro_k0", "duygu karisik=0"), ("uc_F1_micro", "uçtan uca F1 micro"),
                    ("uc_F1_macro", "uçtan uca F1 macro")]:
        out.append(f"{name:<24}{ts[k]:>10.3f}{vs[k]:>10.3f}")
    out.append(f"{'altın nötr çift':<24}{ts['altin_notr']:>10}{vs['altin_notr']:>10}   (nötr kapalı -> hepsi kaçar)")

    # --- 2. Karşılaştırma ---
    out.append("\n2) KARŞILAŞTIRMA (test, yeni altın etiketler)")
    out.append(f"{'kurulum':<28}{'duygu mic/mac':>16}{'k0':>7}{'uç F1 mic/mac':>16}{'konu F1 mic':>12}")
    for name, v, pats in [("V2 + A/b (ANA)", split, PATTERNS), ("(a) V2 + bölme yok/b", "yok", PATTERNS),
                          ("(b) V1 + A/b", split, PATTERNS_V1)]:
        _, s = metrics(test, predict(test, v, mode, cache, pats))
        out.append(f"{name:<28}{s['duygu_acc_micro']:>9.3f}/{s['duygu_acc_macro']:.3f}{s['duygu_acc_micro_k0']:>7.3f}"
                   f"{s['uc_F1_micro']:>9.3f}/{s['uc_F1_macro']:.3f}{s['tespit_F1_micro']:>12.3f}")

    # --- 3. Elmayla elma ---
    out.append("\n3) ELMAYLA ELMA — Görkan'ın kör 20'si, KARAR ÖNCESİ etiketler (n çok küçük: SADECE FİKİR VERİR)")
    human = pd.read_csv(os.path.join(HERE, "..", "data", "aspect_labels", "human_blind_20.csv"), keep_default_na=False)
    changes = pd.read_csv(os.path.join(HERE, "changes_rule_scan.csv"))
    revert = changes[changes.kural == "uyum_karari"].set_index("id")["eski"].to_dict()  # karar öncesi Claude
    lab = labels.set_index("id")
    h = human.assign(claude=[revert.get(i, lab.at[i, "raw"]) for i in human.id], text=[lab.at[i, "text"] for i in human.id])
    all_pred = predict(h, split, mode, cache)
    h["model"] = [p for p, _ in all_pred]
    h["c"] = [parse(r)[0] for r in h.claude]
    h["g"] = [parse(r)[0] for r in h.raw_gorkan]
    out.append(f"{'alt küme':<26}{'n':>4}{'model-Claude':>14}{'model-Görkan':>14}{'Claude-Görkan':>15}")
    for name, sub in [("tümü (20)", h), ("kör 18 (2,3 hariç)", h[~h.no.isin([2, 3])]),
                      ("sadece test (10)", h[h.split == "test"])]:
        mc, mg, cg = (pair_f1(list(sub.model), list(sub.c))[0], pair_f1(list(sub.model), list(sub.g))[0],
                      pair_f1(list(sub.c), list(sub.g))[0])
        out.append(f"{name:<26}{len(sub):>4}{mc:>14.3f}{mg:>14.3f}{cg:>15.3f}")

    # --- 4. Hata kovaları ---
    out.append("\n4) HATA KOVALARI (test)")
    golds = gold(test)
    fp, fn, sent = [], [], []
    for i, (g, (p, used)) in enumerate(zip(golds, tp_pred)):
        r = test.iloc[i]
        for a in ASPECTS:
            if a in p and a not in g:
                m = PATTERNS[a].search(turkish_lower(r.text))
                fp.append((a, r.id, r.text, r.raw, m.group(0)))
            elif a in g and a not in p:
                fn.append((a, r.id, r.text, r.raw))
            elif a in g and a in p and g[a] != p[a]:
                # cümlecikte zıt duygulu başka bir altın konunun kelimesi geçiyor mu?
                conflict = [b for b in g if b != a and g[b] != g[a] and any(PATTERNS[b].search(c) for c in used[a])]
                # yorum hiç bölünemediyse (tek cümlecik) ve zıt duygulu başka altın konu varsa -> yine bölme hatası
                whole = len(used[a]) == 1 and used[a][0] == turkish_lower(r.text) and len(split_clauses(r.text, split)) == 1
                if not conflict and whole:
                    conflict = [b for b in g if b != a and g[b] != g[a]]
                kind = "bölme hatası" if conflict else "V2b hatası"
                sent.append((kind, a, r.id, r.text, r.raw, p[a], " | ".join(used[a])[:120]))

    # FN elle sınıflandırma
    if not os.path.exists(FN_MANUAL):
        pd.DataFrame([(a, i, "", t) for a, i, t, _ in fn], columns=["konu", "id", "kova", "text"]).to_csv(FN_MANUAL, index=False)
        out.append(f"(FN elle sınıflandırma şablonu yazıldı: {FN_MANUAL} — 'kova' sütununu doldurup tekrar çalıştır)")
    manual = pd.read_csv(FN_MANUAL, keep_default_na=False)
    fn_kova = {(r.konu, r.id): r.kova for r in manual.itertuples()}

    out.append(f"Konu FP (yanlış eşleşme): {len(fp)} | konu FN: {len(fn)} | duygu hatası: {len(sent)}")
    out.append(f"\n{'konu':<12}{'FP yanlış_eşl.':>15}{'FN örtük':>10}{'FN kelime_eksik':>17}{'duygu bölme':>13}{'duygu V2b':>11}  baskın")
    for a in ASPECTS:
        c = {"yanlış eşleşme": sum(x[0] == a for x in fp),
             "örtük": sum(x[0] == a and fn_kova.get((a, x[1])) == "örtük" for x in fn),
             "kelime eksik": sum(x[0] == a and fn_kova.get((a, x[1])) == "kelime eksik" for x in fn),
             "bölme hatası": sum(x[1] == a and x[0] == "bölme hatası" for x in sent),
             "V2b hatası": sum(x[1] == a and x[0] == "V2b hatası" for x in sent)}
        top = max(c, key=c.get) if any(c.values()) else "-"
        out.append(f"{a:<12}{c['yanlış eşleşme']:>15}{c['örtük']:>10}{c['kelime eksik']:>17}{c['bölme hatası']:>13}"
                   f"{c['V2b hatası']:>11}  {top}")
    unl = sum(1 for a, i, _, _ in fn if not fn_kova.get((a, i)))
    if unl:
        out.append(f"(UYARI: {unl} FN henüz elle sınıflandırılmadı)")

    def ex(title, items, fmt):
        out.append(f"\n-- {title}: {len(items)} --")
        for it in items[:3]:
            out.append("  " + fmt(it))
    ex("FP / yanlış eşleşme", fp, lambda x: f"{x[1]} [{x[0]}] eşleşen='{x[4]}' altın={x[3]} | {x[2][:110]}")
    for kova in ["örtük", "kelime eksik"]:
        items = [x for x in fn if fn_kova.get((x[0], x[1])) == kova]
        ex(f"FN / {kova}", items, lambda x: f"{x[1]} [{x[0]}] altın={x[3]} tahmin=yok | {x[2][:110]}")
    for kova in ["bölme hatası", "V2b hatası"]:
        items = [x for x in sent if x[0] == kova]
        ex(f"Duygu / {kova}", items, lambda x: f"{x[2]} [{x[1]}] altın={x[4]} tahmin={x[5]} | cümlecik: {x[6]}")

    open(LOG, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
