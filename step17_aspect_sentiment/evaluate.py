"""
ADIM 17 - Değerlendirme: konuya koşullu duygu modeli vs Adım 15 duygu hattı (bölme A + V2b, nötr kapalı; dondurulmuş).

  python evaluate.py val    -> log_val.txt + frozen_config.json (başlangıç, epoch, tohumlar, hash'ler)
  python evaluate.py test   -> log_test.txt — BİR KEZ: yeni test (ana: gorkanai-1e altını) + eski test (kıyas satırı).

Ölçüler (PLAN.md madde 4, önceden sabit):
  (i)  altın konular verilmişken duygu doğruluğu: micro, macro (konu başına), karisik=0, nötr hariç.
       Altın nötr çiftler nötr kapalı olduğu için iki hatta da HATA sayılır.
  (ii) uçtan uca (konu, duygu) çift F1 micro/macro: Adım 16 BERT konuları (dondurulmuş, eşik 0.60) + duygu hattı.
  (iii) eşleştirilmiş bootstrap (yorum bazında, 2000 tekrar, tohum 17): yeni − eski hat; "kazandı" = aralık 0'ı dışlıyor.
  (iv) alt kümeler, (i) ile: zıt duygulu yorumlar / tek duygulu yorumlar / örtük çiftler (konunun V2 kelimesi
       hiçbir cümlecikte geçmiyor).
Seçim kuralı: val (i) micro, 3 tohum ortalaması en yüksek (başlangıç, epoch); 0.005 içinde eşitlikte az epoch, sonra bert.
"""

import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import ASPECTS, DEVICE, EPOCHS, HERE, INITS, ROOT, SEEDS, load_data, pairs, predict
from baseline import PATTERNS, clause_logits, decide, split_clauses, turkish_lower  # noqa: E402
from save_aspect_labels import parse  # noqa: E402

S16 = os.path.join(ROOT, "step16_topic_bert")
FROZEN = os.path.join(HERE, "frozen_config.json")
GOLD_DIR = os.path.join(ROOT, "data", "aspect_labels_step17")
SPLIT, MODE, TOPIC_THR, CEILING = "A", "b", 0.60, 0.967


def sha(path, n=64):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:n]


def baseline_matrix(texts):
    """Adım 15 duygu hattı: her (yorum, konu) için duygu + o konunun kelimesi bir cümlecikte geçiyor mu."""
    cache = clause_logits({c for t in texts for c in split_clauses(t, SPLIT)} | {turkish_lower(t) for t in texts})
    sent, explicit = [], []
    for t in texts:
        clauses = split_clauses(t, SPLIT)
        row_s, row_e = [], []
        for a in ASPECTS:
            cl = [c for c in clauses if PATTERNS[a].search(c)]
            row_e.append(bool(cl))
            row_s.append(decide([cache[c] for c in (cl or [turkish_lower(t)])], MODE))
        sent.append(row_s)
        explicit.append(row_e)
    return np.array(sent), np.array(explicit)


def model_matrix(cfg, texts):
    """Dondurulmuş duygu modeli (tohum ortalaması): her (yorum, konu) için P(pozitif) -> duygu."""
    cols = INITS[cfg["init"]][1]
    probs = []
    for s in cfg["seeds"]:
        d = os.path.join(HERE, f"model_{cfg['init']}_s{s}_e{cfg['epoch']}")
        tok, model = AutoTokenizer.from_pretrained(d), AutoModelForSequenceClassification.from_pretrained(d).to(DEVICE)
        flat_k = [a for _ in texts for a in ASPECTS]
        flat_t = [t for t in texts for _ in ASPECTS]
        probs.append(predict(model, tok, cols, flat_k, flat_t).reshape(len(texts), len(ASPECTS)))
        del model
    p = np.mean(probs, axis=0)
    return np.where(p >= 0.5, "pozitif", "negatif"), p


def gold_matrix(df):
    g = df[ASPECTS].to_numpy().astype(str)
    mixed = np.array([[a in str(m).split(";") for a in ASPECTS] for m in df["karisik_konular"]])
    return g, mixed


def report(name, df, topic_pred, base, new, explicit, out, mask_pairs=None):
    """(i), (ii), (iii), (iv) — bir küme için. mask_pairs: (i)'yi sadece bu çiftlerde hesapla (ortak çiftler satırı)."""
    g, mixed = gold_matrix(df)
    has = g != ""
    if mask_pairs is not None:
        has = has & mask_pairs
    n_pairs = int(has.sum())
    ok = {"eski hat": (base == g) & has, "YENİ": (new == g) & has}
    opp = np.array([{"pozitif", "negatif"} <= set(r[r != ""]) for r in g])        # zıt duygulu yorum
    out.append(f"\n=== {name}: {len(df)} yorum, {n_pairs} altın çift (nötr {int(((g == 'nötr') & has).sum())}, "
               f"karışık {int((mixed & has).sum())}; zıt duygulu yorum {int(opp.sum())}) ===")
    out.append(f"(i) altın konularla duygu doğruluğu{'':<6}{'micro':>8}{'macro':>8}{'karisik=0':>11}{'nötr hariç':>12}"
               f"{'zıt duyg.':>11}{'tek duyg.':>11}{'örtük':>8}{'açık':>8}")
    sub = {"zıt": has & opp[:, None], "tek": has & ~opp[:, None], "örtük": has & ~explicit, "açık": has & explicit}
    for k, o in ok.items():
        macro = np.mean([o[:, j].sum() / has[:, j].sum() for j in range(len(ASPECTS)) if has[:, j].sum()])
        nm, nn = has & ~mixed, has & (g != "nötr")
        out.append(f"  {k:<36}{o.sum() / n_pairs:>8.3f}{macro:>8.3f}{(o & nm).sum() / max(nm.sum(), 1):>11.3f}"
                   f"{(o & nn).sum() / max(nn.sum(), 1):>12.3f}" + "".join(
                       f"{(o & m).sum() / max(m.sum(), 1):>{w}.3f}" for m, w in zip(sub.values(), (11, 11, 8, 8))))
    out.append(f"  {'(alt küme çift sayıları)':<36}{n_pairs:>8}{'':>8}{int((has & ~mixed).sum()):>11}{int((has & (g != 'nötr')).sum()):>12}"
               + "".join(f"{int(m.sum()):>{w}}" for m, w in zip(sub.values(), (11, 11, 8, 8))))
    out.append("  konu başına (eski → YENİ, n): " + ", ".join(
        f"{a} {ok['eski hat'][:, j].sum() / max(has[:, j].sum(), 1):.2f}→{ok['YENİ'][:, j].sum() / max(has[:, j].sum(), 1):.2f} ({int(has[:, j].sum())})"
        for j, a in enumerate(ASPECTS)))
    out.append(f"  iki hattın anlaşmadığı çift: {int(((base != new) & has).sum())} "
               f"(YENİ doğru {int((ok['YENİ'] & ~ok['eski hat']).sum())}, eski doğru {int((ok['eski hat'] & ~ok['YENİ']).sum())})")

    # (ii) uçtan uca
    res = {"n_pairs": n_pairs}
    full = g != ""
    if mask_pairs is None:
        tp = {"eski hat": topic_pred & (base == g), "YENİ": topic_pred & (new == g)}
        f1 = lambda t, idx=slice(None): 2 * t[idx].sum() / max(topic_pred[idx].sum() + full[idx].sum(), 1)
        macro = lambda t: np.mean([2 * t[:, j].sum() / max(topic_pred[:, j].sum() + full[:, j].sum(), 1) for j in range(len(ASPECTS))])
        out.append(f"(ii) uçtan uca çift F1 (Adım 16 konuları, eşik {TOPIC_THR}): "
                   + " | ".join(f"{k} micro {f1(t):.3f} macro {macro(t):.3f}" for k, t in tp.items()))
        # Adım 16 konu modelinin bu kümedeki ölçümü (anahtar kelime V2 çizgisiyle) — duygu hattından bağımsız
        kw = np.array([[bool(PATTERNS[a].search(turkish_lower(t))) for a in ASPECTS] for t in df["text"]])
        tf1 = lambda pr, idx=slice(None): 2 * (pr & full)[idx].sum() / max(pr[idx].sum() + full[idx].sum(), 1)
        tmac = lambda pr: np.mean([2 * (pr & full)[:, j].sum() / max(pr[:, j].sum() + full[:, j].sum(), 1) for j in range(len(ASPECTS))])
        out.append("     KONU TESPİTİ (Adım 16 modeli dondurulmuş; duygudan bağımsız):")
        for k_, pr in (("anahtar kelime V2", kw), ("BERT (Adım 16)", topic_pred)):
            miss = full & ~kw
            out.append(f"       {k_:<20} P {(pr & full).sum() / max(pr.sum(), 1):.3f} R {(pr & full).sum() / full.sum():.3f} "
                       f"F1 micro {tf1(pr):.3f} macro {tmac(pr):.3f} | kelime tutmayan çiftlerde R {(pr & miss).sum() / max(miss.sum(), 1):.3f} (n={int(miss.sum())})")
        out.append("       konu başına F1 (kelime → BERT, n): " + ", ".join(
            f"{a} {2 * (kw & full)[:, j].sum() / max(kw[:, j].sum() + full[:, j].sum(), 1):.2f}→"
            f"{2 * (topic_pred & full)[:, j].sum() / max(topic_pred[:, j].sum() + full[:, j].sum(), 1):.2f} ({int(full[:, j].sum())})"
            for j, a in enumerate(ASPECTS)))
        rng_t = np.random.default_rng(16)
        dt = np.array([tf1(topic_pred, i) - tf1(kw, i) for i in (rng_t.integers(0, len(df), len(df)) for _ in range(2000))])
        lo_t, hi_t = np.quantile(dt, [0.025, 0.975])
        out.append(f"       bootstrap BERT − kelime, konu F1 micro: {tf1(topic_pred) - tf1(kw):+.3f} [{lo_t:+.3f}, {hi_t:+.3f}] "
                   + ("0'ı DIŞLIYOR" if lo_t > 0 or hi_t < 0 else "0'ı içeriyor"))
    # (iii) bootstrap
    rng = np.random.default_rng(17)
    idxs = [rng.integers(0, len(df), len(df)) for _ in range(2000)]
    acc = lambda o, i: o[i].sum() / max(has[i].sum(), 1)
    d_acc = np.array([acc(ok["YENİ"], i) - acc(ok["eski hat"], i) for i in idxs])
    lines = [("(i) micro doğruluk", acc(ok["YENİ"], slice(None)) - acc(ok["eski hat"], slice(None)), d_acc)]
    if mask_pairs is None:
        d_f1 = np.array([f1(tp["YENİ"], i) - f1(tp["eski hat"], i) for i in idxs])
        lines.append(("(ii) uçtan uca micro", f1(tp["YENİ"]) - f1(tp["eski hat"]), d_f1))
    out.append("(iii) eşleştirilmiş bootstrap (2000, tohum 17), YENİ − eski hat, %95 aralık:")
    for label, point, d in lines:
        lo, hi = np.quantile(d, [0.025, 0.975])
        verdict = "0'ı DIŞLIYOR" if lo > 0 or hi < 0 else "0'ı içeriyor -> gürültüden ayrılamıyor"
        out.append(f"  {label:<24}{point:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")
        res[label] = (point, lo, hi)
    return res


def topic_probs_val(val):
    npz = np.load(os.path.join(S16, "val_probs.npz"))
    assert (npz["ids"] == val["id"].to_numpy()).all()
    return np.mean([npz[f"s{s}_e12"] for s in (0, 1, 2)], axis=0)


def run_val():
    train, val = load_data()
    val = val.reset_index(drop=True)
    vp = pairs(val)
    npz = np.load(os.path.join(HERE, "val_probs.npz"), allow_pickle=True)
    assert (npz["ids"] == vp["id"].to_numpy()).all() and (npz["konu"] == vp["konu"].to_numpy()).all()
    gold = vp["duygu"].to_numpy()
    acc = lambda p: float((np.where(p >= 0.5, "pozitif", "negatif") == gold).mean())
    out = [f"ADIM 17 — VAL ({len(val)} yorum, {len(vp)} altın çift; nötr {int((gold == 'nötr').sum())} -> nötr kapalıyken tavan "
           f"{(gold != 'nötr').mean():.3f}). Testlere dokunulmadı.",
           f"(i) ölçüsünün pratik tavanı (iki altının yeni testte ortak konulardaki duygu uyumu): {CEILING}\n",
           "1) IZGARA — val (i) micro doğruluk: tohum 0/1/2 | ortalama (min-max) | 3 tohum olasılık ortalaması"]
    means = {}
    for init in INITS:
        for e in EPOCHS:
            per = [acc(npz[f"{init}_s{s}_e{e}"]) for s in SEEDS]
            means[(init, e)] = np.mean(per)
            out.append(f"  {init:<5} epoch {e:>2}: " + " ".join(f"{v:.3f}" for v in per)
                       + f" | {np.mean(per):.3f} ({min(per):.3f}-{max(per):.3f}) | {acc(np.mean([npz[f'{init}_s{s}_e{e}'] for s in SEEDS], axis=0)):.3f}")
    best = max(means.values())
    init, epoch = sorted([k for k, v in means.items() if v >= best - 0.005], key=lambda k: (k[1], k[0] != "bert"))[0]
    out.append(f"  en yüksek ortalama {best:.3f}; 0.005 içindekiler: {sorted(k for k, v in means.items() if v >= best - 0.005)}")
    out.append(f"  SEÇİLEN: başlangıç = {init}, epoch = {epoch}" + ("  (ızgaranın üst sınırı!)" if epoch == max(EPOCHS) else ""))

    cfg = {"init": init, "epoch": int(epoch), "seeds": list(SEEDS)}
    new, p = model_matrix(cfg, val["text"].tolist())
    base, explicit = baseline_matrix(val["text"].tolist())
    topic_pred = topic_probs_val(val) >= TOPIC_THR
    out.append("\n2) SEÇİLEN MODEL vs ADIM 15 DUYGU HATTI (val)")
    report("VAL 300", val, topic_pred, base, new, explicit, out)
    for name, m in (("ESKİ VAL 100 (Adım 15 altını)", val["origin"] == "eski"), ("YENİ VAL 200 (bu oturumun etiketleri)", val["origin"] == "yeni")):
        m = m.to_numpy()
        report(name, val[m].reset_index(drop=True), topic_pred[m], base[m], new[m], explicit[m], out)

    cfg["model_sha256_16"] = [sha(os.path.join(HERE, f"model_{init}_s{s}_e{epoch}", "model.safetensors"), 16) for s in SEEDS]
    cfg["gold_1e_sha256"] = sha(os.path.join(GOLD_DIR, "gold_1e.csv"))
    cfg["gold_macmini_sha256"] = sha(os.path.join(GOLD_DIR, "gold_macmini.csv"))
    cfg["topic_model"] = "step16_topic_bert/frozen_config.json (epoch 12, eşik 0.60, tohum 0/1/2)"
    json.dump(cfg, open(FROZEN, "w"), indent=1, ensure_ascii=False)
    out.append(f"\nDONDURULAN -> frozen_config.json: {json.dumps(cfg, ensure_ascii=False)}")
    open(os.path.join(HERE, "log_val.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


def gold_df(who, texts):
    g = pd.read_csv(os.path.join(GOLD_DIR, f"gold_{who}.csv"), keep_default_na=False)
    df = texts.merge(g, on="id", validate="one_to_one")
    parsed = df["raw"].map(parse)
    names = {"K": "kargo", "F": "fiyat", "Q": "kalite", "P": "performans", "B": "boyut", "G": "gorunum", "S": "satici"}
    for a in ASPECTS:
        df[a] = parsed.map(lambda q: q[0].get(a, ""))
    df["karisik_konular"] = parsed.map(lambda q: ";".join(q[1]))
    return df


def run_test():
    log = os.path.join(HERE, "log_test.txt")
    assert not os.path.exists(log), "test zaten ölçüldü (log_test.txt var) — ikinci ölçüm yok"
    cfg = json.load(open(FROZEN))
    assert sha(os.path.join(GOLD_DIR, "gold_1e.csv")) == cfg["gold_1e_sha256"], "gold_1e.csv dondurulduktan sonra değişmiş"
    assert sha(os.path.join(GOLD_DIR, "gold_macmini.csv")) == cfg["gold_macmini_sha256"], "gold_macmini.csv değişmiş"
    out = [f"ADIM 17 — TEST, BİR KEZ. DONDURULAN: {json.dumps({k: cfg[k] for k in ('init', 'epoch', 'seeds')})}",
           f"(i) ölçüsünün pratik tavanı (iki altının ortak konulardaki duygu uyumu): {CEILING}"]

    # --- yeni test (ANA) ---
    texts = pd.read_csv(os.path.join(HERE, "test17_set.csv"))[["id", "text"]]
    s16 = json.load(open(os.path.join(S16, "frozen_config.json")))
    tprobs = []
    for s in s16["seeds"]:
        d = os.path.join(S16, f"model_s{s}_e{s16['epoch']}")
        tok, model = AutoTokenizer.from_pretrained(d), AutoModelForSequenceClassification.from_pretrained(d).to(DEVICE)
        model.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = tok([turkish_lower(t) for t in texts["text"].tolist()[i:i + 64]], padding=True, truncation=True,
                          max_length=128, return_tensors="pt").to(DEVICE)
                chunks.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
        tprobs.append(np.concatenate(chunks))
        del model
    topic_p = np.mean(tprobs, axis=0)
    new, p = model_matrix(cfg, texts["text"].tolist())
    base, explicit = baseline_matrix(texts["text"].tolist())
    np.savez(os.path.join(HERE, "test17_probs.npz"), ids=texts["id"].to_numpy(), topic=topic_p, sent_pos=p)
    topic_pred = topic_p >= s16["threshold"]
    d1, dm = gold_df("1e", texts), gold_df("macmini", texts)
    assert (d1["id"] == texts["id"]).all() and (dm["id"] == texts["id"]).all()
    report("YENİ TEST — gorkanai-1e altını (ANA SONUÇ)", d1, topic_pred, base, new, explicit, out)
    report("YENİ TEST — bu oturumun (macmini) altını", dm, topic_pred, base, new, explicit, out)
    g1, gm = gold_matrix(d1)[0], gold_matrix(dm)[0]
    common = (g1 != "") & (g1 == gm)
    report("YENİ TEST — iki altının ORTAK çiftleri (aynı konu + aynı duygu)", d1, topic_pred, base, new, explicit, out, mask_pairs=common)

    # --- eski test (kıyas; hataları 16.5'te okundu -> iyimser olabilir) ---
    old = pd.read_csv(os.path.join(ROOT, "data", "aspect_labels", "aspect_labels.csv"), keep_default_na=False)
    test = old[old["split"] == "test"].reset_index(drop=True)
    tz = np.load(os.path.join(S16, "test_probs.npz"))
    assert (tz["ids"] == test["id"].to_numpy()).all()
    topic_old = np.mean([tz[f"s{s}"] for s in s16["seeds"]], axis=0) >= s16["threshold"]
    new_o, p_o = model_matrix(cfg, test["text"].tolist())
    base_o, exp_o = baseline_matrix(test["text"].tolist())
    np.savez(os.path.join(HERE, "test15_sent_probs.npz"), ids=test["id"].to_numpy(), sent_pos=p_o)
    report("ESKİ TEST (Adım 15; KIYAS — hataları okundu, iyimser olabilir; kıyas uçtan uca 0.697/0.636)",
           test, topic_old, base_o, new_o, exp_o, out)

    # --- birleşik 400 (ikincil) ---
    both = pd.concat([d1[["id", "text"] + ASPECTS + ["karisik_konular"]], test[["id", "text"] + ASPECTS + ["karisik_konular"]]], ignore_index=True)
    report("BİRLEŞİK 400 (yeni test 1e altını + eski test; İKİNCİL)", both, np.vstack([topic_pred, topic_old]),
           np.vstack([base, base_o]), np.vstack([new, new_o]), np.vstack([explicit, exp_o]), out)

    # --- Görkan'ın kör 20'si ---
    human = pd.read_csv(os.path.join(HERE, "human_blind_20.csv"), keep_default_na=False)
    if (human["etiket"].str.strip() != "").all():
        hid = [int(np.where(texts["id"].to_numpy() == i)[0][0]) for i in human["id"]]
        dh = texts.iloc[hid].reset_index(drop=True)
        parsed = [parse(",".join(t for t in re.split(r"[,\s]+", r.strip()) if t)) for r in human["etiket"]]
        for a in ASPECTS:
            dh[a] = [q[0].get(a, "") for q in parsed]
        dh["karisik_konular"] = [";".join(q[1]) for q in parsed]
        report("YENİ TEST — Görkan'ın kör 20'si (n=20, SADECE FİKİR VERİR)", dh, topic_pred[hid], base[hid], new[hid], explicit[hid], out)
    else:
        out.append("\n(Görkan'ın kör 20'si doldurulmadı; o satır yok.)")
    open(log, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    {"val": run_val, "test": run_test}[sys.argv[1]]()
