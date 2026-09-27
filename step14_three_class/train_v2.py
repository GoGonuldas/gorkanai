"""
ADIM 14 - Deney 3: Nötr ile çelişen eğitim verisini temizlemek.

Deney 2'de 3 sınıflı model gerçek nötrlerin sadece %31'ini yakaladı; çoğuna "negatif" dedi.
Tahmin: poz/neg eğitim verisindeki ETİKETSİZ kısa yorumların arasında nötr parçalar var
(Adım 11: kısa "negatif" etiketlilerin ~%15'i nötr) -> model aynı türden cümleye bir yerde
"nötr", yüzlerce yerde "negatif" etiketi görüyor.

İki varyant (ikisi de önceden sabitlendi):
  V2a: etiketsiz kısa (<=10 kelime) poz/neg yorumların HEPSİNİ çıkar.
       Risk: elle etiketli sadece ~31 kısa negatif kalır -> "kısa yorum negatif olmaz" kısayolu.
  V2b: etiketsiz kısa yorumlardan sadece Adım 12'nin K-fold tahmininin (görmediği model)
       etiketle ≥%90 aynı fikirde olduklarını tut. Net olanlar ("berbat") kalır,
       belirsiz parçalar gider.

Nötr VAL seti: elle etiketli 1000 yorumun %20'si (sınıf oranları korunarak) eğitimden ayrılır.
  - varyant seçimi val'de macro-F1 ile
  - "nötr" kararının eşiği: nötr logit'ine eklenen bir sabit (bias), val'de macro-F1 ile seçilir
Test setine (kısa temiz, 200) sadece en sonda bakılır.

GÜNCELLEME (Deney 3b): İlk çalıştırmada val'de sadece 8 negatif vardı ve val yanlış varyantı (V2a)
seçti — V2a'nın "kısa negatif -> nötr" kısayolu val'de görünmedi. Val'e, hiçbir eğitimde olmayan
150 kısa yorum eklendi (data/neutral_labels/batch_05.csv, prepare_val_negatives.py): val artık
145 poz / 96 nötr / 109 neg. batch_05 SADECE val'de kullanılır, eğitime girmez.
Test sonuçlarına ilk çalıştırmada bakıldı -> test artık "görülmüş" sayılır, sayıları iyimserdir.

Çalıştırma: python train_v2.py   (~25 dk; model_v2a/ ve model_v2b/ kaydedilir, seçilen: model_v2/ yok)
"""

import gc
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
C_MODEL_DIR = "../step12_confident_learning/model"
V1_MODEL_DIR = "model"
HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIRS = {"V2a": os.path.join(HERE, "model_v2a"), "V2b": os.path.join(HERE, "model_v2b")}
N_TRAIN = 16000
NEUTRAL_OVERSAMPLE = 8
EPOCHS = 2
SHORT_WORDS = 10
AGREE_MIN = 0.90
C_THRESHOLD = 0.95
LABELS = ["negatif", "nötr", "pozitif"]
label_map = {name: i for i, name in enumerate(LABELS)}
NEG_PATTERN = re.compile(r"\b(?:fena|kötü|berbat|kalitesiz)\w*\s+(?:değil|degil|sayılmaz|sayilmaz)")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Cihaz: {device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def turkish_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


def release_memory():
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


# --- Adım 12'nin eğitim setini (train_b) ve K-fold tahminlerini birebir yeniden kur ---
pool = pd.read_csv("../data/real/train_pool.csv")
pool["has_pattern"] = pool["text"].str.contains(NEG_PATTERN)
pattern_rows = pool[pool["has_pattern"]]
pattern_ratio = (pattern_rows["label"] == "pozitif").mean()
pool = pool.drop(pattern_rows.sample(200, random_state=42).index)
pos = pool[pool["label"] == "pozitif"].sample(frac=1, random_state=42).iloc[500:]
neg = pool[pool["label"] == "negatif"].sample(frac=1, random_state=42).iloc[500:]
train_a = pd.concat([pos.iloc[:N_TRAIN // 2], neg.iloc[:N_TRAIN // 2]])
n_neg_pat = (train_a["has_pattern"] & (train_a["label"] == "negatif")).sum()
n_pos_pat = (train_a["has_pattern"] & (train_a["label"] == "pozitif")).sum()
target_pos = int(round(n_neg_pat * pattern_ratio / (1 - pattern_ratio)))
extra_pool = pos.iloc[N_TRAIN // 2:]
extra = extra_pool[extra_pool["has_pattern"]].head(max(0, target_pos - n_pos_pat))
train_b = pd.concat([train_a, extra]).reset_index(drop=True)
oof = np.load("../step12_confident_learning/oof_probs.npy")
assert len(oof) == len(train_b)
given = train_b["label"].map({"negatif": 0, "pozitif": 1}).values
train_b["agree"] = oof[np.arange(len(train_b)), given]
flagged = set(pd.read_csv("../step12_confident_learning/flagged.csv")["text"])
train_c = train_b[~train_b["text"].isin(flagged)].copy()

# --- Elle etiketli 1000 yorum: %20 val, %80 eğitim (sınıf oranları korunarak) ---
manual = pd.concat([pd.read_csv(f"../data/neutral_labels/batch_0{i}.csv") for i in range(1, 5)])
val_split = pd.concat([manual[manual["label"] == c].sample(frac=0.2, random_state=0) for c in LABELS])
manual_train = manual[~manual["id"].isin(val_split["id"])]
val_extra = pd.read_csv("../data/neutral_labels/batch_05.csv")   # sadece val — eğitime girmez
val_df = pd.concat([val_split, val_extra])
print(f"Val: {val_df['label'].value_counts().to_dict()} | elle-eğitim: {manual_train['label'].value_counts().to_dict()}")

train_c = train_c[~train_c["text"].isin(set(manual["text"]))]
train_c["short"] = train_c["text"].str.split().str.len() <= SHORT_WORDS
variants = {
    "V2a (kısa etiketsizler yok)": train_c[~train_c["short"]],
    "V2b (sadece net kısa)": train_c[~train_c["short"] | (train_c["agree"] >= AGREE_MIN)],
}
neutral = manual_train[manual_train["label"] == "nötr"][["text", "label"]]
manual_sent = manual_train[manual_train["label"] != "nötr"][["text", "label"]]

# --- Testler ---
short_test = pd.read_csv("../data/short_clean_test.csv").rename(columns={"clean_label": "label"})
real_test = pd.read_csv("../data/real/test.csv")


def make_batches(texts, labels=None, batch_size=16, shuffle=False):
    idx = np.arange(len(texts))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        b = idx[start:start + batch_size]
        enc = tokenizer([turkish_lower(texts[i]) for i in b], padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        y = None if labels is None else torch.tensor(labels[b], dtype=torch.long, device=device)
        yield enc, y


@torch.no_grad()
def get_logits(model, texts):
    model.eval()
    return np.concatenate([model(**enc).logits.float().cpu().numpy()
                           for enc, _ in make_batches(texts, batch_size=64)])


def decide(logits, bias):
    lg = logits.copy()
    lg[:, label_map["nötr"]] += bias
    return np.array(LABELS)[lg.argmax(1)]


def scores(true, pred):
    out = {}
    for c in LABELS:
        tp = ((pred == c) & (true == c)).sum()
        p, r = tp / max((pred == c).sum(), 1), tp / max((true == c).sum(), 1)
        out[c] = (p, r, 2 * p * r / max(p + r, 1e-9))
    return (pred == true).mean(), np.mean([f for _, _, f in out.values()]), out


def best_bias(logits, true):
    grid = np.round(np.arange(-3, 3.01, 0.25), 2)
    f1s = [scores(true, decide(logits, b))[1] for b in grid]
    return grid[int(np.argmax(f1s))], max(f1s)


# --- Eğitim: iki varyant, val'de karşılaştır ---
trained = {}
for name, pn in variants.items():
    train_df = pd.concat([pn[["text", "label"]], manual_sent] + [neutral] * NEUTRAL_OVERSAMPLE)
    leak = set(val_df["text"]) & set(train_df["text"])
    assert not leak, f"val metinleri eğitimde: {list(leak)[:3]}"
    short_sent = pn[pn["short"]]
    print(f"\n=== {name}: {len(train_df)} satır | {train_df['label'].value_counts().to_dict()}")
    print(f"    etiketsiz kısa yorum: {len(short_sent)} (neg {(short_sent['label'] == 'negatif').sum()}, "
          f"poz {(short_sent['label'] == 'pozitif').sum()})")
    torch.manual_seed(42)
    np.random.seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    texts, labels = train_df["text"].tolist(), train_df["label"].map(label_map).values
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        for enc, y in make_batches(texts, labels, shuffle=True):
            optimizer.zero_grad()
            loss = model(**enc, labels=y).loss
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        print(f"    epoch {epoch} | loss {total / n:.4f}")
    print(f"    süre: {(time.time() - start) / 60:.1f} dk")
    val_logits = get_logits(model, val_df["text"].tolist())
    bias, val_f1 = best_bias(val_logits, val_df["label"].values)
    _, val_f1_0, _ = scores(val_df["label"].values, decide(val_logits, 0.0))
    print(f"    VAL macro-F1: bias=0 -> {val_f1_0:.3f} | en iyi bias={bias:+.2f} -> {val_f1:.3f}")
    trained[name] = {"val_f1": val_f1, "bias": bias,
                     "test_logits": get_logits(model, short_test["text"].tolist()),
                     "real_logits": get_logits(model, real_test["text"].tolist())}
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = label_map
    save_dir = SAVE_DIRS[name[:3]]
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    trained[name]["dir"] = save_dir
    print(f"    kaydedildi: {save_dir}")
    del model, optimizer
    release_memory()

chosen = max(trained, key=lambda k: trained[k]["val_f1"])
print(f"\nVal'e göre seçilen: {chosen} (bias {trained[chosen]['bias']:+.2f})")

# --- Kıyaslar: model C + %95, Deney 2 modeli (v1) ---
c_model = AutoModelForSequenceClassification.from_pretrained(C_MODEL_DIR).to(device)
c_probs = torch.softmax(torch.tensor(get_logits(c_model, short_test["text"].tolist())), 1).numpy()
del c_model
release_memory()
baseline_pred = np.where(c_probs.max(1) < C_THRESHOLD, "nötr", np.where(c_probs.argmax(1) == 1, "pozitif", "negatif"))
v1_model = AutoModelForSequenceClassification.from_pretrained(V1_MODEL_DIR).to(device)
v1_val_logits = get_logits(v1_model, val_df["text"].tolist())  # (v1 val'i eğitimde gördü -> bias seçimi iyimser)
v1_test_logits = get_logits(v1_model, short_test["text"].tolist())
del v1_model
release_memory()

print("\n=== KISA TEMİZ TEST (97 poz / 35 nötr / 68 neg) — DİKKAT: ilk çalıştırmada görüldü, sayılar iyimser ===")
true = short_test["label"].values
rows = [("Kıyas: C + %95 eşik", baseline_pred), ("Deney 2 (v1), bias 0", decide(v1_test_logits, 0.0))]
for name, t in trained.items():
    rows.append((f"{name}, bias 0", decide(t["test_logits"], 0.0)))
    rows.append((f"{name}, val-bias {t['bias']:+.2f}", decide(t["test_logits"], t["bias"])))
print(f"{'':42s} {'doğr.':>6s} {'mF1':>6s} | {'nötr P/R':>10s} | {'neg P/R':>10s} | {'poz P/R':>10s}")
for name, pred in rows:
    acc, mf1, sc = scores(true, pred)
    pr = " | ".join(f"{sc[c][0]:.2f}/{sc[c][1]:.2f}".rjust(10) for c in ["nötr", "negatif", "pozitif"])
    print(f"{name:42s} {acc:6.3f} {mf1:6.3f} | {pr}")

final_pred = decide(trained[chosen]["test_logits"], trained[chosen]["bias"])
print(f"\nSeçilen model ({chosen}) karışıklık matrisi:")
print(pd.crosstab(pd.Series(true, name="doğrusu"), pd.Series(final_pred, name="tahmin"))
      .reindex(index=LABELS, columns=LABELS, fill_value=0).to_string())
real_pred = decide(trained[chosen]["real_logits"], trained[chosen]["bias"])
answered = real_pred != "nötr"
print(f"\nGerçek test (2 sınıf, gürültülü): nötr deme %{100 * (~answered).mean():.1f} | "
      f"cevap verilenlerde doğruluk {(real_pred[answered] == real_test['label'].values[answered]).mean():.3f}")

# --- Kısayol testi (seçilen model) ---
model = AutoModelForSequenceClassification.from_pretrained(trained[chosen]["dir"]).to(device)
probe = ["çok sağlam", "harika", "berbat", "bayıldım", "iade ettim", "tavsiye etmem", "süper ürün", "çok kötü",
         "fena değil", "idare eder", "kargo 2 günde geldi.", "saygılarımla.", "bunda da aynı bedeni alın.",
         "ürün dün elime ulaştı.", "telefonun rengi siyah.", "Ürün 2019 yılında piyasaya sürülmüştür.",
         "Cihaz yüksek performansı ile kullanıcılar tarafından beğenilmiştir.",
         "ürünü aldıktan bir hafta sonra bozuldu, müşteri hizmetleri de hiç yardımcı olmadı."]
pl = get_logits(model, probe)
pp = torch.softmax(torch.tensor(pl), 1).numpy()
print(f"\n=== Kısayol testi ({chosen}, bias {trained[chosen]['bias']:+.2f}) ===")
for text, lab, p in zip(probe, decide(pl, trained[chosen]["bias"]), pp):
    print(f"  {lab:8s} (nötr olasılığı %{100 * p[1]:3.0f}) | {text}")

print("\n--- Seçilen modelin test hataları ---")
for text, t, p in zip(short_test["text"], true, final_pred):
    if t != p:
        print(f"  doğrusu={t:8s} tahmin={p:8s} | {text}")
