"""
ADIM 13: Temperature scaling — modelin güvenini gerçekçi hale getirmek.

Adım 12'deki model (C) daha doğru ama fazla emin: kalibrasyon hatası (ECE) 0.035'ten
0.069'a çıktı, "Saygılarımla." gibi duygusuz bir cümleye %97 "pozitif" diyor.

Temperature scaling (Guo ve ark., 2017): modelin ham çıktılarını (logit) softmax'tan
önce tek bir sayıya — T'ye — bölüyoruz:
    olasılık = softmax(logit / T)
  T = 1  -> değişiklik yok
  T > 1  -> güvenler yumuşar (%99 -> %90 gibi)
  T < 1  -> güvenler sertleşir
T, sınıfların SIRASINI değiştirmez -> hangi cevabın seçildiği, yani doğruluk aynı kalır.
Sadece "ne kadar emin" kısmı değişir. Model yeniden eğitilmez.

T'yi VAL setinde, negatif log-olabilirliği (NLL) en aza indirerek buluyoruz.
Sonra "emin değilim" eşiğini de (Adım 11 kuralıyla) val'de yeniden seçiyoruz.
Test setlerine sadece doğrulama için bakıyoruz.

Uyarı: val etiketleri gürültülü (Adım 11-12). Model hatalı etiketlere güvenle "karşı
çıktığında" bu cezalandırılır -> bulunan T biraz büyük (temkinli) çıkabilir.

Çıktı: step13_temperature/calibration.json  (app bunu okur)
"""

import json
import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "../step12_confident_learning/model"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
TARGET_ACC = 0.93   # Adım 11'deki hedef: cevap verilenlerde doğruluk >= %93
NEG_PATTERN = re.compile(r"\b(?:fena|kötü|berbat|kalitesiz)\w*\s+(?:değil|degil|sayılmaz|sayilmaz)")
label_map = {"negatif": 0, "pozitif": 1}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()


def turkish_lower(text):
    return text.replace("I", "ı").replace("İ", "i").lower()


@torch.no_grad()
def get_logits(texts, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [turkish_lower(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        out.append(model(**enc).logits.float().cpu())
    return torch.cat(out)


def fit_temperature(logits, labels):
    # log(T) üzerinden optimize ediyoruz ki T hep pozitif kalsın
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.exp().item()


def probs(logits, t):
    return torch.softmax(logits / t, dim=1).numpy()


def ece(p, y):
    conf, ok = p.max(1), p.argmax(1) == y
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.001]
    return sum(((conf >= lo) & (conf < hi)).mean() * abs(conf[(conf >= lo) & (conf < hi)].mean()
               - ok[(conf >= lo) & (conf < hi)].mean())
               for lo, hi in zip(bins[:-1], bins[1:]) if ((conf >= lo) & (conf < hi)).any())


def reliability(p, y, title):
    conf, ok = p.max(1), p.argmax(1) == y
    print(f"  {title}")
    for lo, hi in [(0.5, 0.7), (0.7, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.001)]:
        m = (conf >= lo) & (conf < hi)
        if m.any():
            print(f"    güven %{100 * lo:.0f}-%{min(100 * hi, 100):.0f}: {m.sum():4d} örnek | "
                  f"ort. güven {conf[m].mean():.3f} | gerçek doğruluk {ok[m].mean():.3f}")


# --- Veri: val (Adım 10/11'deki ile birebir aynı), testler ---
pool = pd.read_csv("../data/real/train_pool.csv")
holdout = pool[pool["text"].str.contains(NEG_PATTERN)].sample(200, random_state=42)
pool = pool.drop(holdout.index)
pos = pool[pool["label"] == "pozitif"].sample(frac=1, random_state=42)
neg = pool[pool["label"] == "negatif"].sample(frac=1, random_state=42)
val_df = pd.concat([pos.iloc[:500], neg.iloc[:500]])
test_df = pd.read_csv("../data/real/test.csv")
short = pd.read_csv("../data/short_clean_test.csv")
short_sent = short[short["clean_label"] != "nötr"]
short_neutral = short[short["clean_label"] == "nötr"]

val_logits = get_logits(val_df["text"].tolist())
val_y = torch.tensor(val_df["label"].map(label_map).values)
test_logits = get_logits(test_df["text"].tolist())
test_y = test_df["label"].map(label_map).values
short_logits = get_logits(short_sent["text"].tolist())
short_y = short_sent["clean_label"].map(label_map).values
neutral_logits = get_logits(short_neutral["text"].tolist())

# --- 1) T'yi val'de bul ---
T = fit_temperature(val_logits, val_y)
print(f"=== 1) Val'de bulunan sıcaklık: T = {T:.3f} ===")
print("(T > 1 -> model fazla eminmiş, güvenler yumuşatılacak)\n")

# --- 2) Önce / sonra kalibrasyon ---
print("=== 2) Kalibrasyon hatası (ECE, düşük = iyi) ===")
for name, lg, y in [("val (T burada seçildi)", val_logits, val_y.numpy()),
                    ("gerçek test (gürültülü)", test_logits, test_y),
                    ("kısa temiz (elle)", short_logits, short_y)]:
    print(f"  {name:26s}: önce {ece(probs(lg, 1.0), y):.3f} -> sonra {ece(probs(lg, T), y):.3f}")
print()
reliability(probs(test_logits, 1.0), test_y, "Gerçek test — ÖNCE (T=1)")
reliability(probs(test_logits, T), test_y, f"Gerçek test — SONRA (T={T:.2f})")

# --- 3) Eşiği val'de yeniden seç (ölçeklenmiş olasılıklarla) ---
val_p = probs(val_logits, T)
val_conf, val_ok = val_p.max(1), val_p.argmax(1) == val_y.numpy()
threshold = None
for t in np.arange(0.50, 0.991, 0.01):
    m = val_conf >= t
    if m.any() and val_ok[m].mean() >= TARGET_ACC:
        threshold = round(float(t), 2)
        break
print(f"\n=== 3) Yeni 'emin değilim' eşiği (val'e göre, hedef >= {TARGET_ACC}): %{100 * threshold:.0f} ===")


def app_view(lg, y, t_scale, thr):
    p = probs(lg, t_scale)
    hi = p.max(1) >= thr
    return hi.mean(), (p.argmax(1) == y)[hi].mean()


print(f"{'':34s} {'önce (T=1, eşik %95)':>22s} {'sonra (T, yeni eşik)':>22s}")
for name, lg, y in [("gerçek test: kapsam / doğruluk", test_logits, test_y),
                    ("kısa temiz: kapsam / doğruluk", short_logits, short_y)]:
    c0, a0 = app_view(lg, y, 1.0, 0.95)
    c1, a1 = app_view(lg, y, T, threshold)
    print(f"{name:34s} {f'%{100 * c0:.0f} / {a0:.3f}':>22s} {f'%{100 * c1:.0f} / {a1:.3f}':>22s}")
n0 = (probs(neutral_logits, 1.0).max(1) < 0.95).mean()
n1 = (probs(neutral_logits, T).max(1) < threshold).mean()
print(f"{'nötr yorumlarda emin değilim':34s} {f'%{100 * n0:.0f}':>22s} {f'%{100 * n1:.0f}':>22s}")

# --- 4) Örnek cümleler ---
examples = ["Saygılarımla.", "Kargo 2 günde geldi.", "Bunda da aynı bedeni alın.", "Kumaşı kötü değil.",
            "Eh işte fena değil.", "Hiç memnun değilim.", "Kargo çok hızlıydı, ürün harika!"]
ex_logits = get_logits(examples)
print("\n=== 4) Örnekler: güven önce -> sonra ===")
for text, p0, p1 in zip(examples, probs(ex_logits, 1.0), probs(ex_logits, T)):
    lab = ["negatif", "pozitif"][p0.argmax()]
    flag = "" if p1.max() >= threshold else "  -> EMİN DEĞİLİM"
    print(f"  {lab:8s} %{100 * p0.max():5.1f} -> %{100 * p1.max():5.1f}{flag:18s} | {text}")

with open(OUT_PATH, "w") as f:
    json.dump({"model_dir": "step12_confident_learning/model", "temperature": round(T, 4),
               "threshold": threshold, "target_acc": TARGET_ACC}, f, indent=2)
print(f"\nKaydedildi: {OUT_PATH}")
