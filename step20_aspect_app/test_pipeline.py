"""
ADIM 20 (2) - Uygulamanın konu hattı ölçtüğümüz modeli mi çalıştırıyor? MAC MİNİ'DE çalışır (modeller orada).

  python test_pipeline.py  -> log_test_pipeline.txt

1. EŞDEĞERLİK (PLAN §3.1): uygulamanın kendi AspectPipeline nesnesi (app.aspects) val 300'de çalıştırılır.
   - konu olasılıkları == step19_topic_v2/val_probs_B.npz, seçilen epoch'ta 3 tohum ortalaması
   - duygu olasılıkları (val altın çiftleri) == step17_aspect_sentiment/val_probs.npz, dondurulmuş init/epoch ortalaması
   atol 1e-3; tutmazsa assert -> DUR.
2. DUMAN TESTİ: FastAPI TestClient ile /health, /predict, /aspects (biçim, boş girdi 422, 2000+ karakter 422).
3. GECİKME: val'den 20 yorum /aspects, medyan ve maks (ilk istek ısınma olarak ayrı).
4. HAFİF MOD (PLAN §2, sadece val): tek tohum vs 3 tohum — konu F1 micro/macro ve (i) duygu doğruluğu.
   Kural (önceden): iki ölçüde de fark < 0.01 ise hafif mod "kullanılabilir"; varsayılan yine 3 tohum.
Testlere (hiçbiri) dokunulmaz; val zaten Adım 16-19'da okunmuş küme.
"""

import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step9_app"))

from fastapi.testclient import TestClient  # noqa: E402

import app as webapp  # noqa: E402  (modelleri BİR KEZ yükler: V2b + konu 3 + duygu 3)
from aspect_pipeline import ASPECTS, S17, S19, T16, AspectPipeline  # noqa: E402

ATOL = 1e-3
out = ["ADIM 20 — uygulama hattı doğrulaması (val 300; testlere dokunulmadı)", f"cihaz: {T16.DEVICE}"]
pipe = webapp.aspects
_, val = T16.load_data()
texts = val["text"].tolist()

# 1) eşdeğerlik — konu
f19 = json.load(open(os.path.join(S19, "frozen_config.json")))
npz19 = np.load(os.path.join(S19, "val_probs_B.npz"))
assert (npz19["ids"] == val["id"].to_numpy()).all()
ref_topic = np.mean([npz19[f"s{s}_e{f19['epoch']}"] for s in f19["seeds"]], axis=0)
app_topic = pipe.topic_probs(texts)
d_topic = float(np.abs(app_topic - ref_topic).max())
flips_topic = int(((app_topic >= pipe.threshold) != (ref_topic >= pipe.threshold)).sum())

# 1) eşdeğerlik — duygu (val altın çiftleri, Adım 17 val_probs.npz sırasıyla)
f17 = json.load(open(os.path.join(S17, "frozen_config.json")))
npz17 = np.load(os.path.join(S17, "val_probs.npz"), allow_pickle=True)
id2text = dict(zip(val["id"], val["text"]))
konu, pair_texts = list(npz17["konu"]), [id2text[i] for i in npz17["ids"]]
ref_sent = np.mean([npz17[f"{f17['init']}_s{s}_e{f17['epoch']}"] for s in f17["seeds"]], axis=0)
app_sent = pipe.sentiment_probs(konu, pair_texts)
d_sent = float(np.abs(app_sent - ref_sent).max())
flips_sent = int(((app_sent >= 0.5) != (ref_sent >= 0.5)).sum())

out += ["\n1) EŞDEĞERLİK (atol 1e-3):",
        f"  konu : {app_topic.shape} olasılık, maks |fark| {d_topic:.2e}, eşikte (0.70) değişen karar {flips_topic}",
        f"  duygu: {len(konu)} altın çift, maks |fark| {d_sent:.2e}, 0.5'te değişen karar {flips_sent}"]
ok_equiv = d_topic <= ATOL and d_sent <= ATOL
out.append(f"  → {'GEÇTİ: uygulama ölçtüğümüz modelleri çalıştırıyor' if ok_equiv else 'GEÇMEDİ — DUR'}")
if not ok_equiv:
    open(os.path.join(HERE, "log_test_pipeline.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    raise AssertionError("eşdeğerlik tutmadı")

# 2) duman testi
client = TestClient(webapp.app)
r_health = client.get("/health")
r_pred = client.post("/predict", json={"text": "Kargo çok hızlıydı, ürün harika!"})
r_asp = client.post("/aspects", json={"text": "Kargo çok hızlıydı ama ürün kırık geldi, satıcı da cevap vermedi."})
r_empty = client.post("/aspects", json={"text": ""})
r_long = client.post("/aspects", json={"text": "a" * 2001})
body = r_asp.json()
assert r_health.status_code == 200 and r_pred.status_code == 200 and r_asp.status_code == 200
assert set(body) == {"text", "konular", "genel", "hafif_mod"}
assert all(set(k) == {"konu", "olasilik", "duygu", "guven"} for k in body["konular"])
assert r_empty.status_code == 422 and r_long.status_code == 422
out += ["\n2) DUMAN TESTİ: /health 200, /predict 200, /aspects 200 (biçim doğru), boş 422, 2001 karakter 422 → GEÇTİ",
        f"  örnek (elle yazılmış cümle): {json.dumps(body['konular'], ensure_ascii=False)}"]

# 3) gecikme
sample = texts[:21]
client.post("/aspects", json={"text": sample[0]})   # ısınma
times = []
for t in sample[1:]:
    t0 = time.perf_counter()
    client.post("/aspects", json={"text": t})
    times.append(time.perf_counter() - t0)
out.append(f"\n3) GECİKME (/aspects, 20 val yorumu, 3 tohum): medyan {statistics.median(times) * 1000:.0f} ms, "
           f"maks {max(times) * 1000:.0f} ms")

# 4) hafif mod — sadece val
y = T16.targets(val).astype(bool)
gold_pairs = np.array([val.loc[val["id"] == i, k].iloc[0] for i, k in zip(npz17["ids"], konu)])


def topic_f1(pred):
    tp = (pred & y).sum(0)
    per = 2 * tp / np.maximum(pred.sum(0) + y.sum(0), 1)
    return 2 * tp.sum() / max(pred.sum() + y.sum(), 1), per.mean()


acc = lambda p: float((np.where(p >= 0.5, "pozitif", "negatif") == gold_pairs).mean())
light = AspectPipeline(light=True)
lt, ls = light.topic_probs(texts), light.sentiment_probs(konu, pair_texts)
(mi_f, ma_f), (mi_l, ma_l) = topic_f1(app_topic >= pipe.threshold), topic_f1(lt >= pipe.threshold)
a_f, a_l = acc(app_sent), acc(ls)
usable = abs(mi_f - mi_l) < 0.01 and abs(a_f - a_l) < 0.01
out += ["\n4) HAFİF MOD (val 300; tek tohum vs 3 tohum):",
        f"  konu F1 micro {mi_f:.3f} → {mi_l:.3f}, macro {ma_f:.3f} → {ma_l:.3f}; (i) duygu doğruluğu {a_f:.3f} → {a_l:.3f}",
        f"  kural (fark < 0.01, micro ve (i)) → {'KULLANILABİLİR (ASPECT_LIGHT=1); varsayılan yine 3 tohum' if usable else 'KULLANILMAMALI'}"]
t0 = time.perf_counter()
for t in sample[1:]:
    light.analyze(t)
out.append(f"  hafif mod hattı ortalama {(time.perf_counter() - t0) / 20 * 1000:.0f} ms/yorum (HTTP olmadan)")

open(os.path.join(HERE, "log_test_pipeline.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
