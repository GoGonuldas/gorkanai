"""
ADIM 9: Modeli bir uygulamaya dönüştürmek (FastAPI).
ADIM 14 devamı (2026-09-28): 3 sınıflı modele (V2b) geçiş.
ADIM 20 (2026-10-04): /aspects — konu bazlı analiz (Adım 19 konu modeli + Adım 17 konuya koşullu duygu modeli),
  hattın kendisi aspect_pipeline.py'de. /predict (genel duygu) aynen duruyor.

Gerçek yorumlarla eğittiğimiz BERT modelini bir web servisine koyuyoruz.
Artık model bir script'in içinde değil — herhangi bir program (web sayfası,
mobil uygulama, başka bir servis) ona HTTP isteği atıp cevap alabilir.

Çalıştırmak için:
    cd step9_app && source ../.venv/bin/activate
    uvicorn app:app --reload
Sonra tarayıcıda: http://127.0.0.1:8000        (web arayüzü)
               http://127.0.0.1:8000/docs   (otomatik API dokümantasyonu)

Önemli ders — "eğitimde ne yaptıysan, kullanırken de onu yap":
Model hep KÜÇÜK HARFLİ yorumlarla eğitildi. Kullanıcı "Bu Ürün Harika"
yazarsa model alışık olmadığı bir girdi görür. Bu yüzden her metni önce
eğitim verisine benzetiyoruz (turkish_lower). Sentetik zor testte bu tek
satır doğruluğu 0.81 -> 0.875 çıkardı.

Adım 11-13'teki "emin değilim" eşiği (2 sınıflı model, güven < eşikse belirsiz)
KALDIRILDI: nötr sınıfı artık o işlevi doğrudan üstleniyor (hiç görülmemiş
testte nötr kesinliği %38 -> %58). Eşiği 3 sınıf için yeniden seçmek ayrı bir
val ölçümü gerektirirdi; basitlik için modele doğrudan güveniyoruz.
"""

import os

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from aspect_pipeline import AspectPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
# Adım 14 Deney 3b: nötr sınıfını da öğrenen, val'de seçilen model
# (öncekiler: step12_confident_learning/model, step10_negation/model, step8_real_data/model)
# Yerelde bu klasörden, HF Spaces'te MODEL_DIR ortam değişkeniyle Hub'daki repodan (Urartu65/gorkanai-tr-sentiment) yüklenir.
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HERE, "..", "step14_three_class", "model_v2b"))

# Val'de macro-F1 ile seçilen sabit: nötr kararı ancak bu kadar logit avantajıyla veriliyor
# (Adım 14 Deney 3b, train_v2.py). Test setine bakılarak yeniden seçilmedi.
NEUTRAL_BIAS = 3.00

# Model uygulama açılırken BİR KEZ yüklenir, her istekte değil (yükleme birkaç saniye sürer)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
id2label = model.config.id2label
LABELS = [id2label[i] for i in range(len(id2label))]
NEUTRAL_IDX = LABELS.index("nötr")


def turkish_lower(text: str) -> str:
    # Python'ın .lower()'ı Türkçeyi bilmez: "I" -> "i" (doğrusu "ı"),
    # "İ" -> "i̇" (görünmez ekstra nokta karakteriyle). Önce bu ikisini elle düzeltiyoruz.
    return text.replace("I", "ı").replace("İ", "i").lower()


@torch.no_grad()
def predict(text: str) -> dict:
    encoded = tokenizer(turkish_lower(text), truncation=True, max_length=128, return_tensors="pt")
    logits = model(**encoded).logits[0].clone()
    logits[NEUTRAL_IDX] += NEUTRAL_BIAS
    probs = torch.softmax(logits, dim=0).tolist()
    best = max(range(len(LABELS)), key=lambda i: probs[i])
    return {
        "label": LABELS[best],
        "confidence": round(probs[best], 4),
        "probabilities": {name: round(p, 4) for name, p in zip(LABELS, probs)},
    }


# Konu hattı 6 BERT daha yükler (~2.6 GB); ASPECT_LIGHT=1 ile tohum başına 1 model (bkz. aspect_pipeline.py)
aspects = AspectPipeline(light=os.environ.get("ASPECT_LIGHT") == "1")

app = FastAPI(title="gorkanai — Türkçe Duygu Analizi")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=["Kargo çok hızlıydı, ürün harika!"])


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    return {"text": req.text, **predict(req.text)}


@app.post("/aspects")
def aspects_endpoint(req: PredictRequest):
    return {"text": req.text, "konular": aspects.analyze(req.text), "genel": predict(req.text),
            "hafif_mod": aspects.light}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))
