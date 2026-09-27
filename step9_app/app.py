"""
ADIM 9: Modeli bir uygulamaya dönüştürmek (FastAPI).

Gerçek yorumlarla eğittiğimiz BERT modelini (Adım 12: step12_confident_learning/model/)
bir web servisine koyuyoruz. Artık model bir script'in içinde değil —
herhangi bir program (web sayfası, mobil uygulama, başka bir servis) ona
HTTP isteği atıp cevap alabilir.

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
"""

import os

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
# Adım 12: confident learning ile temizlenmiş veride eğitilen model
# (öncekiler: step10_negation/model, step8_real_data/model)
MODEL_DIR = os.path.join(HERE, "..", "step12_confident_learning", "model")
LABELS = ["negatif", "pozitif"]

# Adım 11: güven bu eşiğin altındaysa "belirsiz" diyoruz. Eşik VAL setinde seçildi
# (step11_confidence/choose_threshold.py). Model değişince eşik de YENİDEN seçilmeli:
# Adım 12 modeli daha "emin" konuştuğu için aynı doğruluk hedefi (>=0.93) %95 eşik gerektirdi.
# Gerçek testte: cevap verilen %82 yorumda doğruluk 0.944, "belirsiz" denenlerde 0.642.
CONFIDENCE_THRESHOLD = 0.95

# Model uygulama açılırken BİR KEZ yüklenir, her istekte değil (yükleme birkaç saniye sürer)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()


def turkish_lower(text: str) -> str:
    # Python'ın .lower()'ı Türkçeyi bilmez: "I" -> "i" (doğrusu "ı"),
    # "İ" -> "i̇" (görünmez ekstra nokta karakteriyle). Önce bu ikisini elle düzeltiyoruz.
    return text.replace("I", "ı").replace("İ", "i").lower()


@torch.no_grad()
def predict(text: str) -> dict:
    encoded = tokenizer(turkish_lower(text), truncation=True, max_length=128, return_tensors="pt")
    probs = torch.softmax(model(**encoded).logits, dim=1)[0].tolist()
    best = max(range(len(LABELS)), key=lambda i: probs[i])
    confident = probs[best] >= CONFIDENCE_THRESHOLD
    return {
        "label": LABELS[best] if confident else "belirsiz",
        "leaning": LABELS[best],              # emin olmasa da hangi tarafa yakın
        "confident": confident,
        "confidence": round(probs[best], 4),
        "probabilities": {name: round(p, 4) for name, p in zip(LABELS, probs)},
    }


app = FastAPI(title="gorkanai — Türkçe Duygu Analizi")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=["Kargo çok hızlıydı, ürün harika!"])


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    return {"text": req.text, **predict(req.text)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))
