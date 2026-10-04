"""
ADIM 20: Konu bazlı analiz hattı — uygulamanın kullandığı TEK yer.

Bir yorum için:
  1. Konu tespiti: Adım 19 v2 B modeli (3 tohumun olasılık ortalaması, eşik frozen_config.json'dan).
  2. Bulunan her konu için duygu: Adım 17 modeli, girdi [CLS] <konu ifadesi> [SEP] <yorum> [SEP]
     (3 tohumun P(pozitif) ortalaması, >= 0.5 -> pozitif).

Önemli ders — "ölçtüğün modeli mi çalıştırıyorsun?":
Ön işleme (turkish_lower, max_length, konu ifadeleri) burada YENİDEN YAZILMIYOR; eğitim script'lerinin
predict/encode fonksiyonları import ediliyor. Ayarlar (epoch, eşik, tohumlar) da elle değil, dondurulmuş
frozen_config.json dosyalarından okunuyor. step20_aspect_app/test_pipeline.py bu hattın val 300'de
değerlendirmedeki olasılıkların aynısını verdiğini assert ediyor.

Hafif mod (ASPECT_LIGHT=1): konu ve duygu için sadece ilk tohum. Ölçülen yapılandırma 3 tohum olduğu için
varsayılan her zaman 3 tohum; hafif mod sadece val kıyası (test_pipeline.py) fark < 1 puan derse kullanılmalı.
"""

import importlib.util
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
S17 = os.path.join(ROOT, "step17_aspect_sentiment")
S19 = os.path.join(ROOT, "step19_topic_v2")

# Adım 17'nin train.py'si Adım 16'nınkini "train16" adıyla zaten yüklüyor; ikisi de train.py olduğu için adla yüklenir.
_spec = importlib.util.spec_from_file_location("train17", os.path.join(S17, "train.py"))
T17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T17)
T16 = T17._train16
ASPECTS, DEVICE = T16.ASPECTS, T16.DEVICE
DISPLAY = {"kargo": "kargo", "fiyat": "fiyat", "kalite": "kalite", "performans": "performans",
           "boyut": "boyut", "gorunum": "görünüm", "satici": "satıcı"}


def _load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
    return tok, model


class AspectPipeline:
    def __init__(self, light=False):
        f19 = json.load(open(os.path.join(S19, "frozen_config.json")))
        f17 = json.load(open(os.path.join(S17, "frozen_config.json")))
        n = 1 if light else None
        self.light = light
        self.threshold = f19["threshold"]
        self.topic_models = [_load(os.path.join(ROOT, d)) for d in f19["model_dirs"][:n]]
        self.sent_models = [_load(os.path.join(S17, f"model_{f17['init']}_s{s}_e{f17['epoch']}")) for s in f17["seeds"][:n]]
        self.cols = T17.INITS[f17["init"]][1]   # (negatif, pozitif) logit sütunları

    @torch.no_grad()
    def topic_probs(self, texts):
        """(len(texts), 7) konu olasılıkları — tohum ortalaması."""
        return sum(T16.predict(m, t, texts) for t, m in self.topic_models) / len(self.topic_models)

    @torch.no_grad()
    def sentiment_probs(self, konular, texts):
        """Her (konu, yorum) çifti için P(pozitif) — tohum ortalaması."""
        return sum(T17.predict(m, t, self.cols, konular, texts) for t, m in self.sent_models) / len(self.sent_models)

    def analyze(self, text):
        p = self.topic_probs([text])[0]
        found = [a for a, q in zip(ASPECTS, p) if q >= self.threshold]
        ps = self.sentiment_probs(found, [text] * len(found)) if found else []
        out = [{"konu": DISPLAY[a], "olasilik": round(float(p[ASPECTS.index(a)]), 4),
                "duygu": "pozitif" if q >= 0.5 else "negatif", "guven": round(float(max(q, 1 - q)), 4)}
               for a, q in zip(found, ps)]
        return sorted(out, key=lambda r: -r["olasilik"])
