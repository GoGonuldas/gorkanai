---
language: tr
license: other
base_model: dbmdz/bert-base-turkish-cased
pipeline_tag: text-classification
tags: [aspect-based-sentiment, multi-label, turkish, product-reviews]
---
# gorkanai-tr-aspect-topic

Türkçe ürün yorumlarında **hangi konulardan bahsedildiğini** bulan çok etiketli (multi-label) BERT. 7 konu, her biri ayrı
sigmoid çıkış. [gorkanai](https://github.com/GoGonuldas/gorkanai) öğrenme projesinin Adım 19 modeli; konu duygusu için
eşi: `Urartu65/gorkanai-tr-aspect-sentiment`, genel duygu için `Urartu65/gorkanai-tr-sentiment`.

**Bu tek tohumlu sürümdür** (tohum 0, epoch 16). Projede ölçülen ana yapılandırma 3 tohumun olasılık ortalamasıydı.

## Çıkışlar (sıra önemli)
`0 kargo · 1 fiyat · 2 kalite · 3 performans · 4 boyut · 5 görünüm · 6 satıcı` — olasılık ≥ **0.70** ise konu var.

## Kullanım
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ASPECTS = ["kargo", "fiyat", "kalite", "performans", "boyut", "gorunum", "satici"]
turkish_lower = lambda t: t.replace("I", "ı").replace("İ", "i").lower()   # eğitimdeki ön işleme — ŞART
tok = AutoTokenizer.from_pretrained("Urartu65/gorkanai-tr-aspect-topic")
model = AutoModelForSequenceClassification.from_pretrained("Urartu65/gorkanai-tr-aspect-topic").eval()
text = "Kargo çok hızlıydı ama ürün kırık geldi, satıcı da cevap vermedi."
with torch.no_grad():
    p = torch.sigmoid(model(**tok(turkish_lower(text), truncation=True, max_length=128, return_tensors="pt")).logits)[0]
print([a for a, q in zip(ASPECTS, p) if q >= 0.70])
```

## Sonuçlar
| Ölçüm | Değer |
|---|---|
| Val 300, **bu tek tohum** | konu F1 micro 0.844, macro 0.800 |
| Val 300, 3 tohum ortalaması | micro 0.852, macro 0.806 |
| 500 yorumluk temiz test, 3 tohum ortalaması (bir kez ölçüldü) | micro 0.852, macro 0.809; satıcı F1 0.51 (recall 0.37) |

## Eğitim
1400 yorum (8-40 kelime), Claude oturumlarının kör etiketleri, kurallar: projedeki `step17_aspect_sentiment/LABEL_RULES.md`
(v1). BCE + pos_weight = sqrt(neg/pos), lr 3e-5, batch 16, max_len 128.

## Sınırlamalar
Altın etiketler Claude etiketi; bir insana karşı (100 yorum) "kabul edilebilir" konu-duygu F1 0.823 (uygulama hattı,
3 tohum). Satıcı ve görünüm zayıf; kalite/performans sınırı yoruma açık. Ürün yorumları dışında denenmedi.
Veri kaynağı: Hugging Face `fthbrmnby/turkish_product_reviews` — kullanım koşulları o veri setine bağlıdır.
