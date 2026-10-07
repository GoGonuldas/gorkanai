---
language: tr
license: other
base_model: Urartu65/gorkanai-tr-sentiment
pipeline_tag: text-classification
tags: [aspect-based-sentiment, turkish, product-reviews]
---
# gorkanai-tr-aspect-sentiment

Türkçe bir ürün yorumunda **belirli bir konunun** duygusunu (pozitif / negatif) veren BERT. Girdi bir çift:
`[CLS] <konu ifadesi> [SEP] <yorum> [SEP]`. [gorkanai](https://github.com/GoGonuldas/gorkanai) projesinin Adım 17 modeli;
konuları bulmak için eşi: `Urartu65/gorkanai-tr-aspect-topic`.

**Bu tek tohumlu sürümdür** (tohum 0, epoch 8). Ölçülen ana yapılandırma 3 tohumun olasılık ortalamasıydı.
Başlangıç: 3 sınıflı genel duygu modeli `Urartu65/gorkanai-tr-sentiment`; başlık 3 çıkışlı kaldı ama **sadece negatif (0)
ve pozitif (2) logit'leri** kullanılır (nötr eğitilmedi).

## Konu ifadeleri (eğitimdekiyle aynı yazılmalı)
`kargo → "kargo ve teslimat"`, `fiyat → "fiyat"`, `kalite → "kalite"`, `performans → "performans ve özellikler"`,
`boyut → "boyut"`, `gorunum → "görünüm"`, `satici → "satıcı ve hizmet"`

## Kullanım
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
turkish_lower = lambda t: t.replace("I", "ı").replace("İ", "i").lower()   # eğitimdeki ön işleme — ŞART
tok = AutoTokenizer.from_pretrained("Urartu65/gorkanai-tr-aspect-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("Urartu65/gorkanai-tr-aspect-sentiment").eval()
text = "Kargo çok hızlıydı ama ürün kırık geldi, satıcı da cevap vermedi."
enc = tok("satıcı ve hizmet", turkish_lower(text), truncation="only_second", max_length=160, return_tensors="pt")
with torch.no_grad():
    p_pos = torch.softmax(model(**enc).logits[0, [0, 2]], dim=0)[1].item()
print("pozitif" if p_pos >= 0.5 else "negatif", round(p_pos, 3))
```

## Sonuçlar ((i) = altın konular verildiğinde duygu doğruluğu)
| Ölçüm | Değer |
|---|---|
| Val 300, **bu tek tohum** | (i) 0.891 |
| Val 300, 3 tohum ortalaması | (i) 0.889 |
| 500 yorumluk temiz test, 3 tohum ortalaması (bir kez ölçüldü) | (i) 0.896 (eski cümlecik hattı 0.862; zıt duygulu yorumlarda 0.695 → 0.812) |
| Bir insana karşı, ortak konularda duygu uyumu (100 yorum) | %94 |

## Sınırlamalar
Konu başına nötr yok. Bazı kelimelere takılır ("iade" → olumsuz satıcı). Altın etiketler Claude etiketi, kurallar
projedeki `LABEL_RULES.md` (v1). Veri kaynağı: `fthbrmnby/turkish_product_reviews` — kullanım koşulları o veri setine bağlıdır.
