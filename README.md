# gorkanai

Sıfırdan başlayarak adım adım kurulan bir **Türkçe duygu analizi** (sentiment analysis) projesi.
En basit yöntemden (Bag-of-Words) başlayıp BERT fine-tuning'e, gerçek veriye, veri temizliğine ve
bir web uygulamasına kadar ilerliyor. Her adımda bir sınırla karşılaşıp sebebini anlamak ve
bir sonraki adımda çözmek amaçlandı.

Ayrıntılı sonuçlar, her adımın dersleri ve kararlar: **[PROGRESS.md](PROGRESS.md)**

## Adımlar

| # | Klasör | Ne yapıyor |
|---|---|---|
| 1-7 | `step1_bow_logreg` … `step7_finetune` | Sentetik veride BoW → TF-IDF → NN → Word2Vec → LSTM → Attention → BERT |
| 8 | `step8_real_data` | Gerçek ürün yorumlarıyla fine-tuning (sentetik model gerçek veride 0.58 → gerçek veriyle ~0.89) |
| 9 | `step9_app` | FastAPI servisi + web arayüzü, "emin değilim" durumu |
| 10 | `step10_negation` | Çift olumsuzlama ("hiç fena değil") — örnekleme hatası bulundu ve düzeltildi |
| 11 | `step11_confidence` | Kalibrasyon analizi, "emin değilim" eşiğinin val setinde seçilmesi |
| 12 | `step12_confident_learning` | Eğitim verisindeki hatalı etiketleri bulup temizlemek |
| 13 | `step13_temperature` | Temperature scaling (denendi, uygulamaya alınmadı — nedeni PROGRESS.md'de) |
| 14 | `step14_three_class` | Nötr sınıfı: Vikipedi nötrleriyle kısayol öğrenme, elle etiketlenmiş nötrlerle 3 sınıflı model (devam ediyor) |

## Kurulum

Python 3.12 gerekli.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn torch gensim transformers datasets fastapi uvicorn
```

Gerçek veri seti (Hugging Face `fthbrmnby/turkish_product_reviews`) indirilip bölünür:

```bash
cd data && python prepare_real_dataset.py
```

## Modeller

Eğitilmiş modeller (~420 MB) depoda **yok**; ilgili adımın script'i ile yeniden üretilir.
Uygulamanın kullandığı model Adım 12'ninki:

```bash
# Adım 12 modeli için önce Adım 10'un eğitim seti kurulumu kullanılır (script kendi içinde yapar)
cd step12_confident_learning && python train.py      # ~30-40 dk (Apple M2 Pro, MPS)
```

## Uygulamayı çalıştırmak

```bash
cd step9_app && uvicorn app:app --reload
```

- http://127.0.0.1:8000 — web arayüzü
- http://127.0.0.1:8000/docs — API dokümantasyonu (`POST /predict {"text": "..."}`)

## Elle etiketlenmiş veriler

| Dosya | İçerik |
|---|---|
| `data/hard_test.csv` | 16 sentetik zor cümle (olumsuzlama, eş anlamlılar) |
| `data/negation_test.csv` | 25 cümle: çift olumsuzlama + basit olumsuzlama |
| `data/short_clean_test.csv` | 200 gerçek kısa yorum, orijinal + temiz etiket (pozitif/negatif/nötr) |
| `data/neutral_labels/` | 1000 gerçek kısa yorum, 3 sınıf etiketli (active learning ile seçildi) |

Temiz etiketler Claude tarafından verildi; bir kısmı proje sahibi tarafından gözden geçirilip düzeltildi.
