# gorkanai

Sıfırdan başlayarak adım adım kurulan bir **Türkçe duygu analizi** (sentiment analysis) projesi.
En basit yöntemden (Bag-of-Words) başlayıp BERT fine-tuning'e, gerçek veriye, veri temizliğine ve
bir web uygulamasına kadar ilerliyor. Her adımda bir sınırla karşılaşıp sebebini anlamak ve
bir sonraki adımda çözmek amaçlandı.

**Tek sayfalık özet (Adım 1-23): [step23_wrapup/OZET.md](step23_wrapup/OZET.md)** ·
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
| 14 | `step14_three_class` | Nötr sınıfı: Vikipedi nötrleriyle kısayol öğrenme, elle etiketlenmiş nötrlerle 3 sınıflı model V2b (uygulamada) |
| 15 | `step15_aspect` | Konu (aspect) bazlı duygu: 7 konu, anahtar kelime + cümlecik bölme temel çizgisi |
| 16 | `step16_topic_bert` | BERT ile çok etiketli konu tespiti, active learning ile 800 yeni etiket |
| 17 | `step17_aspect_sentiment` | Konuya koşullu duygu modeli ([CLS] konu [SEP] yorum); `LABEL_RULES.md` (v1) |
| 18 | `step18_big_test` | 500 yorumluk "kasa testi" + insan kalibrasyonu (Görkan–Claude uyumu) |
| 19 | `step19_topic_v2` | Konu tespiti v2: nadir konular için hedefli veri (satıcı F1 0.17 → 0.51) |
| 20 | `step20_aspect_app` | Konu bazlı analiz uygulamada (`/aspects`), eşdeğerlik testi |
| 21 | `step21_human_test` | İlk insan testi: model vs Görkan vs Claude etiketleyici |
| 22 | `step22_acceptable` | "Kabul edilebilir etiket" ölçüsü (model 0.823, Claude 0.857) |
| 23 | `step23_wrapup` | Kapanış: tek sayfa özet, `LABEL_RULES_v2.md` (kalite tanımı Görkan'la netleştirildi) |

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

Eğitilmiş modeller (~420 MB'lık klasörler) depoda **yok**; ilgili adımın script'i ile yeniden üretilir.
Uygulama 7 BERT kullanır (şu an hepsi Mac mini'de):

| Uygulamadaki rol | Klasör | Üreten script |
|---|---|---|
| Genel duygu (3 sınıf) | `step14_three_class/model_v2b` | `step14_three_class/train_v2.py` |
| Konu tespiti (3 tohum) | `step19_topic_v2/model_B_s{0,1,2}_e16` | `step19_topic_v2/train19.py B` |
| Konuya koşullu duygu (3 tohum) | `step17_aspect_sentiment/model_v2b_s{0,1,2}_e8` | `step17_aspect_sentiment/train.py` |

Genel duygu modeli Hugging Face Hub'da da var: [Urartu65/gorkanai-tr-sentiment](https://huggingface.co/Urartu65/gorkanai-tr-sentiment).

## Uygulamayı çalıştırmak

```bash
cd step9_app && ../.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000   # ~30 sn'de açılır (7 BERT)
```

- http://127.0.0.1:8000 — web arayüzü (ev ağından: `http://<makine-adı>.local:8000`)
- http://127.0.0.1:8000/docs — API dokümantasyonu
- `POST /predict {"text": "..."}` — genel duygu; `POST /aspects {"text": "..."}` — konular + konu başına duygu
- `ASPECT_LIGHT=1` — konu hattında tek tohum (~4 kat hızlı, val'de < 1 puan kayıp)

## Elle etiketlenmiş veriler

| Dosya | İçerik |
|---|---|
| `data/hard_test.csv` | 16 sentetik zor cümle (olumsuzlama, eş anlamlılar) |
| `data/negation_test.csv` | 25 cümle: çift olumsuzlama + basit olumsuzlama |
| `data/short_clean_test.csv` | 200 gerçek kısa yorum, orijinal + temiz etiket (pozitif/negatif/nötr) |
| `data/neutral_labels/` | 1000 gerçek kısa yorum, 3 sınıf etiketli (active learning ile seçildi) |

Temiz etiketler Claude tarafından verildi; bir kısmı proje sahibi tarafından gözden geçirilip düzeltildi.
