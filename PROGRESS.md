# gorkanai — İlerleme Notları

**Amaç:** Sıfırdan (Python + ML'e yeni) başlayarak, basit bir NLP modelinden
başlayıp adım adım daha gelişmiş yöntemlere geçerek "kendi AI'ını" inşa etmek.
Alan: Türkçe duygu analizi (sentiment analysis). Odak: öğrenmek — her adımda
gerçek bir sınırla karşılaşıp sebebini anlamak, sonra bir sonraki yöntemle çözmek.

**Son durum (2026-09-29):** Adım 15 (konu bazlı duygu analizi, anahtar kelime + cümlecik + V2b temel çizgisi) tamamlandı:
testte uçtan uca F1 0.653 (bağımsız insana karşı 0.638, insan-insan 0.667). Sıradaki: Adım 16 (onay bekliyor).

**Önceki durum (2026-09-28):** Adım 1-14 tamamlandı, uygulama 3 sınıflı model V2b'yi kullanıyor, V2b hiç
görülmemiş veride doğrulandı, model Hugging Face Hub'da herkese açık
([Urartu65/gorkanai-tr-sentiment](https://huggingface.co/Urartu65/gorkanai-tr-sentiment)). Uygulamanın
internete açılması PRO abonelik gerektiği için ERTELENDİ (aşağıda). Proje duraklatıldı — sıradaki işler
**"Yapılacaklar"** bölümünde.

## Ortam kurulumu
- Python **3.12** (Homebrew ile kuruldu — sistem Python'ı 3.14 idi, scikit-learn
  gibi paketler için henüz hazır wheel yok, derlemesi çok uzun sürüyordu)
- Sanal ortam: `gorkanai/.venv` — aktifleştirmek için: `source .venv/bin/activate`
- Kurulu paketler: `pandas`, `scikit-learn`, `torch`, `gensim`, `transformers`

## Klasör yapısı
```
gorkanai/
  .venv/
  data/
    reviews.csv                 # 330 örnek, etiketli eğitim verisi (pozitif/negatif)
    hard_test.csv                # 16 örnek, HİÇ eğitimde kullanılmayan zor test seti
                                  # (olumsuzlama + yeni kelimeler içeriyor)
    unlabeled_corpus.txt         # 4400 satır, etiketsiz korpüs (Word2Vec için)
    generate_dataset.py          # reviews.csv'yi üreten script
    generate_unlabeled_corpus.py # unlabeled_corpus.txt'yi üreten script
  step1_bow_logreg/train.py
  step2_tfidf/train.py
  step3_pytorch/train.py
  step4_embeddings/train.py
  step5_lstm/train.py
  step6_attention/train.py
  step7_finetune/train.py
  step8_real_data/part1_synthetic_on_real.py
  step8_real_data/part2_finetune_real.py
  step8_real_data/model/          # 16k gerçek yorumla eğitilmiş BERT (~440MB)
  step9_app/app.py, index.html    # FastAPI servisi + web arayüzü
  step10_negation/, step11_confidence/, step12_confident_learning/
  step13_temperature/calibrate.py, calibration.json
  step14_three_class/experiment_wiki_neutral.py (log_wiki.txt)
  step15_aspect/                  # konu bazlı duygu: prepare_*.py, save_aspect_labels.py, agreement.py, baseline.py
  data/aspect_labels/             # 300 konu-duygu etiketli yorum (val 100 / test 200) + human_blind_20.csv
  data/negation_test.csv, data/short_clean_test.csv
  data/prepare_real_dataset.py    # gerçek veriyi indirip data/real/ altına böler
  data/real/test.csv              # 1000 gerçek yorum, DENGELİ (500/500), eğitimde ASLA kullanılmaz
  data/real/train_pool.csv        # ~232k gerçek yorum (%94 pozitif) — eğitim örnekleri buradan
```

Her `train.py` bağımsız çalıştırılabilir:
```bash
cd stepN_.../ && source ../.venv/bin/activate && python train.py
```

## Adım adım sonuçlar ve dersler

| # | Yöntem | Zor test doğruluğu | Ana ders |
|---|---|---|---|
| 1 | Bag-of-Words + Lojistik Regresyon | (bkz. not) | **Overfitting**: ilk veri setinde her cümlede farklı kelime kullanınca (50 örnek, 163 kelime) model %100 eğitim / %10 test skoru verdi — ezberledi. Çözüm: kelime dağarcığını tekrar ettiren, daha büyük (240 örnek) bir veri seti + çapraz doğrulama. |
| 2 | TF-IDF + n-gram | 0.50 | Bigram'lar basit olumsuzlamayı ("iyi değil") kısmen yakalıyor ama **hiç görülmemiş kelimeler** (çekim farkı: "kötü" vs "kötüydü") için hâlâ sıfır sinyal. BoW/TF-IDF sadece harf dizisini tanır, anlamı değil. |
| 3 | PyTorch feedforward NN | 0.31–0.44 | Daha güçlü model (gizli katman) aynı TF-IDF girdisiyle **daha da fazla ezberledi**. Dropout/weight decay/early stopping denendi ama val seti aynı dağılımdan olduğu için işe yaramadı — regularization sadece "gürültü ezberini" önler, OOV kelime sorununu çözemez. |
| 4 | Word2Vec embeddings (ortalama pooling) | **0.69** | Etiketsiz korpüsle (4400 cümle) eğitilen embedding'ler "kötü"yü "berbat/rezalet/kalitesiz"e yakın yerleştirdi — model artık hiç görmediği eş anlamlı kelimelerle de genelleyebiliyor. Kalan sorun: ortalama alma kelime SIRASINI kaybediyor (olumsuzlama hâlâ zayıf). |
| 5 | LSTM (sıralı okuma) | 0.69 (farklı hatalar) | Cümle sonu olumsuzlamayı ("pek iyi değil") öğrendi ama farklı cümle yapısına ("kötü değildi, güzeldi" — virgüllü, iki cümlecik) genelleyemedi. |
| 6 | Attention | 0.69 (farklı hatalar) | Attention ağırlıkları doğru kelimelere odaklandığını GÖSTERDİ ("kötü"+"değildi" %94 ağırlık) — ama model az örnekten öğrendiği için her kelime+"değil" kombinasyonunu ayrı ayrı öğrenmesi gerekiyordu, genel bir kural çıkaramadı. |
| 7 | **BERT fine-tuning** (`dbmdz/bert-base-turkish-cased`) | **0.88** | Milyarlarca cümlede önceden eğitilmiş model + son katmanı bizim etikete göre ayarlama (4 epoch, lr=2e-5). Tüm olumsuzlama örnekleri doğru. Kalan 2 hata deyimsel ifadeler ("değmez", "beklenti altında kalmak") — basit olumsuzlama değil, dünya bilgisi gerektiriyor. |

**En büyük genel ders:** Mimariyi geliştirmek (NN → embedding → LSTM → attention)
sınırlı veriyle platoya çarpıyor (~0.69). Asıl sıçrama (→0.88), zaten geniş veriden
öğrenmiş bir modeli almakla geldi. Kendi verinle eğitim yerine transfer learning.

## Adım 8: Gerçek veri (seçenek b) — Kısım 1-2 TAMAMLANDI (2026-09-27)
Veri: `fthbrmnby/turkish_product_reviews` (HF, ~233k tekilleştirilmiş ürün yorumu). `pip install datasets` eklendi.
Özellikleri: %94 pozitif (çok dengesiz), hepsi küçük harf, yazım hataları/uzatmalar/emoji, ort. 22 kelime.
Etiketlerde gürültü var (nötr/karışık yorumlar da pozitif/negatif etiketli).

- **Kısım 1** (`part1_synthetic_on_real.py`): Adım 7 modeli (sentetikte eğitilmiş) gerçek testte **0.58**
  (negatif 0.74 / pozitif 0.42). Sentetik zor testte hâlâ 0.88. Ders: **domain shift** — 330 şablon cümle
  gerçek dili temsil etmiyor; "çok memnunum" gibi bariz pozitifler bile kaçtı.
- **Kısım 2** (`part2_finetune_real.py`, log: `part2_log.txt`): gerçek yorumlarla fine-tune, dengeli örnek, 2 epoch, max_length=128.
  | Eğitim verisi | Gerçek test | Sentetik zor test | Süre (MPS) |
  |---|---|---|---|
  | 330 sentetik (Kısım 1) | 0.58 | 0.88 | ~1 dk |
  | 1000 gerçek | 0.879 | 0.81 | 0.9 dk |
  | 4000 gerçek | 0.889 | 0.81 | 3 dk |
  | 16000 gerçek | 0.895 | 0.81 | 21 dk |
  Dersler: (1) Doğru ALANDAN 1000 örnek, 330 sentetikten çok daha değerli (0.58→0.88). (2) Azalan getiri:
  16 kat veri = +1.6 puan (1000 örnekli testte ±1 puan zaten istatistiksel gürültü). (3) Tavan etiket kalitesi:
  kalan hataların çoğu karışık/nötr veya yanlış etiketli yorumlar. (4) Sentetik zor testte takas: deyimleri
  ("değmez", "beklentimin altında kaldı") artık biliyor ama çift olumsuzlamayı ("kötü değildi", "hiç fena değildi")
  kaçırıyor — gerçek ürün yorumlarında bu yapılar nadir. Model ne görürse onu öğreniyor.
  16k model kaydedildi: `step8_real_data/model/` (~440MB) — seçenek (a) uygulama için hazır.

## Adım 9: Uygulama (seçenek a) — TAMAMLANDI (2026-09-27)
`step9_app/`: FastAPI. Çalıştır: `cd step9_app && source ../.venv/bin/activate && uvicorn app:app --reload`
→ http://127.0.0.1:8000 (arayüz), /docs (API dokümanı), POST /predict {"text": ...}, GET /health.
Ek paketler: `fastapi`, `uvicorn`.
Dersler: (1) **Train/serve tutarlılığı**: model küçük harfli veriyle eğitildi → girdiyi de küçült; sentetik zor
test 0.81→0.875. (2) Python `.lower()` Türkçeyi bozuyor ("İ"→"i̇", "I"→"i") → `turkish_lower()`.
(3) **Güven ≠ doğruluk**: "Bu film hiç fena değildi." → %93 güvenle NEGATİF (yanlış). Olasılık çıktısı
modelin ne kadar emin olduğunu söyler, haklı olup olmadığını değil.

## Adım 10: Çift olumsuzlama — TAMAMLANDI (2026-09-27)
`step10_negation/train.py`, yeni test: `data/negation_test.csv` (25 elle yazılmış cümle).
**Teşhis:** Dengeli örnekleme "fena değil" yorumlarının pozitif oranını %71'den (havuz) %12'ye (eğitim seti)
düşürmüştü — negatiflerin %54'ünü, pozitiflerin %4'ünü aldığımız için. Model çarpık veriyi doğru öğrenmişti.
**Düzeltme (B):** havuzdan kalıbı ("fena/kötü/berbat/kalitesiz ... değil/sayılmaz") içeren gerçek pozitifler
eklenerek oran havuzdakine (%73) getirildi. 4000 örnekte sadece +108 yorum.
| Model (4000) | Gerçek test | Negasyon holdout (200 gerçek) | Elle çift olumsuz | Elle basit olumsuz | Sentetik zor |
|---|---|---|---|---|---|
| A eski | 0.893 | 0.665 | 0.80 | 1.00 | 0.875 |
| B düzeltilmiş | 0.884 | **0.795** | 0.80 | 1.00 | **1.00** |
Dersler: gerçek veride +13 puan; genel doğrulukta -0.9 (gürültü sınırında); aşırı düzeltme yok (basit olumsuz %100).
Elle yazılmış test ayırt edici değil: cümlelerin ikinci yarısı ("gayet net", "çok memnunum") cevabı veriyor —
test tasarımı zayıflığı.
| Model (16k) | Gerçek test | Negasyon holdout | Elle çift olumsuz | Elle basit olumsuz | Sentetik zor |
|---|---|---|---|---|---|
| A eski | 0.896 | 0.660 | 0.867 | 1.00 | 0.812 |
| B düzeltilmiş (+454 yorum) | 0.890 | **0.820** | **1.00** | 1.00 | **1.00** |
B modeli `step10_negation/model/` → **step9_app artık bunu kullanıyor** (eski model step8_real_data/model'de duruyor).
Kalan zayıflık: kısa, ipucusuz cümlelerde güven düşük ("Kumaşı kötü değil." → pozitif ama %55);
"fena değil ama çok iyi de değil" gibi karışık yorumlar negatif — ki gerçek veride de öyle etiketleniyor.
Genel ders: **Veriyi dengelemek başka bir şeyi dengesizleştirebilir** — örnekleme kararlarını alt gruplar
bazında da kontrol et.

## Adım 11: Kısa cümlelerde düşük güven — (1) TAMAMLANDI (2026-09-27)
`step11_confidence/analyze.py` (kalibrasyon analizi, step10 modeli, hepsi görülmemiş veri).
Bulgu: model **çekingen değil**, hatta biraz fazla emin (ECE 0.035; %80-90 güven aralığında gerçek doğruluk 0.77).
%50-60 güvenli tahminlerin sadece %35'i doğru → düşük güven hata değil, "burada tahmin ediyorum" sinyali.
Kısa yorumlar (<=5 kelime) gerçekten daha zor: doğruluk 0.841 (uzunlarda 0.896), ort. güven 0.870 — uygun biçimde düşük.
Güveni yapay yükseltmek modeli yanıltıcı yapardı. "Emin değilim" eşiği: %70 altında cevap vermezse
kapsam %90, doğruluk 0.890→0.924; %80 eşik: kapsam %84, doğruluk 0.943.
Önerilen yön: (1) app'e "emin değilim" durumu, (2) kısa yorumların DOĞRULUĞUNU artırmak (veri), (3) temperature scaling.
**(1) yapıldı:** eşik VAL setinde seçildi (`choose_threshold.py`, hedef >=0.93 doğruluk) → **%80**. Gerçek testte
doğrulama: kapsam %83.8, doğruluk 0.943 (eşiksiz 0.890); "belirsiz" denen 162 yorumda doğruluk 0.617.
App: `/predict` artık `label` = pozitif/negatif/**belirsiz**, `leaning`, `confident` döndürüyor; arayüz "EMİN DEĞİLİM" gösteriyor.
Ders: eşik gibi "ayarlar" da test setine bakılarak seçilmemeli — val'de seç, test'te doğrula.
**(2) kısa yorumlar — teşhis yapıldı:** örnekleme çarpıklığı YOK (kısa yorumlar havuzun %9.4'ü, eğitimin %9.7'si).
Asıl sebep **etiket gürültüsü**: veri seti uzun yorumları cümlelere bölüp her cümleye yorumun etiketini vermiş gibi
("kargo çok hızlıydı teşekkürler" → negatif, "saygılarımla!" → negatif). 200 görülmemiş kısa yorum Claude tarafından
elle etiketlendi → `data/short_clean_test.csv` (text, original_label, clean_label: pozitif/negatif/nötr).
Sonuç: %19'u nötr, "negatif" etiketlilerin %16'sı aslında pozitif.
| Kısa yorumlarda (step10 modeli) | Doğruluk |
|---|---|
| Orijinal etiketlerle (200) | 0.815 |
| Temiz etiketlerle (165, nötr hariç) | **0.927** |
| Temiz + %80 eşik (kapsam %84) | **0.978** |
Nötrlerin %61'ine zaten "belirsiz" diyor. Gerçek hatalar 10 tane: yazım hatası ("fena degıl"), güçlü kelimenin
bağlamı ezmesi ("çok küçük tanklar için ideal"), dünya bilgisi ("amortisörleri değişmiş gibi").
Ders: **Modeli düzeltmeden önce ölçüyü kontrol et** — "kısa yorumlarda kötü" sonucunun çoğu testin hatasıydı.
Etiketler: Claude etiketledi, kullanıcı 57 değişikliği gözden geçirip 5'ini düzeltti (#132 uyarı→nötr, #181 şikayet→negatif,
#47/#87/#198 hafif pozitif→pozitif). Sonuç: %17.5 nötr, "negatif" etiketlilerin %19'u aslında pozitif.
Olası devam: eğitim verisindeki gürültülü etiketleri bulup temizlemek (confident learning), temperature scaling.

## Adım 12: Confident learning — TAMAMLANDI (2026-09-27)
`step12_confident_learning/train.py` (log: `log.txt`; ilk deneme GPU bellek hatası: `log_run1_oom.txt`).
3-fold ile her eğitim örneği için "görmediği model" tahmini (`oof_probs.npy`), sınıf başına eşik (~0.85),
etiketin tersine eşiği aşan 744 örnek (%4.5) çıkarıldı → `flagged.csv`. Kısa yorumlarda oran %7.0, uzunlarda %4.3.
İşaretlenenlerin çoğu gerçek hata, ama bazı DOĞRU zor örnekler de gitti (ironi: "mükemmel derecede sızdırıyor").
| | B (Adım 10) | C (temiz) |
|---|---|---|
| Gerçek test (gürültülü) | 0.890 | 0.891 |
| Kısa temiz test (elle, 165) | 0.927 | **0.964** |
| Negasyon holdout / elle / sentetik | 0.820 / 1.00 / 1.00 | 0.815 / 1.00 / 1.00 |
| Kalibrasyon hatası (ECE, gerçek test) | 0.035 | **0.069** (daha fazla aşırı güven) |
| Val'de seçilen eşik (hedef ≥0.93) | %80 | **%95** |
| Kısa temiz, app eşiğiyle | kapsam %84, doğr. 0.978 | kapsam %83, doğr. **0.993** |
App artık C'yi %95 eşikle kullanıyor. Dersler: (1) temiz veri → doğrulukta küçük ama tutarlı kazanç (temiz ölçüde);
(2) temiz veri → model daha emin konuşuyor, eşik yeniden seçilmeli; (3) Python'da fonksiyon içinde `del model`
dışarıdaki referansı silmez → MPS bellek taşması, loss=nan.
Kalan sorun: aşırı güven — "Saygılarımla." → %97 pozitif. Sıradaki doğal adım: **temperature scaling**.

## Adım 13: Temperature scaling — YAPILDI, app'e UYGULANMADI (2026-09-27)
`step13_temperature/calibrate.py`: val'de NLL ile T = **2.47** bulundu (model C için).
| ECE | önce (T=1) | sonra (T=2.47) |
|---|---|---|
| val (gürültülü, T burada seçildi) | 0.076 | 0.020 |
| gerçek test (gürültülü) | 0.069 | 0.018 |
| **kısa temiz (elle)** | **0.033** | **0.075** (kötüleşti) |
App davranışı (eşik val'de yeniden seçildi → %76): kapsam/doğruluk neredeyse birebir aynı (gerçek %82/0.944 → %83/0.943).
"Saygılarımla." %97 → %81 pozitif, ama %76 eşiğini hâlâ geçiyor.
Dersler: (1) Model C'nin "aşırı güveni" büyük ölçüde GÜRÜLTÜLÜ ETİKETLERE göreymiş — temiz etiketlerde zaten iyi
kalibre (ECE 0.033); gürültülü val'e göre ayarlanan T, temiz veride modeli gereğinden çekingen yapıyor.
(2) İki sınıfta T sıralamayı değiştirmez; eşik de yeniden seçilince "emin değilim" kararları neredeyse aynı kalır —
T sadece GÖSTERİLEN sayıları değiştirir. (3) "Saygılarımla" sorunu kalibrasyon sorunu değil: iki sınıflı model
"nötr" kavramını hiç görmedi. Gerçek çözüm: **nötr sınıfı olan 3 sınıflı model** (nötr etiketli veri gerekiyor,
örn. HF `winvoker/turkish-sentiment-analysis-dataset` Notr sınıfı içeriyor — incelenmedi).
Karar: app T=1, eşik %95 ile kalıyor (C modeli).

## Adım 14: 3 sınıflı model (nötr) — MODEL HAZIR, app'e alınmadı (2026-09-27)
Aday veri: HF `winvoker/turkish-sentiment-analysis-dataset` (489k). SORUN: nötrlerin %99.7'si **Vikipedi** cümlesi
(%99'u " ." ile bitiyor, %100 büyük harfle başlıyor); pozitif/negatifler yorum/tweet. Ayrıca `urun_yorumlari`
kaynağı bizim ürün yorumu veri setimizle aynı görünüyor → kullanırsak test sızıntısı riski.
**Deney 1** (`experiment_wiki_neutral.py`): 3000 poz + 3000 neg (bizim havuz) + 3000 Vikipedi nötr (biçim ipuçları
temizlendi), 2 epoch. Vikipedi nötr testi **0.986**, ama elle etiketlenmiş **35 gerçek nötr yorumda 0.114** (4/35).
Kısayol testi: "Cihaz yüksek performansı ile kullanıcılar tarafından beğenilmiştir." → %99 NÖTR (pozitif olmalı);
"kargo 2 günde geldi." → %83 pozitif. Ders: **shortcut learning** — model "duygusuz" değil "ansiklopedik üslup"
öğrendi; biçimi temizlemek yetmedi, içerik/üslup farkı kaldı. Aynı dağılımdan test (0.986) bunu gizler.
**Etiketleme** (`prepare_candidates.py`, `save_labels.py`): active learning ile 1000 kısa yorum seçildi (%50 model C'nin
<%90 emin olduğu, %50 rastgele), Claude sohbette etiketledi → `data/neutral_labels/batch_01-04.csv`:
642 poz / **319 nötr** / 39 neg. Kararsız yorumlarda nötr oranı %48-52, rastgelede %11-16 (active learning ~3.4x verim).
`save_labels.py` 10'luk grup uzunluğu kontrolü 3 yazım hatasını yakaladı (etiket kayması olmadı).
**Deney 2** (`train.py`, log: `log_train.txt`, model: `step14_three_class/model/`): train_c + 1000 elle etiket,
nötr x8 oversample, 2 epoch. Kıyas: C + %95 eşik ("emin değilim" = nötr). Kısa temiz testte (97p/35x/68n):
| | Kıyas (C+%95) | 3 sınıflı |
|---|---|---|
| Doğruluk | 0.800 | 0.830 |
| Macro-F1 | **0.766** | 0.735 |
| Nötr kesinlik / yakalama | 0.46 / **0.69** | **0.73** / 0.31 |
3 sınıflı model nötrlerin çoğunu (14/35) NEGATİF diyor. Üslup kısayolu yok ("Cihaz ... beğenilmiştir" → pozitif,
"saygılarımla." → %99 nötr), "kısa = nötr" kısayolu da yok ("harika", "berbat" doğru).
Olası sebep: poz/neg eğitim verisinde hâlâ nötr parçalar var (kısa "negatif" etiketlilerin ~%15'i nötr) → model
"nötr parça → negatif" öğreniyor, 319 temiz nötrle çelişiyor. App DEĞİŞTİRİLMEDİ (kıyas nötrde daha iyi).
**Deney 3** (`train_v2.py`, log: `log_v2.txt`): elle etiketlerin %20'si val (128p/64x/**8n**). Önceden sabit iki varyant:
V2a = etiketsiz kısa (<=10 kelime) poz/neg'lerin hepsi çıkarıldı; V2b = sadece Adım 12 K-fold tahmininin ≥%90
katıldığı kısalar tutuldu (2991). Val'de macro-F1 ile seçim + nötr logit'ine bias.
| Kısa temiz test | doğr. | macro-F1 | nötr P/R | neg P/R | poz P/R |
|---|---|---|---|---|---|
| Kıyas C+%95 | 0.800 | 0.766 | 0.46/0.69 | 0.92/0.88 | 0.92/0.78 |
| Deney 2 (v1) | 0.830 | 0.735 | 0.73/0.31 | 0.78/0.96 | 0.88/0.93 |
| V2a + val-bias +2.75 (**val'in seçtiği**) | 0.725 | 0.678 | 0.39/0.83 | 0.97/**0.44** | 0.91/0.89 |
| V2b + val-bias +0.50 | **0.860** | **0.811** | 0.70/0.54 | 0.93/0.94 | 0.86/0.92 |
Val seçimi YANLIŞ varyantı seçti: V2a "kısa negatif → nötr" kısayolunu öğrendi ("berbat", "çöp", "iade ettim" → nötr),
ama val'de sadece 8 negatif olduğu için görünmedi. Ders: **val seti, önemsediğin her hatayı temsil etmeli** —
kalibrasyonu olmayan sınıf için ayar yapılamaz. V2b testte en iyi AMA bunu testten öğrendik → V2b'nin test sayıları
artık biraz iyimser. (İlk çalıştırmanın logu: `log_v2_run1_val8neg.txt`.)
**Deney 3b** (aynı `train_v2.py`, log: `log_v2.txt`): val'e hiçbir eğitimde olmayan 150 kısa "negatif" etiketli yorum
eklendi (`prepare_val_negatives.py` → `data/neutral_labels/batch_05.csv`, SADECE val): 101 neg / 32 nötr / 17 poz
(yine: kısa "negatif"lerin %21'i nötr, %11'i pozitif). Val artık 145p / 96x / 109n. Eğitim aynı; sızıntı assert'i eklendi.
Val macro-F1: V2a 0.721, **V2b 0.778** (bias +3.00 — grid'in üst sınırı; test görüldüğü için grid genişletilmedi).
→ Val bu kez **V2b**'yi seçti. Kısa temiz test (görülmüş → iyimser):
| | doğr. | macro-F1 | nötr P/R | neg P/R | poz P/R |
|---|---|---|---|---|---|
| Kıyas C+%95 (app) | 0.800 | 0.766 | 0.46/0.69 | 0.92/0.88 | 0.92/0.78 |
| **V2b + bias +3.00** | **0.860** | **0.825** | **0.67/0.69** | 0.92/0.87 | 0.89/0.92 |
Kısayol testi 18/18 doğru ("berbat"/"çöp" negatif, "kargo 2 günde geldi." nötr, üslup kısayolu yok).
Gerçek test: %5.4 nötr, cevap verilenlerde 0.903. Model: `step14_three_class/model_v2b/` (bias +3.00 ile kullanılmalı).
`model_v2a/` kullanılmamalı. App henüz DEĞİŞTİRİLMEDİ. Eksik: görülmemiş yeni bir test seti.

## Adım 14 devamı: Görülmemiş test seti üzerinde V2b doğrulaması — TAMAMLANDI (2026-09-28)
Bu makinede proje sıfırdan kuruldu (`.venv`, `data/real/train_pool.csv`, tüm `step*/model*/` gitignore'da
olduğu için hiç yoktu) — Homebrew Python 3.12 + venv + paketler yeniden kuruldu, `train_pool.csv`
`prepare_real_dataset.py` ile yeniden üretildi (deterministik: `test.csv` git'teki ile birebir aynı çıktı).

**Yeni test seti** (`step14_three_class/prepare_new_test.py` → `candidates/batch_06.csv`, id 2000-2199):
havuzun ilk 9000'lik dilimi (val+eğitim+pattern-extra payı) tamamen atlanıp ondan SONRAKİ kısımdan,
hem "pozitif" hem "negatif" etiketli havuz yorumlarından 100'er tane seçildi — hiçbir eğitim/val/test/
candidate setinde yok (assert ile doğrulandı). Claude sohbette etiketledi (200/200) → `data/neutral_labels/batch_06.csv`
(`save_labels.py 6 ...`): **104 pozitif / 74 negatif / 22 nötr** (%11 nötr — "rastgele" örneklemde beklenen
%11-16 aralığıyla tutarlı, active-learning örneklemindeki %48-52'den düşük, beklendiği gibi).

**Modeller yeniden eğitildi** (gitignore'da oldukları için): `step12_confident_learning/train.py` (oof_probs.npy
diskte olduğu için K-fold atlandı, sadece model C ~15 dk) ve `step14_three_class/train_v2.py` (V2a ~13 dk,
V2b ~16 dk). Her iki script de sonunda eksik kıyas modelleri yüzünden (`step10_negation/model`,
`step14_three_class/model` — ikisi de bu makinede yok ve gerekmiyor) hata verip durdu, ama asıl modeller
(C, V2a, V2b) hatadan ÖNCE kaydedildiği için sorun olmadı. İlginç not: bu çalıştırmada val macro-F1 V2a'yı
seçti (0.769 vs V2b 0.767 — çok yakın), önceki çalıştırmada V2b seçilmişti (0.778 vs 0.721); muhtemelen
MPS backend'in tam deterministik olmaması. Görev zaten V2b'yi sabit ölçtüğü için etkisi yok.

**Ölçüm** (`step14_three_class/eval_new_test.py`, bias YENİDEN seçilmedi — val'de önceden sabitlenen +3.00 kullanıldı):
| Yeni test (104 poz / 22 nötr / 74 neg, hiç görülmemiş) | doğr. | macro-F1 | nötr P/R | neg P/R | poz P/R |
|---|---|---|---|---|---|
| Kıyas: C + %95 eşik (mevcut app) | 0.825 | 0.746 | 0.38/0.64 | 0.87/0.84 | 0.97/0.86 |
| **V2b + bias +3.00** | **0.870** | **0.808** | **0.58/0.68** | 0.87/0.88 | 0.95/0.90 |
**Sonuç: V2b, önceki "görülmüş" kısa temiz testteki sonucuna (0.860/0.825) çok yakın bir skorla, hiç
görülmemiş veride de kıyas modelini üç sınıfta da (özellikle nötr kesinlikte %38→%58) geçti.** Önceki
şüphe ("V2b'nin sayıları iyimser olabilir, çünkü val'i genişletme kararı bu teste bakılarak verildi")
doğrulanmadı — gerçek kazanç, ölçüm artefaktı değil. V2b'yi uygulamaya almak için elde yeterli kanıt var.

## Uygulamaya 3 sınıf eklendi — TAMAMLANDI (2026-09-28)
`step9_app/app.py`: `MODEL_DIR` → `step14_three_class/model_v2b`, `LABELS` artık modelin `config.id2label`'ından
okunuyor (negatif/nötr/pozitif), tahminden önce nötr logit'ine sabit **+3.00** ekleniyor (val'de seçilen bias,
Adım 14 Deney 3b). Adım 11-13'teki "emin değilim" güven eşiği **kaldırıldı** — nötr sınıfı o işlevi zaten
üstleniyor (karar: kullanıcıyla konuşuldu, basitlik tercih edildi; eşiği tutmak 3 sınıf için ayrı bir
val-tabanlı yeniden seçim gerektirirdi). `index.html`: tek bar yerine üç renkli olasılık çubuğu (yeşil/sarı/kırmızı).

Tarayıcıda test edildi (uvicorn + Chrome): "Kargo çok hızlıydı, ürün harika!" → POZİTİF %99.7,
"İade etmek zorunda kaldım, hiç memnun değilim." → NEGATİF %99.7, "Ürün bugün elime ulaştı." → NÖTR %100.0,
"Saygılarımla." → NÖTR %99.99 (2 sınıflı eski modelde bu %97 YANLIŞ pozitif çıkıyordu — Adım 13'te tespit
edilen üslup/kısayol sorunu artık çözülmüş görünüyor).

## Modeli Hugging Face Hub'a yükle + uygulamayı yayınla — KISMEN TAMAMLANDI (2026-09-28)
`hf auth login` ile giriş yapıldı (kullanıcı: Urartu65). `step14_three_class/model_v2b` bir model kartıyla
(README.md — kullanım örneği, eğitim verisi, hiç görülmemiş test sonuçları) birlikte
[Urartu65/gorkanai-tr-sentiment](https://huggingface.co/Urartu65/gorkanai-tr-sentiment)'e yüklendi (ilk
denemede Xet depolama sunucusunda geçici bir ağ hatası oldu, tekrar denemede tamamlandı).

`step9_app/app.py`: `MODEL_DIR` artık `MODEL_DIR` ortam değişkeninden okunuyor (yoksa yerel `model_v2b`
klasörüne düşüyor) — Hub'dan yükleyip doğrulandı (`Saygılarımla.` → nötr %99.99, yerel modelle birebir aynı
sonuç). `step9_app/Dockerfile`, `requirements.txt`, `README.md` (Spaces YAML front-matter'lı) hazırlandı
ama **kullanılmadı**: `hf repo create --type space --sdk docker` **402 Payment Required** verdi — HF artık
ücretsiz cpu-basic'te bile Docker/Gradio Space'leri için PRO abonelik istiyor (sadece static Space'ler
ücretsiz). Kullanıcıyla konuşuldu, karar: **şimdilik uygulamayı yayınlamayı ertele**, model Hub'da herkese
açık olması yeterli. Dockerfile/requirements.txt hazır — PRO'ya geçilirse veya başka bir Docker destekli
ücretsiz servise (Render/Fly.io) taşınırsa doğrudan kullanılabilir.

## Adım 15: Konu (aspect) bazlı duygu analizi — TAMAMLANDI (2026-09-29)
**Görev:** yorumda HANGİ konudan bahsedildiğini ve o konudaki duyguyu bulmak. Örnek: "kargo çok hızlıydı ama
kumaşı ince" → kargo: pozitif, kalite: negatif. Klasör: `step15_aspect/`, etiketler: `data/aspect_labels/`.
Koordinasyon: iki Claude oturumu. gorkanai-fd planladı, gorkanai-0b uyguladı; kararları Görkan onayladı.

**15.1 Veri keşfi** (`prepare_explore.py` → `explore_300.csv`, id 3000-3299): 300 yorum, 8-40 kelime,
150 "poz" + 150 "neg" havuz etiketli. Hiçbir eğitim/val/test/candidate setinde yoklar (assert).
Havuz iki farklı permütasyonla karıştırılmıştı: Adım 8/10/11/13 holdout düşülmeden, Adım 12/14 düşülerek.
İkisinde de pos/neg'in ilk 9000'i atlandı, negasyon kalıplı TÜM satırlar dışarıda tutuldu. Bu kısıtlarla
8-40 kelimelik görülmemiş "neg" yorum sadece 1402 tane kaldı.
Kelime sayımı + okuma → **7 konu**: K kargo/teslimat (paketleme dahil), F fiyat/değer, Q kalite,
P performans/özellik, B boyut (miktar ve ağırlık dahil), G görünüm, S satıcı/hizmet.
Ders: en sık iki konu anahtar kelimeyle en zor yakalananlar. Kargo çoğu zaman örtük ("2 günde elime
ulaştı"), performansın kelimeleri ise ürün kategorisine göre değişiyor.

**15.2 Etiketli set** (`prepare_label_set.py`, `save_aspect_labels.py` → `data/aspect_labels/aspect_labels.csv`,
id 4000-4299): 15.1'den ayrık 300 yorum. **Val 100 / test 200 bölmesi etiketlemeden ÖNCE sabitlendi**
(havuz etiketine göre tabakalı). Claude sohbette (konu, duygu) çiftleri olarak etiketledi. Adım 14'teki
10'luk grup kontrolü kullanıldı. Karışık (aynı konu hem övülüp hem eleştirilmiş) için baskın duygu yazıldı
ve ayrıca `karisik` sütunu tutuldu. Kurallar `save_aspect_labels.py` docstring'inde. Özellikle:
ekran/görüntü KALİTESİ → P, G sadece dış görünüm; kutunun İÇERİĞİ (eksik, yanlış ürün) → S, kutunun
DURUMU → K. Görkan 30'luk bir örneği inceledi (`review_sample.csv`), düzeltme gerekmedi.
- **Konu içi nötr neredeyse yok:** 469 (konu, duygu) çiftinin 15'i (%3). İnsanlar bir konudan genelde fikir
  belirtmek için bahsediyor.
- Karışık: 300 yorumun 26'sı (%9), neredeyse hepsi performans ("kokusu güzel ama kalıcı değil").

**Tutarlılık taraması** (`changes_rule_scan.csv`): 15.2 sırasında eklenen üç kural (miktar → B,
hasarlı/sağlam gelme → K, ağırlık → B) regex adayları okunarak 300'e sistematik uygulandı: 2 değişiklik.
İçerik/durum kuralı taraması: 4 değişiklik.

**Kör etiketleme ve Q/P tanım değişikliği** (`agreement.py`, `data/aspect_labels/human_blind_20.csv`):
Görkan, etiketlerimi görmeden 20 yorumu (val 10 + test 10, tabakalı) sadece kurallarla etiketledi. Başta 40
olarak planlanmıştı, Görkan'ın isteğiyle 20'ye indirildi. Görkan'ın etiketlerimi gördüğü 30 yorum dışarıda
tutuldu. Şeffaflık notu: format örneği yanlışlıkla 2 ve 3 numaranın gerçek etiketleriydi, bu yüzden o
ikisi kör değil.

| Uyum (Claude vs Görkan, 20 yorum) | konu F1 | çift F1 | kappa Q | kappa P | kappa K | kappa F |
|---|---|---|---|---|---|---|
| Eski kurallar | 0.562 | 0.531 (18 kör yorumda 0.483) | **0.25** | **0.26** | 0.69 | 0.57 |
| **Yeni Q/P tanımı, karar öncesi** | 0.697 | **0.667** | 0.52 | 0.38 | 0.69 | 0.57 |
| Anlaşmazlıklar karara bağlandıktan sonra | 0.746 | 0.716 | 0.63 | 0.63 | 0.69 | 0.57 |

(kappa sadece n≥5 konular için; boyut, görünüm ve satıcı için n yetersiz. Ortak konularda duygu uyumu 24/25.)

Anlaşmazlıkların neredeyse tamamı KONU SINIRINDAYDI, duyguda değil. Görkan Q'yu "ürün genel olarak iyi mi"
anlamında kullanıyordu; eski kurallarda ise genel övgü "konu yok", işe yarama ise P'ydi. Görkan'ın
kararıyla **yeni tanım** getirildi:
- **Q** = ürünün GENEL iyi/kötü olması ("süper ürün", "memnun kaldım", "pişman oldum") + genel olarak işe
  yarayıp yaramaması ("işe yaramıyor", "bozuldu", "şarj etmiyor", "pil ömrü kısa") + malzeme/işçilik.
- **P** = ADI KONAN belirli bir özellik: güç, ses, kamera/ekran, hız, uyumluluk, kurulum/kullanım
  kolaylığı, koku/kalıcılık, tat.
- Kalıp tavsiye ("alın", "tavsiye ederim") tek başına Q değil.
300'ün tamamı yeni tanıma göre tek tek yeniden okundu: **128 değişiklik**. Tüm turlarla toplam **136**
altın etiket değişikliği; hepsi kural sütunuyla `changes_rule_scan.csv`'de. Kalan 11 anlaşmazlığın dağılımı
(`disagreements_resolved.csv`): 8 Görkan hatası (çoğu küçük konu/kural detayı: kargo ücreti → K,
iade → S, "…duruyor" → G), 1 Claude hatası, 1 ikisinin de hatası, 1 kural belirsiz (kalıp tavsiye).
Yeni dağılım (val/test): Q 59/130, P 43/86, F 27/48, K 22/34, B 12/25, G 7/17, S 6/15; konusuz 7 yorum.
**İnsan tavanı olarak 0.667 (çift F1) kullanılıyor, ama İYİMSER** (bkz. sınırlama 6). İkinci, bağımsız bir
kör etiketleme turu planlandı, Görkan'ın isteğiyle iptal edildi.

**15.3 Temel çizgi (eğitim yok)** (`baseline.py`, log: `log_baseline_val.txt`):
- Yöntem: anahtar kelime regex'i ile konu tespiti (kelime başından eşleşme, Türkçe ekler için) → yorumu
  cümleciklere böl → konunun geçtiği cümlecik(ler)e V2b → birden çok cümlecik varsa olasılıklar toplanır.
- Anahtar kelime listeleri SADECE `explore_300`'den türetildi. Yeni Q/P tanımıyla V2 listesi yazıldı:
  genel yargılar Q'ya eklendi, "işe yar/fayda/etki/şarj/çalış" P'den Q'ya taşındı. Fark logda. Val'e bakarak
  kelime eklenmedi.
- Val sadece iki genel karar için kullanıldı: bölme kuralı (A: . ! ? ; + "ama/fakat/ancak/lakin/yalnız";
  B: A + virgül; yok) ve nötr seçeneği (a: V2b bias 0, 3 sınıf; b: sadece poz/neg argmax). V2b'nin +3.00
  nötr bias'ı KULLANILMADI: o tüm yorum için seçilmişti (~%11 nötr), konu içi nötr ise %3.

| Val (100 yorum), her satır kendi val seçimiyle | ayar | duygu micro/macro | karisik=0 | uçtan uca F1 micro/macro | insan tavanı |
|---|---|---|---|---|---|
| eski kurallar + eski liste (V1) | A/b | 0.856/0.846 | 0.867 | 0.673/0.640 | — |
| yeni kurallar + eski liste (V1) | A/b | 0.850/0.842 | 0.856 | 0.572/0.571 | 0.667 |
| **yeni kurallar + yeni liste (V2)** | **A/b** | **0.889/0.856** | 0.893 | **0.696**/0.625 | 0.667 |

Konu tespiti P/R (V2, val): kargo 0.91/0.91 (n=22), fiyat 0.93/0.96 (27), kalite 0.83/0.82 (60),
performans 0.77/**0.57** (42), boyut 0.70/0.58 (12), görünüm 0.50/0.86 (7), satıcı 0.50/0.43 (7).
**Seçilen ayar: bölme A + nötr kapalı (sadece poz/neg).** A ile B val'de birebir eşit; eşitlikte daha basit
olan seçildi. Nötrü kapatmak en fazla ~%3 kaybettirir, (a)'nın verdiği nötrlerin hiçbiri skoru artırmadı.
Test sonuçları aşağıda (15.4).

**Dersler:**
- **Cümleciğe bölmek işe yarıyor:** bölme yok → A ile duygu doğruluğu micro +4-7 puan, macro +6-7 puan.
  Aynı yorumdaki "ama"dan sonraki şikâyet, önceki övgüyle karışmıyor.
- **0.673 aslında "Claude'a benzeme" skoruydu.** Model, etiketleyenle aynı kafadan yazılmış kelimeleri
  kullandığı için insan-insan uyumundan (0.531) yüksek çıktı. Etiketler bir insanın sezgisiyle
  karşılaştırılınca Q/P kavramları çöktü (kappa 0.25/0.26). Bu yüzden V2'nin 0.696'sı da tavanı (0.667)
  "geçmiş" sayılmamalı: model Claude'un altın etiketlerine göre ölçülüyor, tavan ise Claude-Görkan uyumu.
  Elmayla elma kıyası için modelin Görkan'ın etiketlerine karşı skoru gerekir.
- **Tek etiketleyicili kavram tanımları güvenilmez.** Kör ikinci etiketleyici olmadan Q/P'nin belirsiz
  olduğunu hiç görmeyecektik; konu tespiti sayıları ise yüksek görünmeye devam edecekti.
- **Anahtar kelime tanımın aynasıdır:** tanım değişince eski liste Q'da recall 0.37'ye düştü; listeyi yeni
  tanıma göre yazınca 0.82'ye çıktı. Performans (P) recall'u iki listede de ~0.57: adı konan özelliklerin
  kelimeleri kategoriye bağlı. Adım 16'nın (öğrenilmiş konu modeli) ana gerekçesi bu.

**Sınırlamalar:**
1. Set 50/50 poz/neg havuz etiketli, gerçek dağılım ise %94 pozitif. Sayılar gerçek trafiği temsil etmiyor.
2. Negasyon kalıbı içeren yorumlar ("fena değil", "kötü değil") sızıntıyı önlemek için dışarıda. Yani
   negasyon ile konu etkileşimi ölçülmüyor.
3. 8 kelimeden kısa yorumlar dışarıda. Önceki adımlarda en zor grup buydu.
4. Anahtar kelime listelerini yazan Claude, val ve test yorumlarını etiketlerken okudu. Listeler
   `explore_300`'den türetildi, ama bu dolaylı bir aşinalık.
5. Test cümleciklerinin V2b logit'leri cache'lendi (`clause_logits.npz`, commit'lenmiyor), ama hiçbir metrik
   hesaplanmadı ve bakılmadı.
6. **İnsan tavanı (0.667) iyimser:** kurallar aynı 20 yoruma bakılarak ve Görkan'ın kararlarıyla netleşti.
   Yeni tanım bağımsız bir kör setle doğrulanmadı; bu 20 yorumun 2'si de kör değil.

### Adım 15.4: Test ölçümü — TAMAMLANDI (2026-09-29)
`step15_aspect/evaluate.py`, log: `log_test.txt`. Ölçümden önce dondurulan: V2 anahtar kelime listesi
(sha256[:16] `a7f039e0719a8dca`) + val'de seçilen ayar (bölme A, nötr kapalı). Test **bir kez** ölçüldü.
Sonuçları gördükten sonra modelde hiçbir değişiklik yapılmadı; yeniden çalıştırmalar sadece elle hata
sınıflandırmasını rapora eklemek içindi.

| Konu (test) | n | P | R | F1 | duygu doğru | val P/R | not |
|---|---|---|---|---|---|---|---|
| kargo | 34 | 0.81 | 0.74 | 0.77 | 21/25 | 0.91/0.91 | |
| fiyat | 48 | 0.90 | 0.96 | 0.93 | 41/46 | 0.93/0.96 | |
| kalite | 130 | 0.84 | 0.68 | 0.75 | 80/89 | 0.83/0.82 | |
| performans | 87 | 0.82 | 0.72 | 0.77 | 50/63 | 0.77/0.57 | |
| boyut | 25 | 0.67 | 0.80 | 0.73 | 14/20 | 0.70/0.58 | gürültülü (n<30) |
| görünüm | 17 | 0.64 | 0.82 | 0.72 | 14/14 | 0.50/0.86 | gürültülü (n<30) |
| satıcı | 18 | 0.46 | 0.33 | 0.39 | 5/6 | 0.50/0.43 | gürültülü (n<30) |

| Test (200) vs val (100) | konu F1 micro/macro | duygu micro/macro | duygu karisik=0 | **uçtan uca F1 micro/macro** |
|---|---|---|---|---|
| **Test — V2 + A/b (ANA SONUÇ)** | 0.763/0.722 | 0.856/0.851 | 0.858 | **0.653/0.616** |
| Val — V2 + A/b | 0.783/0.724 | 0.889/0.856 | 0.893 | 0.696/0.625 |
| Test — (a) V2 + bölme yok | 0.763 | 0.848/0.800 | 0.845 | 0.647/0.576 |
| Test — (b) eski liste V1 + A/b | 0.643 | 0.850/0.858 | 0.845 | 0.547/0.554 |

Testte 11 altın nötr çift var; nötr kapalı olduğu için hepsi kaçıyor (kabul edilen bedel).

**Elmayla elma** (Görkan'ın kör 20 yorumu, karar öncesi etiketler; n çok küçük, SADECE FİKİR VERİR):

| alt küme | n | model-Claude | model-Görkan | Claude-Görkan |
|---|---|---|---|---|
| tümü | 20 | 0.747 | **0.638** | **0.667** |
| kör 18 (2 ve 3 hariç) | 18 | 0.765 | 0.645 | 0.633 |
| sadece test | 10 | 0.732 | 0.649 | 0.667 |

Model, Claude'a (0.747) Görkan'a (0.638) olduğundan çok daha fazla benziyor. Bağımsız bir insana karşı
ise insan-insan uyumunun biraz altında ya da civarında. Yani val'deki "tavanı geçti" görüntüsü gerçekten bir
ölçüm artefaktıydı.

**Hata kovaları** (test): 67 konu FP, 96 konu FN, 38 duygu hatası.
- **FP (hepsi "yanlış eşleşme"):** kelime geçiyor ama konu kastedilmiyor. En çok "büyük" (6; "en büyük eksisi"
  gibi), "iade" (4; şikâyet değil, bilgi), "mağaza" (3), "ucuz" (3).
- **FN, elle sınıflandırıldı** (`test_fn_manual.csv`): **örtük 44** (konu hiç adlandırılmıyor, çıkarım
  gerekiyor: "arka kapak oturmuyor" → Q, "köpeğim tenezzül etmedi" → Q, "fritöz gönderdi" → S,
  "eve kadar gelmesi güzel" → K) ve **kelime eksik 52**. Kelime eksik olanların alt türleri: 27'si
  listede hiç olmayan kelime ("sallantı", "arayüz", "uyku modu", "güçlü"), 15'i yazım varyantı (ascii
  "guzel/basarili", birleşik noktalı "i̇"), 10'u Türkçe ek/yumuşama ("özellik" → "özelliği",
  "ulaştı" → "ulaşıyor", "tasarım" → "tasarlanmış").
- **Duygu hatası** (otomatik sezgisel): **bölme hatası 15** (cümlecik zıt duygulu iki konuyu birlikte taşıyor:
  "şarj süresi iyi, ısınma problemi…") ve **V2b hatası 23** (cümlecik tek görüşlü ama yanlış; "hediyeden
  faydalanamadım" → negatif).
- Konu bazında baskın kova: kalite → FN (örtük 21 ≈ kelime eksik 20); performans → FN 24 (kelime eksik 14,
  örtük 10) + FP 14 + V2b hatası 10; boyut, görünüm ve satıcı → yanlış eşleşme; satıcı'da FN'lerin çoğu örtük.

**Dersler (15.4):**
- Val'deki iki ders testte de geçerli, ama biri zayıflayarak. **Liste tanımın aynası:** V1 → V2 ile konu F1
  0.643 → 0.763, uçtan uca 0.547 → 0.653. **Bölme işe yarıyor:** duygu macro 0.800 → 0.851 ve uçtan uca
  macro 0.576 → 0.616; ama micro kazanç testte küçük (0.848 → 0.856). Val'de +7 puandı; val 100 yorumla bu
  farkı abartmış.
- Val → test düşüşü küçük (uçtan uca micro 0.696 → 0.653). Beklenen yönde: anahtar kelimeler explore'dan
  türetildi, ama bölme ve nötr kararları val'de verildi.
- Kargo testte val'den belirgin kötü (R 0.91 → 0.74): örtük teslimat ifadeleri ("sabah yola çıktı",
  "belirtilen zamanda aldım") ve yazım varyantları ("hizli").

**Adım 16 için gerekçe:** Konu FN'lerinin %46'sı (44/96) **örtük**. Bunlar anahtar kelimeyle çözülemez,
çünkü konu hiç adlandırılmıyor; bağlamdan çıkarım gerekiyor. "Kelime eksik"lerin yarısı (27) yeni kelime
ister ve liste bitmez: kategoriye göre değişen özellik adları. Ucuz kısım ise 25 yazım/ek hatası; bunlar
normalleştirme (ascii katlama, kök bulma) ile kısmen düzelir, ama bu test sonrası bir iyileştirme olur ve
ölçülürse "iyimser" satırı olarak raporlanmalı. 67 FP de bağlam ister: aynı kelime ("büyük", "iade") bazen
konu, bazen değil. Duygu tarafında 15 bölme hatası, sabit bağlaç kuralının sınırı. 23 V2b hatası ise
yorum düzeyinde eğitilmiş bir modelin cümleciklerde kaymasından geliyor. → **Adım 16: BERT ile çok etiketli
konu tespiti** (bağlamdan örtük konuları öğrenmek); ardından belki **konuya koşullu duygu**: (konu, yorum)
çiftini birlikte okuyan bir model, bölmeye gerek bırakmaz. Eğitim verisi 300 etiketle çok az; Adım 16'nın
ilk sorusu etiketli veriyi nasıl büyüteceğimiz olacak (Adım 14'teki active learning yöntemi).

## Adım 16: BERT ile çok etiketli konu tespiti — TAMAMLANDI (2026-09-30)
**Görev:** Adım 15'in anahtar kelimeli konu tespitini öğrenilmiş bir modelle değiştirmek; asıl hedef ÖRTÜK konular
(konu hiç adlandırılmıyor: "arka kapak oturmuyor" → kalite). Klasör: `step16_topic_bert/`, etiketler:
`data/aspect_labels_step16/`. Koordinasyon: gorkanai-1e değerlendirdi, bu oturum (Mac mini) uyguladı, kararları
Görkan onayladı. Val/test sayıları görülmeden sabitlenen tüm kurallar: `step16_topic_bert/PLAN.md`.

**16.1 Veri büyütme (800 yeni etiket, id 5000-5799).** Elde sadece val 100 / test 200 vardı; ikisi de eğitime
giremez. Hepsi 8-40 kelime, yarı "pozitif" yarı "negatif" havuz etiketli, daha önce hiçbir adımda görülmemiş
(`common.check_no_leak`: ham + normalleştirilmiş metinle assert). Bölme ve kaynak etiketlemeden ÖNCE sabitlendi;
etiketleme dosyalarında sadece id + text vardı (kaynak/bölme/havuz etiketi görülmeden etiketlendi).
- **Val eki 200** (rastgele) → val = 300. Gerekçe: Adım 14 dersi (val'de satıcı 7 örnekti).
- **Eğitim tur 1, 300:** 150 rastgele + 150 anahtar kelimeyle hedefli (`prepare_round1.py`).
- **Eğitim tur 2, 300:** 150 rastgele + 150 "kararsız" (`prepare_round2.py`): tur 1 ile eğitilen kaba modelin
  20000 adayda `u = min_k |p_k − 0.5|` en küçük verdiği yorumlar (Adım 14'teki active learning).
- **Etiketleyici kayması kontrolü** (`drift_check.py`): Adım 15'in altınını başka bir oturum vermişti. Eski val'den
  30 yorum altına bakmadan yeniden etiketlendi: çift F1 0.867 ama bu oturum %17 FAZLA etiket veriyordu (61'e 52).
  "Q sadece açık genel yargı varsa" kalibrasyonundan sonra ikinci 30'da: çift F1 0.897, etiket oranı 53/54.
  (Kıyas: Claude-Görkan 0.667.)

| % yorumda konu var | n | K | F | Q | P | B | G | S | ort. konu |
|---|---|---|---|---|---|---|---|---|---|
| test (Adım 15 altını) | 200 | 17 | 24 | 65 | 44 | 12 | 8 | 9 | 1.79 |
| yeni val (rastgele) | 200 | 16 | 20 | 70 | 41 | 8 | 8 | 4 | 1.68 |
| tur 1 rastgele | 150 | 15 | 19 | 73 | 31 | 10 | 9 | 6 | 1.62 |
| tur 1 hedefli: boyut / görünüm / satıcı / kargo | 30/30/30/20 | | | | | **63** | **40** | **47** | (K **75**) |
| tur 1 hedefli: hiçbir kelime tutmuyor | 40 | 2 | 0 | 68 | 25 | 2 | 2 | 8 | 1.07 |
| tur 2 rastgele | 150 | 13 | 19 | 71 | 44 | 12 | 9 | 3 | 1.71 |
| tur 2 kararsız | 150 | 24 | 30 | 65 | 38 | 7 | 13 | 7 | 1.85 |

Eğitim (600) konu başına örnek: K 114, F 127, Q 411, P 214, B 72, G 68, S 44. Val (300): K 55, F 68, Q 199,
P 124, B 28, G 24, S 15.

**16.2 Model** (`train.py`, log: `log_train.txt`): `dbmdz/bert-base-turkish-cased` + 7 sigmoid çıkış (her konu ayrı
evet/hayır), BCE, pos_weight = sqrt(neg/pos), lr 3e-5 sabit, batch 16. 3 tohum; epoch adayları 3/5/8/12.

**16.3 Val** (`evaluate.py val`, log: `log_val.txt`). Epoch seçimi (val micro-F1, eşik 0.5, 3 tohum ort. ve min-max):
3 → 0.740 (0.699-0.772), 5 → 0.789, 8 → 0.816, **12 → 0.822 (0.812-0.832)**. Global eşik eğrisi düz (0.15-0.75
arası hepsi 0.821-0.834); seçilen 0.60. Dondurulan (`frozen_config.json`): epoch 12, tek global eşik 0.60,
3 tohumun olasılık ortalaması.

| Val | konu F1 micro/macro | örtük recall | uçtan uca micro/macro |
|---|---|---|---|
| Anahtar kelime V2, val 300 | 0.750/0.691 | 0 | 0.650/0.575 |
| **BERT, val 300** | **0.834/0.785** | 0.607 (n=145) | 0.735/0.671 |
| Anahtar kelime → BERT, eski val 100 (Adım 15 altını) | 0.783 → 0.815 (+3.2) | 0.405 | 0.696 → 0.732 |
| Anahtar kelime → BERT, yeni val 200 (bu oturumun etiketleri) | 0.733 → 0.844 (+11.1) | 0.689 | 0.626 → 0.736 |

"Örtük recall" = altın (yorum, konu) çiftlerinden V2 anahtar kelimesinin TUTMADIĞI alt kümede recall (anahtar
kelime çizgisi orada tanım gereği 0). Val'in iki yarısı arasındaki fark yüzünden test beklentisi testten önce
0.79-0.81 olarak yazıldı (PLAN.md).

**16.4 Test — BİR KEZ** (`evaluate.py test`, log: `log_test.txt`, olasılıklar: `test_probs.npz`). Ölçümden önce
dondurulan ayarlar ve başarı ölçütü commit 66f5020'de. Sonuçtan sonra hiçbir şey değiştirilmedi.

| Test (200) | P | R | **konu F1 micro/macro** | R (kelime tutuyor, n=263) | **R (kelime tutmuyor, n=96)** | **uçtan uca micro/macro** |
|---|---|---|---|---|---|---|
| Anahtar kelime V2 (Adım 15.4) | 0.797 | 0.733 | 0.763/0.722 | 1.000 | 0.000 | 0.653/0.616 |
| **BERT, 3 tohum ort., eşik 0.60 (ANA)** | 0.833 | 0.791 | **0.811/0.763** | 0.848 | **0.635** | **0.697/0.636** |
| BERT konu başına eşik (iyimser) | 0.786 | 0.841 | 0.813/0.740 | 0.890 | 0.708 | 0.697/0.619 |
| BERT VEYA anahtar kelime (bilgi) | 0.753 | 0.903 | 0.821/0.783 | 1.000 | 0.635 | 0.707/0.665 |

Tek tohumlar aynı eşikte: micro 0.803 / 0.810 / 0.807, macro 0.748 / 0.744 / 0.764.

| Konu (test) | n | anahtar kelime P/R/F1 | BERT P/R/F1 | BERT örtük recall | not |
|---|---|---|---|---|---|
| kargo | 34 | 0.81/0.74/0.77 | 0.88/0.88/**0.88** | 6/9 | |
| fiyat | 48 | 0.90/0.96/0.93 | 0.94/0.96/0.95 | 1/2 | |
| kalite | 130 | 0.84/0.68/0.75 | 0.78/0.84/**0.81** | 32/41 | precision düştü, recall arttı |
| performans | 87 | 0.82/0.72/0.77 | 0.85/0.76/0.80 | 15/24 | |
| boyut | 25 | 0.67/0.80/0.73 | 0.89/0.68/0.77 | 1/5 | gürültülü (n<30) |
| görünüm | 17 | 0.64/0.82/0.72 | 0.77/0.59/**0.67** | 2/3 | gürültülü; BERT DAHA KÖTÜ |
| satıcı | 18 | 0.46/0.33/0.39 | 0.75/0.33/0.46 | 4/12 | gürültülü; recall aynı |

**Eşleştirilmiş bootstrap** (yorum bazında, 2000 tekrar, tohum 16), BERT − anahtar kelime, %95 aralık. Ölçüt
testten önce yazıldı: "kazandı" demek için aralık 0'ı dışlamalı.
- konu F1 micro **+0.048 [+0.009, +0.089]** → 0'ı dışlıyor.
- konu F1 macro +0.041 [−0.027, +0.105] → **0'ı içeriyor, fark gürültüden ayrılamıyor.**
- uçtan uca F1 micro **+0.044 [+0.004, +0.085]** → 0'ı dışlıyor (alt sınır 0'a çok yakın).

**Elmayla elma** (Görkan'ın kör etiketleri, testteki 10 yorum; n=10, SADECE FİKİR VERİR): çift F1
anahtar kelime model-Görkan 0.649 / model-Claude 0.762; **BERT model-Görkan 0.545 / model-Claude 0.632**;
Claude-Görkan 0.703. Bu 10 yorumda BERT anahtar kelimeden KÖTÜ. (Not: Adım 15.4 tablosundaki Claude sütunları
"karar öncesi" altınla hesaplanmıştı, burada güncel altın kullanıldı; model-Görkan 0.649 iki tabloda aynı.)

**Sonuç:** BERT, konu tespitinde micro F1'i 0.763 → 0.811'e çıkardı (beklenti aralığının üst sınırı, 0.79-0.81) ve
anahtar kelimenin hiç yakalayamadığı 96 örtük çiftin 61'ini (%64) buldu. Kazanç sık konulardan (kalite, kargo,
performans) geliyor; nadir konularda (boyut, görünüm, satıcı) kanıt yok — macro farkı gürültüden ayrılamıyor,
görünümde BERT daha kötü. Uçtan uca kazanç +4.4 puan, sınırda anlamlı. Val 300'deki +8.4 puan iyimserdi.

**Dersler:**
- **Val, modelle aynı etiketleyiciden geliyorsa iyimserdir.** Aynı model, bu oturumun etiketlediği yeni val'de +11.1,
  Adım 15 oturumunun etiketlediği eski val'de +3.2 puan kazandırdı; test (+4.8) eski val'e yakın çıktı. İki ayrı
  Claude oturumu bile birbirinin "ikinci etiketleyicisi" gibi davranıyor. Eski-100 / yeni-200 ayrımını raporlamak
  test beklentisini doğru kurmamızı sağladı (tahmin 0.79-0.81, sonuç 0.811).
- **Etiketleyici kaymasını etiketlemeden ÖNCE ölç.** 30 yorumluk kör kontrol, bu oturumun %17 fazla etiket verdiğini
  gösterdi; 800 yorumu etiketledikten sonra fark edilseydi hepsi çöpe giderdi. Yine de kalan fark (P kuralı tur 1
  ile tur 2 arasında netleşti) eğitim setinin iki yarısını hafif farklı yaptı.
- **Kararsızlık örneklemesi nadir sınıfları getirmez.** `min_k |p_k − 0.5|` en sık ve en bulanık sınıra kilitlendi:
  150 kararsız yorumun 124'ü Q ya da P'de kararsızdı; B/G/S artmadı. Bu, Adım 15'in bulgusuyla tutarlı (en bulanık
  sınır Q/P — model, insanın kararsız olduğu yerde kararsız). Nadir sınıf için anahtar kelimeli hedefleme çok daha
  etkiliydi (boyut %10 → %63, satıcı %6 → %47). Konu başına kotalı kararsızlık daha iyi olurdu.
- **Başarısız deneme — az eğitilmiş modelle active learning:** tur 2'nin ilk aday seçimi 5 epoch'luk kaba modelle
  yapıldı; eğitim kaybı 0.74 → 0.57, tüm olasılıklar 0.5 civarı (medyan u 0.033), seçim anlamsızdı. Hiçbir yorum
  etiketlenmeden atıldı, 15 epoch ile tekrarlandı (`log_prepare_round2_run1_undertrained.txt`). Ders: kararsızlık
  ancak model bir şey öğrendiyse bilgi taşır; seçimden önce eğitim kaybına ve u dağılımına bak.
- **Micro ile macro farklı hikâye anlatır.** Micro kazanç anlamlı, macro değil: 600 eğitim yorumu kalite (411 örnek)
  için yeterli, satıcı (44) ve görünüm (68) için değil. "Model daha iyi" cümlesi sadece sık konular için doğru.
- **Güven aralığı olmadan 4 puanlık fark iddia edilmez.** n=200 ile uçtan uca +0.044'ün aralığı [+0.004, +0.085].
  Ölçütü testten önce yazmak, sonucu gördükten sonra "kazandı" tanımını esnetmeyi engelledi.
- ~~Örtük konuyu bulmak duyguyu çözmez; bölme artık darboğaz~~ — **bu ders 16.5'te YANLIŞLANDI** (tahmindi, ölçülmemişti):
  örtük konularda tüm-yorum yedek kuralı duygu hatasının kaynağı değil (bkz. 16.5, madde 5).

**Sınırlamalar:**
1. Adım 15'in 1-3. sınırlamaları aynen geçerli (50/50 poz/neg örneklem, negasyon kalıpları ve 8 kelimeden kısa
   yorumlar dışarıda).
2. Eğitim etiketleri tek etiketleyiciden (Claude, bu oturum); test altını başka bir Claude oturumundan. Görkan'ın
   30'luk gözden geçirmesi (`review_sample16.csv`) bu yazı yazılırken henüz yapılmamıştı.
3. Yeni etiketlerde altına göre P hafif eksik, Q hafif fazla (tur 1 ve val eki); S muhtemelen eksik (rastgele
   gruplarda %3-6, altında %7-9). Tur 1 etiketleri yeniden etiketlenmedi.
4. Epoch 12 ızgaranın üst sınırıydı; val görüldükten sonra ızgara genişletilmedi. Daha uzun eğitim denenmedi.
5. Eşik eğrisi düz olduğu için 0.60 seçimi büyük ölçüde gürültü; konu başına eşikler testte işe yaramadı
   (macro 0.763 → 0.740).
6. Bağımsız insana karşı kanıt yok: Görkan'ın 10 kör test yorumunda BERT anahtar kelimeden kötü (0.545'e 0.649).
   n=10 ile hüküm verilemez, ama "BERT insana daha çok benziyor" da denemez.
7. "Altın şüpheli" val yorumları (4089, 4196, 4200, 4177) düzeltilmedi; test altınında benzer hatalar olabilir.
8. Ablasyonlar (hedefli/kararsız yarıyı çıkarma, daha uzun eğitim) yapılmadı. Hata kovaları: 16.5.

### Adım 16.5: BERT test hata kovaları — TAMAMLANDI (2026-09-30)
`step16_topic_bert/error_buckets.py`, log: `log_error_buckets.txt`, elle sınıflandırma: `test_bert_errors_manual.csv`
(132 hatanın hepsi). Eğitim yok, ayar yok; sadece dondurulmuş test tahminlerinin analizi. Test altınına dokunulmadı.
**Buradan çıkan her fikir "test görüldü → yeni test gerekir" notuyla okunmalı.**

| Test | konu FP | konu FN | duygu hatası (yedek kural / bölme / V2b) |
|---|---|---|---|
| Anahtar kelime V2 | 67 | 96 | 38 (0 / 15 / 23) |
| BERT | 57 | 75 | 40 (5 / 13 / 22) |

Konu başına FP → FN (kelime → BERT): kargo 6→4 / 9→4; fiyat 5→3 / 2→2; **kalite 17→31 / 41→21**; performans
14→12 / 24→21; boyut 10→2 / 5→8; görünüm 8→3 / 3→7; satıcı 7→2 / 12→12.

**1. Geçiş tablosu** (anahtar kelimenin 96 FN'si, Adım 15.4 kovaları): örtük 44'ün **24'ü düzeldi (%55)**; kelime
eksik 52'nin 37'si (%71): yeni kelime 17/27, yazım varyantı 13/15, ek/yumuşama 7/10. Anahtar kelimenin 67 FP'sinin
49'u BERT'te yok; ama BERT **39 yeni FP** ve **40 yeni FN** (kelime tutuyordu, BERT kaçırdı: 263'ün %15'i; kalite 12,
performans 12, görünüm 6, satıcı 4, boyut 4) üretti. Yani BERT anahtar kelimenin üstüne eklenmiyor; hataların yerini
değiştiriyor. 75 FN'nin 20'sinde olasılık 0.40-0.59 (eşiğin hemen altı).

**2. Kalite FP'leri (31; precision 0.84 → 0.78):** **cömert Q 17**, model hatası 12, altın şüpheli 2. Yani düşüşün
tamamı etiket farkından: sadece 12 gerçek model hatası sayılsa Q precision'ı ~0.90 olurdu (anahtar kelime 0.84).
"Cömert Q"nun iki kalıbı: (i) işe yarama/şikâyet ifadesi altında P, yeni etiketlerde Q ("çok işe yarıyor",
"yetersiz", "hakkını veriyor"); (ii) **kitaplar**: altın içerik yorumunu P sayıyor ("çok ağır ilerliyor" → P),
bu oturum Q etiketledi. 12 yorumda "Q/P takası" var (aynı yorumda kalite FP + performans FN); tersi sadece 2.
gorkanai-1e 31 FP'yi bağımsız okudu, sınıflamaya katıldı ve şu okumayı ekledi: "cömert Q"ların çoğu yazılı kurala göre
savunulabilir Q — yani bu bir etiketleyici kaymasından çok **kuralın kendi belirsizliği**: "işe yarama → Q" maddesi ile
"adı konan özellik → P" maddesi aynı cümleye uyuyor (Adım 15'teki kappa 0.25 bulgusunun devamı).

**3. Görünüm FN (7; recall 0.82 → 0.59) ve boyut FN (8):** görünümde 5'i AÇIK ifade ("muşamba gibi duruyor",
"rastgele renkler geliyor", "hoş duruyor", "görüntüsü", "ucuz duruyor"), 2'si örtük. Boyutta 6'sı açık ("küçücük",
"ebat ideal", "orta boyutlarda", "inceliği ve hafifliği"), 2'si örtük (miktar). Bunlar anahtar kelimenin zaten
yakaladığı kolay örnekler: 68 ve 72 eğitim örneği bu konuları öğrenmeye yetmemiş; model çok konulu yorumlarda
nadir konuyu atlıyor.

**4. Satıcı (FN 12, FP 2):** kaçan 12'nin 8'i örtük (eksik aksesuar "kablosu da olsaymış", yanlış marka geldi,
"siteden memnunum", "güvenilir"), 4'ü açık (servis, kılavuz çıkmadı, eksiksiz). FP sadece 2 (soru cümlesi, teşekkür
kalıbı). Model satıcıyı nadiren söylüyor (8 tahmin), söylediğinde çoğunlukla doğru; recall'da kazanç yok.

**5. Duygu tarafı — beklenenin TERSİ:** BERT'in doğru bulduğu 284 konudan 40'ında duygu yanlış (%14.1). Konunun
kelimesi hiçbir cümlecikte geçmeyen (örtük) 61 doğru konuda hata **7 (%11.5)**; kelimesi geçen 223'te 33 (%14.8).
Tüm-yorum yedek kuralına bağlanabilen hata sadece 5. Yani **örtük konularda duygu hattı daha kötü çalışmıyor**;
16.4'te yazdığım "bölme artık darboğaz" dersi veriyle desteklenmedi. Hataların dağılımı anahtar kelimeyle aynı:
V2b hatası 22, bölme hatası 13; ayrıca 40'ın 11'i altın nötr (nötr kapalı olduğu için hiç yakalanamaz).
Konuya koşullu duygu modelinin gerekçesi bu yüzden "örtük konular" DEĞİL, hâlâ Adım 15.4'teki iki kova:
V2b'nin cümlecikte kayması (22) ve zıt duygulu konuları aynı cümlecikte taşıyan bölme (13).

**6. Elmayla elma (10 kör yorum, nitel):** BERT'in Görkan'dan ayrıştığı yerler çoğunlukla aynı Q/P kayması:
4144 (Görkan ve altın: performans negatif; BERT: kalite pozitif), 4062 (BERT kaliteyi kaçırdı), 4141 (BERT
performans ve görünümü kaçırdı, kalite ekledi), 4279 (görünüm yerine fiyat: "ucuz duruyor").

**Altın şüpheli listesi** (12 çift, düzeltilmedi): 4006, 4009, 4045, 4060, 4082, 4092, 4124, 4230 ve kitap yorumları
4180, 4182, 4185, 4199.

**Dersler (16.5):**
- **Precision düşüşü model değil, etiket farkıydı.** Hatalara tek tek bakmadan "BERT kalitede daha çok yanlış alarm
  veriyor" sonucuna varırdık; oysa 31 FP'nin 19'u iki etiketleyici oturumunun Q/P sınırını farklı çizmesinden.
  Q/P sınırı Adım 15'te insan-insan uyumunu, Adım 16'da oturum-oturum uyumunu bozdu: görevin en zayıf halkası tanım.
- **Ölçmeden gerekçe yazma.** "Örtük konuda duygu tüm yorumdan alınıyor, o yüzden uçtan uca kazanç küçük" makul
  görünüyordu ve PROGRESS'e ders diye yazıldı; sayım bunu yanlışladı (%11.5'e %14.8).
- **Öğrenilmiş model, kuralın kolay yaptığını unutabiliyor.** 40 yeni FN'nin çoğu açık ifade. Bu yüzden "BERT VEYA
  anahtar kelime" satırı testte en yüksek macro'yu verdi (0.783) — ama bu teste bakılarak görüldü; iddia etmek için
  yeni bir test gerekir.
- **Kitap gibi bir ürün kategorisi, konu tanımını sessizce bozabilir.** "Performans" kitap için ne demek, kurallarda
  yok; iki oturum farklı karar verdi.

**Test görüldü → yeni test gerekir (fikirler, UYGULANMADI):** (a) BERT ile anahtar kelimeyi birleştirmek (VEYA ya da
kelime eşleşmesini modele özellik olarak vermek); (b) nadir konular için konu başına kotalı hedefli etiketleme;
(c) Q/P tanımına "kitap/içerik" ve "işe yarama" için açık örnekler ekleyip iki etiketleyiciyle yeniden uyum ölçmek;
(d) eşiği düşürmek (20 FN eşiğin hemen altında) — bu doğrudan teste bakarak ayar olur, yeni val + yeni test ister.

## Adım 17: Konuya koşullu duygu modeli — TEST ÖLÇÜLDÜ (2026-10-01)
Plan: `step17_aspect_sentiment/PLAN.md`. Dondurulan (`frozen_config.json`): v2b başlangıcı, epoch 8, tohum 0/1/2
olasılık ortalaması; konu modeli Adım 16 (dondurulmuş). Test, Görkan'ın onayıyla (gorkanai-8f üzerinden) Mac mini'de
**bir kez** koşturuldu: `python step17_aspect_sentiment/evaluate.py test` → `log_test.txt`, olasılıklar
`test17_probs.npz`, `test15_sent_probs.npz`. Öncesinde iki altın dosyasının sha256'sı `frozen_config.json` ile
eşleşti; hiçbir ayar, eşik ya da kod değişmedi, tekrar koşturma yok.

**Ana sonuç — (i) altın konularla duygu doğruluğu, micro, yeni test, gorkanai-1e altını (363 çift):**
eski hat **0.868** → YENİ **0.887**, fark **+0.019 [−0.016, +0.054]** (eşleştirilmiş bootstrap, 2000, tohum 17).
**Aralık 0'ı içeriyor → başarı ölçütü karşılanmadı: fark yok (gürültüden ayrılamıyor).** Pratik tavan 0.967.

**Test öncesi güncellenmiş beklentiyle (PLAN.md sonu) kıyas:**
| Ölçü (yeni test, 1e altını) | Güncel beklenti | Sonuç | |
|---|---|---|---|
| (i) micro farkı | +1 ile +3 puan, aralık büyük olasılıkla 0'ı içerir | +1.9, aralık 0'ı içeriyor | tuttu |
| (ii) uçtan uca micro farkı | 0 ile +2 puan | 0.706 → 0.720, +0.015 [−0.013, +0.043] | tuttu |
| Zıt duygulu alt küme | kazanç beklenmiyor | 0.705 → 0.744 (78 çift) | beklenenden iyi, n küçük |
| Örtük çiftler | küçük kötüleşme olası | 0.874 → 0.849 (119 çift) | tuttu |
| Eski test uçtan uca | 0.697 → 0.69-0.72 | 0.697 → 0.723 | üst sınırın hemen üstü |
| Konu F1 micro (Adım 16 BERT) | ~0.80-0.83 | 0.802 | alt sınırda |

İlk beklenti (PLAN madde 6) "(i) aralığı büyük olasılıkla 0'ı dışlar" diyordu; bu tutmadı — val sonrası güncellemenin
doğru yönde olduğu testte de görüldü. Nokta tahmini (0.887) ilk beklentinin 0.86-0.90 aralığında.

**İkincil satırlar (ana iddia değil):**
- Yeni test, bu oturumun (macmini) altını: 0.867 → 0.902, +0.035 [+0.000, +0.069] — sınırda, 0'ı içeriyor.
- Yeni test, iki altının ortak çiftleri (320): 0.887 → 0.922, +0.034 [+0.003, +0.068] — 0'ı dışlıyor.
- Eski test (Adım 15; hataları okunmuştu, iyimser olabilir): 0.858 → 0.891, +0.033 [+0.003, +0.063] — 0'ı dışlıyor;
  uçtan uca +0.026 [+0.000, +0.052].
- Birleşik 400 (yeni 1e + eski): +0.026 [+0.004, +0.049], uçtan uca +0.020 [+0.001, +0.039] — ikisi de 0'ı dışlıyor.
  Bu satır önceden ikincil olarak tanımlanmıştı; ana sonucun yerine geçmez.
- Görkan'ın kör 20'si (31 çift): 0.774 → 0.806, +0.032 [−0.062, +0.160] — sadece fikir verir.

Okuma: farkın yönü bütün satırlarda aynı (+2 ile +3.5 puan) ve yeni model **genel micro'da** hiçbir altında
kötüleşmiyor (alt kümelerde kötüleştiği yer var: örtük çiftler, aşağıda ve 17.5); ama önceden
seçilen ana ölçü (1e altını, yeni test) gürültüden ayrılamıyor. Kayda geçen sonuç **"fark yok"**. Diğer altınlardaki
0'ı dışlayan aralıklar sonradan ana iddiaya terfi ettirilmez (eski test görülmüş, ortak çiftler/birleşik ikincil).

**Ek rapor (önceden planlanmamış) — konu tespiti bloğu:** `evaluate.py`'ye bu blok (anahtar kelime çizgisi, kelime
tutmayan çiftlerde recall, bootstrap) dondurmadan SONRA, test ölçülmeden önce eklendi (commit 30e8f6b); duygu modelinden
bağımsız, ayar içermiyor. Yeni test 1e altınında: anahtar kelime V2 F1 micro 0.712 → BERT (Adım 16) **0.802**,
fark +0.090 [+0.051, +0.131] (0'ı dışlıyor); kelime tutmayan 119 çiftte BERT recall 0.504 (anahtar kelime 0).
Macmini altınında +0.095 [+0.056, +0.134], eski testte +0.048 [+0.009, +0.089]. Bu, Adım 16 konu modelinin ilk temiz
(görülmemiş testte) ölçümü; satıcıda BERT anahtar kelimeden kötü kalmaya devam ediyor (0.36 → 0.33).

Hata kovaları: 17.5 (aşağıda).

**Sınırlamalar (Adım 17):**
1. Ana altın (gorkanai-1e) **insan altını değil**: bağımsız ama aynı ailedeki bir Claude oturumu; eski testin hatalarını
   ve tur 1 etiketlerini okumuştu. Model de Claude etiketleriyle eğitildi. "Claude'a benzeme" ile "doğru okuma" ayrılamıyor.
2. Görkan'ın kör 20'si (31 çift) tek insan kıyası: YENİ–Görkan duygu uyumu **0.806**, oysa YENİ–1e tüm testte 0.887
   (aynı 20 yorumda 0.861). Ama **1e–Görkan da sadece 0.826** (19/23) — fark büyük ölçüde Görkan'ın Claude altınlarından
   farklı okumasından geliyor, modelin kendisinden değil (17.5 madde 5). n=20, hüküm verilemez.
3. n küçük: 363 çiftte ±3.5 puanlık aralık; +2 puanlık gerçek bir kazanç bu test boyutuyla ayırt edilemez.
4. Nötr kapalı: altın nötr 6 çift (1e) iki hatta da ulaşılamaz hata; tavan 0.967.
5. Val'den seçilen tek yapılandırma (v2b, epoch 8); başka başlangıç/epoch denenmedi, denenmeyecek (test görüldü).
6. Yeni test 17.5'ten sonra okunmuş sayılır; sonraki her iddia için yeni bir test gerekir.

**Dersler (Adım 17):**
- **Adım 18 dersi (genel):** 200'lük testte "fark yok" bir güç sorunuydu; aynı donmuş model 500'lük kasa testte
  +0.035 [+0.016, +0.054]. 17.5'te n≈40 hata üzerinden çıkarılan alt küme "ders"leri tekrarlamadı → küçük alt kümelerden
  ders çıkarırken aralık hesaplanmalı.
- **"Fark yok" da sonuçtur, önceden yazılmış ölçüt olmasa kaybolurdu.** İkincil satırların üçü (eski test, ortak çiftler,
  birleşik 400) 0'ı dışlıyordu; ana ölçü önceden sabitlenmemiş olsaydı "kazandı" diye raporlamak çok kolaydı.
- **Val'den sonra beklentiyi güncellemek işe yaradı.** İlk beklenti ("aralık 0'ı dışlar") fazla iyimserdi; val'de görülen
  tablo testte neredeyse aynen tekrarladı (val +0.021, test +0.019).
- **(Adım 18'de YANLIŞLANDI)** ~~(a) Örtük çiftlerde kötüleşme tutarlı~~ — 500'lük kasa testte örtük çiftler (273) 0.894 → **0.919**;
  val ve 200'lük testteki düşüş (net ~3 çift) küçük n'nin gürültüsüydü. Eski metin, kayıt için: val 0.869 → 0.848, test 0.874 → 0.849. Eski hat örtük konuda tüm yorumun
  duygusunu alıyor; tek duygulu yorumlarda bu çoğunlukla doğru. Yeni model konu ifadesine bakıp yorumdaki olumsuz
  yüzeyli kalıplara ("eksiği yok", "gerek yok", "yapmıyor") ya da başka konunun duygusuna kayıyor (17.5 madde 3).
- **(Adım 18'de YANLIŞLANDI)** ~~(c) Zıt duygulu hedef tutarlı kazanç vermedi~~ — kasa testte zıt duygulu çiftler (128)
  0.695 → **0.812** (+11.7); macmini altınında 0.673 → 0.770. Eski metin, kayıt için: yeni test (1e) 0.705 → 0.744 (+4), val 0.728 → 0.744 (+1.6),
  eski test 0.667 → 0.667 (0). Modelin varlık nedeni olan alt kümede kazanç küçük ve kararsız; 17.5'te YENİ'nin en büyük
  hata kovası hâlâ "başka konunun duygusu" (41'in 15'i, zıt duygulu yorumlarda 20'nin 14'ü).
- **(Adım 18'de ZAYIFLADI)** Konu-koşullu girdi tek başına yetmiyor — kasa testte zıt duygulu kazanç +11.7, ayrışan 93 çiftin
  55'inde YENİ doğru (23'ünde eski); "başka konunun duygusu" hâlâ kalan hata olabilir ama kasa okunmadığı için bilinmiyor. Eski metin: [konu] [SEP] [yorum] girdisiyle 600 yorumluk eğitimde model, konunun
  hangi cümleye ait olduğunu güvenilir öğrenmedi; düzelttiği 8 "başka konu" hatasına karşı 6 yenisini yaptı.

### Adım 17.5: Duygu modeli test hata kovaları — TAMAMLANDI (2026-10-01)
`step17_aspect_sentiment/error_buckets.py`, log: `log_error_buckets.txt`, elle sınıflandırma: `test17_errors_manual.csv`
(70 satır: yeni testte iki hattın en az birinin yanlış olduğu 63 çift + val'de örtük/eski doğru/YENİ yanlış 7 çift).
Eğitim yok, ayar yok, test sayıları yeniden hesaplanmadı (script log_test'in 0.887 / 0.868'ini assert ediyor). Sadece
dondurulmuş `test17_probs.npz` ve `val_probs.npz`; eski hat deterministik olarak yeniden üretildi. Altınlara dokunulmadı.
**Yeni test (6000-6199) bu adımdan sonra OKUNMUŞ sayılır**: hataları tek tek okundu; artık temiz test değil. Buradan
çıkan her fikir "test görüldü → yeni test gerekir" notuyla okunmalı.

**1-2. Kovalar** (yeni test, 1e altını, altın konular verilmiş; "YENİ doğru" sütununda kova eski hattın hatası için):
| kova | YENİ yanlış (41) | ikisi yanlış (26) | YENİ düzeltti (22) | YENİ bozdu (15) |
|---|---|---|---|---|
| altın nötr | 6 | 6 | 0 | 0 |
| karışık | 5 | 1 | 2 | 4 |
| başka konunun duygusu | 15 | 9 | 8 | 6 |
| örtük/dolaylı duygu | 9 | 6 | 7 | 3 |
| olumsuzlama/ironi | 4 | 2 | 2 | 2 |
| altın şüpheli | 2 | 2 | 0 | 0 |
| diğer model hatası | 0 | 0 | 3 | 0 |

- YENİ'nin 41 yanlışının 26'sında gerçek duygu pozitif; YENİ 29'unda negatif demiş. Yanlışların 23'ünde emin
  (p ≥ 0.9 ya da ≤ 0.1). Konu başına: kalite 19, performans 10, fiyat 6. Macmini altını bu 41'in 5'inde aynı konuya farklı
  duygu vermiş, 11'inde o konuyu hiç vermemiş (yani ~%40'ında konu/duygu etiketi de tartışmalı).
- **Düzelttiği:** eski hattın "başka konunun duygusu" (8; aynı cümlecikte zıt duygulu iki konu) ve "örtük/dolaylı"
  (7; "beklediğimden de kötü", "fotoğraftaki gibi", "kaliteli ürün tercih edin") hataları. **Bozduğu:** yine en çok
  "başka konunun duygusu" (6) ve karışık (4). Net: başka konu +2, örtük/dolaylı +4, karışık −2.
- Kalan 41'in 11'i (altın nötr 6 + karışık 5) yapı gereği ulaşılamaz/belirsiz; 2'si altın şüpheli.

**3. Örtük çiftler** (test 0.874 → 0.849; val 0.869 → 0.848): eski doğru/YENİ yanlış test 8 (karşılığında YENİ
doğru/eski yanlış 5), val 7 (karşılığında 4). Desen val'de tekrarlıyor ve tek bir kovada toplanmıyor: test'te karışık 2,
başka konu 2, örtük/dolaylı 2, olumsuzlama/ironi 2; val'de başka konu 3, olumsuzlama/ironi 3, örtük/dolaylı 1.
Ortak okuma: konunun kelimesi yokken eski hat tüm yorumun duygusunu alıyor (çoğu zaman doğru); yeni model ise
**olumsuz yüzeyli olumlu kalıplara** ("eksiği yoktu", "elektriklenme yapmıyor", "gerek yok", "israf etmeyin") ve
yorumdaki **başka konunun** duygusuna kayıyor. 15 çiftin 5'i sınırda (0.41-0.51).

**4. Zıt duygulu yorumlar** (33 yorum, 78 çift; 0.705 → 0.744): YENİ'nin 20 yanlışının **14'ü "başka konunun duygusu"**,
3 örtük/dolaylı, 2 altın şüpheli, 1 karışık. Bu alt kümede düzelttiği 9, bozduğu 6. Modelin hedefi olan hata türü
hâlâ en büyük kova; tipik kalıp "X güzel ama/yalnız/tek kusuru Y" (6027, 6067, 6115, 6156, 6174, 6189).

**5. Görkan'ın kör 20'si** (31 Görkan çifti; YENİ–Görkan 25/31 = 0.806; **1e–Görkan 19/23 = 0.826**; YENİ–1e aynı
20 yorumda 0.861). YENİ'nin Görkan'dan ayrıştığı 6 çiftin 3'ünde 1e modelle aynı (6103 kitap kalite, 6026 Q/P'nin
ters dağıtımı ×2), 2'sinde Görkan'la aynı (6113 "parasına göre belki iyi", 6046 "bebekler bile çekip çıkarıyor"),
1'inde 1e o konuyu vermemiş (6195). Yani model–Görkan ayrışmasının yarısı **Claude altınlarının da Görkan'dan ayrıştığı**
yerde: model büyük olasılıkla Claude-tarzı okumayı öğrenmiş; Görkan kitap ve Q/P'de farklı okuyor. n=20, sadece fikir.

**Altın şüpheli** (2 çift, düzeltilmedi): 6001 kalite ("idare eder", macmini Qx), 6006 performans ("eldeki hissi iyi",
macmini P vermemiş).

**Test görüldü → yeni test gerekir (fikirler, UYGULANMADI):** (a) olumsuz yüzeyli olumlu kalıplar ve "X güzel ama Y"
yapısı için hedefli eğitim örnekleri; (b) örtük konuda eski hattın tüm-yorum duygusunu modele özellik olarak vermek ya da
iki hattı birleştirmek; (c) nötr/karışık sınıfını açmak (11 hata yapı gereği); (d) Q/P ve kitap tanımı için insanla
(Görkan) yeniden uyum ölçmek — model–insan farkının yarısı tanım farkı.

## Adım 18: Büyük temiz test + insan çapası — (1) seçim ve (2a) kalibrasyon TAMAMLANDI (2026-10-01)
Plan: `step18_big_test/PLAN.md`. Model eğitilmez. Seçim (`prepare_sets.py`, 1432083): kalibrasyon 100 (id 7000-7099) +
ana test 500 (id 8000-8499), 50/50 ve 250/250 havuz etiketli, hiç görülmemiş; sızıntı assert'leri geçti; kalan
"negatif" havuz 302. Ana testin metinleri okunmadı (KASA kuralı: hataları tek tek okunmayacak, en fazla 4 karşılaştırma).

### Adım 18 (2a): kalibrasyon — üç kör etiketleyici
Etiketler: `data/aspect_labels_step18/calib_fresh.csv` (taze Claude oturumu, projede hiçbir etiketi görmemiş),
`calib_macmini.csv` (bu oturum), `calib_gorkan.csv` (Görkan; kısa yönerge `GORKAN_YONERGE.md`). Rapor:
`step18_big_test/agreement.py` → `log_agreement.txt`.

| Çift | çift F1 | konu F1 | ortak konuda duygu uyumu | kappa Q / P | birebir aynı | Q/P takası |
|---|---|---|---|---|---|---|
| fresh – macmini | 0.935 | 0.946 | 0.989 | 0.89 / 0.88 | 81/100 | 2 |
| Görkan – fresh | 0.642 | 0.735 | 0.873 | 0.53 / 0.40 | 37/100 | 4 |
| Görkan – macmini | 0.644 | 0.718 | 0.897 | 0.44 / 0.32 | 37/100 | 5 |

**Beklentiyle kıyas (PLAN madde 4):** Claude–Claude 0.88-0.92 bekleniyordu → 0.935, üstünde. Görkan–Claude 0.60-0.70
→ 0.64, aralıkta. Duygu uyumu ~0.85 → 0.87-0.90, biraz üstünde. "Q/P kappası Görkan'la < 0.4" kısmen tuttu (P 0.40/0.32;
Q 0.53/0.44 tutmadı).

**Ana desen:** Görkan 136 çift verdi, Claude'lar 185/187. Ayrışma çoğunlukla **kapsam**: Görkan ikincil konuları yazmıyor
(sadece Claude'ların verdiği: performans 28/27, görünüm 11/12, boyut 6, kargo 4, satıcı 4-5) ve adı konan özelliği Q'ya
topluyor ya da hiç yazmıyor (ör. "çok ses çıkarıyor, hiç beğenmedim" → Görkan `Qn`, Claude'lar `Pn,Qn`). Gerçek Q/P
takası az (4-5). Konu ortak olduğunda duygu uyumu yüksek (0.87-0.90). Görkan'ın iki Claude'dan da ayrıştığı ama iki
Claude'un birebir anlaştığı 48 yorum konuşmanın gündemiydi (log'da listeli).

**Görkan'ın kararları (kalibrasyon konuşması, laptop oturumu gorkanai-8f üzerinden):**
1. Kapsam: görüş bildirilen HER konu yazılır (şimdiki kural).
2. Adı konan özelliğin şikâyeti ("çok ses çıkarıyor", "titreme") → P (şimdiki kural 1).
3. Tek başına "tavsiye ederim" → Q değil (şimdiki kural 2).
4. 7097 (`Qn,Fn`) ve 7011 (`Fp`): Görkan "bilinçliydi" dedi → işaretleme hatası değil; etiketler aynen kaldı.

**Sonuç: `LABEL_RULES.md` DEĞİŞMEDEN donduruldu** (sha256 `step18_big_test/frozen_rules.txt`'de). Kalibrasyon tartışması
kural dosyasına yazılmadı; ana testi etiketleyecek taze oturum sadece o dosyayı okuyacak. Kalibrasyon 100 artık okundu,
test değil (ileride val/eğitim olabilir).

**Sınırlamalar:**
- Görkan sadece kısa yönergeyi kullandı, Claude'lar ayrıntılı `LABEL_RULES.md`'yi. Görkan–Claude farkının bir kısmı
  kural bilgisinden geliyor ("kim haklı" değil, "aynı kuralı biliyorlar mı").
- Claude–Claude 0.935 **doğruluk değil**, aynı kuralı uygulamadaki **tutarlılık**. İki Claude ortak bir okuma tarzını
  paylaşıyor olabilir; insan çapası (Görkan) bunun sadece 0.64'ünü paylaşıyor.
- Mac mini oturumu önceki adımların etiket/hata tartışmalarını görmüştü (kör ama taze değil); taze oturum görmemişti.
  İkisinin yüksek uyumu bu farkın küçük olduğunu düşündürüyor.
- Tek insan, n=100.

### Adım 18 (2b + 5): ana test 500 — etiketleme ve ölçüm 1-3 (2026-10-01)
**Etiketler:** ana altın taze oturum (`test18_fresh.csv`), ikinci altın bu oturum (`test18_macmini.csv`); ikisi birbirini
görmeden, donmuş `LABEL_RULES.md` ile. sha256'lar `step18_big_test/frozen_test.json`'da. İki altının uyumu (sadece toplu,
`log_test_agreement.txt`): konu F1 0.966, çift F1 0.935, ortak konuda duygu uyumu 0.968 ((i)'nin pratik tavanı), konu
başına kappa 0.87-0.99. Ana altında 43 nötr çift (%4.7) iki hat için de ulaşılamaz.

**Ölçüm:** `step18_big_test/evaluate.py` (Adım 17 `report()` aynen; bootstrap 2000/tohum 17, konu tohum 16; başta bütün
sha256'lar assert edildi). Önce kalibrasyon 100'de kuru çalıştırma (`log_calib_dryrun.txt`, kod kontrolü), sonra Görkan'ın
onayıyla ana testte **bir kez** (`log_test.txt`, `test18_probs.npz`). Hiçbir ayar/kod değişmedi.

**ANA İDDİA — (i) altın konularla duygu doğruluğu micro, fresh altını, 917 çift:** Adım 15 hattı **0.862** → Adım 17
duygu modeli **0.896**, fark **+0.035 [+0.016, +0.054]** — **aralık 0'ı DIŞLIYOR: Adım 17 modeli kazandı.**
(Adım 17'nin 200'lük testinde aynı fark +0.019 [−0.016, +0.054] idi, "ayırt edilemez"; 500'lük test aralığı ±0.019'a indirdi.)

| Ölçüm (ana altın) | Beklenti (PLAN madde 4) | Sonuç | |
|---|---|---|---|
| 1. (i) micro farkı | +1 ile +3; 0'ı dışlama ~yarı yarıya | +0.035 [+0.016, +0.054] | dışladı; nokta beklentinin hemen üstünde |
| 2. konu F1 micro, BERT − kelime | +0.06 ile +0.09, 0'ı dışlar | 0.753 → 0.815, +0.062 [+0.035, +0.087] | tuttu |
| 3. (ii) uçtan uca micro farkı | +1 ile +2.5, dışlamayabilir | 0.701 → 0.730, +0.029 [+0.012, +0.046] | dışladı; nokta beklentinin biraz üstünde |

**Alt kümeler (ana altın, (i)):** zıt duygulu çiftler (128) 0.695 → **0.812** (+11.7); tek duygulu 0.888 → 0.910; örtük
(273) 0.894 → 0.919 (Adım 17 testinde ve val'de örtükte görülen kötüleşme burada TEKRARLAMADI); açık 0.848 → 0.887; nötr
hariç 0.904 → 0.941. Konu başına (i): yedi konunun hepsinde YENİ ≥ eski. Ayrışan 93 çiftin 55'inde YENİ, 23'ünde eski doğru.
**Konu tespiti (ölçüm 2) konu başına F1, kelime → BERT:** kargo 0.85→0.93, fiyat 0.96→0.95, kalite 0.75→0.82, performans
0.70→0.77, boyut 0.66→0.77, görünüm 0.63→0.77, **satıcı 0.43→0.17** (BERT satıcıda belirgin kötü; Adım 16-17'deki işaret
burada daha net). macro 0.710 → 0.741. Kelime tutmayan 273 çiftte BERT recall 0.535.

**İkincil satırlar (ana iddiaya terfi yok):** macmini altını (i) +0.040 [+0.020, +0.059], (ii) +0.031 [+0.013, +0.048],
konu +0.061 [+0.036, +0.086]; iki altının ortak çiftleri (865) (i) 0.881 → 0.919, +0.038 [+0.018, +0.058]. Hepsi aynı yönde
ve 0'ı dışlıyor.

**Okuma:** Adım 17'nin "fark yok"u bir güç sorunuymuş: aynı model, aynı kurallar, 2.5 kat büyük ve okunmamış testte
+3.5 puan ve aralık 0'ı dışlıyor. Kazancın en büyüğü modelin hedefi olan zıt duygulu yorumlarda. Ama: (a) altınlar yine
Claude etiketi (insan çapası kalibrasyonda Görkan–Claude çift F1 0.64); "kazandı" = "Claude'un kurallarına göre okumada
kazandı". (b) 17.5'teki "örtükte kötüleşme" bulgusu bu testte tekrarlamadı — o bulgu küçük n'nin gürültüsü olabilir.
(c) Uçtan uca 0.73'te tavanı belirleyen artık konu tespiti (konu F1 0.815; satıcı çok zayıf).

**Kasa:** hata kovası yapılmadı, yorum bazlı çıktı üretilmedi. **Bu test 3 karşılaştırma harcadı (ölçüm 1-3); en fazla 1
hak kaldı** (PLAN madde 3.4: sonraki bir model, kendi planıyla, bir kez). Sonra test emekliye ayrılır.

## Yapılacaklar (2026-10-01'de güncellendi)

0. ~~Adım 16: BERT ile çok etiketli konu tespiti~~ — **yapıldı** (16.1-16.5, yukarıda).
   **Adım 17: konuya koşullu duygu modeli — TEST ÖLÇÜLDÜ** (yukarıda; ana sonuç "fark yok"); 17.5 hata kovaları
   yapıldı, yeni test artık okunmuş.
   **Adım 18: büyük temiz test** — TAMAMLANDI: ölçüm 1-3 yapıldı (ana iddia (i) +0.035 [+0.016, +0.054], 0'ı dışlıyor).
   Ana test 500'de 1 karşılaştırma hakkı kaldı. Sonraki adım için ayrı plan + onay gerekiyor. Açık kalanlar (ayrı onayla): Adım 16 ablasyonları; Görkan'ın
   `step16_topic_bert/review_sample16.csv` gözden geçirmesi.
1. ~~Yeni, hiç görülmemiş bir test seti (~200 kısa yorum) etiketle ve V2b'yi ölç.~~ — **yapıldı** (yukarıda).
2. ~~Uygulamaya 3 sınıf ekle.~~ — **yapıldı** (yukarıda).
3. ~~Modeli Hugging Face Hub'a yükle.~~ — **yapıldı** (yukarıda). Uygulamayı internete açmak ERTELENDİ (PRO gerekiyor).
4. ~~Git commit + push~~ — düzenli olarak yapılıyor. Önce `git status` ile büyük dosya (model, .venv) girmediğini kontrol et.
5. (İsteğe bağlı) Yerel `.git` 1.5 GB — geri alınan eski commit'in (01cd6b70, .venv + modeller içeriyordu) nesneleri.
   Artık gerek yoksa: `git reflog expire --expire=now --all && git gc --prune=now`.

Daha sonrası için fikirler: modeli Hugging Face Hub'a yükleyip uygulamayı yayınlamak; yeni bir NLP görevi
(yorum özetleme, "hangi özellikten şikâyet ediliyor?" gibi konu/aspect analizi).

## Devam ederken hatırlanacaklar
- `data/hard_test.csv`'yi asla eğitim/tuning'e karıştırma — o gerçek genelleme ölçütümüz, "sızdırırsak" tüm karşılaştırmalar anlamsızlaşır.
- Her adımın script'i önceki adımların sonucuyla karşılaştırma yazdırıyor (`Önceki adımlar -> ...`) — bu tutarlılığı yeni adımlarda da sürdür.
- Model indirmeleri (torch, transformers, BERT ağırlıkları) internet gerektiriyor ve büyük gelenler (~440MB BERT) arka planda (background task) çalıştırılmalı, timeout'a düşmesin.
