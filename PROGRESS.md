# gorkanai — İlerleme Notları

**Amaç:** Sıfırdan (Python + ML'e yeni) başlayarak, basit bir NLP modelinden
başlayıp adım adım daha gelişmiş yöntemlere geçerek "kendi AI'ını" inşa etmek.
Alan: Türkçe duygu analizi (sentiment analysis). Odak: öğrenmek — her adımda
gerçek bir sınırla karşılaşıp sebebini anlamak, sonra bir sonraki yöntemle çözmek.

**Son durum (2026-09-28):** Adım 1-14 tamamlandı, uygulama 3 sınıflı model V2b'yi kullanıyor, V2b hiç
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

## Yapılacaklar (2026-09-28'de güncellendi)

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
