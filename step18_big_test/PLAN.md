# Adım 18 — Büyük, temiz test seti + insan çapası: PLAN (2026-10-01, Görkan ONAYLADI; kod/etiket başlamadı)

## 0. Neden (Adım 15-17'nin ortak duvarı)
- Üç adımdır sonuç aynı: 200 yorumluk testte (~360 çift) eşleştirilmiş farkın %95 aralığı ±3.5 puan. Adım 17'nin +1.9'u,
  gerçek olsa bile bu boyutta görünmez. Ayrıca bütün altınlar Claude etiketi; insanla tek kıyas n=20.
- Bu adım **model eğitmez**. Amacı: (a) küçük farkları ayırt edebilecek kadar büyük, hiç okunmamış bir test;
  (b) etiket kurallarının bir insanla (Görkan) ne kadar uyuştuğunu n=20 yerine n=100'de ölçmek.

## 1. Güç hesabı (önceden, dürüst)
Adım 17 testinden ölçek: 363 çiftte aralık yarı genişliği ~0.035; genişlik ~1/√n ile küçülür.
| test boyutu | ~çift | aralık yarı genişliği | %80 olasılıkla yakalanan fark |
|---|---|---|---|
| 200 (şimdiki) | 360 | ±0.035 | ~+5 puan |
| **500 (öneri)** | 900 | ±0.022 | **~+3 puan** |
| 1200 | 2150 | ±0.014 | ~+2 puan |
- **Sonuç:** 500 yorum +3 puanlık farkı güvenilir yakalar; +2 puanlık fark için ~1200 gerekir. 1200 hem etiket maliyeti
  hem havuz yüzünden gerçekçi değil (aşağıda). **500 öneriyorum; +2'lik bir fark yine "ayırt edilemez" çıkabilir.**
- Havuz sınırı: hiç görülmemiş "negatif" havuz etiketli yorum **602** kaldı (pozitif ~140 bin). Bu plan 300'ünü kullanır,
  ~300 gelecek için kalır.

## 2. İki parça, bu sırayla
### 2a. Kalibrasyon seti — 100 yorum (id 7000-7099), Görkan + Claude
- Aynı tarif: 8-40 kelime, 50 pozitif + 50 negatif havuz etiketli, hiç görülmemiş; `check_no_leak`.
- **Görkan 100'ü kör etiketler** (~50 dk; 20'si ~10 dk sürmüştü; 25'lik 4 dosya, istediği zaman). Yönerge: Adım 17'deki
  `GORKAN_YONERGE.md` aynen.
- Aynı 100'ü iki Claude etiketler: **taze bir oturum** (projedeki hiçbir etiketi/hatayı görmemiş; sadece `LABEL_RULES.md`)
  ve Mac mini oturumu. Etiketler Görkan'ınkinden bağımsız, önce commit'lenir.
- Uyum raporu: konu F1, çift F1, konu başına kappa, ortak konularda duygu uyumu; Görkan–Claude, Claude–Claude.
- **Sonra Görkan'la ayrışmalar konuşulur** (özellikle Q/P ve kitap/içerik). Kurallarda **sadece netleştirme** yapılabilir
  (yeni konu yok, tanım değişikliği yok); her değişiklik `LABEL_RULES.md`'de tarihle yazılır. Bu 100 bundan sonra
  **test değildir** (okundu); ileride val/eğitim olarak kullanılabilir.
- Neden önce: test etiketleri kurallara bağlı. Kural belirsizliği varsa 500'ü etiketlemeden önce görmek ucuz.

### 2b. Ana test — 500 yorum (id 8000-8499), sadece Claude, KASA
- Aynı tarif: 250 pozitif + 250 negatif havuz etiketli, hiç görülmemiş (kalibrasyon 100'ü dahil her şeyle sızıntı kontrolü).
- **Ana altın: taze oturum** (2a'daki), donmuş kurallarla. **İkinci altın: Mac mini oturumu.** İkisi birbirini görmeden,
  önce taze oturum commit'ler. Birleştirme/uzlaştırma yok. sha256'lar `frozen_test.json`'a yazılır.
- Görkan bu 500'ü etiketlemez (maliyet); insan çapası 2a'dan gelir.
- **KASA KURALI:** bu testin hataları **tek tek okunmaz**, hata kovası yapılmaz, satır bazlı çıktı log'a yazılmaz.
  Sadece toplu sayılar ve bootstrap aralıkları. Hata analizi gerekirse val üzerinde yapılır.
  (Adım 15 ve 17 testleri, hataları okunduğu için birer adımda tükendi; bu sefer test birden fazla ölçüme dayanmalı.)

## 3. Bu test hangi ölçümlere harcanacak (önceden sabit liste)
Her ölçüm bir kez, dondurulmuş model, önceden yazılmış beklentiyle. Listede olmayan ölçüm yeni onay ister.
1. **Adım 17 duygu modeli vs Adım 15 duygu hattı** (zaten donmuş; hemen ölçülebilir). Ana iddia: (i) altın konularla
   duygu doğruluğu micro, ana altınla. Bu, "Adım 17'nin +2'si gerçek mi" sorusunu 500'lük testte yanıtlar.
2. **Adım 16 BERT konu modeli vs anahtar kelime** (donmuş): konu F1 micro + konu başına.
3. **Uçtan uca** (konu + duygu) çift F1: Adım 16 + 17 hattı vs Adım 15 hattı.
4. Sonraki modeller (17.5'teki fikirler: hedefli "X güzel ama Y" örnekleri, örtükte eski hat birleşimi) —
   her biri kendi planında, bu teste karşı bir kez. **Toplam en fazla 4 karşılaştırma**; sonra test emekliye ayrılır.
- Ölçüm 1-3 tek seferde, 2b etiketleri commit'lendikten hemen sonra yapılır (model değişmediği için bekleme gereksiz).

## 4. Beklenti (önceden, sayısal)
- Kalibrasyon: Claude–Claude çift F1 ~0.88-0.92 (Adım 17: 0.903); **Görkan–Claude çift F1 ~0.60-0.70** (n=20'de
  0.57-0.62); duygu uyumu ortak konularda ~0.85. Q/P kappası Görkan'la düşük (<0.4).
- Ölçüm 1: fark +1 ile +3 puan; aralık ±0.022 → **0'ı dışlama olasılığı yaklaşık yarı yarıya**.
- Ölçüm 2: BERT − kelime +0.06 ile +0.09, aralık 0'ı dışlar.
- Ölçüm 3: +1 ile +2.5 puan, 0'ı dışlamayabilir.

## 5. Kapsam dışı
Yeni model eğitimi, eşik/ayar değişikliği, eski testlerin yeniden etiketlenmesi, nötr sınıfını açmak.

## Sıra ve duraklar
1. (onay) → kalibrasyon 100 + ana test 500 seçimi, sızıntı assert'leri → commit. (2b'nin metinleri seçilir ama
   etiketlenmez.)
2. Kalibrasyon: Claude etiketleri (taze oturum, Mac mini) → commit → Görkan'ın 100'ü → uyum raporu → **DUR**.
3. (Görkan'la ayrışmaları konuşma; kural netleştirmeleri, onay) → `LABEL_RULES.md` dondurulur.
4. Ana test 500: taze oturum → Mac mini → sha256 dondurma → **DUR**.
5. (onay) → ölçüm 1-3, bir kez → rapor, PROGRESS. Hata kovası YOK (kasa kuralı).

## Görkan'a düşen iş
~100 yorum kör etiketleme (~50 dk, 4 parça) ve 2. adımdan sonra ~20 dk ayrışma konuşması. Taze bir Claude oturumu
açmak (Mac mini'de ya da laptopta, projedeki etiketlere bakmaması söylenerek).
