# Adım 19 — Konu tespiti v2 (nadir konular, özellikle satıcı): PLAN (2026-10-02, Görkan ONAYLADI)

## 0. Neden
- Adım 18 kasa testinde uçtan uca 0.730; tavanı artık konu tespiti belirliyor (konu F1 micro 0.815).
- **Satıcı (S) F1 0.43 → 0.17** (BERT anahtar kelimeden kötü, n=38). Boyut 0.77, görünüm 0.77 de zayıf.
- Sorun tanım değil, veri: iki Claude altını satıcıda kappa **0.94** veriyor (kural net). Ama eğitimde S sadece **44**,
  G 68, B 72 örnek (Q 411). Adım 16.5: model nadir konuyu çok konulu yorumda atlıyor; S'yi nadiren söylüyor
  (testte 8 tahmin), söylediğinde çoğunlukla doğru → sorun recall.
- Ek şüphe: Adım 16 eğitim etiketleri, kasa testinin altınını veren "taze" oturumdan farklı bir oturumdan
  (etiketleyici kayması Adım 16'da ölçülmüştü). Bu planda yeni etiketler LABEL_RULES.md'ye bağlı tek bir oturumdan.

## 1. Veri (eğitim 600 → ~1400)
a) **Okunmuş, artık test olmayan setler eğitime girer** (yeni etiket gerektirmez): Adım 15/16 testi 200, Adım 17
   testi 200 (17.5'te okundu), Adım 18 kalibrasyon 100 (PLAN 18 §2a: "ileride val/eğitim"). +500 yorum.
   Bunlar val'e GİRMEZ (val sabit kalır, kıyas bozulmasın).
b) **Konu başına kotalı hedefli etiketleme, 300 yeni yorum (id 9000-9299):**
   - S 120 (anahtar kelime hedefli: satıcı, mağaza, iade, garanti, servis, eksik, kutusundan, yanlış ürün, kılavuz ...)
   - G 60, B 60 (anahtar kelime hedefli)
   - **Örtük S 30:** kelime tutmayan ama Adım 16 modelinin S olasılığı 0.15-0.60 olan yorumlar
     (konu başına kotalı kararsızlık — Adım 16 dersi: global kararsızlık nadir konuyu getirmiyor).
   - 30 rastgele (dağılım kontrolü).
   - Havuz: hedefli seçim pozitif havuzdan (~140 bin); negatif havuzdan en fazla 60 (kalan 302 → ≥242 gelecek için).
     Her grupta %20 negatif (S 24, G 12, B 12, rastgele 6, örtük S 6 = 60).
   - **İki parça:** 270 (S/G/B/rastgele, V2 anahtar kelime listesiyle) laptopta `prepare_select.py`; örtük S 30
     Mac mini'de `prepare_implicit_s.py` (Adım 16 modelleri orada), 270'i de dışlayarak.
   - Sızıntı: `check_no_leak` + Adım 16/17/18 tüm setleriyle çakışma assert'i (özellikle test18 500).
c) **Kör etiketleme**, batch dosyalarında sadece id + text, karışık sıra. Etiketleyici: **taze oturum**
   (gorkanai-fresh, sadece LABEL_RULES.md; kasa testinin ana altınını veren oturum → eğitim/test etiket uyumu).
   Başlamadan önce 30'luk kayma kontrolü: calib 100'den 30 yorum, altına bakmadan → çift F1 ≥ 0.85 beklenir.

## 2. Model ve ayar (Adım 16 tarifi AYNEN, tek değişken veri)
- `dbmdz/bert-base-turkish-cased`, 7 sigmoid, BCE, pos_weight = sqrt(neg/pos), lr 3e-5, batch 16, max_len 128,
  3 tohum (0,1,2), olasılık ortalaması. Epoch adayları 5/8/12/16 (veri 2.3 kat; 12 üst sınırdı).
- Tek global eşik, val micro-F1 ile (ızgara 0.05-0.95). Konu başına eşik sadece "iyimser" satır.
- **Ablasyon (val'de, karar için):** A = sadece 1a (eski veri + okunmuş setler), B = 1a + 1b (tam).
  Böylece kazancın "daha çok veri"den mi "hedefli nadir konu"dan mı geldiği görülür.
- **Aday ikinci model (val'de karar):** BERT VEYA anahtar kelime, sadece S/G/B için (Adım 16.5 fikri (a)).
  Val'de macro'yu artırıp precision'ı < 0.70'e düşürmüyorsa teste o girer; karar teste bakmadan yazılır.

## 3. Val
- Val 300 (Adım 16) aynen. S 15, G 24, B 28 → nadir konu sayıları gürültülü; konu başına sonuçlar "fikir verir"
  notuyla. Ana seçim ölçütü val micro-F1; macro ve S recall'u raporlanır.
- Hata analizi gerekirse SADECE val'de (kasa kuralı).

## 4. Test — kasa 500'ün SON hakkı (PLAN 18 §3 madde 4), bir kez
- Dondurulan model vs Adım 16 BERT (donmuş, test18_probs.npz'deki olasılıklar), ana altın (fresh), eşleştirilmiş
  bootstrap 2000, tohum 16.
- **ANA İDDİA: konu F1 macro** (nadir konular hedef olduğu için micro değil). Kazandı = aralık 0'ı dışlar.
- İkincil: S F1 (n=38), G, B F1; konu F1 micro (**koruma: micro düşmemeli**, aralığın alt sınırı > −0.01);
  uçtan uca micro (Adım 17 duygu modeli sabit).
- **Beklenti (önceden):** macro 0.741 → 0.77-0.80 (+3 ile +6), 0'ı dışlama olasılığı ~%60; S F1 0.17 → 0.40-0.60;
  micro 0.815 → 0.82-0.83; uçtan uca +0.5 ile +1.5 (dışlamayabilir).
- Sonra test **emekliye ayrılır**; Adım 20+ için yeni test gerekir (negatif havuz ~240 kaldı — bunu bilerek).
- Kasa kuralı aynen: satır bazlı çıktı yok, hata kovası yok.

## 5. Kapsam dışı
Duygu modeli değişikliği, kural/konu tanımı değişikliği, nötr sınıfı, eski altınların yeniden etiketlenmesi.

## Sıra ve duraklar
1. (onay) → 300 yorum seçimi + sızıntı assert'leri → commit.
2. Taze oturum: 30 kayma kontrolü → **DUR** (≥0.85 değilse konuşulur) → 300 etiket → commit.
3. Eğitim A ve B (+ VEYA adayı), 3 tohum, Mac mini → val raporu → dondurma (`frozen_config.json`) → **DUR**.
4. (onay) → kasa testi bir kez → rapor, PROGRESS.

## Oturum iş bölümü (öneri)
- **gorkanai-87 (laptop):** plan, seçim script'i, val/test değerlendirme, PROGRESS.
- **piped-oasis (gorkanai-fresh, Mac mini):** kayma kontrolü + 300 kör etiket. Repo'yu GÜNCELLEMEZ (kasa altınını
  vermiş oturum; yeni batch'ler ona dosya olarak verilir).
- **glowing-teapot (gorkanai, Mac mini):** eğitim (modeller orada; laptopta model klasörü yok).

## Görkan'a düşen iş
Onay (madde 1 ve 4'ten önce). İsteğe bağlı: `step16_topic_bert/review_sample16.csv` (30 yorum) — eğitim etiketleri
için tek insan kontrolü.

## Ekler (2026-10-02, seçim sırasında; hiçbir etiket/model sonucu görülmeden)
- **Sapma:** satıcı kelimeli negatif havuzda sadece 16 yorum var → hedefli S grubunda 24 yerine 16 negatif,
  eksik pozitifle tamamlandı. Parça 1 negatif 46; kalan negatif havuz 256 (parça 2 en fazla 6 daha alır).
- Parça 1: `prepare_select.py` → `select19.csv` (id 9000-9269), `batch_01..05.csv` (54'er). Parça 2 (Mac mini):
  `prepare_implicit_s.py` → select19.csv'ye eklenir, `batch_06.csv` (30).
