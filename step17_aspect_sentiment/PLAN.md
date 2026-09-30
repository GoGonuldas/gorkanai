# Adım 17 — Konuya koşullu duygu modeli: PLAN (2026-09-30, ONAY BEKLİYOR; kod/eğitim başlamadı)

## 0. Önce dürüst risk
Kazanılabilecek pay küçük. Adım 16.5'e göre BERT'in doğru bulduğu 284 konuda duygu hatası 40 (%14.1): 22 V2b
hatası, 13 bölme hatası, 5 yedek kural; bunların 11'i altın nötr (nötr kapalıyken yakalanamaz), 4'ü karışık yorum.
Hedeflenebilir hata ~25-30 çift. Yarısını düzeltsek uçtan uca F1 ~+3 puan; n=200'de Adım 16'nın +4.4 puanlık
farkının aralığı [+0.004, +0.085] idi. **Yani gerçek bir iyileşme bile testte gürültüden ayrılamayabilir.**
Sonuç "fark yok" çıkarsa bu da kayda girer.

## 1. Görev
- Girdi: (konu, yorum) çifti, BERT cümle çifti olarak: `[CLS] <konu ifadesi> [SEP] <yorum> [SEP]`.
  Konu ifadeleri (sabit): kargo ve teslimat / fiyat / kalite / performans ve özellikler / boyut / görünüm /
  satıcı ve hizmet. Yorum önceki adımlardaki gibi Türkçe küçük harfe çevrilir, max 160 token.
- Çıktı: **2 sınıf (pozitif / negatif).** Nötr KAPALI. Gerekçe (val görülmeden): eğitimde 1050 çiftin 30'u nötr
  (%3; performans ve satıcıda 0) — öğrenilemez; Adım 15'te nötrü açmak hiç kazandırmamıştı. Altın nötr çiftler
  eğitime girmez; değerlendirmede hata sayılır (Adım 15 ile aynı muhasebe) ve ayrıca "nötr hariç" satırı verilir.
- Karışık çiftler (eğitimde 50) baskın duygularıyla eğitime girer (altın etiket bu); karisik=0 ayrı raporlanır.

## 2. Veri (yeni eğitim etiketi GEREKMİYOR)
| | yorum | (konu, duygu) çifti | poz / neg / nötr | karışık | zıt duygulu (poz+neg) yorum |
|---|---|---|---|---|---|
| eğitim (Adım 16 etiketleri) | 600 | 1050 | 596 / 424 / 30 | 50 | 87 |
| val (eski 100 + yeni 200) | 300 | 513 | 295 / 208 / 10 | 19 | 51 |
| eski test (Adım 15) | 200 | 359 | 207 / 141 / 11 | 21 | 25 |

Eğitimde konu başına poz/neg: kargo 74/28, fiyat 94/23, kalite 219/188, performans 103/111, boyut 30/40,
görünüm 54/12, satıcı 22/22. Eğitim/val bölmesi Adım 16'daki ile aynı (sızıntı assert'leri aynen).
- **Başlangıç ağırlığı adayları (önceden sabit, val seçer):** (i) `dbmdz/bert-base-turkish-cased` (düz);
  (ii) V2b (`step14_three_class/model_v2b`, duygu için eğitilmiş; 3 sınıflı başlığı korunur, karar poz/neg
  logit'leri arasında). V2b'nin eğitim verisi havuzun "görülmüş" kısmından; Adım 15/16'nın tüm setleri görülmemiş
  havuzdan seçildiği için çakışma yok (assert ile yeniden doğrulanacak).

## 3. Test sorunu — öneri: (c) yeni test ANA SONUÇ + eski test kıyas satırı
Eski testin 200 yorumu 16.5'te tek tek okundu ve 132 hatası elle sınıflandı; mimari kararları (bu plan dahil) o
okumadan etkilendi. Eski testte tek başına ölçüm iyimser olur.
- **Yeni test: 200 yorum** (id 6000-6199), aynı tarif (8-40 kelime, 100 "pozitif" + 100 "negatif" havuz etiketli,
  hiç görülmemiş; negatif havuzda 702 kaldı). Eğitim/val ile ve eski her şeyle `check_no_leak`.
- **Kim etiketler — etiketleyici kayması dersi:** eğitim etiketleri bu oturumdan. Yeni testi de bu oturum etiketlerse
  test "eğitimle aynı kafadan" olur (Adım 16'da yeni val +11, eski val +3 puan göstermişti). Öneri:
  **ana altın = gorkanai-1e'nin kör etiketleri** (sadece docstring kuralları + 16.5'teki iki netleştirme; bu
  oturumun etiketlerini görmeden). Bu oturum da aynı 200'ü kör etiketler → (1) iki oturum arası uyum (çift F1,
  Q/P kappa) bedavaya ölçülür, (2) "bu oturumun altınıyla" sonuç ikinci satır olur ve iyimserlik payı görünür.
  İki altın birleştirilmez/uzlaştırılmaz.
- **Görkan (isteğe bağlı, ~10 dk):** yeni testten 20 yorumu kör etiketlemesi. Olursa "model-Görkan" satırı
  (n=20, sadece fikir verir); olmazsa plan aynen yürür.
- Yeni test, Adım 16 konu modelinin de ikinci ve temiz ölçümü olur (dondurulmuş haliyle, ayar yok).
- Eski test: tek ölçüm, "hatalar okundu → iyimser olabilir" notuyla kıyas satırı. Birleşik 400 yorumluk satır
  (daha dar aralık için) ikincil sonuç olarak verilir.

## 4. Ölçüler ve kıyas (önceden sabit)
Kıyas her yerde **Adım 15 duygu hattı** (bölme A + V2b, nötr kapalı, dondurulmuş).
- (i) **Altın konular verilmişken duygu doğruluğu**: micro / macro (konu başına), karisik=0 ayrı, nötr hariç ayrı.
- (ii) **Uçtan uca (konu, duygu) çift F1** micro / macro: Adım 16 BERT konuları (dondurulmuş, eşik 0.60) + yeni duygu.
  Eski testte kıyas 0.697 / 0.636.
- (iii) **Eşleştirilmiş bootstrap** (yorum bazında, 2000 tekrar, tohum 17): yeni − eski duygu hattı farkı, (i) micro
  ve (ii) micro için %95 aralık. **"Kazandı" = aralık 0'ı dışlıyor**; içeriyorsa "gürültüden ayrılamıyor".
- (iv) **Hedef alt kümeler**, (i) ölçüsüyle ayrı: (a) çok konulu + zıt duygulu yorumlar (bölme hatasının hedefi;
  eski testte 25 yorum), (b) tek duygulu yorumlar (kontrol: burada kötüleşmemeli), (c) konunun kelimesi hiçbir
  cümlecikte geçmeyen (örtük) çiftler.
- Ana iddia: (i) micro, yeni testte, gorkanai-1e altınıyla.

## 5. Eğitim ayarları (önceden sabit)
- 3 tohum (0, 1, 2); teste 3 tohumun olasılık ortalaması girer.
- lr 3e-5 sabit, batch 16, AdamW wd 0.01, sınıf ağırlığı yok (poz/neg 596/424, dengeye yakın).
- Epoch ızgarası **2 / 4 / 6 / 8 / 12 / 16** (aynı çalıştırmanın ara kayıtları). Adım 16 dersi: 1050 çift ≈ 66
  adım/epoch; 16'daki 12 epoch ≈ 450 adım buradaki ~7 epoch'a denk, yani ızgara iki yana da taşıyor.
- Seçim: (başlangıç × epoch) içinden val'de (i) micro doğruluğu (3 tohum ortalaması) en yüksek olan; eşitlikte
  (0.005 içinde) daha az epoch ve düz BERT. Seçilen epoch üst sınırdaysa not düşülür, ızgara genişletilmez.
- Val raporu: tüm ızgara (ort. ± min-max), eski-100 / yeni-200 ayrı, (iv) alt kümeleri, val bootstrap.
- Dondurma: `frozen_config.json` (başlangıç, epoch, tohumlar, model hash'leri). **Testlerden ÖNCE durulur**, onay beklenir.
  Yeni test etiketleri model eğitilmeden önce toplanıp commit'lenir (etiketleyenler model çıktısı görmez).

## 6. Beklenti (test öncesi, sayısal)
- Kıyas hattının altın konularla doğruluğu ~0.84-0.86 (Adım 15.4'te bulduğu konularda 0.856).
- Yeni model (i) micro: **0.86-0.90** (beklenen kazanç +2 ile +4 puan); zıt duygulu alt kümede daha büyük (+5-10)
  ama n çok küçük. Uçtan uca micro eski testte 0.697 → **0.70-0.73**.
- Tahmin: (i) için aralık büyük olasılıkla 0'ı dışlar (çiftler üzerinde n≈350), (ii) için dışlamayabilir.
- Tavan: nötr kapalıyken (i) en fazla ~0.97.

## 7. Kapsam dışı
Adım 16'nın açık ablasyonları (daha uzun eğitim, hedefli/kararsız yarıyı çıkarma), BERT+anahtar kelime birleşimi,
eşik değişikliği, Q/P birleştirme: bu adıma karışmaz. Konu modeli dondurulmuş haliyle kullanılır.

## Sıra
1. (onay) → yeni test 200 seçimi + sızıntı assert'leri → commit.
2. Kör etiketleme: gorkanai-1e (ana altın) ve bu oturum (ikinci) → uyum raporu → commit.
3. Eğitim (2 başlangıç × 3 tohum) → val raporu → dondur → DUR.
4. (onay) → yeni test + eski test, bir kez → rapor, PROGRESS, hata kovaları ayrı onayla.
