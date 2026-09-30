# Adım 16 — önceden sabitlenen kurallar (2026-09-30, val/test sayıları görülmeden yazıldı)

Görev: BERT ile çok etiketli konu tespiti (7 konu). Plan Görkan tarafından onaylandı (800 yeni yorum);
aşağıdaki ekler gorkanai-1e (değerlendirme oturumu) önerisiyle eklendi.

## Veri
- 800 yeni yorum, id 5000-5799: val eki 200 (rastgele) + eğitim tur 1 300 (150 rastgele + 150 anahtar kelime
  hedefli) + eğitim tur 2 300 (150 rastgele + 150 kararsız). Val = eski 100 + yeni 200 = 300. Test = Adım 15'in 200'ü.
- **Kör etiketleme:** etiketleme dosyaları (`batch_XX.csv`) sadece id + text içerir, sıra karışık. Kaynak
  (rastgele/hedefli/kararsız), bölme ve havuz etiketi sadece `label_set16.csv`'de; etiketleme bitene kadar okunmaz.
- **Tur 2 "kararsız" tanımı:** tur 1 modeli (tur 1'in 300 eğitim yorumuyla, tohum 0) aday havuzunun rastgele
  20000'lik alt kümesinde çalıştırılır; her yorum için `u = min_k |p_k - 0.5|` (7 konu üzerinden) hesaplanır;
  `u` en küçük olanlardan yarı "pozitif" yarı "negatif" havuz etiketli 150 yorum alınır. Tur 2'nin rastgele
  150'si aynı alt kümenin kalanından. Tur 1 ile ve val/test ile metin çakışması assert ile yasak.
- Sızıntı assert'leri: `common.check_no_leak` (ham ve normalleştirilmiş metin).

## Model ve ayar
- Türkçe BERT (önceki adımlarla aynı taban), 7 sigmoid çıkış, BCE.
- **Tohum:** 3 tohum (0, 1, 2). Val sonuçları ortalama ± yayılım (min-max) olarak raporlanır.
  **Teste giren model: 3 tohumun olasılık ortalaması** (tek tohum seçimi yok).
- **Eşik — ANA SONUÇ: tek global eşik**, val micro-F1'i en yüksek yapan (ızgara 0.05-0.95, adım 0.05),
  3 tohum ortalaması olasılıklar üzerinde. Konu başına eşik sadece ikinci satır, "iyimser olabilir" notuyla
  (val'de satıcı/görünüm/boyut ~20-35 örnek).
- Epoch sayısı val micro-F1 ile (global eşik 0.5'te) seçilir; aday: 3, 5, 8.

## Raporlanacaklar (val'de, sonra testte bir kez)
- Konu F1 micro/macro, konu başına P/R; kıyas = Adım 15 V2 anahtar kelime listesi (dondurulmuş), aynı kümede.
- **Örtük konu ölçüsü:** her altın (yorum, konu) çifti "V2 anahtar kelimesi tutuyor / tutmuyor" diye ayrılır;
  recall iki alt kümede ayrı raporlanır (anahtar kelime çizgisi "tutmuyor"da tanım gereği 0). BERT'in
  "tutmuyor" alt kümesindeki recall'u Adım 16'nın asıl sonucu.
- Eski val 100 ve yeni val 200 bir kez ayrı ayrı (iki etiketleyici oturumu arasında fark var mı).
- Tur 1 / tur 2'de hedefli-kararsız yarının konu dağılımı, rastgele yarıyla kıyaslanır.
- Uçtan uca: BERT konuları + Adım 15'in dondurulmuş duygu hattı (bölme A, nötr kapalı).

## Test
Ölçümden ÖNCE durulur: dondurulan ayarlar (epoch, eşik, tohumlar) gorkanai-1e'ye bildirilir, onay beklenir.
Test bir kez ölçülür. Kıyas: konu F1 0.763/0.722, uçtan uca 0.653/0.616.

## Etiketleyici kayması kontrolü (etiketlemeden önce yapıldı, `log_drift_check.txt`)
Eski val'den 30 yorum altın etiketlere bakılmadan yeniden etiketlendi: konu F1 0.903, çift F1 0.867
(Claude-Görkan 0.667); kappa Q 0.73, P 0.80. Fark yönü: bu oturum daha ÇOK etiket veriyor (61'e 52) —
özellikle belirli özellik övgüsünün yanına ek Q ve bilgi amaçlı bahsedilen konular (iade, indirim).
Kalibrasyon kararı: Q sadece açık bir GENEL yargı varsa; sadece bilgi olarak geçen konu yazılmaz/az yazılır.

## Ekler (2026-09-30, tur 1 etiketlemesi sırasında; yine val/test model sayıları görülmeden)
- **İkinci kayma kontrolü** (`log_drift_check_b.txt`, kalibrasyon kararından sonra, eski val'den başka 30 yorum):
  konu F1 0.916, çift F1 0.897; etiket sayısı 53 yeni / 54 altın (oran 0.98; ilk turda 61/52 = 1.17). Q kappa 1.00.
- **Tanı satırı (val raporu):** global eşik eski-100'de ve yeni-200'de ayrı ayrı seçilseydi hangi değer çıkardı ve
  F1 ne olurdu. Ana kural değişmez (val 300, tek global eşik); ikisi belirgin ayrışırsa testten önce konuşulur.
- **"Altın şüpheli" listesi** (val altınına dokunulmadı; test sonrası hata analizinde bakılacak):
  4089 ("8 inç tabletlere olmuyor": altın Bn, bence Pn/uyumluluk), 4196 ("sessiz ama tozları çekmiyor": altın Pp),
  4200 ("incelik olarak iyi": altın Gp, bence Bp), 4177 ("çok ince ... daha kalın matları tercih edin": altında B yok).
- Altından öğrenilen ve tur 1'de uygulanan ince kurallar: aksesuar/çanta/kılavuz eksikliği -> S (altın 4190);
  sade "X günde geldi / teslim aldım" -> K nötr, "ertesi gün / çok hızlı" -> K pozitif.

## Ekler (2026-09-30, tur 2 sırasında; val/test model sayıları hâlâ görülmedi)
- **train.py sabitleri (kayıt için):** lr 3e-5 sabit (ısınma/azalma yok), batch 16, max_len 128, AdamW wd 0.01,
  pos_weight = sqrt(neg/pos), epoch adayları aynı çalıştırmanın ara kayıtları.
- **Epoch adaylarına 12 eklendi (3/5/8/12).** Gerekçe SADECE eğitim kaybı: tur 2 kaba modeli (300 yorum) 5 epoch'ta
  0.74 -> 0.57 ile neredeyse hiç öğrenmemişti, 15 epoch'ta 0.107'ye indi. 600 yorumla 8 epoch ≈ 300 adım (kaba modelin
  15. epoch'una denk); 3 ve 5 epoch büyük olasılıkla az eğitilmiş kalır, 12 üst tarafı da görmek için.
- **Tur 2 ilk denemesi atıldı** (`log_prepare_round2_run1_undertrained.txt`): 5 epoch'luk kaba modelde tüm olasılıklar
  0.5 civarındaydı (medyan u 0.033) -> "kararsız" seçimi anlamsız. Hiçbir yorum etiketlenmeden 15 epoch ile yeniden yapıldı
  (medyan u 0.12; `log_prepare_round2.txt`).
- **P kuralı (tur 2'de açıkça uygulandı, gorkanai-1e notu):** adı konan özellik (ses, koku/kalıcılık, hız, güç,
  ekran/kamera, uyumluluk, kurulum/kullanım kolaylığı, tat) arıza/şikâyet biçiminde geçse de P; genel
  "çalışmıyor/işe yaramadı" Q; ikisi birlikte olabilir. Tur 1 etiketleri yeniden etiketlenmedi ->
  SINIRLAMA: tur 1'de (val eki 200 dahil) P altına göre hafif eksik, Q hafif fazla olabilir.
