# Adım 22 — "Kabul edilebilir etiket" ölçüsü: PLAN (2026-10-05, Görkan seçti/onayladı)

## 0. Neden
Adım 21: model–Görkan çift F1 0.636, ama ayrışma okumasında 7 "kural" ayrışmasının 3'ü Görkan'ın hatası, 4'ünde Görkan
modelin fazladan konusunu "eklenebilir" buldu. Tek-doğru-cevaplı çift F1, kabul edilebilir etiketi yanlış sayıyor.
Bu adım **eğitim yapmaz**; aynı 100 yorumu (9500-9599, artık okunmuş) Görkan'ın yargısıyla yeniden puanlar.

## 1. Görkan'ın işi (kör kaynak)
- Her yorum için üç etiketleyicinin (Görkan, taze Claude, model) bütün (konu, duygu) çiftlerinin birleşimi. Üçünün de
  verdiği çiftler (85) otomatik "kabul". Kalan **142 aday** çift için Görkan: **k** = kabul edilebilir (bu yorum için
  yazılması yanlış olmaz), **y** = yanlış (konu yok ya da duygu ters).
- **Kaynak gizli:** adayın kimden geldiği yazmaz (kendi çiftleri de dahil, karışık sıra). Görkan'ın Adım 21'de 12 yorumun
  kaynağını gördüğü bilinir (sınırlama).
- Dosya: `step22_acceptable/judge_gorkan.csv` (no, id, text, aday, karar). ~30-35 dk.

## 2. Ölçü (önceden sabit)
- **Kabul kümesi A** = otomatik kabul + Görkan'ın k dediği adaylar.
- **Zorunlu küme R** = Görkan'ın KENDİ çiftlerinden A'da kalanlar (kendi hatasını y ile eleyebilir).
- Her etiketleyici için: **kabul precision** = çiftlerinin A'daki oranı; **zorunlu recall** = R'nin bulunan oranı;
  **kabul F1** = ikisinin harmonik ortalaması.
- ANA: kabul F1, model − taze Claude, eşleştirilmiş bootstrap (2000, tohum 22). İkincil: konu başına y sayısı (modelin
  gerçek hataları nerede), y dediği çiftlerde kaynak dağılımı.

## 3. Beklenti (önceden)
- Model kabul F1 0.80-0.88 (çift F1 0.636'dan büyük sıçrama); Claude 0.82-0.90; fark −0.05 ile +0.02.
- Modelin y çiftleri en çok performans ve kalitede; "iade/performans/küçük" kalıpları görünür.
- Görkan kendi çiftlerinin birkaçını (3-6) y ile eler.

## 4. Sınırlamalar (baştan)
Tek insan; "kabul edilebilir" yargısı cömert olmaya yatkın (y demek k demekten zor); set okunmuş; n=100.
