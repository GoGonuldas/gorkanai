# gorkanai — tek sayfada yolculuk (Adım 1-23, 2026-09 / 2026-10)

**Hedef:** Python ve makine öğrenmesine yeni başlayan biri olarak, sıfırdan adım adım "kendi yapay zekanı" kurmak.
Alan: Türkçe ürün yorumlarında duygu analizi. Odak: öğrenmek — her adımda bir sınıra çarpıp nedenini anlamak.
Ayrıntılar, bütün sayılar ve dersler: [PROGRESS.md](../PROGRESS.md).

## Sonuç: elimizde ne var?
Bir yorumu yazınca **genel duyguyu** (pozitif/nötr/negatif) ve **hangi konulardan bahsedildiğini + her konunun duygusunu**
("kargo 👍, satıcı 👎") veren, Mac mini'de çalışan bir web uygulaması (`step9_app`). Arkasında üç ince ayarlı Türkçe BERT var:
genel duygu (Adım 14 V2b), konu tespiti (Adım 19), konuya koşullu duygu (Adım 17).

| Ölçü | Sonuç |
|---|---|
| Genel duygu (uygulamadaki V2b, 3 sınıf), hiç görülmemiş 200 yorum | doğruluk 0.870, macro-F1 0.808 (Adım 14) |
| Konu tespiti, 500 yorumluk kasa testi | F1 micro 0.852, macro 0.809 (Adım 19) |
| Konuya koşullu duygu, kasa testi | doğruluk 0.896 (Adım 18) |
| Uçtan uca (konu + duygu), kasa testi | çift F1 0.763 (Adım 19) |
| Bir insana (Görkan) karşı, "kabul edilebilir" ölçü | F1 0.823 — Claude etiketleyiciden (0.857) ayırt edilemiyor (Adım 22) |
| İnsanla ortak konuda duygu uyumu | %94 (Adım 21) |

## Dört dönem
1. **Temeller (Adım 1-7, sentetik veri):** Bag-of-Words → TF-IDF → sinir ağı → Word2Vec → LSTM → attention → BERT.
   Mimariyi büyütmek küçük veride ~0.69'da duvara çarptı; sıçrama (0.88) önceden eğitilmiş modeli (BERT) uyarlamakla geldi.
2. **Gerçek dünya (Adım 8-14):** gerçek yorumlar, uygulama, olumsuzlama ("hiç fena değil"), güven/kalibrasyon, hatalı
   etiketleri bulma, nötr sınıfı. Ders: hataların çoğu modelde değil VERİDE (örnekleme çarpıklığı, etiket gürültüsü, kısayollar).
3. **Konu bazlı analiz (Adım 15-20):** anahtar kelime → öğrenilmiş konu modeli → konuya koşullu duygu → nadir konular
   (satıcı 0.17 → 0.51) → uygulamaya koymak. Ders: küçük testler küçük farkları göremez; 500'lük "kasa testi" ve önceden
   yazılmış beklentiler, Adım 17'nin "fark yok"unun aslında güç eksikliği olduğunu gösterdi.
4. **İnsana karşı (Adım 21-23):** ilk insan testi, "kabul edilebilir etiket" ölçüsü, kalite (Q) kurallarının
   Görkan'la netleştirilmesi (`LABEL_RULES_v2.md`). Ders: insan etiketi de kusursuz değil; ölçü, sorulan soruyu belirler.

## En önemli 8 ders
1. **Transfer learning:** kendi küçük verinle sıfırdan eğitmek yerine önceden eğitilmiş modeli uyarlamak (Adım 7).
2. **Eğitimde ne yaptıysan kullanırken de onu yap** (`turkish_lower`, Adım 9) — Adım 20'de ön işlemeyi kopyalamak
   yerine import ederek ve eşdeğerlik testiyle (fark tam 0) kalıcı çözüldü.
3. **Önce veriye bak:** "model hatası" sanılanların çoğu örnekleme ya da etiket sorunuydu (Adım 10, 12, 14, 16.5).
4. **Kısayol öğrenme:** Vikipedi nötrleri biçimden tanındı, gerçek nötrde 0.11 (Adım 14).
5. **Test kutsaldır:** test bir kez ölçülür; hataları okunursa tükenir; kasa testi 4 önceden yazılmış karşılaştırmaya
   dayandı ve sonra emekli oldu (Adım 18-19).
6. **Güven aralığı olmadan fark iddia edilmez;** beklentiyi ölçümden ÖNCE yaz (Adım 16-22).
7. **Val'in etiketleyicisi testinkinden farklıysa val her iki yöne de yanıltır** (Adım 16: iyimser, Adım 19: kötümser).
8. **Görev tanımı modelden önemli olabilir:** kalite/performans (Q/P) sınırı Adım 15'ten 22'ye her insan kıyasında en
   büyük ayrışmaydı; Adım 23'te kural Görkan'la yeniden yazıldı.

## Bilerek bırakılanlar
Satıcı recall'u (~0.37), görünüm; konu başına nötr; kelimeye takılma hataları ("iade", "performans", "küçük");
v2 kurallarıyla yeniden etiketleme + yeniden eğitim; uygulamayı internette yayınlamak (HF Spaces PRO istiyor).
Görülmemiş negatif yorum havuzu ~220 — yeni bir test için sınırlı.

## Sonraki yolculuk için fikirler
Kendi kod ajanını yazmak (Ollama ile yerel ücretsiz model ya da Claude API); küçük bir dil modelini sıfırdan eğitmek;
metin üretme görevleri (yorum özetleme).
