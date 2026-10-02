# Adım 20 — Konu bazlı analizi uygulamaya koymak: PLAN (2026-10-02, ONAY BEKLİYOR; kod başlamadı)

## 0. Neden
- Adım 15-19 boyunca konu bazlı hat sadece script'lerde ve log'larda yaşadı. Uygulama (`step9_app`) hâlâ Adım 14'ün
  tek etiketli genel duygu modelini (V2b) kullanıyor: "kargo hızlıydı ama ürün kırık geldi" için tek bir cevap veriyor.
- Hedef: bir yorumu yazınca **hangi konulardan bahsedildiğini ve her konunun duygusunu** canlı görmek.
- Bu adım **yeni bir iddia değil**, mühendislik: model eğitimi yok, ayar yok, yeni test yok. Kullanılan her parça
  zaten dondurulmuş ve ölçülmüş: konu = Adım 19 v2 B (kasa testi macro 0.809), duygu = Adım 17 (kasa testi (i) 0.896).
- Öğrenme hedefi: birkaç modeli tek bir hatta bağlamak, "script'te ölçtüğüm şey ile uygulamanın yaptığı şey aynı mı?"
  sorusunu test ile cevaplamak (eğitim/sunum tutarsızlığı — Adım 9'daki turkish_lower dersinin büyük hali).

## 1. Ne değişir
- `step9_app/app.py` (yerinde güncellenir, Adım 9/14'teki gibi):
  - Mevcut `/predict` (genel duygu, V2b) **aynen kalır**.
  - Yeni **`/aspects`**: `{text}` → `{"konular": [{"konu", "olasilik", "duygu", "guven"}, ...], "genel": <predict çıktısı>}`.
    1. Konu: v2 B, 3 tohumun olasılık ortalaması, eşik 0.70 (`step19_topic_v2/frozen_config.json`'dan OKUNUR, elle yazılmaz).
    2. Her bulunan konu için duygu: Adım 17 modeli, (konu ifadesi, yorum) çifti, 3 tohum ortalaması P(pozitif) ≥ 0.5.
    3. Hiç konu bulunmazsa boş liste + genel duygu.
  - Yeni `aspect_pipeline.py`: yükleme + tahmin tek yerde. Kod `step16_topic_bert/train.py` ve
    `step17_aspect_sentiment/train.py`'deki ön işlemeyi (turkish_lower, max_len 128 / 160, konu ifadeleri PHRASE)
    **import eder, kopyalamaz** — eğitimle sunum aynı fonksiyonu kullanır.
- `index.html`: sonuç kutusunun altında konu "çipleri" (ör. `kargo 👍 %94`, `kalite 👎 %81`), genel duygu çubukları aynen.
- Bilinçli sınırlamalar arayüzde bir satırla yazılır: konu başına nötr yok (Adım 17 iki sınıflı), 8 kelimeden kısa
  yorumlar ve olumsuzlama kalıpları eğitimde azdı, satıcı recall'u düşük (~0.37).

## 2. Kaynak ve hız
- 7 BERT bellekte: V2b 1 + konu 3 + duygu 3 ≈ 3 GB RAM. Bir yorum = 1 + 3 + 3×(bulunan konu sayısı) ileri geçiş.
- **Hafif mod (isteğe bağlı, val ile karar):** konu ve duygu için tek tohum (tohum 0). Ölçülen yapılandırma 3 tohum
  olduğu için hafif mod sadece val'de kıyaslanır (konu F1, (i)); fark < 1 puan ise ortam değişkeniyle açılabilir,
  varsayılan her zaman 3 tohum. Teste dokunulmaz (kasa emekli, yeni iddia yok).
- Gecikme ölçülür: 20 örnek yorumda medyan/maks süre (CPU ve MPS).

## 3. Doğrulama (bu adımın asıl işi)
1. **Eşdeğerlik testi** (`test_pipeline.py`): uygulamanın `aspect_pipeline`'ı val 300 üzerinde çalıştırılır;
   konu olasılıkları `step19_topic_v2/val_probs_B.npz`'deki e16 ortalamasıyla, duygu olasılıkları
   `step17_aspect_sentiment/val_probs.npz` ile **atol 1e-3 ile aynı** olmalı (assert). Tutmazsa uygulama, ölçtüğümüz
   modeli çalıştırmıyor demektir → DUR.
2. **Uçtan uca duman testi**: FastAPI TestClient ile `/predict`, `/aspects`, `/health` — biçim ve boş/çok uzun girdi.
3. **Görkan'ın 10 yorumu** (ölçüm DEĞİL, sadece deneyim): kendi yazdığı yorumlarla arayüzü dener; garip çıkanlar
   PROGRESS'e "gözlem" olarak yazılır, hiçbir ayar değişmez.

## 4. Nerede çalışır
- Modeller Mac mini'de (laptopta `step*/model*/` yok). **Öneri: uygulama Mac mini'de çalışır**, laptoptan tarayıcıyla
  ev ağı üzerinden açılır (`uvicorn app:app --host 0.0.0.0`, adres `http://<mac-mini>.local:8000`).
- Alternatif: 6 model klasörü (~2.6 GB) laptopa kopyalanır. Karar Görkan'ın.
- Yayınlamak (HF Spaces) kapsam dışı: PRO gerekiyordu (Adım 14 sonrası karar aynen geçerli).

## 5. Kapsam dışı
Yeni eğitim, eşik/ayar değişikliği, nötr konu duygusu, satıcı iyileştirmesi (açık konu olarak PROGRESS'te),
yayınlama, Mac mini'deki kullanılmayan Adım 19 model klasörlerinin silinmesi (ayrı onay).

## Sıra ve duraklar
1. (onay) → `aspect_pipeline.py` + `/aspects` + `test_pipeline.py` (laptopta yazılır, push).
2. Mac mini: eşdeğerlik testi + duman testi + gecikme → **DUR** (eşdeğerlik tutmazsa burada biter).
3. (isteğe bağlı) hafif mod val kıyası.
4. `index.html` konu çipleri → Mac mini'de uygulama açılır → Görkan dener → PROGRESS.

## Görkan'a düşen iş
Onay; "nerede çalışsın" kararı (§4); sonunda ~10 dk arayüzü kendi yorumlarıyla denemek.
Tahmini süre: yarım gün (çoğu Mac mini'deki testlerin ve onayların beklemesi).
