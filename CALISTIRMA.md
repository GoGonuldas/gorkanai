# gorkanai — Nasıl çalıştırılır

Bu doküman uygulamayı (genel duygu + konu bazlı analiz) çalıştırmayı anlatır: hazır bir makinede, sıfırdan kurulmuş
bir makinede ve modeller eksikse yeniden üreterek. Projenin hikâyesi için [step23_wrapup/OZET.md](step23_wrapup/OZET.md),
ayrıntılar için [PROGRESS.md](PROGRESS.md).

## İçindekiler
1. [Hızlı başlangıç (Mac mini — her şey hazır)](#1-hızlı-başlangıç-mac-mini--her-şey-hazır)
2. [Uygulamayı kullanmak](#2-uygulamayı-kullanmak)
3. [Yeni bir makinede kurulum](#3-yeni-bir-makinede-kurulum)
4. [Modelleri edinmek](#4-modelleri-edinmek)
5. [Kurulumu doğrulamak](#5-kurulumu-doğrulamak)
6. [Sorun giderme](#6-sorun-giderme)

---

## 1. Hızlı başlangıç (Mac mini — her şey hazır)
Mac mini'de (`~/workspace/gorkanai`) sanal ortam ve uygulamanın 7 modeli hazır durumda.

```bash
cd ~/workspace/gorkanai
git pull
cd step9_app
../.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
```

- İlk açılış **~30 saniye** sürer (7 BERT modeli belleğe yüklenir, ~3 GB RAM).
- `Application startup complete` yazınca hazırdır.
- Mac mini'de: http://127.0.0.1:8000 · Aynı Wi-Fi'deki laptoptan: **http://gorkans-mac-mini.local:8000**
  (ya da Mac mini'nin IP'siyle, ör. `http://192.168.5.11:8000` — IP değişebilir: `ipconfig getifaddr en1`).
- Kapatmak: terminalde `Ctrl+C`.

> `--host 0.0.0.0` uygulamayı **ev ağına açar** (aynı ağdaki her cihaz erişebilir). Sadece Mac mini'den kullanacaksan
> `--host 127.0.0.1` yaz.

## 2. Uygulamayı kullanmak

### Web arayüzü
Kutuya bir yorum yaz → **Analiz et**. Üstte genel duygu (pozitif / nötr / negatif ve olasılık çubukları), altta
**konu çipleri**: bulunan her konu ve o konunun duygusu, ör. `kargo 👍 %94`, `satıcı 👎 %99`.

7 konu: kargo, fiyat, kalite, performans, boyut, görünüm, satıcı.

### API
| Uç nokta | Ne yapar |
|---|---|
| `POST /predict` | Genel duygu (3 sınıf) |
| `POST /aspects` | Konular + her konunun duygusu + genel duygu |
| `GET /health` | `{"status": "ok"}` |
| `GET /docs` | Otomatik API dokümantasyonu (tarayıcıda dene) |

```bash
curl -s -X POST http://127.0.0.1:8000/aspects \
  -H "Content-Type: application/json" \
  -d '{"text": "Kargo çok hızlıydı ama ürün kırık geldi, satıcı da cevap vermedi."}'
```
Yanıt (kısaltılmış):
```json
{"konular": [{"konu": "satıcı", "olasilik": 0.9963, "duygu": "negatif", "guven": 0.9979},
             {"konu": "kargo",  "olasilik": 0.9951, "duygu": "negatif", "guven": 0.8556}],
 "genel": {"label": "negatif", "confidence": 0.98, "probabilities": {...}},
 "hafif_mod": false}
```
- `olasilik`: konunun yorumda geçme olasılığı (≥ 0.70 olanlar listelenir).
- `guven`: o konunun duygusundan ne kadar emin olduğu.
- Metin 1-2000 karakter olmalı; boş ya da daha uzunsa `422` döner.

### Bilinen sınırlamalar
- Konu başına **nötr yok** (sadece pozitif/negatif) — bilgi veren cümlelerde yanlış duygu verebilir.
- **Satıcı** konusu sık kaçırılır (testte recall ~0.37).
- 8 kelimeden kısa yorumlar ve olumsuzlama kalıpları ("hiç fena değil") eğitimde azdı.
- Bazı kelimelere takılır: "iade" → olumsuz satıcı, "performans" kelimesi → performans konusu, "küçük" → olumsuz boyut.
- Kurallar `step17_aspect_sentiment/LABEL_RULES.md` (v1) ile eğitildi; ör. "ürün kırık geldi" kural gereği **kargo**
  konusudur, kalite değil.

### Hafif mod
```bash
ASPECT_LIGHT=1 ../.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
```
Konu hattında 3 yerine 1 model: ~4 kat hızlı, daha az bellek; val'de kayıp < 1 puan (Adım 20). Varsayılan 3 modeldir
(ölçülen yapılandırma bu). Normal modda bile bir yorum ~0.1 saniye sürer, genelde gerek yok.

## 3. Yeni bir makinede kurulum
Gerekenler: macOS (Apple Silicon önerilir; Intel/Linux'ta CPU ile çalışır ama yavaş), **Python 3.12**, git,
~6 GB disk (modeller), ~4 GB boş RAM, ilk kurulumda internet.

```bash
git clone https://github.com/GoGonuldas/gorkanai.git
cd gorkanai
python3.12 -m venv .venv
.venv/bin/pip install pandas numpy scikit-learn torch transformers huggingface_hub fastapi uvicorn httpx
```
(`scikit-learn`, `gensim`, `datasets` sadece eski adımların script'leri için; uygulama için şart değil.)

Depoda **modeller yok** (her biri ~420 MB, GitHub kabul etmiyor). Uygulamayı açmadan önce bir sonraki bölümdeki
modellerin hepsi yerinde olmalı; eksikse uygulama açılırken hata verir.

## 4. Modelleri edinmek
Uygulama 7 model kullanır:

| Rol | Klasör (depo köküne göre) |
|---|---|
| Genel duygu (3 sınıf) | `step14_three_class/model_v2b` |
| Konu tespiti (3 tohum) | `step19_topic_v2/model_B_s0_e16`, `…_s1_e16`, `…_s2_e16` |
| Konuya koşullu duygu (3 tohum) | `step17_aspect_sentiment/model_v2b_s0_e8`, `…_s1_e8`, `…_s2_e8` |

Üç yol var, en kolayından en zoruna:

### Yol A — Mac mini'den kopyala (önerilen; birebir aynı modeller)
Mac mini'de *Sistem Ayarları → Genel → Paylaşma → Uzaktan Giriş* açık olmalı. Yeni makinede, depo kökünde:
```bash
MINI=gorkan@gorkans-mac-mini.local:workspace/gorkanai
rsync -av "$MINI/step14_three_class/model_v2b" step14_three_class/
for s in 0 1 2; do
  rsync -av "$MINI/step19_topic_v2/model_B_s${s}_e16" step19_topic_v2/
  rsync -av "$MINI/step17_aspect_sentiment/model_v2b_s${s}_e8" step17_aspect_sentiment/
done
```
(~3 GB.) Kopyalamak yerine harici diskle de taşıyabilirsin; klasör adları ve yerleri aynı kalmalı.

### Yol B — Genel duygu modelini Hugging Face'ten indir
Genel duygu modeli Hub'da herkese açık: [Urartu65/gorkanai-tr-sentiment](https://huggingface.co/Urartu65/gorkanai-tr-sentiment).
```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('Urartu65/gorkanai-tr-sentiment', local_dir='step14_three_class/model_v2b')"
```
Konu ve konu-duygu modelleri (6 tanesi) Hub'da **yok** — onlar için Yol A ya da Yol C.

### Yol C — Yeniden eğit (Mac mini'de saatler sürer)
Eğitim verileri depoda. Sıra önemli (Adım 17, genel duygu modelinden başlar):
```bash
# 0) genel duygu modeli yerinde olmalı (Yol A ya da B)
# 1) konuya koşullu duygu: iki başlangıç × 3 tohum × epoch 2-16 eğitir; seçilen v2b/e8 klasörleri de oluşur
cd step17_aspect_sentiment && ../.venv/bin/python train.py && cd ..
# 2) konu tespiti v2: sadece B yapılandırması gerekir (3 tohum × epoch 5/8/12/16)
cd step19_topic_v2 && ../.venv/bin/python train19.py B && cd ..
```
- İnternet gerekir (temel model `dbmdz/bert-base-turkish-cased` ilk seferde ~440 MB iner).
- Ara epoch klasörleri de kaydedilir (diski doldurur); uygulama sadece yukarıdaki tablodakileri kullanır, diğerleri silinebilir.
- **Dikkat:** yeniden eğitilen modeller orijinallerle birebir aynı olmaz (MPS tam deterministik değil); sayılar çok yakın
  ama 5. bölümdeki eşdeğerlik testi bu durumda "geçmedi" diyebilir — bu beklenen bir durumdur, uygulama yine çalışır.

## 5. Kurulumu doğrulamak
Uygulamanın, projede ölçülen modelleri çalıştırdığını kontrol eden test (val 300 yorum, birkaç dakika):
```bash
cd step20_aspect_app && ../.venv/bin/python test_pipeline.py
```
Beklenen (Adım 20'deki gibi):
- `1) EŞDEĞERLİK … → GEÇTİ` — olasılıklar kayıtlı değerlendirmeyle aynı (Yol A'da fark tam 0).
- `2) DUMAN TESTİ … → GEÇTİ`
- `3) GECİKME` — Mac mini'de medyan ~100 ms.
Çıktı `step20_aspect_app/log_test_pipeline.txt`'ye yazılır (bu dosya git'te kayıtlı; değişirse commit'leme).

Hızlı elle kontrol: uygulama açıkken `curl -s http://127.0.0.1:8000/health` → `{"status":"ok"}`.

## 6. Sorun giderme
| Belirti | Sebep / çözüm |
|---|---|
| Açılışta `OSError: Repo id must be in the form 'repo_name' … '…/step19_topic_v2/model_B_s0_e16'` (ya da başka bir model klasörü) | O model klasörü yok; kod onu Hub adı sanıyor → 4. bölüm. |
| `ModuleNotFoundError` | Paket eksik → 3. bölümdeki `pip install`; uygulamayı `../.venv/bin/uvicorn` ile başlat (sistem Python'u değil). |
| `address already in use` (port 8000 dolu) | `lsof -i :8000` ile süreci bul, `kill <PID>`; ya da `--port 8001`. |
| Laptoptan açılmıyor (zaman aşımı) | Aynı Wi-Fi'de mi? Mac mini'de *Ağ → Güvenlik Duvarı* Python'a gelen bağlantıya izin veriyor mu? Laptopta terminalden `curl` çalışmayıp tarayıcı çalışabilir (macOS "Yerel Ağ" izni) — tarayıcıyla dene. |
| `gorkans-mac-mini.local` bulunamıyor | IP adresiyle dene (`ipconfig getifaddr en1`, Mac mini'de). |
| Çok yavaş / bellek dolu | `ASPECT_LIGHT=1` ile başlat; başka büyük uygulamaları kapat. Apple Silicon olmayan makinede CPU kullanılır, yavaş olması normal. |
| Arayüz açılıyor ama "Hata: sunucuya ulaşılamadı" | Sunucu kapanmış ya da çökmüş — terminaldeki hata mesajına bak. |

### Not: Docker / internete yayınlama
`step9_app/Dockerfile` ve `requirements.txt` eski, **sadece genel duygu** (`/predict`) sürümü için hazırlanmıştı
(Hugging Face Spaces PRO istediği için kullanılmadı). Konu bazlı analiz depodaki diğer klasörlerin kodunu ve 6 modeli
daha gerektirdiği için bu dosyalar şu haliyle `/aspects`'i çalıştırmaz.
