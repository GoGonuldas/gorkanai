# LABEL_RULES v2 (2026-10-06, Adım 23 — Görkan'ın kararlarıyla)

**v1 (`step17_aspect_sentiment/LABEL_RULES.md`) DEĞİŞTİRİLMEDİ:** Adım 17-22'nin bütün etiketleri ve kasa testinin
sha256'sı v1'e bağlı. v2 sadece BUNDAN SONRAKİ etiketlemeler içindir; v1 ile etiketlenmiş veriyle karıştırılırsa
etiketler tutarsız olur (aşağıdaki "Etki").

## v1 → v2 değişiklikleri (kaynak: Adım 21-22 insan testi, Görkan'ın kalite (Q) ayrışmalarındaki kararları)
1. **Bağlı yargı (Q sınırı):** "beğendim / çok memnunum / süper" gibi genel bir yargı, yakınındaki BAŞKA bir konuya
   bağlanabiliyorsa sadece o konu yazılır, Q yazılmaz. "kolda güzel duruyor, beğendim" → `Gp`; "en kısa zamanda geldi,
   çok beğendim" → `Kp`; "taksitle aldım, çok memnunum" → `Fp`. Q sadece ürünün BÜTÜNÜ hakkında açık yargı varsa
   ("ürün süper", "iyi ki almışım", "keşke beklemeseydim").
2. **Ana işlev = Q (P değil):** ürünün ASIL İŞİ olan özelliğin değerlendirmesi **sadece Q**: parfümde koku/kalıcılık,
   kulaklıkta/hoparlörde ses, çamaşır makinesinde yıkama, süpürgede çekim, ısıtıcıda ısıtma. **P artık yalnız yan
   özellikler** içindir (parfüm şişesi, makinenin gürültüsü, kurulum kolaylığı, uyumluluk, kumanda...).
   v1'in sınır kuralı 1'deki "kokusu kalıcı değil → Pn" örneği v2'de `Qn`.
3. **Kitap/içerik:** kitabın konusu ya da içeriği hakkındaki kişisel görüş ("sıkıldım", "sürükleyici", "anlatımı güzel")
   **ürün değerlendirmesi sayılmaz** → yazılmaz (başka konu yoksa `0`). Q yalnız ürünün kendisi için (baskı, kâğıt,
   cilt, sayfa eksikliği). v1 sınır kuralı 3 bu yüzden KALDIRILDI.

## Etki (dürüst not)
- 2 numara Q/P sınırını ürün türüne bağlıyor ("bu ürünün asıl işi ne?"). Bu, v1'de en çok ayrışılan sınırdı; v2 onu
  Görkan'ın okumasına yaklaştırıyor ama yeni bir yargı noktası ekliyor. Yeni bir etiketleyiciyle ilk 30 yorumda uyum
  ölçülmeden büyük etiketlemeye geçilmemeli.
- Mevcut 1400 eğitim yorumu + val + insan testi v1 ile etiketli. v2 ile bir model eğitmek için bunların (en azından
  Q, P ve kitap içeren yorumların) yeniden etiketlenmesi gerekir.

---
## v1 metni (değişiklikler yukarıdaki 3 maddeyle geçersiz kılınan yerler dışında aynen geçerli)
# Adım 17 — Yeni test (id 6000-6199) için etiketleme kuralları (TEK KAYNAK)

İki etiketleyici de (gorkanai-1e = ana altın, Mac mini oturumu = ikinci) sadece bu dosyayı kullanır.
Temel: `step15_aspect/save_aspect_labels.py` docstring'i (2026-09-29 Q/P tanımı) + Adım 16'da netleşen maddeler.
Buradaki örneklerin hiçbiri test/val yorumu değildir (uydurma cümleler).

## Biçim
Her yorum için virgülle ayrılmış (konu, duygu) çiftleri; konu yoksa `0`.
- konu: `K` kargo/teslimat, `F` fiyat/değer, `Q` kalite, `P` performans/özellik, `B` boyut, `G` görünüm, `S` satıcı/hizmet
- duygu: `p` pozitif, `n` negatif, `x` nötr; sonuna `*` = karışık (aynı konu hem övülmüş hem eleştirilmiş; yazılan baskın duygu)
- örnek: `Kp,Qn*` → kargo pozitif, kalite negatif (karışık). Aynı konu bir yorumda bir kez.
- Dosya: `id,raw` sütunlu CSV (ör. `6003,"Kp,Qn*"`), 200 satır, id'ler batch dosyalarındakiyle birebir.

## Konular
- **K**: kargo hızı, teslimat, kargo ücreti, paketleme; ürünün/paketin hasarlı ya da sağlam GELMESİ.
  Sade "3 günde geldi / teslim aldım" → `Kx`; "ertesi gün geldi", "çok hızlı", "geç geldi" → `Kp` / `Kn`.
- **F**: fiyat, değer, indirim, hediye/kampanya. "fiyatına göre iyi" → `Fp` (tek başına Q DEĞİL); "bu paraya değmez" → `Fn`;
  "indirimdeyken aldım" → `Fx`.
- **Q**: ürünün GENEL iyi/kötü olması ("süper ürün", "memnun kaldım", "pişman oldum", "beğendim") + GENEL olarak işe
  yarayıp yaramaması ("işe yaramıyor", "etkisi yok", "bozuldu", "şarj etmiyor", "pil ömrü kısa", "kedim bayılıyor",
  "sızdırıyor") + malzeme, işçilik, dayanıklılık.
- **P**: ADI KONAN belirli bir özellik/teknik ölçü: güç (çekim/emiş), ses/gürültü, kamera/ekran/görüntü kalitesi, hız,
  uyumluluk, kurulum/kullanım kolaylığı, ergonomi/konfor, koku/kalıcılık, tat, ısınma, bir fonksiyonun olması/olmaması.
- **B**: boyut, beden, ölçü + MİKTAR ("içinden çok az çıktı") + AĞIRLIK ("hafif", "çok ağır") + kalınlık/incelik.
- **G**: sadece DIŞ görünüm: renk, tasarım, şıklık, "resimdeki gibi (değil)". Ekran/görüntü KALİTESİ G değil, P.
- **S**: satıcı, müşteri hizmetleri, servis, garanti, iade süreci; yanlış ürün gönderilmesi; kutunun İÇERİĞİ
  (eksik/eksiksiz parça, aksesuar, çanta, kılavuz çıkmaması). Kutunun DURUMU → K.

## Sınır kuralları
1. **Q/P (Adım 16 netleştirmesi):** adı konan özellik, arıza ya da şikâyet biçiminde geçse de **P** ("hoparlörden ses
   gelmiyor" → `Pn`, "kokusu kalıcı değil" → `Pn`). Genel "çalışmıyor / işe yaramadı / bozuldu" → **Q**.
   İkisi aynı yorumda birlikte olabilir ("emişi zayıf, pişmanım" → `Pn,Qn`).
2. **Q sadece açık bir GENEL yargı ya da genel işe yarama ifadesi varsa** yazılır. Belirli bir özelliğin övgüsü tek
   başına Q getirmez ("ekranı çok net" → sadece `Pp`). Kalıp tavsiye ("alın", "tavsiye ederim/etmem") tek başına Q değil.
3. **Kitap ve içerik ürünleri (Adım 16.5 netleştirmesi):** içeriğin değerlendirilmesi ("sürükleyici", "sıkıcı",
   "anlatımı güzel") ürünün genel iyi/kötü olmasıdır → **Q**. (Adım 15 altınında bunlar P yazılmıştı; eğitim
   etiketleri Q. Yeni testte Q.)
4. Konudan görüş bildirmeden bahsediliyorsa nötr (`x`); sadece bilgi/anlatım için geçen, değerlendirme taşımayan bahis
   ("iade edeceğim" bir sebep cümlesinin parçası olarak) konu sayılmaz. Emin değilsen yazma.
5. Sadece "teşekkürler hepsiburada", soru, kullanım ipucu → `0`.
6. Karışık: aynı konuda hem övgü hem eleştiri varsa baskın olanı yaz ve `*` ekle.
