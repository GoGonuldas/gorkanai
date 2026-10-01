# Görkan için: 100 yorumun kör etiketlenmesi (4 parça × 25, toplam ~50 dakika, istediğin zaman)

**Amaç:** etiket kurallarının bir insanla (seninle) ne kadar uyuştuğunu n=20 yerine n=100'de ölçmek. Bu yüzden hiçbir Claude etiketine ve model çıktısına
bakmadan, sadece kendi okumanla doldurman önemli. Doğru/yanlış yok; emin olmadığın yerde en yakın bulduğunu yaz.

**Dosyalar:** bu makinede (Mac mini) `~/workspace/gorkanai/step18_big_test/calib_gorkan_01.csv` …
`calib_gorkan_04.csv` (25'er yorum, `no` 1-100). Parçaları istediğin zaman, istediğin sırayla yapabilirsin; her biri
~12 dakika. Numbers/Excel ya da bir metin düzenleyicide aç, her satırın **`etiket`** sütununu doldur, aynı adla
**kaydet** (CSV olarak). Başka bir şey yapmana gerek yok; hangi parçayı kaydettiğini söylemen yeterli. Dosyayı açmak
istemezsen sohbete `1: Kp,Qn` biçiminde satırlar yazman da olur.

## Ne yazılacak
Her yorum için: **hangi konulardan bahsediyor** ve **o konuda duygu ne**. Virgülle ayır. Hiç konu yoksa `0`.

| Harf | Konu | Örnek ifadeler |
|---|---|---|
| K | kargo / teslimat / paketleme | "ertesi gün geldi", "kutusu ezik geldi" |
| F | fiyat / değer | "fiyatına göre iyi", "bu paraya değmez" |
| Q | kalite: ürün GENEL olarak iyi mi, işe yarıyor mu, sağlam mı | "süper ürün", "pişman oldum", "bozuldu" |
| P | ADI KONAN bir özellik | ses, koku, hız, kamera/ekran, kurulum kolaylığı, uyumluluk |
| B | boyut / beden / miktar / ağırlık | "bir beden büyük", "çok hafif" |
| G | dış görünüm / renk / tasarım | "çok şık", "resimdeki gibi değil" |
| S | satıcı / hizmet / iade / eksik parça | "müşteri hizmetleri ilgilendi", "kablosu çıkmadı" |

Duygu: **p** = pozitif, **n** = negatif, **x** = nötr (bahsediyor ama görüş yok).

## Uydurma örnekler (bunlar listedeki yorumlar değil)
- "Kargo hızlıydı ama kumaşı çok ince, beğenmedim." → `Kp,Qn`
- "Sesi çok yüksek çıkıyor, onun dışında fiyatına göre idare eder." → `Pn,Fp`
- "Teşekkürler hepsiburada." → `0`

Ayrıntılı kurallar `step17_aspect_sentiment/LABEL_RULES.md` dosyasında; okumak zorunda değilsin, takıldığında bakabilirsin.
