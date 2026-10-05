# Adım 21 — ayrışma okuması (ölçümden ve rapordan SONRA, Görkan'la; 2026-10-05)
Seçim: model–Görkan çift farkı en büyük yorumlar; "kural" = Claude (fresh) da modelle aynı/yakın, "model" = sadece model farklı.
Hiçbir altın etiket DEĞİŞTİRİLMEDİ; ölçüm (log_evaluate.txt) olduğu gibi kalır. Bu set artık "okunmuş".

## A. Kural kaynaklı (Görkan'ın yanıtları)
| # | id | Görkan | model | Claude | Görkan'ın yanıtı |
|---|---|---|---|---|---|
| 1 | 9500 | Qp,Pp | Pn | Pn* | hata — doğrusu Pn (model ile aynı) |
| 2 | 9576 | Pp | Qp,Pn | Pn | hata — doğrusu Pn |
| 3 | 9511 | Gp,Qp | Qp,Pp,Bp,Gp | Bp,Gp,Pp,Qp | bilinçli (ana fikir), ama Bp ve Pp eklenebilir |
| 4 | 9517 | Fp,Pp | Fp,Pn | Fp,Pn,Qn | hata — doğrusu Fp,Pn (model ile birebir aynı) |
| 5 | 9524 | Qn | Pn | Qp,Pn* | bilinçli, Pn eklenebilir |
| 6 | 9527 | Qp | Pp | Pp | bilinçli, Pp eklenebilir |
| 7 | 9531 | Qp,Kp | Fp,Qp | Qp,Fp | bilinçli, Fp eklenebilir |

## B. Sadece model (laptop oturumunun okuması, Görkan'a soru yok)
- 9577 iade hızlı yapıldı → model Sn (Görkan ve Claude Sp) + olmayan Kn: "iade" kelimesini olumsuz okuyor.
- 9506 "ürün küçük, büyük bir şey beklemeyin. kullanışlı" → model Bn: uyarıyı şikâyet sanıyor.
- 9566 "ekonomi performans olarak çok iyi" → model Pp (Görkan ve Claude Fp,Qp): "performans" kelimesine takılıyor.
- 9552 sadece bilgi (ölçü, renk) → model Qn,Bn: nötr üretemediği için (bilinen sınırlama).
- 9533 → model "şık"ı (G) kaçırdı; "bıçakların olmaması" eksikliğini B saydı (S/P olmalı).
