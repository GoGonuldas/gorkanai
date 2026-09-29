"""
ADIM 15.2 (2/2) - Elle verilen konu-duygu etiketlerini kaydeder (Adım 14 save_labels.py yöntemi).

Kullanım: python save_aspect_labels.py <batch_no> <grup1> <grup2> ...
Her grup "/" ile ayrılmış TAM 10 yorum etiketi (son grup kısa olabilir) -> etiket kayması yakalanır.
Bir yorumun etiketi: virgülle ayrılmış (konu, duygu) çiftleri, konu yoksa "0".
  konu:  K=kargo/teslimat  F=fiyat/değer  Q=kalite/malzeme  P=performans/işlev
         B=boyut/beden     G=görünüm/tasarım  S=satıcı/hizmet
  duygu: p=pozitif  n=negatif  x=nötr ; sonuna "*" = karışık (aynı konu hem övülüp hem eleştirilmiş,
         yazılan duygu baskın olan)
  örnek: "Kp,Qn*"  ->  kargo: pozitif, kalite: negatif (karışık)

Etiketleme notları (kararlar):
  - Konudan duygusuz bahsediliyorsa nötr ("10 günde geldi", "indirimdeyken aldım").
  - "fiyatına göre iyi" -> F pozitif; "bu fiyata değmez" -> F negatif. Hediye/kampanya -> F.
  - Kargo ücreti, paketleme -> K. Paketin/ürünün sağlam-hasarsız ya da hasarlı gelmesi -> K.
  - Q/P SINIRI (2026-09-29, Görkan kararı — kör etiketleme uyumsuzluğundan sonra):
      Q = ürünün GENEL iyi/kötü olması ("süper ürün", "ürün iyi değil", "memnun kaldım", "pişman oldum",
          "beğendim") ve GENEL olarak işe yarayıp yaramaması ("işe yaramıyor", "etkisi yok", "bozuldu",
          "şarj etmiyor", "pil ömrü kısa", "köpeğim bayılıyor", "sızdırıyor") + malzeme, işçilik, kırılma.
      P = ADI KONAN belirli bir özellik/teknik ölçü: çekim/emiş gücü, ses, kamera/ekran, hız, uyumluluk,
          kurulum/kullanım kolaylığı, ergonomi, koku/kalıcılık, tat, ayar/fonksiyon eksikliği.
    Kalıp tavsiye cümleleri ("alın", "tavsiye ederim/etmiyorum") tek başına Q DEĞİL.
  - Ekran/görüntü KALİTESİ (TV, kamera, monitör) -> P. G sadece dış görünüm, renk, "resimdeki gibi".
  - B = boyut + MİKTAR ("yağ çok az", "3gr yerine 1.5gr", "10 tane az") + AĞIRLIK ("hafif", "çok ağır").
  - Eksik/eksiksiz parça veya içerik (kutudan ne çıktığı), yanlış ürün gönderilmesi, iade, müşteri hizmetleri,
    servis, garanti, satıcının müdahalesi -> S. (Sınır: kutunun İÇERİĞİ -> S, kutunun DURUMU -> K.)
  - Sadece "teşekkürler hepsiburada", soru, kullanım ipucu -> konu yok. (Eski "içeriksiz genel övgü -> konu
    yok" kuralı 2026-09-29'da KALDIRILDI: genel övgü/yergi artık Q.)

Çıktı: data/aspect_labels/batch_N.csv ve hepsinin birleşimi data/aspect_labels/aspect_labels.csv
(her konu için bir sütun: pozitif/negatif/nötr/boş, `karisik` 0/1, `karisik_konular`).
"""

import glob
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "data", "aspect_labels")
ASPECTS = {"K": "kargo", "F": "fiyat", "Q": "kalite", "P": "performans",
           "B": "boyut", "G": "gorunum", "S": "satici"}
SENT = {"p": "pozitif", "n": "negatif", "x": "nötr"}
PAIR = re.compile(r"^([KFQPBGS])([pnx])(\*?)$")


def parse(token):
    if token == "0":
        return {}, []
    out, mixed = {}, []
    for pair in token.split(","):
        m = PAIR.match(pair)
        assert m, f"geçersiz etiket: {pair!r} ({token!r})"
        a, s, star = m.groups()
        assert ASPECTS[a] not in out, f"aynı konu iki kez: {token!r}"
        out[ASPECTS[a]] = SENT[s]
        if star:
            mixed.append(ASPECTS[a])
    return out, mixed


def merge_all():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "batch_*.csv")))
    labels = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files])
    meta = pd.read_csv(os.path.join(HERE, "label_set.csv"))[["id", "pool_label", "split"]]
    df = meta.merge(labels, on="id")
    parsed = df["raw"].map(parse)
    for name in ASPECTS.values():
        df[name] = parsed.map(lambda p: p[0].get(name, ""))
    df["karisik_konular"] = parsed.map(lambda p: ";".join(p[1]))
    df["karisik"] = (df["karisik_konular"] != "").astype(int)
    df.to_csv(os.path.join(OUT_DIR, "aspect_labels.csv"), index=False)
    return df


if __name__ == "__main__":
    batch_no = int(sys.argv[1])
    groups = [g.split("/") for g in sys.argv[2:]]
    for i, g in enumerate(groups[:-1]):
        assert len(g) == 10, f"{i}. grup ({i * 10}-{i * 10 + 9}) 10 değil: {len(g)} -> {g}"
    tokens = [t for g in groups for t in g]
    for t in tokens:
        parse(t)

    batch = pd.read_csv(os.path.join(HERE, f"batch_{batch_no}.csv"))
    assert len(tokens) == len(batch), f"{len(tokens)} etiket, {len(batch)} yorum"
    batch["raw"] = tokens
    os.makedirs(OUT_DIR, exist_ok=True)
    batch.to_csv(os.path.join(OUT_DIR, f"batch_{batch_no}.csv"), index=False)

    df = merge_all()
    print(f"batch {batch_no} kaydedildi. Toplam etiketli: {len(df)}")
    counts = {a: df[a].value_counts().to_dict() for a in ASPECTS.values()}
    print(pd.DataFrame(counts).T.fillna(0).astype(int).drop(columns="", errors="ignore").to_string())
    print(f"konusuz yorum: {(df['raw'] == '0').sum()} | karışık: {df['karisik'].sum()}")
