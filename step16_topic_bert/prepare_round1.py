"""
ADIM 16 (1/?) - Etiketlenecek yorumları seçmek, tur 1: val eki (200) + eğitim tur 1 (300).

Elimizde sadece val 100 / test 200 etiketli yorum var, ikisi de eğitime giremez -> eğitim seti sıfırdan etiketlenecek.
  - Val eki 200: TAMAMEN rastgele, mevcut val ile aynı tarif (8-40 kelime, yarı "pozitif" yarı "negatif" havuz
    etiketli) -> val 100 -> 300. (Adım 14 dersi: val, önemsediğin her sınıfı temsil etmeli; val'de satıcı 7 örnekti.)
  - Eğitim tur 1, 300: 150 rastgele + 150 hedefli. Henüz konu modeli yok, o yüzden "kararsızlık" yerine Adım 15'in
    DONDURULMUŞ V2 anahtar kelime listesi kullanılıyor: nadir konular (satıcı, görünüm, boyut, kargo) + hiçbir
    anahtar kelimenin tutmadığı yorumlar (örtük konu adayı). Yarısı rastgele: sadece kelimeyle seçersek model
    "kelime varsa konu var" kısayolunu öğrenir.
  - Tur 2 (prepare_round2.py): tur 1 ile eğitilen kaba modelin kararsız kaldığı yorumlar.

Bölme (train/val) ve kaynak (rastgele/hedefli) etiketlemeden ÖNCE sabitleniyor; etiketleme dosyalarında
(batch_XX.csv) sadece id + text var, sıra karışık -> etiketleyen hangi yorumun nereden geldiğini görmüyor.

Çıktı: label_set16.csv (id 5000-5499) + batch_01..10.csv (50'şer).
"""

import os

import pandas as pd

from common import HERE, LABEL_SET, candidate_pool, check_no_leak  # step15_aspect'i sys.path'e ekler
from baseline import detect  # noqa: E402

TARGETS = {"satici": 30, "gorunum": 30, "boyut": 30, "kargo": 20, "kelime_yok": 40}   # toplam 150

cand, exclude, exclude_norm = candidate_pool()
print(f"Aday havuzu (8-40 kelime, görülmemiş): {cand['label'].value_counts().to_dict()}")


def take(df, n, seed):
    """Yarı 'pozitif' yarı 'negatif' havuz etiketli n yorum (negatif yetmezse pozitifle tamamla)."""
    neg = df[df["label"] == "negatif"]
    neg = neg.sample(min(n // 2, len(neg)), random_state=seed)
    pos = df[df["label"] == "pozitif"].sample(n - len(neg), random_state=seed + 1)
    return pd.concat([pos, neg])


parts = []
val = take(cand, 200, 35).assign(split="val", source="rastgele")
parts.append(val)
rest = cand.drop(val.index)
rnd = take(rest, 150, 37).assign(split="train", source="rastgele")
parts.append(rnd)
rest = rest.drop(rnd.index)

found = rest["text"].map(detect)
for i, (name, n) in enumerate(TARGETS.items()):
    mask = (found.map(len) == 0) if name == "kelime_yok" else found.map(lambda s: name in s)
    part = take(rest[mask.reindex(rest.index)], n, 40 + 2 * i).assign(split="train", source=f"hedefli:{name}")
    parts.append(part)
    rest = rest.drop(part.index)

chosen = pd.concat(parts).sample(frac=1, random_state=50).reset_index(drop=True)
chosen["id"] = range(5000, 5000 + len(chosen))
chosen["round"] = 1
chosen = chosen.rename(columns={"label": "pool_label"})[["id", "text", "pool_label", "split", "source", "round"]]

assert len(chosen) == 500
check_no_leak(chosen, exclude, exclude_norm)

chosen.to_csv(LABEL_SET, index=False)
for b in range(0, len(chosen), 50):
    chosen.iloc[b:b + 50][["id", "text"]].to_csv(os.path.join(HERE, f"batch_{b // 50 + 1:02d}.csv"), index=False)
print(chosen.groupby(["split", "source", "pool_label"]).size().to_string())
print(f"{len(chosen)} yorum -> label_set16.csv + batch_01..{len(chosen) // 50:02d}.csv | sızıntı kontrolleri geçti")
