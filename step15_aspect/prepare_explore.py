"""
ADIM 15.1 - Aspect (konu) bazlı duygu analizi: görev tanımı ve veri keşfi (eğitim yok).

Amaç: "kargo çok hızlıydı ama kumaşı ince" -> kargo: pozitif, kalite: negatif.
Önce verinin hangi konulardan bahsettiğine bakıyoruz: HİÇBİR eğitim/val/test/candidate setinde
olmayan 300 yorum seçip kelime/kalıp sayımıyla 5-7 konu öneriyoruz.

Hangi yorumlar "görülmemiş"?
  - Adım 8/10/11/13 havuzu negasyon holdout'u ÇIKARMADAN, Adım 12/14 ÇIKARARAK karıştırıyor -> iki farklı
    permütasyon. İkisinde de pozitif/negatiflerin ilk 9000'i (val 500 + eğitim en fazla 8000 + pay) dışarıda.
  - Negasyon kalıbı içeren TÜM satırlar dışarıda (holdout + pattern-extra bunlardan seçiliyordu).
  - Bilinen tüm metin kümeleri (testler, elle etiketlenenler, candidate'ler, flagged) dışarıda.
Adım 15.2 bu dosyadaki `unseen_pool()` fonksiyonunu kullanacak ve buradaki 300'ü de dışarıda tutacak.

Uzun yorumlar seçiyoruz (8-40 kelime): 1-2 kelimelik yorumda birden çok konu nadiren geçer.
Havuz %94 pozitif; şikâyetleri görmek için yarı "pozitif" yarı "negatif" etiketli.

Çıktı: step15_aspect/explore_300.csv (id 3000-3299) + ekrana kelime ve konu sayımları.
"""

import glob
import os
import re
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
NEG_PATTERN = re.compile(r"\b(?:fena|kötü|berbat|kalitesiz)\w*\s+(?:değil|degil|sayılmaz|sayilmaz)")
SAFE_HEAD = 9000
MIN_WORDS, MAX_WORDS = 8, 40


def p(*parts):
    return os.path.join(ROOT, *parts)


def known_texts():
    """Daha önce herhangi bir amaçla görülmüş/etiketlenmiş tüm metinler."""
    files = [p("data/real/test.csv"), p("data/short_clean_test.csv"), p("data/negation_test.csv"),
             p("data/hard_test.csv"), p("step12_confident_learning/flagged.csv"),
             p("step14_three_class/candidates/all_candidates.csv")]
    files += glob.glob(p("data/neutral_labels/batch_*.csv"))
    files += glob.glob(p("step14_three_class/candidates/batch_*.csv"))
    return set().union(*(set(pd.read_csv(f)["text"]) for f in files))


def unseen_pool(extra_exclude=()):
    """Havuzdan hiçbir adımda kullanılmamış yorumlar (+ used: kullanılmış olabilecek metinler)."""
    pool = pd.read_csv(p("data/real/train_pool.csv"))
    has_pattern = pool["text"].str.contains(NEG_PATTERN)
    holdout = pool[has_pattern].sample(200, random_state=42)
    used = set(pool.loc[has_pattern, "text"])
    for base in (pool, pool.drop(holdout.index)):   # iki permütasyon (holdout'suz / holdout'lu)
        for label in ("pozitif", "negatif"):
            used |= set(base[base["label"] == label].sample(frac=1, random_state=42).iloc[:SAFE_HEAD]["text"])
    exclude = used | known_texts() | set(extra_exclude)
    unseen = pool[~pool["text"].isin(exclude)].drop_duplicates("text")
    assert not set(unseen["text"]) & exclude
    return unseen, exclude


if __name__ == "__main__":
    unseen, exclude = unseen_pool()
    n_words = unseen["text"].str.split().str.len()
    cand = unseen[(n_words >= MIN_WORDS) & (n_words <= MAX_WORDS)]
    print(f"Görülmemiş yorum: {len(unseen)} | {MIN_WORDS}-{MAX_WORDS} kelime: {len(cand)} "
          f"({cand['label'].value_counts().to_dict()})")

    chosen = pd.concat([cand[cand["label"] == "pozitif"].sample(150, random_state=15),
                        cand[cand["label"] == "negatif"].sample(150, random_state=16)])
    chosen = chosen.sample(frac=1, random_state=17).reset_index(drop=True)
    chosen["id"] = range(3000, 3000 + len(chosen))
    assert not set(chosen["text"]) & exclude, "keşif seti daha önce görülmüş bir metin içeriyor"
    chosen[["id", "text", "label"]].rename(columns={"label": "pool_label"}).to_csv(
        os.path.join(HERE, "explore_300.csv"), index=False)
    print(f"{len(chosen)} yorum -> explore_300.csv (id 3000-3299)\n")

    # --- Kelime sayımı: yorumlar en çok neden bahsediyor? ---
    low = lambda t: t.replace("I", "ı").replace("İ", "i").lower()
    stop = set("""ve bir bu çok da de ama için ile gibi daha en ben o ne mi mı var yok çok olarak olan
                 olduğu diye kadar sonra şu her hiç biraz gayet tam zaten değil ise ki ya yani artık
                 ürün ürünü ürünün aldım aldık tavsiye ederim teşekkürler iyi güzel""".split())
    tokens = [re.findall(r"\w+", low(t)) for t in chosen["text"]]
    words = Counter(w for ts in tokens for w in set(ts) if w not in stop and len(w) > 2)
    print("En sık 60 kelime (kaç yorumda geçtiği):")
    print(", ".join(f"{w} {c}" for w, c in words.most_common(60)), "\n")

    # --- Aday konular: kelime kökleri ile ön sayım (kök eşleşmesi: "kargo" -> kargoya, kargosu...) ---
    aspects = {
        "kargo/teslimat": r"kargo|teslim|paket(?!le)|geldi|ulaştı|elime|hızlı gel|gecik|kurye",
        "kalite/malzeme": r"kalite|kumaş|malzeme|sağlam|dayanık|plastik|dikiş|kırıl|bozul|yırt",
        "fiyat": r"fiyat|para|ucuz|pahalı|indirim|uygun|kampanya|tl\b",
        "beden/uyum": r"beden|kalıp|dar\b|bol\b|numara|büyük|küçük|uymadı|oldu mu|boy",
        "paketleme": r"paketle|kutu|ambalaj|poşet|özenle",
        "satıcı/iletişim": r"satıcı|mağaza|iletişim|müşteri hizmet|iade|değişim|firma|ilgi",
        "performans/işlev": r"çalış|şarj|pil|batarya|ses|performans|hız(?!lı gel)|ısın|çekim|ekran|kullanış",
        "görünüm/renk": r"renk|görün|resim|fotoğraf|tasarım|şık|görsel",
    }
    counts = []
    for name, pat in aspects.items():
        hits = chosen[[bool(re.search(pat, low(t))) for t in chosen["text"]]]
        counts.append((name, len(hits), (hits["label"] == "negatif").sum()))
    n_any = sum(any(re.search(pat, low(t)) for pat in aspects.values()) for t in chosen["text"])
    print(f"{'konu':<18} {'yorum':>5} {'(neg. etiketli)':>16}")
    for name, n, n_neg in counts:
        print(f"{name:<18} {n:>5} {n_neg:>16}")
    print(f"\nEn az bir konu kelimesi geçen yorum: {n_any}/{len(chosen)}")
