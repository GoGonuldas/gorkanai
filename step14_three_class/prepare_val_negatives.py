"""
ADIM 14 - Val setini güçlendirmek: kısa NEGATİF örnekler.

Deney 3'te val setinde sadece 8 negatif vardı; V2a'nın "kısa negatif -> nötr" kısayolu val'de
görünmedi ve val yanlış varyantı seçti. Çözüm: val'e ~150 kısa yorum daha eklemek, havuzun
"negatif" etiketlilerinden (çoğu gerçekten negatif, bir kısmı nötr olacak — Adım 11'de ~%19'u pozitif çıkmıştı).

Kısıt: bu yorumlar HİÇBİR varyantın eğitiminde olmamalı. Adım 10-14 eğitim setleri negatifleri
karıştırılmış havuzun ilk 500 (val) + 8000 (eğitim) satırından alıyor; biz ondan SONRAKİLERDEN seçiyoruz.
Test setleri ve daha önce etiketlenen tüm yorumlar da dışarıda.

Çıktı: candidates/batch_05.csv (id 1000'den başlar)
"""

import glob
import os
import re

import pandas as pd

N = 150
MAX_WORDS = 10
HERE = os.path.dirname(os.path.abspath(__file__))
NEG_PATTERN = re.compile(r"\b(?:fena|kötü|berbat|kalitesiz)\w*\s+(?:değil|degil|sayılmaz|sayilmaz)")

pool = pd.read_csv("../data/real/train_pool.csv")
holdout = pool[pool["text"].str.contains(NEG_PATTERN)].sample(200, random_state=42)
rest = pool.drop(holdout.index)
neg = rest[rest["label"] == "negatif"].sample(frac=1, random_state=42)
unused_neg = neg.iloc[500 + 8000:]

seen = (set(pd.read_csv("../data/real/test.csv")["text"])
        | set(pd.read_csv("../data/short_clean_test.csv")["text"])
        | set(pd.concat([pd.read_csv(f) for f in glob.glob("../data/neutral_labels/batch_*.csv")])["text"])
        | set(pd.read_csv("../step12_confident_learning/flagged.csv")["text"]))
cand = unused_neg[~unused_neg["text"].isin(seen)]
cand = cand[cand["text"].str.split().str.len() <= MAX_WORDS].drop_duplicates("text")
# Güvenlik: bu metinler eğitimde kullanılan 8000'lik negatif örneklemde de yok mu?
assert not set(cand["text"]) & set(neg.iloc[:500 + 8000]["text"])
print(f"Eğitimde hiç kullanılmamış kısa 'negatif' yorum: {len(cand)}")

chosen = cand.sample(N, random_state=5).reset_index(drop=True)
chosen["id"] = range(1000, 1000 + N)
chosen[["id", "text"]].to_csv(os.path.join(HERE, "candidates", "batch_05.csv"), index=False)
src = pd.read_csv(os.path.join(HERE, "candidates", "all_candidates.csv"))
if not (src["id"] >= 1000).any():
    pd.concat([src, chosen.assign(source="val_negatif")[["id", "text", "source"]]]).to_csv(
        os.path.join(HERE, "candidates", "all_candidates.csv"), index=False)
print(f"{N} aday -> candidates/batch_05.csv")
