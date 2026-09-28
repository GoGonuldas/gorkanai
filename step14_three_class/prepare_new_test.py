"""
Yapılacaklar #1: Adım 14'te hiç görülmemiş, taze bir test seti.

data/short_clean_test.csv Adım 14'te görüldü (val'i genişletme kararı bu sete bakılarak
verildi) -> V2b'nin o setteki sonucu iyimser. Burada aynı mantıkla ama HİÇBİR eğitim/val/
test/candidate setinde olmayan kısa yorumlar seçiyoruz: hem "pozitif" hem "negatif"
etiketlilerden (~%20 nötr bekleniyor, Adım 11-14'teki gibi).

Güvenlik: pos/neg havuzunun ilk 9000'i (val 500 + train 8000 + pattern-extra payı) kullanılmış
olabilir; 9000'den sonrasından seçip yine de tüm bilinen metin kümeleriyle kesişim kontrolü
yapıyoruz (assert).

Çıktı: candidates/batch_06.csv (id 2000'den başlar), candidates/all_candidates.csv'ye eklenir.
"""

import glob
import os
import re

import pandas as pd

N = 200
MAX_WORDS = 10
SAFE_TAIL = 9000  # val(500) + train(8000) + pattern-extra payından kesin daha ileri
HERE = os.path.dirname(os.path.abspath(__file__))
NEG_PATTERN = re.compile(r"\b(?:fena|kötü|berbat|kalitesiz)\w*\s+(?:değil|degil|sayılmaz|sayilmaz)")

pool = pd.read_csv("../data/real/train_pool.csv")
holdout = pool[pool["text"].str.contains(NEG_PATTERN)].sample(200, random_state=42)
rest = pool.drop(holdout.index)
pos = rest[rest["label"] == "pozitif"].sample(frac=1, random_state=42)
neg = rest[rest["label"] == "negatif"].sample(frac=1, random_state=42)
unused_pos = pos.iloc[SAFE_TAIL:]
unused_neg = neg.iloc[SAFE_TAIL:]

seen = (set(pd.read_csv("../data/real/test.csv")["text"])
        | set(pd.read_csv("../data/short_clean_test.csv")["text"])
        | set(pd.read_csv("../data/negation_test.csv")["text"])
        | set(pd.concat([pd.read_csv(f) for f in glob.glob("../data/neutral_labels/batch_*.csv")])["text"])
        | set(pd.read_csv("../step12_confident_learning/flagged.csv")["text"])
        | set(pd.read_csv(os.path.join(HERE, "candidates", "all_candidates.csv"))["text"])
        | set(holdout["text"]))

cand_pos = unused_pos[~unused_pos["text"].isin(seen)]
cand_pos = cand_pos[cand_pos["text"].str.split().str.len() <= MAX_WORDS].drop_duplicates("text")
cand_neg = unused_neg[~unused_neg["text"].isin(seen)]
cand_neg = cand_neg[cand_neg["text"].str.split().str.len() <= MAX_WORDS].drop_duplicates("text")
print(f"Kullanılabilir kısa aday: pozitif {len(cand_pos)}, negatif {len(cand_neg)}")

# Güvenlik: bu metinler eğitimde/val'de kullanılan ilk 9000'lik dilimde de yok mu?
assert not set(cand_pos["text"]) & set(pos.iloc[:SAFE_TAIL]["text"])
assert not set(cand_neg["text"]) & set(neg.iloc[:SAFE_TAIL]["text"])

chosen_pos = cand_pos.sample(N // 2, random_state=6).assign(source="yeni_test_pozitif")
chosen_neg = cand_neg.sample(N // 2, random_state=7).assign(source="yeni_test_negatif")
chosen = pd.concat([chosen_pos, chosen_neg]).sample(frac=1, random_state=8).reset_index(drop=True)
chosen["id"] = range(2000, 2000 + len(chosen))

chosen[["id", "text"]].to_csv(os.path.join(HERE, "candidates", "batch_06.csv"), index=False)
src = pd.read_csv(os.path.join(HERE, "candidates", "all_candidates.csv"))
if not (src["id"] >= 2000).any():
    pd.concat([src, chosen[["id", "text", "source"]]]).to_csv(
        os.path.join(HERE, "candidates", "all_candidates.csv"), index=False)
print(f"{len(chosen)} aday -> candidates/batch_06.csv (id 2000-{2000 + len(chosen) - 1})")
