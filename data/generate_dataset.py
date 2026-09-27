"""
Eğitim veri setini üretir: data/reviews.csv

Adım 1'deki basit kalıplara ek olarak, olumsuzlama (negasyon) içeren
kalıplar ekliyoruz: "kötü değil" -> pozitif, "iyi değil" -> negatif.
Amaç: modele "değil" kelimesinin anlamı tersine çevirdiğini öğretmek.
"""

import csv
import random

subjects = ["film", "kitap", "restoran", "otel", "oyun", "dizi", "telefon", "uygulama",
            "konser", "ürün", "kurs", "hizmet", "şarkı", "müze", "kafe"]

pos_phrases = [
    "gerçekten harika",
    "çok kaliteli",
    "son derece etkileyici",
    "beklediğimden çok daha iyi",
    "tam bir mükemmellik",
    "inanılmaz derecede başarılı",
    "gayet güzel ve tatmin edici",
    "harika ötesi bir deneyim",
]
neg_phrases = [
    "gerçekten kötü",
    "çok kalitesiz",
    "son derece sıkıcı",
    "beklediğimden çok daha kötü",
    "tam bir hayal kırıklığı",
    "inanılmaz derecede başarısız",
    "berbat ve tatmin etmeyen",
    "kötü ötesi bir deneyim",
]

# Olumsuzlama kalıpları: "değil" kelimesi anlamı çeviriyor
pos_negation_phrases = [
    "hiç kötü değil",
    "kötü değil, gayet iyi",
    "berbat değil, oldukça başarılı",
]
neg_negation_phrases = [
    "hiç iyi değil",
    "iyi değil, oldukça kötü",
    "harika değil, oldukça sıkıcı",
]

random.seed(42)
rows = []
for subj in subjects:
    for phrase in pos_phrases:
        rows.append((f"Bu {subj} {phrase}.", "pozitif"))
    for phrase in neg_phrases:
        rows.append((f"Bu {subj} {phrase}.", "negatif"))
    for phrase in pos_negation_phrases:
        rows.append((f"Bu {subj} {phrase}.", "pozitif"))
    for phrase in neg_negation_phrases:
        rows.append((f"Bu {subj} {phrase}.", "negatif"))

random.shuffle(rows)

with open("reviews.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"Toplam {len(rows)} örnek yazıldı -> reviews.csv")
