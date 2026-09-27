"""
Etiketsiz (label'sız) bir korpüs üretir: data/unlabeled_corpus.txt

Word2Vec ETİKET görmez, sadece ham metin ister. Öğrendiği şey:
"hangi kelimeler benzer bağlamlarda (benzer komşu kelimelerle) geçiyor?"

Buradaki fikir: "iyi", "güzel", "harika", "keyifli" gibi kelimeleri hep
AYNI kalıplarda kullanarak, Word2Vec'in bunları birbirine yakın vektörlere
yerleştirmesini sağlıyoruz (distributional hypothesis). Aynı şey olumsuz
kelimeler (kötü, berbat, yetersiz, sıkıcı...) için de geçerli.

Not: Hem kelimenin kök hali (iyi) hem çekimli hali (iyiydi) dahil ediliyor
-> gerçek dünyada büyük korpuslardan bunu bedavaya alırız, biz burada
küçük ölçekte elle simüle ediyoruz.
"""

subjects = ["film", "kitap", "restoran", "otel", "oyun", "dizi", "telefon", "uygulama",
            "konser", "ürün", "kurs", "hizmet", "şarkı", "müze", "kafe", "tatil",
            "yemek", "kahve", "kamera", "bilgisayar"]

pos_words = [
    "iyi", "iyiydi", "güzel", "güzeldi", "harika", "harikaydı", "mükemmel", "mükemmeldi",
    "kaliteli", "başarılı", "başarılıydı", "etkileyici", "keyifli", "keyifliydi",
    "verimli", "verimliydi", "hoş", "hoştu", "rahat", "tatmin edici", "muhteşem",
    "akıcı", "akıcıydı", "şaşırtıcı derecede iyi",
]
neg_words = [
    "kötü", "kötüydü", "berbat", "berbattı", "kalitesiz", "başarısız", "başarısızdı",
    "sıkıcı", "sıkıcıydı", "yetersiz", "yetersizdi", "rezalet", "vasat",
    "hayal kırıklığı", "hayal kırıklığıydı", "değersiz", "can sıkıcı",
    "içler acısı", "beklentimin altında", "izlemeye değmez",
]

templates = [
    "Bu {subj} çok {word}.",
    "Bu {subj} gerçekten {word}.",
    "{subj} bence {word}.",
    "Bu {subj} son derece {word}.",
    "Herkes bu {subj} için {word} diyor.",
]

lines = []
for subj in subjects:
    for word in pos_words + neg_words:
        for template in templates:
            lines.append(template.format(subj=subj, word=word))

with open("unlabeled_corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Toplam {len(lines)} etiketsiz cümle yazıldı -> unlabeled_corpus.txt")
