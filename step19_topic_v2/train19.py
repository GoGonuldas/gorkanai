"""
ADIM 19 (3) - Konu modeli v2: Adım 16 tarifi AYNEN (train_model/predict step16_topic_bert/train.py'den), tek değişken veri.

Kullanım: python train19.py A   (eğitim = origin != adim19_new, 1100)
          python train19.py B   (eğitim = hepsi, 1400)
Her biri: 3 tohum (0, 1, 2), epoch adayları 5/8/12/16 (PLAN §2). Her adayda val 300 olasılıkları + model kaydı.
Val = Adım 16 val 300 (load_data), aynen. Bu script testlere (Adım 16/18) DOKUNMAZ.
Çıktı: val_probs_{A,B}.npz, model_{A,B}_s{seed}_e{epoch}/ (git'e girmez), log_train_{A,B}.txt (stdout yönlendir).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step16_topic_bert"))

from common import norm  # noqa: E402
from train import ASPECTS, DEVICE, SEEDS, load_data, predict, targets, train_model  # noqa: E402

EPOCHS = (5, 8, 12, 16)
TRAIN19 = os.path.join(ROOT, "data", "aspect_labels_step19", "train19.csv")

if __name__ == "__main__":
    config = sys.argv[1]
    assert config in ("A", "B")
    train = pd.read_csv(TRAIN19, keep_default_na=False)
    if config == "A":
        train = train[train["origin"] != "adim19_new"].reset_index(drop=True)
    _, val = load_data()
    assert not set(train["text"].map(norm)) & set(val["text"].map(norm)), "eğitim-val sızıntısı"
    print(f"Yapılandırma {config} | eğitim {len(train)} {train['origin'].value_counts().to_dict()} | val {len(val)} | cihaz {DEVICE}")
    print("Eğitimde konu başına örnek:", dict(zip(ASPECTS, targets(train).sum(0).astype(int).tolist())), flush=True)
    probs = {}
    for seed in SEEDS:
        def save(epoch, model, tokenizer, seed=seed):
            if epoch in EPOCHS:
                probs[f"s{seed}_e{epoch}"] = predict(model, tokenizer, val["text"].tolist())
                out = os.path.join(HERE, f"model_{config}_s{seed}_e{epoch}")
                model.save_pretrained(out)
                tokenizer.save_pretrained(out)
        train_model(train, seed, max(EPOCHS), save)
    np.savez(os.path.join(HERE, f"val_probs_{config}.npz"), ids=val["id"].to_numpy(), **probs)
    print(f"val olasılıkları -> val_probs_{config}.npz ; modeller -> model_{config}_s*_e*/")
