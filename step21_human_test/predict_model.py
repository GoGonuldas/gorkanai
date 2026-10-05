"""
ADIM 21 (3) - Modelin etiketleri: dondurulmuş UYGULAMA hattı (step9_app/aspect_pipeline.py, 3 tohum), bir kez.
MAC MİNİ'DE çalışır. Çıktı: data/aspect_labels_step21/human_model.csv (id, raw — Görkan'ınkiyle aynı biçim)
                           + human_model_probs.npz (konu olasılıkları, konu başına P(pozitif)).
Model konu başına nötr üretmez (bilinen sınırlama); bu script hiçbir ayar içermez.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step9_app"))
from aspect_pipeline import ASPECTS, AspectPipeline  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "step15_aspect"))
from save_aspect_labels import ASPECTS as LETTER2NAME, parse  # noqa: E402

NAME2LETTER = {v: k for k, v in LETTER2NAME.items()}
OUT = os.path.join(ROOT, "data", "aspect_labels_step21", "human_model.csv")
assert not os.path.exists(OUT), "model etiketleri zaten var — tek çalıştırma"

texts = pd.read_csv(os.path.join(HERE, "human_set.csv"))
pipe = AspectPipeline()
p_topic = pipe.topic_probs(texts["text"].tolist())
p_sent = np.full(p_topic.shape, np.nan)
raws = []
for n, text in enumerate(texts["text"]):
    found = [a for k, a in enumerate(ASPECTS) if p_topic[n, k] >= pipe.threshold]
    if found:
        p_sent[n, [ASPECTS.index(a) for a in found]] = pipe.sentiment_probs(found, [text] * len(found))
    raws.append(",".join(NAME2LETTER[a] + ("p" if p_sent[n, ASPECTS.index(a)] >= 0.5 else "n") for a in found) or "0")
for r in raws:
    parse(r)
pd.DataFrame({"id": texts["id"], "raw": raws}).to_csv(OUT, index=False)
np.savez(os.path.join(HERE, "human_model_probs.npz"), ids=texts["id"].to_numpy(), topic=p_topic, sent_pos=p_sent)
n_pairs = sum(len(parse(r)[0]) for r in raws)
print(f"model: 100 yorum, {n_pairs} çift, konusuz {raws.count('0')}; eşik {pipe.threshold} -> {OUT}")
