"""ADIM 16 - ortak yardımcılar: görülmemiş aday havuzu ve sızıntı kontrolleri."""

import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "step15_aspect"))

from baseline import turkish_lower  # noqa: E402
from prepare_explore import MAX_WORDS, MIN_WORDS, unseen_pool  # noqa: E402

LABEL_SET = os.path.join(HERE, "label_set16.csv")


def norm(text):
    """Neredeyse aynı yorumları yakalamak için: küçük harf + boşlukları tek boşluğa indir."""
    return re.sub(r"\s+", " ", turkish_lower(text)).strip()


def step15_sets():
    explore = pd.read_csv(os.path.join(ROOT, "step15_aspect", "explore_300.csv"))
    val_test = pd.read_csv(os.path.join(ROOT, "step15_aspect", "label_set.csv"))
    return explore, val_test


def candidate_pool(extra_exclude=()):
    """8-40 kelimelik, hiçbir adımda görülmemiş yorumlar (+ dışarıda tutulan metin kümesi)."""
    explore, val_test = step15_sets()
    unseen, exclude = unseen_pool(extra_exclude=set(explore["text"]) | set(val_test["text"]) | set(extra_exclude))
    n_words = unseen["text"].str.split().str.len()
    cand = unseen[(n_words >= MIN_WORDS) & (n_words <= MAX_WORDS)]
    exclude_norm = {norm(t) for t in exclude}
    cand = cand[~cand["text"].map(norm).isin(exclude_norm)]
    cand = cand[~cand["text"].map(norm).duplicated()]
    return cand, exclude, exclude_norm


def check_no_leak(chosen, exclude, exclude_norm):
    """Yeni seçilenler görülmüş hiçbir metinle (val/test dahil) çakışmamalı; kendi içinde de tekil olmalı."""
    explore, val_test = step15_sets()
    hard = pd.read_csv(os.path.join(ROOT, "data", "hard_test.csv"))
    real_test = pd.read_csv(os.path.join(ROOT, "data", "real", "test.csv"))
    texts, norms = set(chosen["text"]), set(chosen["text"].map(norm))
    assert not texts & exclude, "daha önce görülmüş metin var"
    assert not norms & exclude_norm, "normalleştirince görülmüş bir metinle aynı olan var"
    for name, df in [("Adım 15 val/test", val_test), ("explore_300", explore), ("hard_test", hard),
                     ("data/real/test", real_test)]:
        assert not norms & set(df["text"].map(norm)), f"{name} ile çakışma var"
    assert chosen["text"].map(norm).is_unique, "kendi içinde tekrar eden metin var"
    assert chosen["id"].is_unique
    for a in ("train", "val"):
        for b in ("train", "val"):
            if a < b:
                assert not set(chosen.loc[chosen["split"] == a, "text"].map(norm)) & \
                           set(chosen.loc[chosen["split"] == b, "text"].map(norm))
