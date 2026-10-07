"""
Konu ve konu-duygu modellerinin TEK TOHUMLU sürümlerini Hugging Face'e GİZLİ (private) depo olarak yükler.
MAC MİNİ'DE çalışır (modeller orada). Önce Görkan kendisi giriş yapmalı: `.venv/bin/hf auth login`.
  python upload_hf.py          -> neyin yükleneceğini gösterir (kuru çalıştırma)
  python upload_hf.py --yukle  -> gerçekten yükler
Herkese açmak ayrı bir karardır (Hugging Face'te depo ayarlarından); bu script açmaz.
"""

import os
import sys

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
JOBS = [("Urartu65/gorkanai-tr-aspect-topic", "step19_topic_v2/model_B_s0_e16", "hf_cards/topic_README.md"),
        ("Urartu65/gorkanai-tr-aspect-sentiment", "step17_aspect_sentiment/model_v2b_s0_e8", "hf_cards/sentiment_README.md")]

api = HfApi()
print("giriş:", api.whoami()["name"])
for repo, folder, card in JOBS:
    path = os.path.join(ROOT, folder)
    files = sorted(os.listdir(path))
    size = sum(os.path.getsize(os.path.join(path, f)) for f in files) / 1e6
    print(f"{repo}  <-  {folder} ({size:.0f} MB: {files}) + {card} (README.md)")
    if "--yukle" in sys.argv:
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=path, commit_message="tek tohumlu sürüm")
        api.upload_file(repo_id=repo, path_or_fileobj=os.path.join(HERE, card), path_in_repo="README.md",
                        commit_message="model kartı")
        print("  yüklendi (gizli):", f"https://huggingface.co/{repo}")
if "--yukle" not in sys.argv:
    print("kuru çalıştırma — yüklemek için: python upload_hf.py --yukle")
