# src/extract_embeddings.py
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from preprocessing import DataPreprocessor


def embed_images(image_paths, batch_size=64):
    device = torch.device("cpu")

    weights = models.ResNet18_Weights.DEFAULT
    tfms = weights.transforms()

    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    embs = []
    with torch.no_grad():  
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding"):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(tfms(img))
            x = torch.stack(imgs, dim=0).to(device)        # (B,3,224,224)
            feat = model(x).cpu().numpy()                  # (B,512)
            embs.append(feat)

    return np.concatenate(embs, axis=0)


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    npz = np.load(PROCESSED_DATA_DIR / "train_processed.npz", allow_pickle=True)
    imgs_train = npz["images_train"]
    imgs_val = npz["images_val"]

    emb_train = embed_images(imgs_train, batch_size=64)
    emb_val = embed_images(imgs_val, batch_size=64)

    np.save(PROCESSED_DATA_DIR / "img_emb_train.npy", emb_train)
    np.save(PROCESSED_DATA_DIR / "img_emb_val.npy", emb_val)

    prep = DataPreprocessor()
    prep.load_preprocessor()

    test_df = prep.load_data("test")
    test_df = prep.clean_data(test_df)
    test_df = prep.engineer_features(test_df)

    X_test_tab, _, imgs_test = prep.prepare_features(test_df, is_training=False)

    np.save(PROCESSED_DATA_DIR / "X_test_tab.npy", X_test_tab)
    np.save(PROCESSED_DATA_DIR / "test_ids.npy", test_df["id"].values)

    emb_test = embed_images(imgs_test, batch_size=64)
    np.save(PROCESSED_DATA_DIR / "img_emb_test.npy", emb_test)

    print("Saved embeddings + test tabular arrays to data/processed/")


if __name__ == "__main__":
    main()
