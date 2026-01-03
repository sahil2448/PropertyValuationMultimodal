# src/multimodal_train.py
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import r2_score, root_mean_squared_error

from config import PROCESSED_DATA_DIR
from preprocessing import DataPreprocessor


class MultiModalDataset(Dataset):
    def __init__(self, X_tab, y, image_paths, img_tfms):
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y.reshape(-1), dtype=torch.float32)
        self.image_paths = image_paths
        self.img_tfms = img_tfms

    def __len__(self):
        return len(self.X_tab)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.img_tfms(img)
        tab = self.X_tab[idx]
        if self.y is None:
            return img, tab
        return img, tab, self.y[idx]


class MultiModalRegressor(nn.Module):
    def __init__(self, tab_dim: int):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.img_head = nn.Sequential(
            nn.Linear(num_feats, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.tab_head = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, img, tab):
        img_feat = self.backbone(img)          # (B, 512)
        img_emb = self.img_head(img_feat)      # (B, 256)
        tab_emb = self.tab_head(tab)           # (B, 64)
        x = torch.cat([img_emb, tab_emb], dim=1)
        out = self.regressor(x).squeeze(1)     # (B,)
        return out



def run_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"train epoch {epoch}", total=len(loader))
    for img, tab, y in pbar:
        img = img.to(device)
        tab = tab.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(img, tab)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)
        pbar.set_postfix(loss=float(loss.item()))

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds = []
    ys = []
    for img, tab, y in loader:
        img = img.to(device)
        tab = tab.to(device)
        pred = model(img, tab).cpu().numpy()
        preds.append(pred)
        ys.append(y.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def main():
    device = torch.device("cpu")
    print("Device:", device)

    data = np.load(PROCESSED_DATA_DIR / "train_processed.npz", allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"].reshape(-1)   # scaled
    imgs_train = data["images_train"]
    X_val = data["X_val"]
    y_val = data["y_val"].reshape(-1)       # scaled
    imgs_val = data["images_val"]

    # ---- FAST DEV MODE (CPU) ----
    FAST_DEV = True
    N_FAST_TRAIN = 2000
    N_FAST_VAL = 500

    if FAST_DEV:
        X_train = X_train[:N_FAST_TRAIN]
        y_train = y_train[:N_FAST_TRAIN]
        imgs_train = imgs_train[:N_FAST_TRAIN]

        X_val = X_val[:N_FAST_VAL]
        y_val = y_val[:N_FAST_VAL]
        imgs_val = imgs_val[:N_FAST_VAL]
    # -----------------------------


    tab_dim = X_train.shape[1]

    weights = models.ResNet18_Weights.DEFAULT
    img_tfms = weights.transforms()

    train_ds = MultiModalDataset(X_train, y_train, imgs_train, img_tfms)
    val_ds = MultiModalDataset(X_val, y_val, imgs_val, img_tfms)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0) 

    model = MultiModalRegressor(tab_dim=tab_dim).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-3)

    prep = DataPreprocessor()
    prep.load_preprocessor()

    best_rmse = float("inf")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = 5 
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, epoch)
        pred_val_scaled, y_val_scaled = eval_epoch(model, val_loader, device)

        pred_val_price = prep.target_scaler.inverse_transform(pred_val_scaled.reshape(-1, 1)).reshape(-1)
        y_val_price = prep.target_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).reshape(-1)

        rmse = root_mean_squared_error(y_val_price, pred_val_price)
        r2 = r2_score(y_val_price, pred_val_price)

        print(f"Epoch {epoch}/{epochs} | train_mse={train_loss:.4f} | val_RMSE={rmse:.2f} | val_R2={r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), out_dir / "multimodal_best.pt")
            print(f"  saved best -> {out_dir / 'multimodal_best.pt'}")

    print("Best val RMSE:", best_rmse)


if __name__ == "__main__":
    main()
