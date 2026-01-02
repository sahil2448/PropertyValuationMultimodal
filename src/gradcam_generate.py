# src/gradcam_generate.py
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

import matplotlib.pyplot as plt


class MultiModalRegressor(nn.Module):
    """Must match src/multimodal_train.py architecture exactly."""
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
        img_feat = self.backbone(img)         # (B,512)
        img_emb = self.img_head(img_feat)     # (B,256)
        tab_emb = self.tab_head(tab)          # (B,64)
        x = torch.cat([img_emb, tab_emb], dim=1)
        out = self.regressor(x).squeeze(1)    # (B,)
        return out


def compute_gradcam(model, target_layer, img_tensor, tab_tensor):
    """
    Grad-CAM for regression:
    - forward pass => scalar output
    - backward on that scalar
    - weights = GAP over gradients
    - cam = ReLU(sum_k w_k * A_k)
    [web:472][web:475]
    """
    img_tensor = img_tensor.requires_grad_(True)
    tab_tensor = tab_tensor.requires_grad_(True)

    activations = None
    gradients = None

    def fwd_hook(_, __, output):
        nonlocal activations
        activations = output

    def bwd_hook(_, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)

    pred = model(img_tensor, tab_tensor)   # (1,)
    score = pred.sum()                     # scalar
    score.backward()

    h1.remove()
    h2.remove()

    w = gradients.mean(dim=(2, 3), keepdim=True)          # (1,C,1,1) [web:472]
    cam = (w * activations).sum(dim=1, keepdim=True)      # (1,1,H,W)
    cam = F.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return float(pred.item()), cam


def overlay_and_save(pil_img, cam, out_path):
    w, h = pil_img.size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)

    plt.figure(figsize=(5, 5))
    plt.imshow(pil_img)
    plt.imshow(cam_img, cmap="jet", alpha=0.40)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    device = torch.device("cpu")

    out_dir = Path("outputs/gradcam")
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load("data/processed/train_processed.npz", allow_pickle=True)
    X_val = npz["X_val"]
    imgs_val = npz["images_val"]
    y_val_scaled = npz["y_val"].reshape(-1)

    tab_dim = X_val.shape[1]

    model = MultiModalRegressor(tab_dim=tab_dim).to(device)
    ckpt = Path("outputs/multimodal_best.pt")
    if not ckpt.exists():
        raise FileNotFoundError("Missing outputs/multimodal_best.pt. Run multimodal_train.py once to create it.")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    target_layer = model.backbone.layer4[-1].conv2

    weights = models.ResNet18_Weights.DEFAULT
    tfms = weights.transforms()

    import pickle
    with open("data/processed/preprocessor.pkl", "rb") as f:
        prep = pickle.load(f)
    target_scaler = prep["target_scaler"]
    y_val_price = target_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).reshape(-1)

    k = 5
    low_idx = np.argsort(y_val_price)[:k]
    high_idx = np.argsort(y_val_price)[-k:]

    chosen = list(low_idx) + list(high_idx)

    for rank, idx in enumerate(chosen):
        img_path = imgs_val[idx]
        pil_img = Image.open(img_path).convert("RGB")

        img_tensor = tfms(pil_img).unsqueeze(0).to(device)
        tab_tensor = torch.tensor(X_val[idx], dtype=torch.float32).unsqueeze(0).to(device)

        pred_scaled, cam = compute_gradcam(model, target_layer, img_tensor, tab_tensor)
        pred_price = target_scaler.inverse_transform(np.array([[pred_scaled]])).item()
        true_price = float(y_val_price[idx])

        out_path = out_dir / f"{rank:02d}_true_{true_price:.0f}_pred_{pred_price:.0f}.png"
        overlay_and_save(pil_img, cam, out_path)
        print("Saved", out_path)

    print("Done. Check outputs/gradcam/ for overlays.")


if __name__ == "__main__":
    main()
