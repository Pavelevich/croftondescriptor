"""Texture-ceiling baseline: a small CNN trained on the raw cell-image crops.

This is the upper-bound comparison for the thesis table — it can use interior
texture/intensity that the shape descriptor deliberately ignores. Comparing
Crofton-FFT (interpretable, rotation-invariant by construction, ~23 dims) to a
CNN on pixels shows how much (if anything) shape-only gives up, and at what
cost in parameters / interpretability / rotation robustness.

Runs on the RTX 5070 if a CUDA-enabled torch is installed; falls back to CPU
(the crops are tiny so CPU is fine for 625 images).

Used by experiments.py when torch is available and the dataset provides per-cell
image crops (not just contours).
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:                      # torch not installed
    TORCH_OK = False

IMG = 64   # crops resized to IMG x IMG


class SmallCNN(nn.Module if TORCH_OK else object):
    """Compact CNN: 3 conv blocks + GAP + linear. ~120k params."""
    def __init__(self, n_classes):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm2d(16), nn.BatchNorm2d(32), nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.c2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.c3(x))), 2)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def _prep(imgs):
    """list of HxWx3 BGR uint8 -> NCHW float tensor in [0,1], resized to IMG."""
    import cv2
    arr = np.stack([cv2.resize(im, (IMG, IMG)) for im in imgs]).astype(np.float32) / 255.0
    arr = arr[..., ::-1].copy()                       # BGR->RGB
    return torch.from_numpy(arr).permute(0, 3, 1, 2)  # NCHW


def device():
    return torch.device("cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu")


def train_eval_cv(images, y, classes, folds=5, epochs=25, seed=0, rotate_test_deg=None):
    """Stratified-CV accuracy/F1 of the CNN on raw crops.

    images: list of HxWx3 BGR arrays. y: label array. Returns dict with acc,
    macro-F1, predictions, device name, param count, inference ms/cell.
    If rotate_test_deg is set, test crops are rotated by that angle (rotation
    robustness for the CNN baseline)."""
    if not TORCH_OK:
        return {"error": "torch not installed"}
    import cv2, time
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score

    dev = device()
    cls_to_i = {c: i for i, c in enumerate(classes)}
    yi = np.array([cls_to_i[c] for c in y])
    X = _prep(images).to(dev)
    yt = torch.tensor(yi, device=dev)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=int)
    n_params = 0
    inf_ms = []
    g = torch.Generator().manual_seed(seed)
    for tr, te in skf.split(np.zeros(len(y)), yi):
        torch.manual_seed(seed)
        net = SmallCNN(len(classes)).to(dev)
        n_params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()
        tr_t = torch.tensor(tr, device=dev)
        net.train()
        for ep in range(epochs):
            perm = tr_t[torch.randperm(len(tr_t), generator=g, device="cpu").to(dev)]
            for i in range(0, len(perm), 32):
                idx = perm[i:i + 32]
                # light on-the-fly augmentation: random hflip
                xb = X[idx]
                if torch.rand(1, generator=g).item() < 0.5:
                    xb = torch.flip(xb, dims=[3])
                opt.zero_grad()
                loss = lossf(net(xb), yt[idx])
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            if rotate_test_deg:
                te_imgs = []
                for k in te:
                    h, w = images[k].shape[:2]
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_test_deg, 1.0)
                    te_imgs.append(cv2.warpAffine(images[k], M, (w, h), borderValue=(255, 255, 255)))
                Xte = _prep(te_imgs).to(dev)
            else:
                Xte = X[torch.tensor(te, device=dev)]
            t0 = time.perf_counter()
            out = net(Xte).argmax(1).cpu().numpy()
            inf_ms.append((time.perf_counter() - t0) * 1000 / len(te))
        preds[te] = out

    pred_lbl = np.array([classes[i] for i in preds])
    return {
        "acc": accuracy_score(y, pred_lbl),
        "f1": f1_score(y, pred_lbl, average="macro"),
        "pred": pred_lbl,
        "device": str(dev),
        "params": n_params,
        "inf_ms": float(np.mean(inf_ms)),
    }
