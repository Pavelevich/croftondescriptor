"""Dataset adapters: turn a downloaded labeled cell dataset into
(contour, label) records the experiment/classifier can consume.

erythrocytesIDB ships per-cell BINARY MASKS (mask-circular / mask-elongated /
mask-other) alongside the cropped cell images. We extract the contour straight
from the mask — clean ground-truth boundaries, no re-segmentation needed (the
default HSV+top-hat mask is tuned for purple stain and would miss pink RBCs).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# erythrocytesIDB shape classes
IDB_CLASSES = ["circular", "elongated", "other"]


def _contour_from_mask(mask_gray: np.ndarray):
    _, binm = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    # masks may be white-cell-on-black or inverted; ensure cell = white minority
    if np.count_nonzero(binm) > binm.size * 0.5:
        binm = cv2.bitwise_not(binm)
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 30:
        return None
    return c.reshape(-1, 2).astype(np.float64)


def _contour_from_stained_crop(img_bgr: np.ndarray, min_area=120, max_frac=0.92):
    """Contour of a purple-stained cell crop via HSV-saturation Otsu (the stain
    is far more saturated than the pale background). Returns the largest
    plausible contour, or None for debris / whole-frame thresholds."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area or area > max_frac * img_bgr.shape[0] * img_bgr.shape[1]:
        return None
    return c.reshape(-1, 2).astype(np.float64)


def find_erythrocytesIDB(root: Path):
    """Return (records, stats) where records = list of (contour, label, image).

    `image` is the BGR cell crop (for the CNN baseline) or None when only a mask
    was available. Searches mask folders first, else the cropped image folders
    (segmented with HSV-saturation Otsu — the stain is highly saturated).
    """
    root = Path(root)
    records = []
    used_masks = 0
    used_images = 0

    for cls in IDB_CLASSES:
        mask_dirs = list(root.rglob(f"mask-{cls}")) + list(root.rglob(f"mask_{cls}"))
        for md in mask_dirs:
            for p in sorted(md.glob("*")):
                if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                    continue
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                c = _contour_from_mask(m)
                if c is not None and len(c) >= 10:
                    records.append((c, cls, None)); used_masks += 1

        if not mask_dirs:
            img_dirs = [d for d in root.rglob(cls) if d.is_dir() and "mask" not in d.name.lower()]
            for d in img_dirs:
                for p in sorted(d.glob("*")):
                    if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                        continue
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    c = _contour_from_stained_crop(img)
                    if c is not None and len(c) >= 10:
                        records.append((c, cls, img)); used_images += 1

    return records, {"from_masks": used_masks, "from_images": used_images}


# --------------------------------------------------------------------------
# Chula-RBC-12: smear images + per-cell coordinate labels (multi-class)
# --------------------------------------------------------------------------

CHULA_CLASSES = {
    0: "Normal", 1: "Macrocyte", 2: "Microcyte", 3: "Spherocyte",
    4: "Target", 5: "Stomatocyte", 6: "Ovalocyte", 7: "Teardrop",
    8: "Burr", 9: "Schistocyte", 11: "Hypochromia", 12: "Elliptocyte",
}


def is_chula(root: Path) -> bool:
    root = Path(root)
    return (root / "Dataset").is_dir() and (root / "Label").is_dir()


def _central_contour(crop_bgr, min_area=150):
    """Segment a crop and return the contour whose centroid is nearest the
    crop centre (the labeled cell), ignoring neighbour fragments."""
    h, w = crop_bgr.shape[:2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    s = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cx, cy = w / 2, h / 2
    best, bestd = None, 1e9
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        d = (M["m10"] / M["m00"] - cx) ** 2 + (M["m01"] / M["m00"] - cy) ** 2
        if d < bestd:
            bestd, best = d, c
    if best is None:
        return None
    return best.reshape(-1, 2).astype(np.float64)


def find_chula(root: Path, crop_half=48, max_per_class=300, min_per_class=25):
    """Return (records, stats), records = list of (contour, class_name, crop).

    Crops a window around each labeled (x,y), segments the central cell, and
    keeps it if a valid contour is found. Per-class capped (Chula is heavily
    imbalanced toward Normal) and rare classes dropped.
    """
    root = Path(root)
    ds, lab = root / "Dataset", root / "Label"
    from collections import defaultdict
    buckets = defaultdict(list)        # class_name -> [(contour, crop)]

    for lab_file in sorted(lab.glob("*.txt")):
        img_path = ds / (lab_file.stem + ".jpg")
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        for line in lab_file.read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x, y, cid = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            cls = CHULA_CLASSES.get(cid)
            if cls is None:
                continue
            if len(buckets[cls]) >= max_per_class:
                continue
            x0, y0 = x - crop_half, y - crop_half
            x1, y1 = x + crop_half, y + crop_half
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H:   # skip border-clipped cells
                continue
            crop = img[y0:y1, x0:x1]
            c = _central_contour(crop)
            if c is not None and len(c) >= 10:
                buckets[cls].append((c, crop))

    records, dropped = [], {}
    for cls, items in buckets.items():
        if len(items) < min_per_class:
            dropped[cls] = len(items)
            continue
        for c, crop in items:
            records.append((c, cls, crop))
    return records, {"kept_classes": sorted(set(r[1] for r in records)),
                     "dropped_rare": dropped,
                     "per_class": {k: len(v) for k, v in buckets.items()}}


if __name__ == "__main__":
    import sys
    from collections import Counter
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets")
    if is_chula(root):
        recs, stats = find_chula(root)
        print("Chula records:", len(recs))
        print("per class (sampled):", Counter(lbl for _, lbl, _ in recs))
        print("dropped rare:", stats["dropped_rare"])
    else:
        recs, stats = find_erythrocytesIDB(root)
        print("records:", len(recs), "| source:", stats)
        print("per class:", Counter(lbl for _, lbl, _ in recs))
