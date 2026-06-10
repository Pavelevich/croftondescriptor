"""Stage (a): multi-cell instance segmentation.

The base pipeline (crofton_gpu.extract_contour) returns only the single
largest contour. A blood smear has many cells per field, so for classification
we need EACH cell as its own contour. This splits the foreground mask (reusing
the proven HSV+top-hat / Otsu masks from crofton_gpu) into instances with a
distance-transform + watershed, which separates touching/overlapping cells.

OpenCV + numpy only — no new dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crofton_gpu


def _foreground_mask(img_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    """Binary cell mask. Try the stain-specific HSV+top-hat first; if it covers
    too little, fall back to Otsu on grayscale (foreground = minority class)."""
    mask = crofton_gpu._mask_hsv_tophat(img_bgr)
    frac = np.count_nonzero(mask) / mask.size
    if 0.005 < frac < 0.95:
        return mask, "hsv+tophat"
    return crofton_gpu._mask_otsu(img_bgr), "otsu"


def segment_cells(img_bgr: np.ndarray,
                  min_area_frac: float = 2e-4,
                  max_area_frac: float = 0.25,
                  min_circularity: float = 0.15,
                  fg_ratio: float = 0.45) -> dict:
    """Return {contours: list[(N,2) int32], markers, method}.

    Each contour is one cell instance. Filters drop debris (too small),
    merged blobs / the whole-field background (too large) and linear streaks
    (very low circularity). NOTE the circularity floor is kept low (0.15) on
    purpose so genuinely non-round cells — sickle/drepanocyte, teardrop,
    elliptocyte — are NOT rejected. Tunables are fractions of the image area.
    """
    h, w = img_bgr.shape[:2]
    img_area = h * w
    min_area = max(min_area_frac * img_area, 20.0)
    max_area = max_area_frac * img_area

    mask, method = _foreground_mask(img_bgr)

    # clean speckle
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)

    # sure background by dilation
    sure_bg = cv2.dilate(mask, k3, iterations=3)

    # sure foreground = peaks of the distance transform (cell interiors).
    # Threshold PER CONNECTED COMPONENT using each blob's own distance max, so
    # small/thin cells (elliptocyte, sickle) aren't lost to a global threshold
    # dominated by the largest cell.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    n_comp, comp = cv2.connectedComponents(mask)
    sure_fg = np.zeros(mask.shape, np.uint8)
    for lbl in range(1, n_comp):
        blob = comp == lbl
        local_max = dist[blob].max()
        if local_max <= 0:
            continue
        sure_fg[blob & (dist > fg_ratio * local_max)] = 255

    unknown = cv2.subtract(sure_bg, sure_fg)

    # one marker per sure-fg component; watershed floods the unknown band
    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1               # background must be 1, not 0
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_bgr, markers)

    contours = []
    for label in range(2, n_markers + 1):   # 1 = background, 2.. = cells
        cell = np.uint8(markers == label) * 255
        if cv2.countNonZero(cell) == 0:
            continue
        cnts, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perim ** 2) if perim > 1e-6 else 0.0
        if circ < min_circularity:
            continue
        contours.append(c.reshape(-1, 2).astype(np.int32))

    return {"contours": contours, "markers": markers, "method": method}


def draw_instances(img_bgr: np.ndarray, contours: list, labels=None,
                   palette=None) -> np.ndarray:
    """Overlay each cell contour in a distinct color (or a per-class color)."""
    out = img_bgr.copy()
    lw = max(2, round(max(out.shape[:2]) / 800))
    default = [(0, 255, 0), (0, 200, 255), (255, 120, 0), (255, 0, 200),
               (0, 255, 255), (180, 255, 0), (120, 0, 255)]
    for i, c in enumerate(contours):
        if labels is not None and palette is not None:
            color = palette.get(labels[i], (200, 200, 200))
        else:
            color = default[i % len(default)]
        cv2.drawContours(out, [c.reshape(-1, 1, 2)], -1, color, lw)
    return out


if __name__ == "__main__":
    # quick visual smoke test over the sample images
    root = Path(__file__).resolve().parents[2]
    samples = list((root / "apple_silicon_version" / "resources" / "sample_images").glob("c*.jpg"))
    samples += [root / "resources" / "e1.jpeg"]
    for p in samples:
        img = cv2.imread(str(p))
        if img is None:
            continue
        r = segment_cells(img)
        print(f"{p.name}: {len(r['contours'])} cells  (mask={r['method']})")
        out = draw_instances(img, r["contours"])
        cv2.imwrite(f"/tmp/seg_{p.stem}.png", out)
