"""Rotation- and scale-invariant shape features for cell classification.

Stage (b) of the classification pipeline. Turns the GPU Crofton signature
C(phi) + the cell contour into a compact, INTERPRETABLE feature vector that is
invariant to rotation and scale BY CONSTRUCTION (not by training augmentation):

  * scale     -> divide every length by the cell diameter
  * rotation  -> a rotation cyclically shifts C(phi); the magnitude of its
                 Fourier spectrum |FFT(C)| is therefore invariant (shift theorem)

Depends only on numpy + OpenCV (already in the venv) — no scikit-learn/torch.
The returned feature *names* travel with the vector so the classifier's
feature-importance can be shown to the user (explainability card).
"""

from __future__ import annotations

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 1) Crofton FFT harmonics  (rotation- + scale-invariant directional shape)
# --------------------------------------------------------------------------

def crofton_fft_features(curve: np.ndarray, diameter: float,
                         n_harmonics: int = 7) -> np.ndarray:
    """Magnitude spectrum of the half-period Crofton signature.

    curve: 361-sample C(phi). The Crofton crossing count is antipodally
    symmetric (a line at phi and phi+180 is the same line), so we fold to the
    [0,180) half-period to halve variance.
    Returns harmonics H[1..n_harmonics-1] normalized by H[0] (drops scale/DC),
    so the result is invariant to rotation (shift theorem) and scale.

    Only LOW-order harmonics are kept (default H1..H6): they carry the shape
    biology (H2 = elongation, H5 = 5-fold lobing) while high harmonics of a
    1°-sampled curve are dominated by discretization noise.
    """
    curve = np.asarray(curve, dtype=np.float64)
    half = curve[:180]
    if diameter and diameter > 1e-9:
        half = half / diameter
    spectrum = np.abs(np.fft.rfft(half))           # rotation-invariant magnitude
    spectrum = spectrum[:n_harmonics]
    dc = spectrum[0] if spectrum[0] > 1e-12 else 1.0
    return (spectrum[1:] / dc).astype(np.float64)  # length n_harmonics-1


# --------------------------------------------------------------------------
# 2) Radial signature harmonics  (asymmetry: teardrop tail, single bump)
# --------------------------------------------------------------------------

def radial_fft_features(contour: np.ndarray, n_harmonics: int = 6) -> np.ndarray:
    """Magnitude spectrum of the centroid-distance signature r(t).

    Captures asymmetry the antipodally-symmetric Crofton curve cannot:
    H[1] is a single off-center bump (teardrop), H[k] a k-fold lobing.
    Rotation-invariant (magnitude) and scale-invariant (normalized by DC).
    """
    pts = np.asarray(contour, dtype=np.float64)
    c = pts.mean(axis=0)
    r = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    spectrum = np.abs(np.fft.rfft(r))
    spectrum = spectrum[:n_harmonics]
    dc = spectrum[0] if spectrum[0] > 1e-12 else 1.0
    return (spectrum[1:] / dc).astype(np.float64)  # length n_harmonics-1


# --------------------------------------------------------------------------
# 3) Scalar morphometrics  (the literature's RBC contour feature set)
# --------------------------------------------------------------------------

def _exact_diameter(pts: np.ndarray) -> float:
    diff = pts[:, None, :] - pts[None, :, :]
    return float(np.sqrt((diff ** 2).sum(-1)).max())


def _curvature_peaks(pts: np.ndarray, smooth: int = 5):
    """(#spicules, regularity_variance) from discrete boundary curvature.

    Counts local maxima of turning angle around the closed contour; regularity
    is the normalized variance of the spacing between consecutive peaks
    (low -> echinocyte/regular burrs, high -> acanthocyte/irregular spurs).
    """
    n = len(pts)
    if n < 8:
        return 0, 0.0
    nxt = np.roll(pts, -1, axis=0)
    edge = nxt - pts
    ang = np.arctan2(edge[:, 1], edge[:, 0])
    dtheta = np.diff(ang, append=ang[:1])
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi          # wrap to (-pi, pi]
    k = np.abs(dtheta)
    if smooth > 1:                                            # circular moving average
        kernel = np.ones(smooth) / smooth
        k = np.convolve(np.concatenate([k[-smooth:], k, k[:smooth]]), kernel, "same")
        k = k[smooth:-smooth]
    thr = k.mean() + k.std()
    prev, nxtk = np.roll(k, 1), np.roll(k, -1)
    peaks = np.where((k > prev) & (k >= nxtk) & (k > thr))[0]
    if len(peaks) < 2:
        return int(len(peaks)), 0.0
    gaps = np.diff(np.concatenate([peaks, [peaks[0] + n]]))
    regularity = float(np.var(gaps) / (np.mean(gaps) ** 2 + 1e-9))
    return int(len(peaks)), regularity


def scalar_morphometrics(contour: np.ndarray, curve: np.ndarray | None = None,
                         diameter: float | None = None) -> dict:
    """Dimensionless shape scalars computed from the contour (+ optional curve).

    contour: (N,2) float array of boundary points (centered or not — all
    features are translation/rotation/scale invariant ratios).
    """
    pts = np.asarray(contour, dtype=np.float64)
    if diameter is None:
        diameter = _exact_diameter(pts)
    diameter = max(diameter, 1e-9)

    cnt = pts.astype(np.float32).reshape(-1, 1, 2)
    cnt_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)

    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 1e-9 else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    hull_perim = float(cv2.arcLength(hull, True))
    solidity = area / hull_area if hull_area > 1e-9 else 0.0
    convexity = hull_perim / perimeter if perimeter > 1e-9 else 0.0

    # ellipse fit -> elongation
    aspect, eccentricity = 1.0, 0.0
    if len(pts) >= 5:
        (_, _), (ax1, ax2), _ = cv2.fitEllipse(cnt)
        major, minor = max(ax1, ax2), min(ax1, ax2)
        if major > 1e-9:
            aspect = minor / major
            eccentricity = float(np.sqrt(max(0.0, 1 - (minor / major) ** 2)))

    # deepest concavity (crescent of a sickle, bite of a keratocyte)
    max_defect, mean_defect = 0.0, 0.0
    try:
        hull_idx = cv2.convexHull(cnt_i, returnPoints=False)
        if hull_idx is not None and len(hull_idx) > 3:
            defects = cv2.convexityDefects(cnt_i, hull_idx)
            if defects is not None:
                depths = defects[:, 0, 3] / 256.0          # cv2 fixed-point depth
                max_defect = float(depths.max()) / diameter
                mean_defect = float(depths.mean()) / diameter
    except cv2.error:
        pass

    n_spicules, spicule_regularity = _curvature_peaks(pts)

    feats = {
        "circularity": circularity,
        "aspect_ratio": aspect,
        "eccentricity": eccentricity,
        "solidity": solidity,
        "convexity": convexity,
        "max_defect_depth": max_defect,
        "mean_defect_depth": mean_defect,
        "area_norm": area / (diameter ** 2),
        "perimeter_norm": perimeter / diameter,
        "n_spicules": float(n_spicules),
        "spicule_regularity": spicule_regularity,
    }
    if curve is not None:
        c = np.asarray(curve, dtype=np.float64)
        feats["crofton_width_ratio"] = float(c[:180].min() / (c[:180].max() + 1e-9))
    return feats


# --------------------------------------------------------------------------
# 4) Assembled feature vector (+ parallel names for explainability)
# --------------------------------------------------------------------------

def feature_vector(contour: np.ndarray, curve: np.ndarray, diameter: float,
                   n_crofton: int = 7, n_radial: int = 6):
    """Return (vector float64, names list[str]) — the function every later
    pipeline stage (classifier, anomaly detector, UI card) imports."""
    h_crofton = crofton_fft_features(curve, diameter, n_crofton)
    h_radial = radial_fft_features(contour, n_radial)
    scalars = scalar_morphometrics(contour, curve, diameter)

    names = ([f"crofton_H{k}" for k in range(1, n_crofton)]
             + [f"radial_H{k}" for k in range(1, n_radial)]
             + list(scalars.keys()))
    vec = np.concatenate([h_crofton, h_radial,
                          np.array(list(scalars.values()), dtype=np.float64)])
    return vec, names
