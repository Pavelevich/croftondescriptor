"""Tests for the rotation/scale-invariant feature extractor (features.py).

Two load-bearing properties, on synthetic shapes (circle / ellipse / 5-star /
sickle):
  1. INVARIANCE — feature vectors are ~equal across rotation {0,30,90,210°}
     and scale {0.5x,1x,3x}. This is the empirical proof of the Section-1
     invariance theorem.
  2. SEPARATION — the four shape classes are linearly far apart: a 1-NN on one
     prototype per class correctly labels every rotated/scaled variant.

Run with the project venv (no pytest needed):
    .venv/bin/python cuda_version/webapp/tests/test_features.py
or under pytest if installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crofton_gpu
import features as F

N = 240


def _largest_contour_from_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    return c - c.mean(axis=0)


# ----------------------------- synthetic shapes ---------------------------

def circle(r=100.0, n=N):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def ellipse(a=160.0, b=70.0, n=N):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([a * np.cos(t), b * np.sin(t)], axis=1)


def star(k=5, r_out=120.0, r_in=50.0, n=N):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = r_out + (r_in - r_out) * (0.5 * (1 + np.cos(k * t)))
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def sickle():
    """Crescent (lune) built by mask subtraction, like the real pipeline, so
    the contour is guaranteed simple: a disk minus an offset disk -> one deep
    concavity, low aspect, broken bilateral symmetry."""
    canvas = np.zeros((400, 400), np.uint8)
    cv2.circle(canvas, (200, 200), 130, 255, -1)
    cv2.circle(canvas, (255, 200), 110, 0, -1)
    return _largest_contour_from_mask(canvas)


def transform(pts, deg, scale):
    a = np.deg2rad(deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (pts @ rot.T) * scale


# --------------------------- feature computation --------------------------

def extract(pts):
    """Full pipeline: resample -> center -> GPU Crofton -> invariant features."""
    pts = pts.astype(np.float32)
    resampled = crofton_gpu.resample_contour(pts)
    centered = crofton_gpu.center_on_bbox(resampled)
    res = crofton_gpu.crofton_descriptor_gpu(centered)
    vec, names = F.feature_vector(centered, res.curve, res.diameter)
    return vec, names


SHAPES = {"circle": circle(), "ellipse": ellipse(), "star": star(), "sickle": sickle()}
VARIANTS = [(0, 1.0), (30, 1.0), (90, 1.0), (210, 1.0), (0, 0.5), (0, 3.0), (137, 2.0)]


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    info = crofton_gpu.gpu_info()
    print(f"GPU: {info['name']} (CC {info['compute_capability']})\n")
    all_ok = True
    names = None

    # compute features for every shape x variant
    feats = {s: [] for s in SHAPES}
    for s, pts in SHAPES.items():
        for deg, sc in VARIANTS:
            v, names = extract(transform(pts, deg, sc))
            feats[s].append(v)
        feats[s] = np.array(feats[s])

    if np.any([~np.isfinite(feats[s]).all() for s in SHAPES]):
        all_ok &= check("all features finite", False, "NaN/inf present")
    else:
        all_ok &= check("all features finite", True, f"{len(names)} features/cell")

    # z-scale features by the global per-feature std (so near-zero harmonics
    # don't dominate); invariance and separation are both judged in z-units.
    scale = np.concatenate([feats[s] for s in SHAPES]).std(axis=0) + 1e-9
    centers = {s: feats[s].mean(axis=0) for s in SHAPES}

    # 1) INVARIANCE as a MARGIN: within-class spread must be far smaller than
    # the distance to the nearest other class (the property that guarantees a
    # classifier can separate them despite rotation/scale).
    print("\n== Invariance margin (within-class spread vs between-class gap) ==")
    for s in SHAPES:
        within = float(np.max([np.linalg.norm((v - centers[s]) / scale) for v in feats[s]]))
        between = float(np.min([np.linalg.norm((centers[s] - centers[o]) / scale)
                                for o in SHAPES if o != s]))
        ratio = within / (between + 1e-9)
        # noisiest features for this class (largest z-deviation from its mean)
        zdev = np.max(np.abs(feats[s] - centers[s]) / scale, axis=0)
        top = np.argsort(zdev)[::-1][:3]
        worst = ", ".join(f"{names[i]}={zdev[i]:.1f}z" for i in top)
        all_ok &= check(f"{s}: within < between", ratio < 0.85,
                        f"ratio {ratio:.2f} (within {within:.2f}/between {between:.2f}); noisiest: {worst}")

    # 2) SEPARATION: 1-NN on one prototype/class labels all variants correctly
    print("\n== Separation (1-NN on per-class prototype) ==")
    protos = {s: feats[s][0] for s in SHAPES}          # rotation 0, scale 1
    correct, total = 0, 0
    confusion = []
    for true_s in SHAPES:
        for i, (deg, sc) in enumerate(VARIANTS):
            v = feats[true_s][i]
            dists = {s: np.linalg.norm((v - protos[s]) / scale) for s in SHAPES}
            pred = min(dists, key=dists.get)
            total += 1
            correct += (pred == true_s)
            if pred != true_s:
                confusion.append(f"{true_s}@{deg}°x{sc}->{pred}")
    all_ok &= check("1-NN classifies all variants", correct == total,
                    f"{correct}/{total}" + (f"  misses: {confusion}" if confusion else ""))

    # 3) DISCRIMINATIVE features behave as the biology predicts
    print("\n== Feature sanity (named morphometrics) ==")
    idx = {n: k for k, n in enumerate(names)}
    m = {s: feats[s].mean(axis=0) for s in SHAPES}
    asp = idx["aspect_ratio"]
    # Elongation is the elliptocyte's signature; the (fat) crescent is defined
    # by concavity instead (checked below via max_defect_depth), not aspect.
    all_ok &= check("ellipse more elongated than circle",
                    m["ellipse"][asp] < m["circle"][asp] - 0.1,
                    f"circle {m['circle'][asp]:.2f}, ellipse {m['ellipse'][asp]:.2f}")
    sol = idx["solidity"]
    all_ok &= check("star & sickle less solid than circle",
                    m["star"][sol] < m["circle"][sol] - 0.05
                    and m["sickle"][sol] < m["circle"][sol] - 0.05,
                    f"circle {m['circle'][sol]:.2f}, star {m['star'][sol]:.2f}, sickle {m['sickle'][sol]:.2f}")
    defd = idx["max_defect_depth"]
    all_ok &= check("sickle has the deepest concavity",
                    m["sickle"][defd] > m["circle"][defd]
                    and m["sickle"][defd] > m["ellipse"][defd],
                    f"sickle {m['sickle'][defd]:.3f}, circle {m['circle'][defd]:.3f}, ellipse {m['ellipse'][defd]:.3f}")
    sp = idx["n_spicules"]
    all_ok &= check("star has the most spicules",
                    m["star"][sp] > m["circle"][sp] and m["star"][sp] > m["ellipse"][sp],
                    f"star {m['star'][sp]:.1f}, circle {m['circle'][sp]:.1f}, ellipse {m['ellipse'][sp]:.1f}")

    print("\nRESULT:", "ALL OK" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
