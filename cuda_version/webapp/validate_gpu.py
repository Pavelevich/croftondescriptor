"""Validation harness for the CUDA Crofton implementation.

1. NumPy reference of the exact same algorithm -> GPU SMap/curve must match.
2. Synthetic shapes with known geometry:
   - circle radius r: C(phi) = 2r for all phi, Cauchy-Crofton perimeter = 2*pi*r
   - ellipse a,b: C(phi) = Feret width = 2*sqrt(a^2 cos^2 + b^2 sin^2)
3. Real sample images end-to-end (contour found, perimeter sane vs OpenCV).

Run: .venv/bin/python cuda_version/webapp/validate_gpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crofton_gpu
from crofton_gpu import CANT_PHI, CROFTON_MAX_POINTS


def crofton_numpy(centered: np.ndarray):
    """Reference implementation of the same vote-map algorithm in NumPy."""
    n = CROFTON_MAX_POINTS
    real_n = min(len(centered), n)
    pts = centered[:real_n].astype(np.float32)

    radius = float(np.hypot(pts[:, 0], pts[:, 1]).max())
    window = 2.0 * radius
    cant_p = max(int(np.ceil(radius)), 1)

    ang = (np.arange(CANT_PHI, dtype=np.float32) * np.float32(np.pi)) / np.float32(180.0)
    # s[phi, j], float32 like the GPU
    s = (pts[:, 0][None, :] * np.cos(ang)[:, None]
         + pts[:, 1][None, :] * np.sin(ang)[:, None]).astype(np.float32)

    bin_w = np.float32(window / cant_p)
    origin = np.float32(-0.5 * window)
    smap = np.zeros((cant_p, CANT_PHI), dtype=np.float32)

    a = s
    b = np.roll(s, -1, axis=1)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    for phi in range(CANT_PHI):
        for j in range(real_n):
            l, h = lo[phi, j], hi[phi, j]
            if h <= l:
                continue
            p0 = int(np.ceil((l - origin) / bin_w - 0.5))
            p1 = int(np.ceil((h - origin) / bin_w - 0.5)) - 1
            p0 = max(p0, 0)
            p1 = min(p1, cant_p - 1)
            if p1 >= p0:
                smap[p0:p1 + 1, phi] += 1.0

    curve = 0.5 * bin_w * smap.sum(axis=0)
    feret = s[:, :real_n].max(axis=1) - s[:, :real_n].min(axis=1)
    return smap, curve, feret, window, cant_p


def make_circle(r=100.0, n=CROFTON_MAX_POINTS):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1).astype(np.float32)


def make_ellipse(a=150.0, b=80.0, n=CROFTON_MAX_POINTS):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([a * np.cos(t), b * np.sin(t)], axis=1).astype(np.float32)


def make_polygon(vertices):
    """Resample + bbox-center an arbitrary polygon (asymmetric shapes exposed
    the old offset-window clipping bug, so they must stay in the suite)."""
    pts = np.asarray(vertices, dtype=np.float32)
    resampled = crofton_gpu.resample_contour(pts)
    return crofton_gpu.center_on_bbox(resampled)


def make_half_disk(r=200.0, n_arc=160):
    t = np.linspace(0, np.pi, n_arc)
    arc = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    base = np.stack([np.linspace(-r, r, 60, endpoint=False), np.zeros(60)], axis=1)
    return np.vstack([arc, base]).astype(np.float32)


def polygon_perimeter(pts):
    seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def ellipse_perimeter(a, b):
    # Ramanujan II approximation
    h = ((a - b) / (a + b)) ** 2
    return np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    info = crofton_gpu.gpu_info()
    print(f"GPU: {info['name']} (CC {info['compute_capability']}, CuPy {info['cupy']})\n")
    all_ok = True

    # --- 1+2: synthetic shapes, GPU vs NumPy vs analytic ---
    triangle = make_polygon([[0, 0], [100, 0], [90, 50]])
    half_disk = make_polygon(make_half_disk(200.0))
    for name, pts, expected_perim in [
        ("circle r=100", make_circle(100.0), 2 * np.pi * 100.0),
        ("ellipse 150x80", make_ellipse(150.0, 80.0), ellipse_perimeter(150.0, 80.0)),
        ("scalene triangle (asymmetric)", triangle, polygon_perimeter(triangle)),
        ("half-disk r=200 (asymmetric)", half_disk, polygon_perimeter(half_disk)),
    ]:
        print(f"== {name} ==")
        res = crofton_gpu.crofton_descriptor_gpu(pts)
        smap_np, curve_np, feret_np, diam_np, cant_p_np = crofton_numpy(pts)

        all_ok &= check("SMap GPU == NumPy",
                        res.smap.shape == smap_np.shape and np.array_equal(res.smap, smap_np),
                        f"shape {res.smap.shape}, max diff "
                        f"{np.abs(res.smap - smap_np).max() if res.smap.shape == smap_np.shape else 'shape mismatch'}")
        all_ok &= check("curve GPU ~= NumPy",
                        np.allclose(res.curve, curve_np, atol=1e-3),
                        f"max diff {np.abs(res.curve - curve_np).max():.2e}")
        all_ok &= check("feret GPU ~= NumPy",
                        np.allclose(res.feret, feret_np, atol=1e-3),
                        f"max diff {np.abs(res.feret - feret_np).max():.2e}")

        rel = abs(res.perimeter_crofton - expected_perim) / expected_perim
        all_ok &= check("Cauchy-Crofton perimeter vs analytic",
                        rel < 0.02,
                        f"GPU {res.perimeter_crofton:.1f} vs analytic {expected_perim:.1f} "
                        f"({100 * rel:.2f}% err)")

        # circle: curve must be flat = 2r; SMap crossings must be exactly 2 inside
        if name.startswith("circle"):
            spread = res.curve.max() - res.curve.min()
            all_ok &= check("circle curve flat (=2r)", spread < 2.5,
                            f"mean {res.curve.mean():.1f} (expect 200), spread {spread:.2f}")
            inside = res.smap[res.smap > 0]
            all_ok &= check("crossing counts are 2 (convex)",
                            np.all(inside == 2.0),
                            f"unique values {np.unique(inside)}")
        print()

    # --- 3: real sample images end-to-end ---
    import cv2
    samples = [
        Path(__file__).resolve().parents[2] / "resources" / "e1.jpeg",
        Path(__file__).resolve().parents[2] / "apple_silicon_version" / "resources"
        / "sample_images" / "test_cell.jpg",
        Path(__file__).resolve().parents[2] / "apple_silicon_version" / "resources"
        / "sample_images" / "c1.jpg",
        Path(__file__).resolve().parents[2] / "apple_silicon_version" / "resources"
        / "sample_images" / "test_circle.png",
    ]
    for path in samples:
        if not path.exists():
            continue
        print(f"== {path.name} ==")
        img = cv2.imread(str(path))
        try:
            r = crofton_gpu.process_image(img)
        except ValueError as exc:
            all_ok &= check("contour found", False, str(exc))
            continue
        # Compare against the resampled polygon the GPU actually measures.
        rel = abs(r["perimeter_crofton"] - r["perimeter"]) / max(r["perimeter"], 1.0)
        all_ok &= check("contour found", True,
                        f"method={r['method']}, contours={r['num_contours']}, "
                        f"area={r['area']:.0f}")
        all_ok &= check("Crofton perimeter vs resampled polygon within 3%",
                        rel < 0.03,
                        f"Crofton {r['perimeter_crofton']:.1f} vs polygon {r['perimeter']:.1f} "
                        f"({100 * rel:.2f}%; raw pixel contour {r['perimeter_raw']:.1f})")
        print(f"  gpu={r['gpu_ms']:.2f}ms cpu={r['cpu_ms']:.1f}ms total={r['total_ms']:.1f}ms")
        print()

    print("RESULT:", "ALL OK" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
