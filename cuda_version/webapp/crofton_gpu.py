"""GPU Crofton descriptor pipeline for NVIDIA cards (CuPy + NVRTC).

Compiles cuda_version/crofton_kernels.cu at runtime — the exact same device
code the native nvcc build uses — so it runs on any architecture the local
CUDA driver supports, including Blackwell (RTX 50xx, sm_120).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import cupy as cp
import numpy as np

KERNELS_PATH = Path(__file__).resolve().parents[1] / "crofton_kernels.cu"

CROFTON_MAX_POINTS = 239
CANT_PHI = 361

_module = None


def _kernels() -> cp.RawModule:
    global _module
    if _module is None:
        _module = cp.RawModule(code=KERNELS_PATH.read_text())
    return _module


def gpu_info() -> dict:
    props = cp.cuda.runtime.getDeviceProperties(0)
    cc = str(cp.cuda.Device(0).compute_capability)  # e.g. "120" -> 12.0
    return {
        "name": props["name"].decode(),
        "compute_capability": f"{cc[:-1]}.{cc[-1]}" if len(cc) > 1 else cc,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cupy": cp.__version__,
    }


# ----------------------------------------------------------------------------
# Preprocessing (port of enhancedPreprocessing from main.cu / main_metal.cpp)
# ----------------------------------------------------------------------------

def _mask_hsv_tophat(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, (100, 20, 20), (180, 255, 255))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    top_hat = cv2.subtract(gray, opened)
    _, bin_top_hat = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    combined = cv2.bitwise_or(mask_hsv, bin_top_hat)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
    return combined


def _mask_otsu(img_bgr: np.ndarray) -> np.ndarray:
    """Fallback for photos the cell-specific pipeline can't segment: Otsu on
    blurred grayscale, polarity chosen so the foreground is the minority class."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if np.count_nonzero(mask) > mask.size // 2:
        mask = cv2.bitwise_not(mask)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, 0
    largest = max(contours, key=cv2.contourArea)
    return largest, len(contours)


def extract_contour(img_bgr: np.ndarray):
    """Returns (contour Nx2 float32, num_contours, method). Tries the repo's
    HSV+top-hat pipeline first; falls back to Otsu if the result is degenerate."""
    min_area = 0.001 * img_bgr.shape[0] * img_bgr.shape[1]

    mask = _mask_hsv_tophat(img_bgr)
    contour, n = _largest_contour(mask)
    method = "hsv+tophat"

    if contour is None or cv2.contourArea(contour) < min_area:
        mask = _mask_otsu(img_bgr)
        c2, n2 = _largest_contour(mask)
        if c2 is not None and (contour is None or cv2.contourArea(c2) > cv2.contourArea(contour)):
            contour, n, method = c2, n2, "otsu-fallback"

    if contour is None:
        return None, 0, "none"
    return contour.reshape(-1, 2).astype(np.float32), n, method


# ----------------------------------------------------------------------------
# Contour resampling + packing (port of resampleContour, float precision)
# ----------------------------------------------------------------------------

def resample_contour(points: np.ndarray, n_points: int = CROFTON_MAX_POINTS) -> np.ndarray:
    seg = np.diff(np.vstack([points, points[:1]]), axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(points[:1], n_points, axis=0)

    targets = np.arange(n_points) * (total / n_points)
    idx = np.searchsorted(cum, targets, side="left")
    idx = np.clip(idx, 1, len(cum) - 1)
    prev = idx - 1
    frac = (targets - cum[prev]) / np.maximum(cum[idx] - cum[prev], 1e-12)

    closed = np.vstack([points, points[:1]])
    p1 = closed[prev]
    p2 = closed[idx]
    return (p1 + frac[:, None] * (p2 - p1)).astype(np.float32)


def center_on_bbox(points: np.ndarray) -> np.ndarray:
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return points - (mn + mx) / 2.0


def exact_diameter(points: np.ndarray) -> float:
    diff = points[:, None, :] - points[None, :, :]
    return float(np.sqrt((diff ** 2).sum(-1)).max())


# ----------------------------------------------------------------------------
# GPU descriptor
# ----------------------------------------------------------------------------

@dataclass
class CroftonResult:
    smap: np.ndarray
    curve: np.ndarray
    feret: np.ndarray
    diameter: float
    cant_p: int
    perimeter_crofton: float
    gpu_ms: float
    contour_resampled: np.ndarray = field(default=None, repr=False)


def crofton_descriptor_gpu(centered: np.ndarray) -> CroftonResult:
    """centered: (N,2) float32 contour, origin-centered. Runs the three CUDA
    kernels (projection, vote map, reduction) on the NVIDIA GPU."""
    n = CROFTON_MAX_POINTS
    real_n = min(len(centered), n)

    borde = np.zeros(2 * n, dtype=np.float32)
    borde[:real_n] = centered[:real_n, 0]
    borde[n:n + real_n] = centered[:real_n, 1]

    diameter = exact_diameter(centered[:real_n])
    # Offset window: |projection| <= max ||p|| for every angle, so 2*R always
    # covers all crossings. The pairwise diameter is NOT a valid window for
    # non-centrally-symmetric shapes (projections at oblique angles are not
    # centered at 0 and crossings would be clipped).
    radius = float(np.hypot(centered[:real_n, 0], centered[:real_n, 1]).max())
    window = 2.0 * radius
    cant_p = max(int(np.ceil(radius)), 1)  # binW stays ~2 px

    mod = _kernels()
    k_proj = mod.get_function("proyectionKernel")
    k_crofton = mod.get_function("kernelCrofton")
    k_reduce = mod.get_function("reduceKernel")

    d_borde = cp.asarray(borde)
    d_sproyx = cp.empty(CANT_PHI * n, dtype=cp.float32)
    d_smap = cp.empty(cant_p * CANT_PHI, dtype=cp.float32)
    d_curve = cp.empty(CANT_PHI, dtype=cp.float32)
    d_feret = cp.empty(CANT_PHI, dtype=cp.float32)

    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()

    block2 = (16, 16, 1)
    grid2 = ((CANT_PHI + 15) // 16, (n + 15) // 16, 1)
    k_proj(grid2, block2, (d_borde, d_sproyx,
                           np.int32(n), np.int32(real_n), np.int32(CANT_PHI)))

    block1 = (64, 1, 1)
    grid1 = ((CANT_PHI + 63) // 64, 1, 1)
    k_crofton(grid1, block1, (d_sproyx, d_smap,
                              np.int32(n), np.int32(real_n), np.int32(CANT_PHI),
                              np.int32(cant_p), np.float32(window)))
    k_reduce(grid1, block1, (d_sproyx, d_smap, d_curve, d_feret,
                             np.int32(n), np.int32(real_n), np.int32(CANT_PHI),
                             np.int32(cant_p), np.float32(window)))

    stop.record()
    stop.synchronize()
    gpu_ms = cp.cuda.get_elapsed_time(start, stop)

    curve = cp.asnumpy(d_curve)
    # Cauchy–Crofton: perimeter = 1/2 ∫0..pi ∫ n(phi,p) dp dphi; the curve
    # already folds in binW/2, so integrating phi over [0, pi) gives the
    # perimeter estimate directly.
    perimeter = float(np.sum(curve[:180]) * (np.pi / 180.0))

    return CroftonResult(
        smap=cp.asnumpy(d_smap).reshape(cant_p, CANT_PHI),
        curve=curve,
        feret=cp.asnumpy(d_feret),
        diameter=diameter,
        cant_p=cant_p,
        perimeter_crofton=perimeter,
        gpu_ms=float(gpu_ms),
        contour_resampled=centered,
    )


# ----------------------------------------------------------------------------
# Full image -> result pipeline
# ----------------------------------------------------------------------------

def process_image(img_bgr: np.ndarray) -> dict:
    t0 = time.perf_counter()

    contour, num_contours, method = extract_contour(img_bgr)
    if contour is None:
        raise ValueError("No se encontraron contornos en la imagen")

    perimeter_raw = float(cv2.arcLength(contour.astype(np.int32), True))
    area = float(cv2.contourArea(contour.astype(np.int32)))

    resampled = resample_contour(contour)
    # Perimeter of the polygon the GPU actually measures (resampling smooths
    # away pixel-staircase noise, so this is below the raw pixel perimeter).
    seg = np.diff(np.vstack([resampled, resampled[:1]]), axis=0)
    perimeter_resampled = float(np.hypot(seg[:, 0], seg[:, 1]).sum())
    centered = center_on_bbox(resampled)
    t1 = time.perf_counter()

    res = crofton_descriptor_gpu(centered)
    t2 = time.perf_counter()

    overlay = img_bgr.copy()
    line_w = max(2, round(max(overlay.shape[:2]) / 800))
    cv2.drawContours(overlay, [contour.astype(np.int32).reshape(-1, 1, 2)],
                     -1, (0, 255, 0), line_w)
    for x, y in resampled.astype(int):
        cv2.circle(overlay, (x, y), line_w, (0, 200, 255), -1)
    # Cap the preview size: a 16 MP PNG becomes tens of MB of base64 and
    # chokes the browser; the metrics stay in full-resolution pixels.
    h, w = overlay.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        overlay = cv2.resize(overlay, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)

    smap = res.smap
    smap_norm = (255 * smap / smap.max()).astype(np.uint8) if smap.max() > 0 else smap.astype(np.uint8)
    smap_img = cv2.applyColorMap(smap_norm, cv2.COLORMAP_VIRIDIS)
    # cant_p scales with the shape's size (e.g. ~2000 rows for a 4K photo);
    # render the heat map at a fixed, browser-friendly size.
    target_h = min(max(smap_img.shape[0], 200), 500)
    smap_img = cv2.resize(smap_img, (722, target_h), interpolation=cv2.INTER_NEAREST)

    return {
        "method": method,
        "num_contours": num_contours,
        "area": area,
        "perimeter": perimeter_resampled,
        "perimeter_raw": perimeter_raw,
        "perimeter_crofton": res.perimeter_crofton,
        "diameter": res.diameter,
        "n_points": CROFTON_MAX_POINTS,
        "cant_p": res.cant_p,
        "curve": res.curve.tolist(),
        "feret": res.feret.tolist(),
        "overlay_bgr": overlay,
        "smap_bgr": smap_img,
        "gpu_ms": res.gpu_ms,
        "cpu_ms": (t1 - t0) * 1000.0,
        "total_ms": (t2 - t0) * 1000.0,
    }
