"""FastAPI server: front page (upload -> GPU contour detection) + JSON API.

Run from the repo root:
    .venv/bin/python cuda_version/webapp/server.py
Then open http://localhost:8000
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crofton_gpu
import segmentation
import features as F
import conditions

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PIXELS_HARD = 80_000_000   # reject above (decompression-bomb guard)
MAX_PIXELS_PROC = 16_000_000   # downscale above (phone photos still work)
ANOMALY_THRESHOLD = 6.0        # z-distance beyond nearest class -> "atypical"

app = FastAPI(title="Crofton Descriptor CUDA")

# Named colors for the common shape classes; any other class gets a color from
# a generated palette so 12-class datasets (Chula) render distinctly too.
_NAMED_COLORS = {
    "discocyte": (0, 220, 0), "Normal": (0, 220, 0),
    "elliptocyte": (0, 200, 255), "Elliptocyte": (0, 200, 255),
    "sickle": (255, 80, 0), "echinocyte": (255, 0, 200),
    "teardrop": (0, 255, 255), "Teardrop": (0, 255, 255),
}
_PALETTE = [(0, 220, 0), (0, 200, 255), (255, 80, 0), (255, 0, 200), (0, 255, 255),
            (180, 255, 0), (120, 0, 255), (0, 140, 255), (200, 200, 0),
            (255, 0, 80), (80, 255, 160), (160, 80, 255)]
_classifier = None          # active CellClassifier
_active_model = None         # id of active model
_model_cache = {}            # id -> CellClassifier
MODELS_DIR = Path(__file__).resolve().parent / "models"


def _discover_models():
    """Return {id: {id, name, classes, source, path}} for every saved model."""
    import joblib
    reg = {}
    paths = list(MODELS_DIR.glob("*.joblib")) if MODELS_DIR.exists() else []
    legacy = Path(__file__).resolve().parent / "cell_model.joblib"
    if legacy.exists():
        paths.append(legacy)
    for p in paths:
        try:
            meta = joblib.load(p)
            reg[p.stem] = {"id": p.stem, "name": meta.get("name", p.stem),
                           "classes": meta.get("classes", []),
                           "source": meta.get("source", "?"), "path": str(p)}
        except Exception:
            continue
    return reg


def _class_colors(classes):
    """Stable BGR color per class: named where known, else from the palette."""
    out = {}
    for i, c in enumerate(classes):
        out[c] = _NAMED_COLORS.get(c, _PALETTE[i % len(_PALETTE)])
    return out


def _get_classifier():
    """Active classifier. Prefers a selected model; defaults to chula > any > legacy."""
    global _classifier, _active_model
    if _classifier is not None:
        return _classifier
    import classify
    reg = _discover_models()
    if reg:
        default = ("chula" if "chula" in reg else
                   "cell_model" if "cell_model" in reg else next(iter(reg)))
        _active_model = default
        _classifier = _load_model(default, reg)
    elif classify.MODEL_PATH.exists():
        _active_model = "cell_model"
        _classifier = classify.CellClassifier()
    else:
        _classifier = False  # not trained yet
    return _classifier


def _load_model(mid, reg=None):
    import classify
    reg = reg or _discover_models()
    if mid not in reg:
        return None
    if mid not in _model_cache:
        _model_cache[mid] = classify.CellClassifier(Path(reg[mid]["path"]))
    return _model_cache[mid]


# --- sample-image gallery (fixed sidebar) ---------------------------------
ROOT = Path(__file__).resolve().parents[2]
_SAMPLES = None


def _build_samples():
    """Curated gallery: a few cells per dataset + repo examples. Returns an
    id->path map; only these ids are ever served (no path traversal)."""
    reg = {}

    def add(group, label, path, n=None):
        p = Path(path)
        if not p.exists():
            return
        files = sorted([f for f in p.glob("*")
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]) if p.is_dir() else [p]
        for f in (files[:n] if n else files):
            sid = f"{group}-{label}-{f.stem}".replace(" ", "_")
            reg[sid] = {"id": sid, "group": group, "label": label, "path": str(f)}

    idb = ROOT / "datasets/erythrocytesIDB/Version 2, erythrocytesIDB/erythrocytesIDB1/individual cells"
    for cls in ("circular", "elongated", "other"):
        add("erythrocytesIDB", cls, idb / cls, n=3)
    chula = ROOT / "datasets/Chula-RBC-12-Dataset/Dataset"
    add("Chula smear", "smear", chula, n=6)
    add("Examples", "cell", ROOT / "resources/e1.jpeg")
    sm = ROOT / "apple_silicon_version/resources/sample_images"
    for f in ("test_cell.jpg", "c3.jpg", "test_circle.png"):
        add("Examples", "sample", sm / f)
    return reg


def _samples():
    global _SAMPLES
    if _SAMPLES is None:
        _SAMPLES = _build_samples()
    return _SAMPLES


@app.get("/api/samples")
def list_samples():
    items = list(_samples().values())
    return {"samples": [{"id": s["id"], "group": s["group"], "label": s["label"]}
                        for s in items]}


@app.get("/api/sample/{sid}")
def get_sample(sid: str, thumb: bool = False):
    s = _samples().get(sid)
    if not s:
        return JSONResponse({"error": "unknown sample"}, status_code=404)
    if not thumb:
        return FileResponse(s["path"])
    img = cv2.imread(s["path"])
    if img is None:
        return JSONResponse({"error": "unreadable"}, status_code=500)
    h, w = img.shape[:2]
    scale = 96 / max(h, w)
    thumb_img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
    from fastapi.responses import Response
    ok, buf = cv2.imencode(".jpg", thumb_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.middleware("http")
async def reject_oversized_uploads(request: Request, call_next):
    # Early rejection before multipart parsing spools the body to disk/RAM.
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 64 * 1024:
        return JSONResponse({"error": "Image too large (max 25 MB)"},
                            status_code=413)
    return await call_next(request)


def _img_b64(img_bgr: np.ndarray, ext: str = ".png", params=None) -> str:
    ok, buf = cv2.imencode(ext, img_bgr, params or [])
    if not ok:
        raise ValueError("Could not encode the output image")
    return base64.b64encode(buf.tobytes()).decode()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    try:
        info = crofton_gpu.gpu_info()
        return {
            "gpu_ok": True,
            "gpu_name": info["name"],
            "compute_capability": info["compute_capability"],
            "cupy": info["cupy"],
            "cuda_runtime": info["cuda_runtime"],
        }
    except Exception as exc:  # GPU missing / driver issue
        return {"gpu_ok": False, "error": str(exc)}


@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        return JSONResponse({"error": "Empty file"}, status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "Image too large (max 25 MB)"}, status_code=413)

    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        img = None
    if img is None:
        return JSONResponse({"error": "Could not decode the image"}, status_code=400)

    pixels = img.shape[0] * img.shape[1]
    if pixels > MAX_PIXELS_HARD:
        return JSONResponse({"error": "Resolution too high (max 80 MP)"}, status_code=422)
    if pixels > MAX_PIXELS_PROC:
        scale = (MAX_PIXELS_PROC / pixels) ** 0.5
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    try:
        r = crofton_gpu.process_image(img)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse({"error": f"GPU failure: {exc}"}, status_code=500)

    # Debug capture: keep the last upload + result for diagnosis.
    try:
        Path("/tmp/crofton_last_upload.bin").write_bytes(data)
        cv2.imwrite("/tmp/crofton_last_overlay.png", r["overlay_bgr"])
        print(f"[process] {file.filename}: {img.shape[1]}x{img.shape[0]} "
              f"method={r['method']} contours={r['num_contours']} "
              f"area={r['area']:.0f} gpu={r['gpu_ms']:.2f}ms", flush=True)
    except OSError:
        pass

    info = crofton_gpu.gpu_info()
    return {
        "device": f"{info['name']} (CUDA, CC {info['compute_capability']})",
        "method": r["method"],
        "overlay_png": _img_b64(r["overlay_bgr"], ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 85]),
        "smap_png": _img_b64(r["smap_bgr"]),
        "crofton_curve": r["curve"],
        "feret": r["feret"],
        "metrics": {
            "num_contours": r["num_contours"],
            "area": r["area"],
            "perimeter": r["perimeter"],
            "perimeter_raw": r["perimeter_raw"],
            "perimeter_crofton": r["perimeter_crofton"],
            "diameter": r["diameter"],
            "n_points": r["n_points"],
            "cant_p": r["cant_p"],
        },
        "timings": {
            "gpu_ms": r["gpu_ms"],
            "cpu_ms": r["cpu_ms"],
            "total_ms": r["total_ms"],
        },
    }


def _decode_upload(data: bytes):
    """Shared decode + size guards. Returns (img, error_response_or_None)."""
    if not data:
        return None, JSONResponse({"error": "Empty file"}, status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        return None, JSONResponse({"error": "Image too large (max 25 MB)"}, status_code=413)
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        img = None
    if img is None:
        return None, JSONResponse({"error": "Could not decode the image"}, status_code=400)
    pixels = img.shape[0] * img.shape[1]
    if pixels > MAX_PIXELS_HARD:
        return None, JSONResponse({"error": "Resolution too high (max 80 MP)"}, status_code=422)
    if pixels > MAX_PIXELS_PROC:
        scale = (MAX_PIXELS_PROC / pixels) ** 0.5
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img, None


@app.get("/api/model")
def model_status():
    clf = _get_classifier()
    if not clf:
        return {"trained": False}
    return {"trained": True, "classes": clf.classes, "active": _active_model}


@app.get("/api/models")
def list_models():
    _get_classifier()  # ensure active set
    reg = _discover_models()
    return {
        "active": _active_model,
        "models": [{"id": m["id"], "name": m["name"],
                    "n_classes": len(m["classes"]), "source": m["source"]}
                   for m in reg.values()],
    }


@app.post("/api/select_model")
async def select_model(req: Request):
    global _classifier, _active_model
    body = await req.json()
    mid = body.get("id")
    reg = _discover_models()
    if mid not in reg:
        return JSONResponse({"error": f"unknown model '{mid}'"}, status_code=404)
    m = _load_model(mid, reg)
    if m is None:
        return JSONResponse({"error": "failed to load"}, status_code=500)
    _classifier = m
    _active_model = mid
    print(f"[model] switched to {mid} ({len(m.classes)} classes)", flush=True)
    return {"active": mid, "classes": m.classes}


@app.post("/api/classify")
async def classify_cells(file: UploadFile = File(...)):
    clf = _get_classifier()
    if not clf:
        return JSONResponse(
            {"error": "Model not trained. Run: .venv/bin/python cuda_version/webapp/classify.py train"},
            status_code=503)

    data = await file.read()
    img, err = _decode_upload(data)
    if err:
        return err

    seg = segmentation.segment_cells(img)
    contours = seg["contours"]
    if not contours:
        return JSONResponse({"error": "No cells found to classify"}, status_code=422)

    cells, labels, counts = [], [], {}
    n_atypical = 0
    for idx, c in enumerate(contours):
        pred = clf.predict_contour(c.astype(np.float64))
        # prefer the One-Class SVM verdict (unsupervised, vs the NORMAL manifold);
        # fall back to the class-mean distance heuristic if no detector is present
        atypical = pred["atypical"] if pred["atypical"] is not None \
            else pred["anomaly_score"] > ANOMALY_THRESHOLD
        n_atypical += int(atypical)
        label = pred["label"]
        labels.append(label)
        counts[label] = counts.get(label, 0) + 1
        x, y, w, h = cv2.boundingRect(c.reshape(-1, 1, 2))
        cells.append({
            "id": idx, "label": label, "confidence": pred["confidence"],
            "atypical": atypical, "anomaly_score": pred["anomaly_score"],
            "ocsvm_score": pred["ocsvm_score"],
            "bbox": [int(x), int(y), int(w), int(h)],
            "drivers": pred["drivers"], "proba": pred["proba"],
        })

    palette = _class_colors(clf.classes)
    overlay = segmentation.draw_instances(img, contours, labels, palette)
    if max(overlay.shape[:2]) > 1600:
        s = 1600 / max(overlay.shape[:2])
        overlay = cv2.resize(overlay, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    assessment = conditions.assess(counts, n_atypical=n_atypical, total=len(cells))

    print(f"[classify] {file.filename}: {len(cells)} cells, counts={counts}, "
          f"atypical={n_atypical}, conditions={[c['condition'] for c in assessment['candidates']]}",
          flush=True)
    return {
        "method": seg["method"],
        "n_cells": len(cells),
        "n_atypical": n_atypical,
        "normal_class": clf.normal_class,
        "counts": counts,
        "assessment": assessment,
        "cells": cells,
        "overlay_png": _img_b64(overlay, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 85]),
        "class_colors": {k: f"rgb({v[2]},{v[1]},{v[0]})" for k, v in palette.items()},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
