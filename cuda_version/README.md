# CUDA Version

NVIDIA/CUDA branch of the Crofton boundary-detection pipeline, plus a web
front page to upload a photo and detect its contour on the GPU.

## What changed (June 2026 fix)

The previous `main.cu` shipped an **empty** `kernelCrofton` stub (the SMap came
back all zeros — see the old `crofton_log.txt`). The device code is now fully
implemented in [`crofton_kernels.cu`](crofton_kernels.cu) as the classical
Crofton construction:

- `proyectionKernel` — projects the 239 resampled contour points onto each of
  the 361 angle directions (one thread per `(phi, point)`).
- `kernelCrofton` — vote map `SMap[p][phi]` = number of intersections between
  the closed polygon and the line with normal direction `phi` at signed offset
  `o_p ∈ [-d/2, +d/2]` (one thread per `phi` column, no atomics).
- `reduceKernel` — per-angle signature `C(phi) = (Δp/2)·Σ_p SMap[p][phi]`
  (equals the Feret width for convex shapes) plus the plain Feret width.

The Cauchy–Crofton formula `L ≈ (π/180)·Σ_{phi<180} C(phi)` recovers the
contour perimeter from the vote map alone, which gives an independent
correctness check against OpenCV's `arcLength`.

`crofton_kernels.cu` is the **single source of truth** for device code: it is
`#include`d by `main.cu` (nvcc build) and compiled at runtime with NVRTC by the
webapp backend (CuPy), so it runs on any architecture the driver supports —
including Blackwell / RTX 50xx (`sm_120`), which needs CUDA 12.8+.

## Web app (front page)

Upload a photo, get the detected contour, the Crofton signature C(φ), the
SMap heat map and metrics — all computed on the NVIDIA GPU.

```bash
# one-time setup (from the repo root); uv: https://docs.astral.sh/uv/
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python 'cupy-cuda13x[ctk]' \
    opencv-python-headless fastapi 'uvicorn[standard]' numpy pillow python-multipart

# run
.venv/bin/python cuda_version/webapp/server.py
# open http://localhost:8000
```

API: `GET /api/health` (GPU status), `POST /api/process` (multipart image →
JSON with contour overlay PNG, SMap PNG, C(φ) curve, Feret widths, metrics,
timings).

### Cell classification (Stages a–f of CLASSIFICATION_PLAN.md)

On top of the shape descriptor, the webapp can segment EVERY cell in a smear
and classify each by shape into discocyte / elliptocyte / sickle / echinocyte /
teardrop, with an anomaly flag for shapes unlike any trained class.

```bash
# train the classifier (synthetic cells by default — no download needed;
# pass --data DIR with class subfolders to train on a real set, e.g. erythrocytesIDB)
.venv/bin/python cuda_version/webapp/classify.py train          # then `eval` to check held-out
```

New modules + endpoints:
- `segmentation.py` — `segment_cells()`: multi-cell instance segmentation
  (distance-transform + watershed, per-component thresholding so small/thin
  cells aren't lost). Reuses the proven crofton_gpu masks.
- `features.py` — rotation/scale-invariant feature vector: low-order Crofton
  FFT harmonics (`|FFT(C(φ))|`, invariant by the shift theorem) + radial
  harmonics + named scalar morphometrics. Verified by `tests/test_features.py`.
- `classify.py` — RandomForest on those features + synthetic labeled-cell
  generator + `CellClassifier`. Trains a One-Class SVM on the NORMAL class for
  unsupervised anomaly detection (flags deformities unlike any normal shape,
  even unlabeled ones). `train --out PATH --name NAME` saves named models.
- `datasets.py` — adapters for erythrocytesIDB (circular/elongated/other) and
  Chula-RBC-12 (12 classes from per-cell coordinate labels).
- `conditions.py` + `condition_rules.json` — smear-level decision-support: maps
  the per-cell morphology distribution to **candidate conditions** (sickle→SCD,
  schistocyte>1%→MAHA, elliptocyte>25%→HE, microcyte+hypochromia→IDA, …) with
  evidence, confirmatory test and source. 13 web-verified rules (StatPearls/ASH/
  Merck). Research/screening only, **not** diagnosis.
- Endpoints: `POST /api/classify` → per-cell class + drivers + anomaly +
  population counts + `assessment` (suspected conditions) + class-colored
  overlay. `GET /api/models`, `POST /api/select_model` → switch trained model
  live. `GET /api/samples`, `GET /api/sample/{id}` → built-in image gallery.
- Front page: fixed **sample-image sidebar** (click to process), a **trained-model
  selector** (Chula-12 / sickle-3 / synthetic), a **Shape descriptor / Cell
  classification** mode toggle, and a **smear-level assessment** card.

Train the three shipped models:
```bash
M=cuda_version/webapp/models
.venv/bin/python cuda_version/webapp/classify.py train --out $M/synthetic.joblib --name "Synthetic (5 classes)"
.venv/bin/python cuda_version/webapp/classify.py train --data "datasets/erythrocytesIDB/Version 2, erythrocytesIDB/erythrocytesIDB1" --out $M/sickle.joblib --name "Sickle / erythrocytesIDB (3 classes)"
.venv/bin/python cuda_version/webapp/classify.py train --data "datasets/Chula-RBC-12-Dataset" --out $M/chula.joblib --name "Chula-RBC-12 (12 classes)"
```

Experiments (Stage g): `experiments.py --data DIR` (feature-set comparison +
rotation robustness), `make_report.py --data DIR --tag NAME` (figures + report),
`conditions_eval.py --data DIR` (per-condition sensitivity). Outputs in `results/`.

Run the feature tests: `.venv/bin/python cuda_version/webapp/tests/test_features.py`

NOTE: shape-only classification covers morphology-driven conditions
(sickle/elliptocyte/teardrop/schistocyte/spiculated); malaria (internal
parasite) and most leukemias (nuclear texture) need texture features — see
`CLASSIFICATION_PLAN.md`. This is research / decision-support, **not** a
clinical diagnostic.

Segmentation tries the repo's HSV + top-hat pipeline first and falls back to
Otsu thresholding when the image is not a stained-cell photo, so arbitrary
photos with a clear subject also work.

## Native CLI build (optional, requires CUDA toolkit 12.8+ and OpenCV)

```bash
cd cuda_version
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/crofton_cuda path/to/image.jpg          # headless
./build/crofton_cuda path/to/image.jpg --show   # with windows
```

Outputs: `contour_result.jpg`, `crofton_descriptor.csv`, `crofton_log.txt`.
