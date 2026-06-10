I have the full ground truth: venv has `cupy 14.1.1`, `cv2 4.13.0`, `numpy 2.4.6`; missing `scipy/sklearn/skimage/xgboost/torch/pandas` (will need installing). `numpy.fft` is available so the FFT feature extractor needs zero new deps. The `.cu` is at `cuda_version/crofton_kernels.cu`. Now I'll write the plan.

# Thesis Plan: Extending the GPU Crofton Descriptor into RBC Cell Classification + Deformity Detection

Target hardware: NVIDIA RTX 5070 (Blackwell, sm_120). Existing pipeline verified: `cuda_version/webapp/crofton_gpu.py` segments the single largest contour, resamples to 239 points, centers on bbox, and computes on-GPU a 361-sample Crofton signature `C(phi)`, per-angle Feret widths, and diameter/area/perimeter via `crofton_kernels.cu`. FastAPI app (`server.py` + `static/index.html`) exposes `/api/process`. Venv (`/home/paul/croftondescriptor/.venv`) has `cupy 14.1.1`, `cv2 4.13.0`, `numpy 2.4.6`; **`scipy/scikit-learn/scikit-image/xgboost/torch` are NOT installed** — they must be added for Stages c–e. The Stage 4 feature extractor below is written to need **only numpy** (uses `numpy.fft`), so it runs today with zero new installs.

---

## 1. Core scientific idea: making C(phi) rotation- AND scale-invariant

### The raw signature and its two symmetries
The current pipeline produces a signal `C(phi)`, phi = 0..360 deg (`CANT_PHI = 361`), the Crofton/Cauchy count of boundary crossings per angle — a 1-D directional shape signature. Two problems block it from being a classifier feature:

- **Rotation:** rotating a cell by `theta` cyclically shifts `C(phi) -> C(phi - theta)`. Two photos of the same sickle cell at different orientations give different raw vectors.
- **Scale:** `C(phi)` amplitude and the scalar morphometrics scale with cell size and image magnification (px/um differs per microscope).

### Fix 1 — Scale: normalize by diameter (already computed)
The pipeline already returns `diameter` (`exact_diameter`, max pairwise distance = Feret max). Divide every length-valued quantity by it:
- Normalized Crofton signature: `Cn(phi) = C(phi) / diameter`
- Normalized Feret widths: `Fn(phi) = feret(phi) / diameter`
- All scalar lengths below are made dimensionless by dividing area by `diameter^2` and perimeter by `diameter`.

This removes magnification entirely — the descriptor becomes a pure **shape** descriptor, comparable across microscopes.

### Fix 2 — Rotation: FFT magnitude spectrum
Because rotation is a circular shift of `C(phi)`, take the Discrete Fourier Transform along phi and keep the **magnitude spectrum**. By the Fourier shift theorem, a circular shift multiplies each harmonic by a unit-modulus phase factor `e^(-i*2*pi*k*theta/N)` — the **magnitude `|FFT(Cn)[k]|` is invariant** to that shift, hence to rotation. Concretely:

```
H[k] = |DFT_phi(Cn(phi))[k]|  for k = 0..K       (numpy.fft.rfft over phi in [0,180) period)
H[k] = H[k] / H[0]            (normalize DC out -> also re-confirms scale invariance)
```

Keep the first `K ~ 10–20` harmonics: `H[0]` ~ mean width (roundness baseline), `H[2]` ~ 2-fold elongation (ellipticity / aspect), `H[k>=3]` ~ spicule count and lobing. Low harmonics dominate biology; truncating denoises. **Use the [0,180) period** because the Crofton count is antipodally symmetric (a line at phi and phi+180 is the same line) — folding to half-period halves variance and doubles SNR.

This is the same family of math as **Fourier descriptors of the boundary** named in the evidence (Wiley 2022 `10.1155/2022/1240020`; arXiv `2106.00389`), but applied to the *Crofton directional signature* rather than the complex contour — that novelty is the thesis core.

### Add these scalar morphometrics (the literature's RBC feature set)
From the evidence (Wiley 2022; eClinPath), the standard RBC-classification contour features, with formulas, all computable from the existing contour + `cv2`:

| Feature | Formula | Captures |
|---|---|---|
| Circularity | `4*pi*Area / Perimeter^2` (=1 for a disc) | roundness; low for sickle/schisto |
| Aspect ratio | `minorAxis / majorAxis` from `cv2.fitEllipse` (or `1/eccentricity`) | elongation; low (<0.3–0.4) = sickle, mid = elliptocyte |
| Eccentricity | `sqrt(1 - (minor/major)^2)` | elongation, orientation-free |
| Convexity | `HullPerimeter / Perimeter` | spicule/notch roughness (<1 spiky) |
| Solidity | `Area / ConvexHullArea` | concavity depth; low for keratocyte bite, schisto |
| Convexity-defect depth | max & mean of `cv2.convexityDefects` depths / diameter | crescent concavity (sickle), bite (keratocyte) |
| Roundness | `4*Area / (pi*majorAxis^2)` | disc vs elongated |
| Curvature signature stats | std/kurtosis of discrete boundary curvature; **count of local maxima** | spicule **count** (echinocyte vs acanthocyte) |
| Spicule regularity | variance of angular spacing & of amplitude of curvature peaks | echinocyte (regular) vs acanthocyte (irregular) — evidence explicitly says this is the discriminator |
| Asymmetry | distance from centroid to area-bisector; or `H` odd-harmonic energy | teardrop single tail, broken bilateral symmetry |
| Crofton perimeter ratio | `perimeter_crofton / perimeter_resampled` | independent roughness cross-check (already computed) |

Final per-cell feature vector: `[H[1..K] (rotation-invariant Crofton harmonics)] ++ [the ~12 dimensionless scalars]` → ~25–35 dims.

### Why this is thesis-worthy
1. **Interpretable** — unlike a black-box CNN, every dimension maps to a named morphology (`H[2]` = elongation, defect-depth = crescent concavity), so a pathologist can read *why* a cell was flagged. Directly enables the Stage-(f) explainability card.
2. **GPU-accelerated & real-time** — `C(phi)` is already produced on the RTX 5070 in milliseconds (`gpu_ms`); the FFT is `O(N log N)` on a 361-vector, negligible. The thesis can claim throughput numbers vs CNN inference.
3. **Provably rotation/scale invariant** — invariance is a *theorem* (Fourier shift + diameter normalization), not learned/augmented. This is a stronger, defensible scientific claim than "we trained with rotation augmentation."
4. **Novel descriptor** — Crofton-signature FFT harmonics as classifier features is not the standard boundary-Fourier approach; the comparison in Stage-(g) (Crofton-FFT vs Hu moments vs CNN) is the publishable result.

---

## 2. What SHAPE can and cannot detect (honest scope)

**Center the thesis on morphology-driven (contour-only) conditions.** From the evidence:

**Detectable from contour alone (in scope — the core classes):**
- **Sickle cell / drepanocyte** — high elongation (aspect <0.3–0.4), pointed tips, one deep concavity (curved long axis). → low aspect ratio + high max convexity-defect depth + low circularity. Disease: **sickle cell anemia (HbSS)**.
- **Elliptocyte / ovalocyte** — smooth ellipse, aspect ~1.5–4x, near-zero concavity, straight long axis. → high eccentricity + high convexity, low defect depth. Disease: **hereditary elliptocytosis** (clean shape signal).
- **Teardrop / dacrocyte** — one tail, single high-curvature vertex, broken symmetry. → moderate elongation + one dominant curvature peak + odd-harmonic asymmetry. Disease: **primary myelofibrosis / marrow infiltration**.
- **Schistocyte / fragment** — small, jagged, angular, low solidity. → reduced area + low convexity/solidity + high curvature variance (flag as "irregular/abnormal," sub-typing is hard per evidence). Disease: **microangiopathic hemolytic anemia (TTP/HUS/DIC)**; fraction >1% clinically concerning.
- **Echinocyte (burr)** — many short, uniform, regularly spaced spicules. → high count of small, periodic curvature peaks, low peak-amplitude/spacing variance.
- **Acanthocyte (spur)** — few, irregular, variable-length spicules. → fewer peaks, high amplitude & spacing variance (the regularity variance is the echinocyte/acanthocyte discriminator).
- **Keratocyte (helmet/bite)** — one notch + horns. → single deep convexity defect on an otherwise smooth boundary.
- **Population poikilocytosis** — variance/spread of the above descriptors across all cells in an image flags heterogeneity even when individual cells aren't classifiable.

**NOT reliably detectable from contour (out of scope; documented future texture add-on):**
- **Spherocyte** — round outline like a normal disc; defining feature is **loss of central pallor** (interior intensity). Pure boundary fails. *(The key RBC counterexample to state explicitly in the thesis.)* `Hereditary spherocytosis` needs EMA-binding/AGLT labs.
- **Target cell / codocyte** — internal **bullseye** of hemoglobin; outline normal. → needs interior intensity. Diseases: thalassemia, liver disease, HbC.
- **Stomatocyte** — internal mouth-shaped pallor slit; outline round.
- **Malaria** — internal ring/trophozoite parasite inside a normal-shaped RBC; gold standard is Giemsa-stained interior microscopy. **Out of scope.**
- **Most leukemias** — diagnosis is nuclear/chromatin texture, not RBC outline. **Out of scope.**

**Thesis framing:** "A rotation/scale-invariant GPU shape descriptor for RBC morphology classification and unsupervised deformity screening, with an explicit, evidence-grounded characterization of the shape-only ceiling and a roadmap for interior-intensity features." The honest negative results (spherocyte/codocyte/stomatocyte) are themselves a contribution — they delimit where shape suffices.

---

## 3. Pipeline stages (each a discrete deliverable)

### (a) Multi-cell instance segmentation — many cells per image
The current `extract_contour` returns only the single largest contour. RBC smears have hundreds of cells per field. Deliverable: `cuda_version/webapp/segmentation.py` with `segment_cells(img_bgr) -> list[contour Nx2]`.
- **Baseline (no new heavy deps):** adaptive/Otsu threshold → `cv2` distance transform + **watershed** to split touching cells → filter by area/circularity → per-cell external contour. Reuses the existing `_mask_*` masks.
- **Strong option:** **Cellpose** (`cyto`/`cyto3` model) for robust instance masks on overlapping cells — runs on the RTX 5070 via its torch backend. Add as optional dependency; gate behind a config flag so the lightweight watershed path always works.
- Each cell contour is fed through the existing `resample_contour` → `center_on_bbox` → `crofton_descriptor_gpu`. Batch many cells per GPU call later as an optimization.

### (b) Per-cell feature vector — Crofton FFT harmonics + scalars
Deliverable: `cuda_version/webapp/features.py` (the Stage-4 first step, detailed below). Input: one centered contour + the `CroftonResult`. Output: the ~25–35-dim invariant vector. **numpy-only**, no new deps.

### (c) Labeled dataset + protocol
From the evidence, the two best public, shape-labeled RBC datasets:
1. **erythrocytesIDB / erythrocytesIDB2** (sickle-cell DB, Gonzalez-Hidalgo et al.) — peripheral-smear RBCs labeled **circular / elongated (sickle) / other**, the canonical dataset behind the arXiv `2601.17032` sickle-cell shape work. Best primary for the supervised sickle/elliptocyte axis.
2. **Sickle Cell Disease dataset (Kaggle, "SCD"/"Sickle cell" sets)** and the **dacrocyte/schistocyte/elliptocyte IDA set** from academia.edu `33380053` — broader poikilocyte classes for the multi-class task.

Protocol:
- **Split by IMAGE/patient, not by cell** (cells from one smear are correlated — leaking them across splits inflates accuracy). 60/20/20 train/val/test, stratified by class.
- **5-fold cross-validation** on train+val for model selection; the held-out test set is touched once for the final number.
- Report class balance and use class weights / focal loss for rare classes (schistocytes, keratocytes are scarce).
- Need **per-cell GT masks/labels** validated by a pathologist — see caveats.

### (d) Classifier — interpretable first, then compared
Deliverable: `cuda_version/webapp/classify.py` (training script + saved model artifact).
- **Start interpretable:** `RandomForest` / `XGBoost` / `SVM(RBF)` on the Stage-(b) feature vector. RandomForest/XGBoost give **feature-importance** for free → directly powers the explainability card. (Requires installing `scikit-learn` + `xgboost` into the venv.)
- **Compare against:**
  1. **1-D CNN on `Cn(phi)`** — does a learned model on the raw signature beat hand-crafted harmonics? (torch, GPU.)
  2. **Hu-moments baseline** (`cv2.HuMoments`) + same classifier — the classic shape baseline named in the evidence.
  3. **2-D CNN on the cell image crop** (e.g., small ResNet) — the texture-capable upper bound; shows how much shape alone gives up.
- The **headline thesis result** is: Crofton-FFT features match/beat Hu moments and approach the image-CNN on morphology classes, while being interpretable, rotation-invariant by construction, and GPU-fast.

### (e) Deformity detection — supervised AND unsupervised
- **Supervised:** the multi-class classifier from (d) (normal-discocyte + each poikilocyte class).
- **Unsupervised anomaly detection** for *unseen* deformities: train **One-Class SVM** and/or a small **autoencoder on the feature vector of NORMAL discocytes only**; flag cells with high reconstruction error / outside the learned normal manifold. This catches deformities absent from training labels and yields a continuous "abnormality score" → maps to the population-poikilocytosis screen (variance of descriptors) from the evidence. Deliverable: `cuda_version/webapp/anomaly.py`.

### (f) `/api/classify` endpoint + explainable UI card
- New FastAPI route in `server.py` mirroring `/api/process`: segment all cells (a) → features (b) → per-cell predicted class + confidence (d) + anomaly score (e).
- UI card in `static/index.html`: predicted class, confidence bar, **the top-3 morphometrics that drove the decision** (from RF/XGBoost feature importance or SHAP), e.g. "Sickle (0.91): aspect_ratio=0.28 ↓, max_defect_depth=0.34 ↑, circularity=0.41 ↓." Overlay the contour colored by class. This interpretability is the practical payoff of the shape-only approach.

### (g) Evaluation — the thesis claim
- **Confusion matrix** + **per-class precision / recall / F1** + macro-F1 (rare classes matter).
- **ROC / AUC** per class (one-vs-rest) and for the binary anomaly detector.
- **5-fold CV** mean ± std; single held-out test number.
- **The comparison table** (the thesis claim): Crofton-FFT vs Hu-moments vs 1-D-CNN-on-C(phi) vs 2-D-CNN-on-image — accuracy, macro-F1, **inference latency on RTX 5070**, and **parameter count / interpretability**. Plus a **rotation-robustness experiment**: rotate test cells 0–360 deg, show Crofton-FFT accuracy stays flat while a non-invariant baseline degrades — empirical proof of the Section-1 invariance theorem.

---

## 4. Concrete first step I can implement now (numpy-only, with unit tests)

Create the rotation/scale-invariant feature extractor and a synthetic-shape test that **must** cleanly separate circle vs ellipse vs star vs sickle. This needs **no new dependencies** (`numpy.fft` + existing `cv2`).

**Files to create:**

- `/home/paul/croftondescriptor/cuda_version/webapp/features.py`
  - `crofton_fft_features(curve: np.ndarray, diameter: float, n_harmonics: int = 16) -> np.ndarray`
    1. take `curve[:180]` (antipodal half-period), divide by `diameter` (scale-invariant);
    2. `h = np.abs(np.fft.rfft(cn))[:n_harmonics]`;
    3. normalize by `h[0]` (drops magnitude/scale), return `h[1:]` as rotation-invariant harmonics.
  - `scalar_morphometrics(contour: np.ndarray, area, perimeter, diameter, feret) -> dict` — circularity, aspect ratio / eccentricity (`cv2.fitEllipse`), convexity, solidity (`cv2.convexHull`/`cv2.contourArea`), max & mean convexity-defect depth (`cv2.convexityDefects`) / diameter, curvature-peak count & regularity variance (from `feret` or discrete boundary curvature), odd-harmonic asymmetry energy.
  - `feature_vector(...) -> (np.ndarray, list[str])` — concatenates harmonics + scalars and returns parallel feature **names** (needed for the explainability card and importance plots).

- `/home/paul/croftondescriptor/cuda_version/webapp/tests/test_features.py`
  - Synthetic generators: `circle()`, `ellipse(aspect)`, `star(n_points)`, `sickle(curvature)` as `Nx2` contours.
  - **Invariance tests (the load-bearing ones):** for each shape, generate at rotations {0,30,90,210 deg} and scales {0.5x,1x,3x}; assert the feature vectors are equal within tolerance (`np.allclose`, rtol ~1e-2) — this is the empirical proof of Section 1.
  - **Separation test:** assert circle vs ellipse vs star vs sickle occupy distinct regions, e.g. pairwise feature-vector L2 distance between classes >> within-class distance (or a trivial 1-NN on the 4 prototypes classifies all rotated/scaled variants correctly). Specifically: ellipse has high `H[2]`, star has energy at `H[5]` (5-point), sickle has high max-defect-depth + low aspect ratio + odd-harmonic asymmetry, circle is near-flat spectrum.

Run with the project venv: `/home/paul/croftondescriptor/.venv/bin/python -m pytest cuda_version/webapp/tests/` (install `pytest` into the venv, or use a plain `if __name__=="__main__"` assert-runner to avoid even that dependency).

This deliverable is self-contained, testable today, and is the literal feature function every later stage imports.

---

## 5. Honest caveats

- **Per-cell GT masks are required.** Shape features are only as good as the segmentation. Single-largest-contour (current code) is insufficient; Stage-(a) multi-cell segmentation must produce clean per-cell masks, and training needs **per-cell labels validated by a hematopathologist**. Mis-segmentation (merged/clipped cells) directly corrupts every descriptor.
- **Shape-only ceiling is real and must be stated.** Spherocyte, codocyte/target, stomatocyte, and malaria are **not** separable from the outline (interior intensity/parasite). The thesis must scope to morphology-driven classes and document texture/intensity features as explicit future work — turning the limitation into a contribution by quantifying exactly where shape fails.
- **Clinical / regulatory framing.** This is **research / decision-support**, not a diagnostic device. No clinical claims; outputs are "morphology flags" to assist a pathologist. Diagnosis of sickle cell anemia, elliptocytosis, spherocytosis, thalassemia, etc. requires confirmatory labs (hemoglobin electrophoresis/HPLC, EMA-binding, AGLT) — cite these. Any deployment would need regulatory review the thesis explicitly disclaims.
- **Dataset bias & rare classes.** Public RBC datasets skew toward sickle vs normal; schistocytes/keratocytes are scarce → report per-class recall, use class weighting, and treat the unsupervised anomaly detector (e) as the safety net for under-represented and unseen deformities.
- **Dependency gap (actionable now):** the venv at `/home/paul/croftondescriptor/.venv` currently has only `cupy/cv2/numpy`. Stage 4 needs nothing more. Stages (c)–(e) require installing `scikit-learn`, `xgboost`, `scipy`, optionally `torch`+`cellpose` for the GPU CNN/segmentation baselines.

**Relevant file paths:** existing — `/home/paul/croftondescriptor/cuda_version/webapp/crofton_gpu.py`, `/home/paul/croftondescriptor/cuda_version/crofton_kernels.cu` (note: import in `crofton_gpu.py` resolves to `parents[1]` = this path), `/home/paul/croftondescriptor/cuda_version/webapp/server.py`, `/home/paul/croftondescriptor/cuda_version/webapp/static/index.html`. To create first — `/home/paul/croftondescriptor/cuda_version/webapp/features.py` and `/home/paul/croftondescriptor/cuda_version/webapp/tests/test_features.py`.