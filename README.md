# Crofton-Based Cell Boundary Detection (CUDA & Apple Silicon Metal)

This repository implements a reproducible pipeline for detecting **outer cell boundaries** in stained microscopy images and characterizing their geometry with a **Crofton-based descriptor**. Two GPU backends are provided: **CUDA** (NVIDIA) and **Metal** (Apple Silicon). The system supports **n-pass iterative refinement** with parameter scheduling and quantitative model selection.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/5df481ac-0123-4a0e-a390-245f1bb1b950" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/ce4535cd-b4da-4295-91ce-426e5beb9182" width="100%"></td>
  </tr>
</table>


---

## 1. Problem Definition

Given a color micrograph \(I \in \mathbb{R}^{H\times W\times 3}\) containing a single stained cell on a nearly uniform background, we seek the **outer boundary** \(\Gamma\) of the cell such that:

1. \(\Gamma\) maximizes edge consistency along a faint, semi-transparent halo;
2. \(\Gamma\) encloses internal structures (e.g., nuclei) but is not biased by them;
3. \(\Gamma\) is stable under moderate illumination variation, background texture, and staining variability.

The output is (i) a polygonal contour \(\Gamma = \{(x_k,y_k)\}_{k=1}^K\), optionally **resampled** to a fixed cardinality \(N\), and (ii) the corresponding **Crofton signature** \(S(p,\phi)\).

---

## 2. Method Overview

The pipeline comprises **preprocessing**, **segmentation**, **contour extraction**, and **Crofton analysis**; a controller executes these steps **n times** under a parameter schedule and selects the best result using quantitative criteria.

### 2.1 Preprocessing

1. **Background estimation in CIE-Lab**. The background color \( \mu_b \) is the median color of a border band (5% of image size).
2. **Color distance map**. Compute \( D(x) = \lVert L^*(x)-\mu_b\rVert_2 \) in Lab; normalize to \([0,255]\).
3. **Edge evidence**. Compute Scharr gradients on a denoised grayscale \(G\) to obtain \( M(x) = \sqrt{(G_x)^2+(G_y)^2} \).
4. **Seed prior (optional)**. When purple-stained nuclei exist, an HSV range isolates the largest purple component; a dilated mask \( \Omega_{\text{seed}} \) acts as an anchor to suppress spurious blobs away from the cell body.

### 2.2 Segmentation

- **Thresholding**: Otsu’s threshold \(T\) is computed on \(D\); to recover the faint halo we use a **lowered threshold** \(T'=\alpha T\) with \(\alpha\in[0.5,0.7]\).
- **Cue fusion**: \( B = \mathbb{1}[D\ge T'] \; \lor \; \mathbb{1}[M \ge \tau_M] \), where \(\tau_M\) is the \(q\)-quantile (default \(q=0.75\)).
- **Spatial constraint** (if the seed is available): keep pixels within a distance band of the seed using a distance transform; this encourages a single, compact component around the cell.
- **Morphology**: apply \( \text{Open}_{r_o} \) (speckle removal) then \( \text{Close}_{r_c} \) (gap filling).
- **Component selection**: choose the largest connected component overlapping the seed (if present), else the global largest component.

### 2.3 Contour Extraction and Normalization

- Extract the external contour \(\Gamma\) with `findContours` (`RETR_EXTERNAL`, `CHAIN_APPROX_NONE`).
- Optionally **smooth** by `approxPolyDP(\epsilon = \rho\cdot P)` where \(P\) is perimeter, \(\rho\in[0.003,0.006]\).
- **Centering**: subtract the bounding-box centroid; **resampling** to \(N\) points is performed by uniform arc-length sampling to obtain \( \tilde{\Gamma}\in\mathbb{R}^{N\times 2} \).

### 2.4 Crofton Signature

Let \(\tilde{\Gamma}=\{(x_j,y_j)\}_{j=1}^N\). For \(\phi \in \{0,1,\dots,\Phi-1\}\) degrees, project onto axis \(u_\phi=(\cos\phi, \sin\phi)\):  
\[ s_{j,\phi} = x_j\cos\phi + y_j\sin\phi. \]

Let \(d\) be the maximum pairwise Euclidean distance in \(\tilde{\Gamma}\); define \(P=\lceil d/2\rceil\) radial bins. A **vote map** \( S \in \mathbb{R}^{P\times \Phi}\) is accumulated by binning \( s_{j,\phi}\) into the corresponding radial bin \(p\). Implementation preserves the memory layout \( S[p,\phi] \leftrightarrow \text{offset } (p\cdot\Phi + \phi) \).

---

## 3. Iterative Refinement (n passes)

We seek robust detection under weak contrast. The controller evaluates a schedule of parameter tuples
\(\theta = (\alpha, q, r_o, r_c, \rho, \text{HSV bounds}, \ldots)\). For each pass \(t\in\{1\dots n\}\):

1. Run the full pipeline with \(\theta_t\).
2. Compute a **score** \(J_t\) combining Crofton peakiness, radial stability, and shape regularity:
    - For each \(\phi\), compute \(v_\phi = \max_p S[p,\phi]\) and \(p_\phi = \arg\max_p S[p,\phi]\).
    - Peakiness: \(\bar{v} = \frac{1}{\Phi}\sum_\phi v_\phi\).
    - Radius jitter: \(\sigma_p = \text{std}_\phi(p_\phi)\).
    - Circularity (contour): \( \mathcal{C} = \frac{4\pi A}{P^2}\in[0,1] \).
    - Composite score: \( J = \bar{v} - \lambda \sigma_p + \gamma \mathcal{C} \) with small \(\lambda,\gamma\).
3. Keep the best \((\Gamma_t, S_t, J_t)\); apply **early stopping** if \(J_t\) does not improve by \(\epsilon\) for \(k\) consecutive passes.

This objective empirically correlates with visually clean, single-shell outer boundaries and penalizes fragmented or wobbly rims.

---

## 4. GPU Backends

### 4.1 CUDA (NVIDIA)

- **Threading model**: each block processes a subset of angles \(\phi\); within a block, threads iterate over contour points \(j\) and accumulate into per-\(\phi\) histograms using shared memory and warp-level reductions; global writes are conflict-free per \(\phi\).
- **Memory layout**: contours are stored as two packed arrays \([x_0\ldots x_{N-1}, y_0\ldots y_{N-1}]\). Projections \(s_{j,\phi}\) are either recomputed or cached.
- **Complexity**: \(O(N\Phi)\) per pass; typically \(N=239, \Phi=361\).

### 4.2 Metal (Apple Silicon)

- **Kernel**: one **thread per angle** \(\phi\) (grid size \(\Phi\)), writing a disjoint column \(S[:,\phi]\) → no atomics.
- **Buffers**: `dBorde` (2N floats), `dSProyX` (\(\Phi N\)), `dSMap` (\(P\Phi\)). Use `MTLStorageModeShared` for simplicity; upgrade to `Private` with staging for throughput.
- **Launch**: round up the grid to a multiple of the threadgroup size (e.g., 384 when \(\Phi=361\)); early-return when `phi >= Phi`.
- **Re-use**: create device/queue/pipeline **once**; allocate buffers at **max P** (e.g., half of the image diagonal) and pass `activeP` each pass.

**Equivalence**: both backends implement identical accumulation and indexing, ensuring numerical comparability of \(S\).

---

## 5. Reproducibility and Evaluation

- **Determinism**: fixed resampling length \(N\), deterministic morphology and contour smoothing, fixed random seeds where applicable.
- **Metrics**: in addition to \(J\), report contour length, area, circularity, gradient mean along \(\Gamma\), and IoU vs. manual annotation (if available).
- **Ablations**: (i) without gradient cue, (ii) without seed constraint, (iii) single-pass vs. n-pass, (iv) CPU vs. GPU backends.
- **Runtime**: the n-pass controller adds linear time in \(n\); typical \(n\in[6,12]\) yields stable results. On M-series, the Metal backend generally completes a pass in ~1–3 s at 2–6 MP images; CUDA disks vary with GPU class.

---

## 6. Build & Run

### 6.1 Common prerequisites

- OpenCV ≥ 4.7, C++17, CMake ≥ 3.18.
- (Optional) Python 3.9+ for the web/UI utilities; Node.js 18+ for the React viewer.

### 6.2 CUDA (Linux/Windows)

```bash
git clone https://example.com/crofton.git
cd crofton/cuda
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CLI demo
./build/crofton_cuda --image path/to/cell.jpg --passes 10 --export contour.csv
```

### 6.3 Apple Silicon / Metal (macOS)

```bash
brew install cmake opencv
cd crofton/metal
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CLI demo
./build/crofton_metal --image path/to/cell.jpg --passes 10 --export contour.csv
```

### 6.4 Optional web viewer (macOS/Linux)

```bash
# Backend (Flask)
pip install flask flask-cors opencv-python numpy pillow
python edge_detection_gui.py

# Frontend (React)
cd image-shape-explorer
npm install
npm run dev
```

---

## 7. Reference CLI Options

```
--image <path>                 Input image
--passes N                     Number of refinement passes (default: 8)
--resample N                   Contour resampling length for Crofton (default: 239)
--alpha-start <0..1>           Lowering factor for Otsu threshold (default: 0.60)
--alpha-step <0..1>            Increment per pass (default: +0.02)
--q-grad <0..1>                Gradient quantile for edge mask (default: 0.75)
--open, --close                Morphology radii (pixels)
--score-lambda <float>         Weight for radius jitter penalty (default: 0.1)
--score-gamma <float>          Weight for circularity (default: 0.2)
--export <file.csv>            Save contour coordinates
--export-smap <file.npy>       Save S(p,phi) map as NumPy array
--log                          Verbose logging
```

---

## 8. Limitations

- Extremely low contrast between cell membrane and background may require adaptive illumination correction prior to our pipeline.
- Seed constraint assumes at least one stained internal region is present; otherwise it is disabled.
- When the background is nonstationary (gradients, vignetting), a spatially varying background model is beneficial (not included by default).

---

## 9. License

CUDA code: non-commercial, no redistribution without explicit permission.  
Metal code: research/academic use; when in doubt, the most restrictive clause applies.

---

## 10. Citation

If this work is useful in your research, please cite:
```
@software{crofton_boundary_detection,
  title = {Crofton-Based Cell Boundary Detection with CUDA and Metal},
  year  = {2025},
  author= {Pavel Chmirenko et al.},
  url   = {https://example.com/crofton},
  note  = {GPU-accelerated iterative segmentation and Crofton analysis}
}
```
