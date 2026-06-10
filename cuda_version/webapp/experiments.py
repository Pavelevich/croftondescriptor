"""Stage (g): the thesis comparison experiments.

Compares the proposed Crofton-FFT shape descriptor against standard baselines
on a labeled cell dataset, and runs the rotation-robustness experiment that is
the empirical proof of the invariance claim.

Feature sets compared (all fed to the SAME RandomForest so only the FEATURES
differ — a fair head-to-head):
  * crofton    : our rotation/scale-invariant vector (features.feature_vector)
  * hu         : 7 log-Hu moments (classic shape baseline, also rot/scale inv.)
  * geometry   : raw scalar morphometrics only (circularity, aspect, solidity…)
  * crofton_raw: the Crofton signature C(φ) sampled directly (NOT FFT) — shows
                 what rotation-invariance buys vs the raw signal.

Metrics: 5-fold stratified CV macro-F1 + accuracy, per-class report, confusion
matrix. Rotation experiment: rotate every test cell by 0..330° and measure how
each feature set's accuracy holds up (invariant features stay flat; the raw
signal degrades).

Usage:
    .venv/bin/python cuda_version/webapp/experiments.py            # synthetic data
    .venv/bin/python cuda_version/webapp/experiments.py --data DIR # real dataset
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crofton_gpu
import features as F
import classify


# --------------------------------------------------------------------------
# Feature extractors (each: contour -> vector). Shared GPU Crofton where used.
# --------------------------------------------------------------------------

def feat_crofton(contour, curve, diameter):
    v, _ = F.feature_vector(contour, curve, diameter)
    return v


def feat_crofton_raw(contour, curve, diameter):
    """Raw half-period Crofton signature, scale-normalized but NOT made
    rotation-invariant — the ablation that exposes the value of |FFT|."""
    c = np.asarray(curve, dtype=np.float64)[:180]
    return c / (diameter + 1e-9)


def feat_hu(contour, curve, diameter):
    cnt = np.round(contour).astype(np.int32).reshape(-1, 1, 2)
    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
    # log-transform (standard) to compress the huge dynamic range
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def feat_geometry(contour, curve, diameter):
    m = F.scalar_morphometrics(contour, curve, diameter)
    return np.array(list(m.values()), dtype=np.float64)


EXTRACTORS = {
    "crofton": feat_crofton,
    "crofton_raw": feat_crofton_raw,
    "hu": feat_hu,
    "geometry": feat_geometry,
}


# --------------------------------------------------------------------------
# Dataset -> contours (+ a cached GPU Crofton curve per cell)
# --------------------------------------------------------------------------

def _contour_record(contour):
    """Resample/center once; run GPU Crofton once. Returns dict reused by every
    feature set so the GPU work isn't repeated per extractor."""
    pts = contour.astype(np.float32)
    resampled = crofton_gpu.resample_contour(pts)
    centered = crofton_gpu.center_on_bbox(resampled)
    res = crofton_gpu.crofton_descriptor_gpu(centered)
    return {"contour": centered, "curve": res.curve, "diameter": res.diameter}


def load_dataset(data_dir, per_class, seed):
    """Returns records[dict(contour,curve,diameter)], y[labels], classes, images.

    `images` is a list aligned with records of BGR cell crops (or None) for the
    CNN baseline; empty/None entries mean no image available for that cell.
    """
    records, y, images = [], [], []
    if data_dir:
        import datasets as DS
        d = Path(data_dir)
        # multi-class Chula-RBC-12 (smear images + per-cell coordinate labels)
        if DS.is_chula(d):
            print(f"Chula-RBC-12 at {d}: cropping labeled cells ...")
            recs, stats = DS.find_chula(d)
            print(f"  {len(recs)} cells (source: {stats})")
            classes = sorted(set(lbl for _, lbl, _ in recs))
            for contour, lbl, img in recs:
                records.append(_contour_record(contour)); y.append(lbl); images.append(img)
        # erythrocytesIDB (circular/elongated/other crops)
        elif list(d.rglob("mask-circular")) or list(d.rglob("circular")):
            print(f"erythrocytesIDB at {d}: extracting contours ...")
            recs, stats = DS.find_erythrocytesIDB(d)
            print(f"  {len(recs)} cells (source: {stats})")
            for contour, lbl, img in recs:
                records.append(_contour_record(contour)); y.append(lbl); images.append(img)
            classes = DS.IDB_CLASSES
        else:
            classes = sorted([s.name for s in d.iterdir() if s.is_dir()])
            print(f"Real dataset {d} : classes {classes}")
            for cls in classes:
                files = sorted([p for p in (d / cls).glob("**/*")
                                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")])
                n = 0
                for p in files:
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    contour, _, _ = crofton_gpu.extract_contour(img)
                    if contour is None or len(contour) < 10:
                        continue
                    records.append(_contour_record(contour)); y.append(cls); images.append(img); n += 1
                print(f"  {cls}: {n} cells")
    else:
        rng = np.random.default_rng(seed)
        classes = classify.CLASSES
        print(f"Synthetic dataset: {per_class}/class, classes {classes}")
        for cls in classes:
            for _ in range(per_class):
                records.append(_contour_record(classify.gen_contour(cls, rng)))
                y.append(cls); images.append(None)
    return records, np.array(y), classes, images


def build_matrix(records, extractor):
    X = []
    for r in records:
        X.append(extractor(r["contour"], r["curve"], r["diameter"]))
    return np.array(X)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def run(data_dir=None, per_class=150, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

    records, y, classes, images = load_dataset(data_dir, per_class, seed)
    print(f"\nTotal cells: {len(y)}  | classes: {len(classes)}\n")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    have_images = any(im is not None for im in images)

    print("=" * 70)
    print(f"{'feature set':<14}{'#dims':>7}{'accuracy':>11}{'macro-F1':>11}{'fit ms/cell':>13}")
    print("-" * 70)
    results = {}
    for name, extractor in EXTRACTORS.items():
        t0 = time.perf_counter()
        X = build_matrix(records, extractor)
        extract_ms = (time.perf_counter() - t0) * 1000 / len(records)
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
        y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        results[name] = {"X": X, "acc": acc, "f1": f1, "y_pred": y_pred, "dims": X.shape[1]}
        print(f"{name:<14}{X.shape[1]:>7}{acc:>11.3f}{f1:>11.3f}{extract_ms:>13.2f}")

    # CNN-on-image texture-ceiling baseline (only if crops + torch available)
    cnn_res = None
    if have_images:
        try:
            import cnn_baseline as CNN
            if CNN.TORCH_OK:
                imgs = [im if im is not None else np.zeros((8, 8, 3), np.uint8) for im in images]
                cnn_res = CNN.train_eval_cv(imgs, y, classes, folds=5, seed=seed)
                if "error" not in cnn_res:
                    results["cnn_image"] = {"acc": cnn_res["acc"], "f1": cnn_res["f1"],
                                            "y_pred": cnn_res["pred"], "dims": cnn_res["params"]}
                    print(f"{'cnn_image':<14}{cnn_res['params']:>7}{cnn_res['acc']:>11.3f}"
                          f"{cnn_res['f1']:>11.3f}{cnn_res['inf_ms']:>13.2f}   ({cnn_res['device']}, params not dims)")
            else:
                print("cnn_image      (skipped — torch not installed)")
        except Exception as e:
            print(f"cnn_image      (skipped — {e})")
    print("=" * 70)

    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest feature set: {best}\n")
    print(f"Per-class report ({best}):")
    print(classification_report(y, results[best]["y_pred"], digits=3))
    print(f"Confusion matrix ({best}) rows=true cols=pred, labels={classes}")
    print(confusion_matrix(y, results[best]["y_pred"], labels=classes))

    # ----- rotation-robustness experiment -----------------------------------
    print("\n" + "=" * 64)
    print("ROTATION ROBUSTNESS  (train at fixed orientation, test rotated by θ)")
    print("-" * 64)
    angles = [0, 30, 60, 90, 150, 210, 270, 330]
    rot_sets = ["crofton", "crofton_raw", "hu", "geometry"]

    # Build a CANONICAL-orientation training set so the contrast isn't masked by
    # rotation augmentation. Synthetic: regenerate cells with canonical=True.
    # Real: cells have a fixed natural orientation already, so reuse the split.
    from sklearn.model_selection import train_test_split
    if data_dir:
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
        train_records = [records[i] for i in tr]; y_tr = y[tr]
        test_records = [records[i] for i in te]; y_te = y[te]
    else:
        rng = np.random.default_rng(seed + 1)
        train_records, y_tr = [], []
        test_records, y_te = [], []
        for cls in classes:
            for _ in range(80):
                train_records.append(_contour_record(classify.gen_contour(cls, rng, canonical=True)))
                y_tr.append(cls)
            for _ in range(40):
                test_records.append(_contour_record(classify.gen_contour(cls, rng, canonical=True)))
                y_te.append(cls)
        y_tr = np.array(y_tr); y_te = np.array(y_te)

    models = {}
    for name in rot_sets:
        Xtr = build_matrix(train_records, EXTRACTORS[name])
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
        clf.fit(Xtr, y_tr); models[name] = clf

    header = "  θ°  " + "".join(f"{n:>13}" for n in rot_sets)
    print(header)
    rot_table = {n: [] for n in rot_sets}
    for ang in angles:
        a = np.deg2rad(ang)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        # rotate each test cell by θ and re-run GPU Crofton on the rotated contour
        rot_records = [_contour_record(rec["contour"] @ R.T) for rec in test_records]
        row = f"{ang:>4}  "
        for name in rot_sets:
            Xte = build_matrix(rot_records, EXTRACTORS[name])
            acc = (models[name].predict(Xte) == y_te).mean()
            rot_table[name].append(acc)
            row += f"{acc:>13.3f}"
        print(row)
    print("-" * 64)
    print("accuracy spread over rotation (max-min; lower = more invariant):")
    for name in rot_sets:
        arr = np.array(rot_table[name])
        print(f"  {name:<13} spread {arr.max()-arr.min():.3f}   mean {arr.mean():.3f}")
    print("=" * 64)

    return results, rot_table, y, classes


if __name__ == "__main__":
    data = None; per_class = 150
    for i, a in enumerate(sys.argv):
        if a == "--data" and i + 1 < len(sys.argv): data = sys.argv[i + 1]
        if a == "--per-class" and i + 1 < len(sys.argv): per_class = int(sys.argv[i + 1])
    run(data, per_class)
