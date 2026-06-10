"""Stage (d/e): cell shape classifier + training, on top of features.py.

Trains an interpretable RandomForest on the rotation/scale-invariant feature
vector. By default it learns from SYNTHETIC labeled cells (normal discocyte,
elliptocyte, sickle/drepanocyte, echinocyte, teardrop/dacrocyte) so the whole
pipeline is verifiable end-to-end with no external download; pass
`--data <dir>` (subfolders = class names, each holding cell images) to train on
a real labeled set such as erythrocytesIDB.

The saved model artifact carries the feature names + per-class training stats,
so /api/classify can report WHICH morphometrics drove each decision
(explainability), and an anomaly score flags shapes unlike any trained class.

Usage:
    .venv/bin/python cuda_version/webapp/classify.py train [--data DIR] [--per-class N]
    .venv/bin/python cuda_version/webapp/classify.py eval
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crofton_gpu
import features as F

MODEL_PATH = Path(__file__).resolve().parent / "cell_model.joblib"
CLASSES = ["discocyte", "elliptocyte", "sickle", "echinocyte", "teardrop"]


# ==========================================================================
# Synthetic labeled cell generator (rasterized -> realistic contour)
# ==========================================================================

def _rasterize(poly: np.ndarray, canvas=420) -> np.ndarray:
    """Fill a polygon centered on a canvas and return its external contour."""
    img = np.zeros((canvas, canvas), np.uint8)
    pts = (poly + canvas / 2).astype(np.int32)
    cv2.fillPoly(img, [pts], 255)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)


def _polar(r, t):
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def gen_contour(cls: str, rng: np.random.Generator, canonical: bool = False) -> np.ndarray:
    """One randomized cell contour of the given class (random size/rotation/noise).

    canonical=True disables the random rotation (cell drawn at a fixed
    orientation) — used by the rotation-robustness experiment so the contrast
    between invariant and non-invariant features is not masked by training-time
    rotation augmentation."""
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    R = rng.uniform(70, 150)
    noise = 1 + rng.normal(0, 0.02, t.shape)        # membrane roughness

    if cls == "discocyte":                          # normal RBC: near-round
        asp = rng.uniform(0.85, 1.0)
        poly = _polar(R * noise, t) * np.array([1.0, asp])
    elif cls == "elliptocyte":                      # elongated oval
        asp = rng.uniform(0.32, 0.55)
        poly = _polar(R * noise, t) * np.array([1.0, asp])
    elif cls == "echinocyte":                       # many small regular spicules
        k = rng.integers(11, 18)
        amp = rng.uniform(0.10, 0.18)
        poly = _polar(R * (1 + amp * np.cos(k * t)) * noise, t)
    elif cls == "teardrop":                         # one pointed tail
        tail = 1 + rng.uniform(0.5, 0.9) * np.exp(-((((t - np.pi) % (2 * np.pi)) - np.pi) ** 2) / 0.05)
        asp = rng.uniform(0.7, 0.9)
        poly = _polar(R * tail * noise, t) * np.array([1.0, asp])
    elif cls == "sickle":                           # crescent via mask subtraction
        canvas = 420
        img = np.zeros((canvas, canvas), np.uint8)
        cx = canvas // 2
        cv2.circle(img, (cx, cx), int(R), 255, -1)
        off = int(R * rng.uniform(0.45, 0.7))
        cv2.circle(img, (cx + off, cx), int(R * rng.uniform(0.9, 1.05)), 0, -1)
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        poly = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        poly = poly - poly.mean(0)
    else:
        raise ValueError(cls)

    # random rotation (skipped for the rotation-robustness experiment)
    if not canonical:
        a = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        poly = poly @ rot.T
    return _rasterize(poly)


# ==========================================================================
# Feature extraction (contour -> invariant vector via the GPU Crofton path)
# ==========================================================================

def contour_to_features(contour: np.ndarray):
    pts = contour.astype(np.float32)
    resampled = crofton_gpu.resample_contour(pts)
    centered = crofton_gpu.center_on_bbox(resampled)
    res = crofton_gpu.crofton_descriptor_gpu(centered)
    return F.feature_vector(centered, res.curve, res.diameter)


def _load_real_dataset(data_dir: Path):
    """Load a real labeled dataset into features. Recognizes Chula-RBC-12 and
    erythrocytesIDB via the datasets adapters; otherwise expects class
    subfolders of cell images."""
    import datasets as DS
    X, y, names = [], [], None
    d = Path(data_dir)
    if DS.is_chula(d):
        recs, _ = DS.find_chula(d)
        for contour, lbl, _img in recs:
            vec, names = contour_to_features(contour)
            X.append(vec); y.append(lbl)
        classes = sorted(set(y))
    elif list(d.rglob("circular")) or list(d.rglob("mask-circular")):
        recs, _ = DS.find_erythrocytesIDB(d)
        for contour, lbl, _img in recs:
            vec, names = contour_to_features(contour)
            X.append(vec); y.append(lbl)
        classes = DS.IDB_CLASSES
    else:
        classes = sorted([s.name for s in d.iterdir() if s.is_dir()])
        for cls in classes:
            for img_path in sorted((d / cls).glob("*")):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                contour, _, _ = crofton_gpu.extract_contour(img)
                if contour is None:
                    continue
                vec, names = contour_to_features(contour)
                X.append(vec); y.append(cls)
    return np.array(X), np.array(y), names, sorted(set(y))


def build_synthetic(per_class: int, rng):
    X, y, names = [], [], None
    for cls in CLASSES:
        for _ in range(per_class):
            vec, names = contour_to_features(gen_contour(cls, rng))
            X.append(vec); y.append(cls)
    return np.array(X), np.array(y), names, CLASSES


# ==========================================================================
# Train
# ==========================================================================

def train(data_dir: str | None = None, per_class: int = 150, seed: int = 0,
          out: str | None = None, name: str | None = None):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix

    out_path = Path(out) if out else MODEL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if data_dir:
        print(f"Loading real dataset from {data_dir} ...")
        X, y, names, classes = _load_real_dataset(Path(data_dir))
    else:
        print(f"Generating synthetic dataset ({per_class}/class, {len(CLASSES)} classes) ...")
        X, y, names, classes = build_synthetic(per_class, rng)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")

    clf = RandomForestClassifier(n_estimators=400, max_depth=None,
                                 class_weight="balanced", random_state=seed, n_jobs=-1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    print("\n=== 5-fold cross-validation ===")
    print(classification_report(y, y_pred, digits=3))
    print("Confusion matrix (rows=true, cols=pred), labels:", classes)
    print(confusion_matrix(y, y_pred, labels=classes))

    clf.fit(X, y)
    order = np.argsort(clf.feature_importances_)[::-1]
    print("\nTop feature importances:")
    for i in order[:8]:
        print(f"  {names[i]:24s} {clf.feature_importances_[i]:.3f}")

    # per-class stats for explainability + anomaly scoring
    mean = X.mean(0); std = X.std(0) + 1e-9
    class_mean = {c: X[y == c].mean(0) for c in classes}

    # ----- unsupervised anomaly detector (Stage e) ----------------------
    # One-Class SVM trained ONLY on NORMAL cells: flags deformities unlike any
    # normal shape, including types absent from the training labels.
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    normal = next((c for c in ("Normal", "discocyte", "circular") if c in classes), None)
    if normal is None:
        normal = max(classes, key=lambda c: int((y == c).sum()))   # most populous
    Xn = X[y == normal]
    scaler = StandardScaler().fit(Xn)
    ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale").fit(scaler.transform(Xn))
    train_pred = ocsvm.predict(scaler.transform(Xn))
    print(f"\nAnomaly detector: OneClassSVM trained on '{normal}' "
          f"({len(Xn)} cells); flags {(train_pred == -1).mean()*100:.0f}% of "
          "train normals as outliers (≈nu).")

    artifact = {
        "clf": clf, "feature_names": names, "classes": classes,
        "global_mean": mean, "global_std": std, "class_mean": class_mean,
        "ocsvm": ocsvm, "anom_scaler": scaler, "normal_class": normal,
        "source": "synthetic" if not data_dir else str(data_dir),
        "name": name or (out_path.stem if out else "default"),
    }
    joblib.dump(artifact, out_path)
    acc = (y_pred == y).mean()
    print(f"\nSaved {out_path}  (CV accuracy {acc:.3f}, source={artifact['source']})")
    return acc


# ==========================================================================
# Inference
# ==========================================================================

class CellClassifier:
    def __init__(self, path: Path = MODEL_PATH):
        import joblib
        a = joblib.load(path)
        self.clf = a["clf"]
        self.names = a["feature_names"]
        self.classes = a["classes"]
        self.mean = a["global_mean"]
        self.std = a["global_std"]
        self.class_mean = a["class_mean"]
        self.ocsvm = a.get("ocsvm")
        self.anom_scaler = a.get("anom_scaler")
        self.normal_class = a.get("normal_class")

    def predict_vector(self, vec: np.ndarray) -> dict:
        proba = self.clf.predict_proba([vec])[0]
        k = int(np.argmax(proba))
        label = self.clf.classes_[k]
        # drivers: features where the cell most resembles the predicted class,
        # weighted by the model's global importance (interpretable explanation)
        z = (vec - self.mean) / self.std
        imp = self.clf.feature_importances_
        score = imp * np.abs(z)
        top = np.argsort(score)[::-1][:3]
        drivers = [{"name": self.names[i], "value": float(vec[i]),
                    "z": float(z[i]), "importance": float(imp[i])} for i in top]
        # distance to nearest class mean (z-space) — kept as a fallback signal
        anom = min(float(np.linalg.norm((vec - self.class_mean[c]) / self.std))
                   for c in self.classes)
        # One-Class SVM verdict against the NORMAL manifold (unsupervised)
        atypical, ocsvm_score = None, None
        if self.ocsvm is not None and self.anom_scaler is not None:
            xs = self.anom_scaler.transform([vec])
            ocsvm_score = float(self.ocsvm.decision_function(xs)[0])  # >0 inlier
            atypical = bool(self.ocsvm.predict(xs)[0] == -1)          # -1 outlier
        return {
            "label": str(label),
            "confidence": float(proba[k]),
            "proba": {str(c): float(p) for c, p in zip(self.clf.classes_, proba)},
            "drivers": drivers,
            "anomaly_score": anom,
            "ocsvm_score": ocsvm_score,
            "atypical": atypical,
            "normal_class": self.normal_class,
        }

    def predict_contour(self, contour: np.ndarray) -> dict:
        vec, _ = contour_to_features(contour)
        return self.predict_vector(vec)


# ==========================================================================
# CLI
# ==========================================================================

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        data = None
        per_class = 150
        out = None; name = None
        for i, a in enumerate(sys.argv):
            if a == "--data" and i + 1 < len(sys.argv):
                data = sys.argv[i + 1]
            if a == "--per-class" and i + 1 < len(sys.argv):
                per_class = int(sys.argv[i + 1])
            if a == "--out" and i + 1 < len(sys.argv):
                out = sys.argv[i + 1]
            if a == "--name" and i + 1 < len(sys.argv):
                name = sys.argv[i + 1]
        train(data, per_class, out=out, name=name)
    elif cmd == "eval":
        clf = CellClassifier()
        rng = np.random.default_rng(123)
        ok = 0; n = 0
        for cls in CLASSES:
            for _ in range(20):
                r = clf.predict_contour(gen_contour(cls, rng))
                ok += (r["label"] == cls); n += 1
        print(f"Held-out synthetic accuracy: {ok}/{n} = {ok/n:.3f}")
    else:
        print(__doc__)
