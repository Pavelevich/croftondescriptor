"""Stage (g+): evaluate the smear-level CONDITION layer (per-condition sensitivity).

The datasets are labeled per CELL, not per disease, so we cannot score against
clinical ground truth. Instead we measure the layer's DETECTION SENSITIVITY in a
controlled way that folds in real classifier error:

  For each condition with a clear single trigger morphology present in the
  dataset, compose synthetic "pseudo-smears" from REAL labeled cells — k% cells
  of the trigger TRUE class + the rest Normal — run the full deployed pipeline
  (classify every cell on GPU -> aggregate -> conditions.assess), and record how
  often the expected condition is flagged, as a function of the seeded abundance.

This answers: "if a smear truly contains X% of morphology M, how reliably does
the system raise the associated condition?" — a real, honest operating-curve.

Usage: .venv/bin/python cuda_version/webapp/conditions_eval.py --data DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as DS
import classify
import conditions

# trigger TRUE class (in the dataset) -> condition key we expect assess() to raise
TRIGGER_TO_CONDITION = {
    "Schistocyte": "Microangiopathic hemolytic anemia",
    "Elliptocyte": "Hereditary elliptocytosis",
    "Spherocyte": "spherocytosis",            # matches HS/AIHA condition text
    "Teardrop": "myelofibrosis",
    "Target": "Thalassemia / liver",
    "Ovalocyte": None,                         # contributes to several; skip exact
}
ABUNDANCES = [0.05, 0.10, 0.20, 0.35, 0.50]
SMEAR_SIZE = 60
REPEATS = 12
NORMAL = "Normal"


def _matches(cond_name: str, key: str) -> bool:
    return key.lower() in cond_name.lower()


def run(data_dir, seed=0):
    print(f"Loading {data_dir} ...")
    recs, _ = DS.find_chula(Path(data_dir))
    by_class = {}
    for contour, lbl, _img in recs:
        by_class.setdefault(lbl, []).append(contour)
    print("cells per class:", {k: len(v) for k, v in by_class.items()})
    if NORMAL not in by_class:
        print("No Normal class — cannot compose pseudo-smears"); return

    clf = classify.CellClassifier(Path(__file__).resolve().parent / "models" / "chula.joblib")
    rng = np.random.default_rng(seed)

    def classify_pool(contours):
        """Classify a list of contours -> counts dict + n_atypical."""
        counts, n_atyp = {}, 0
        for c in contours:
            p = clf.predict_contour(np.asarray(c, np.float64))
            counts[p["label"]] = counts.get(p["label"], 0) + 1
            n_atyp += int(p["atypical"]) if p["atypical"] is not None else 0
        return counts, n_atyp

    targets = [(t, c) for t, c in TRIGGER_TO_CONDITION.items()
               if c and t in by_class and len(by_class[t]) >= 10]
    normals = by_class[NORMAL]

    results = {}   # condition -> list of detection rate per abundance
    fa = {}        # false-alarm: rate on pure-normal smears
    # false-alarm baseline (0% abnormal)
    fa_hits = {t: 0 for t, _ in targets}
    for _ in range(REPEATS):
        pool = [normals[i] for i in rng.integers(0, len(normals), SMEAR_SIZE)]
        counts, n_atyp = classify_pool(pool)
        a = conditions.assess(counts, n_atyp, SMEAR_SIZE)
        for t, cond in targets:
            if any(_matches(c["condition"], cond) for c in a["candidates"]):
                fa_hits[t] += 1

    print("\n" + "=" * 70)
    print("CONDITION-LAYER SENSITIVITY  (detection rate vs seeded abundance)")
    print("real cells; full classify->assess pipeline; %d trials/cell" % REPEATS)
    print("-" * 70)
    hdr = f"{'condition':<34}" + "".join(f"{int(a*100):>5}%" for a in ABUNDANCES) + f"{'  FA(0%)':>9}"
    print(hdr)
    for t, cond in targets:
        pool_trig = by_class[t]
        row = []
        for frac in ABUNDANCES:
            k = max(1, int(round(frac * SMEAR_SIZE)))
            hits = 0
            for _ in range(REPEATS):
                trig = [pool_trig[i] for i in rng.integers(0, len(pool_trig), k)]
                norm = [normals[i] for i in rng.integers(0, len(normals), SMEAR_SIZE - k)]
                counts, n_atyp = classify_pool(trig + norm)
                a = conditions.assess(counts, n_atyp, SMEAR_SIZE)
                if any(_matches(c["condition"], cond) for c in a["candidates"]):
                    hits += 1
            row.append(hits / REPEATS)
        results[cond] = row
        fa[cond] = fa_hits[t] / REPEATS
        label = f"{cond[:32]} ({t})"
        print(f"{label[:34]:<34}" + "".join(f"{r*100:>5.0f}" for r in row) + f"{fa[cond]*100:>8.0f}%")
    print("=" * 70)

    # figure
    out = Path(__file__).resolve().parents[1] / "results" / "conditions"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond, row in results.items():
        ax.plot([a * 100 for a in ABUNDANCES], [r * 100 for r in row], marker="o", label=cond[:28])
    ax.set_xlabel("seeded abundance of trigger morphology (% of smear)")
    ax.set_ylabel("detection rate (%)")
    ax.set_title("Condition-layer sensitivity (real cells, full pipeline)")
    ax.set_ylim(-3, 105); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "condition_sensitivity.png", dpi=130); plt.close(fig)

    md = ["# Condition-layer sensitivity\n",
          f"Pseudo-smears of {SMEAR_SIZE} REAL Chula cells, {REPEATS} trials per point; "
          "each trial = k% trigger-morphology cells + Normal background, run through the "
          "full deployed pipeline (per-cell GPU classification -> conditions.assess).\n",
          "Detection rate = fraction of trials where the expected condition was flagged. "
          "FA(0%) = false-alarm rate on pure-Normal smears.\n",
          "| condition (trigger class) | " + " | ".join(f"{int(a*100)}%" for a in ABUNDANCES) + " | FA(0%) |",
          "|---|" + "---|" * (len(ABUNDANCES) + 1)]
    for t, cond in targets:
        row = results[cond]
        md.append(f"| {cond[:34]} ({t}) | " + " | ".join(f"{r*100:.0f}%" for r in row)
                  + f" | {fa[cond]*100:.0f}% |")
    md.append("\n![sensitivity](condition_sensitivity.png)\n")
    md.append("**Reading.** The curve rising with abundance shows the layer detects a "
              "condition once its morphology is sufficiently present; the FA(0%) column is "
              "the specificity cost on normal smears. Sensitivity is bounded by the per-cell "
              "classifier (≈0.45 on 12 real classes), so this is a SCREENING operating "
              "curve, not diagnostic accuracy.\n")
    (out / "RESULTS.md").write_text("\n".join(md))
    print(f"\nWrote {out}/ RESULTS.md + condition_sensitivity.png")
    return results, fa


if __name__ == "__main__":
    data = None
    for i, a in enumerate(sys.argv):
        if a == "--data" and i + 1 < len(sys.argv): data = sys.argv[i + 1]
    run(data or "datasets/Chula-RBC-12-Dataset")
