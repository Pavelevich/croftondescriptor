"""Smear-level condition inference (decision-support, NOT diagnosis).

Aggregates per-cell morphology classifications into a population profile and
applies transparent, source-cited hematology rules to surface CANDIDATE
conditions with the evidence that triggered each one. This is the layer that
turns "here are the cell types" into "this smear is suggestive of X — confirm
with test Y".

IMPORTANT FRAMING (enforced in every output):
  * This is research / screening decision-support, not a clinical diagnosis.
  * Shape-only morphology cannot establish a disease; CBC indices (MCV/MCH/RDW),
    clinical context, and confirmatory labs are required.
The RULES table is populated from web-verified hematology sources (see
condition_rules.json / the smear-condition-rules workflow).
"""

from __future__ import annotations

# Map every classifier label (Chula-12, erythrocytesIDB, synthetic) to a
# canonical morphology token used by the rules.
LABEL_TO_MORPH = {
    # Chula-RBC-12
    "Normal": "normal", "Macrocyte": "macrocyte", "Microcyte": "microcyte",
    "Spherocyte": "spherocyte", "Target": "target", "Stomatocyte": "stomatocyte",
    "Ovalocyte": "ovalocyte", "Teardrop": "teardrop", "Burr": "burr",
    "Schistocyte": "schistocyte", "Hypochromia": "hypochromia", "Elliptocyte": "elliptocyte",
    # synthetic demo
    "discocyte": "normal", "echinocyte": "burr", "sickle": "sickle",
    # erythrocytesIDB
    "circular": "normal", "elongated": "elongated", "other": "poikilocyte",
}

# Rules: each fires when the summed fraction of its trigger morphologies exceeds
# `threshold` (a fraction of all cells). `requires_all` = every listed morph must
# itself exceed `min_each`. Populated/locked-in from verified sources.
# Schema per rule:
#   key, condition, triggers[list of morph], threshold(float), severity,
#   requires_all(list|None), min_each(float), confirm(str), source(str)
RULES = [
    # placeholder — replaced by verified ruleset in load_rules()
]


def _morph_fractions(counts: dict, label_to_morph=None) -> tuple[dict, int]:
    """counts: {class_label: n}. Returns (morph_fraction dict, total)."""
    m = label_to_morph or LABEL_TO_MORPH
    total = sum(counts.values()) or 1
    frac = {}
    for label, n in counts.items():
        token = m.get(label, label.lower())
        frac[token] = frac.get(token, 0.0) + n / total
    return frac, total


def assess(counts: dict, n_atypical: int = 0, total: int | None = None,
           label_to_morph=None) -> dict:
    """Return a decision-support assessment from the per-class counts.

    Output: {abnormal_fraction, candidates:[{condition, score, evidence,
    confirm, source, severity}], summary, disclaimer}.
    """
    frac, tot = _morph_fractions(counts, label_to_morph)
    total = total or tot
    atypical_frac = (n_atypical / total) if total else 0.0

    candidates = []
    for rule in RULES:
        present = {t: frac.get(t, 0.0) for t in rule["triggers"]}
        score = sum(present.values())
        fires = score >= rule.get("threshold", 0.0)
        if rule.get("requires_all"):
            fires = fires and all(frac.get(t, 0.0) >= rule.get("min_each", 0.0)
                                  for t in rule["requires_all"])
        if not fires:
            continue
        ev = ", ".join(f"{t} {present[t]*100:.0f}%" for t in rule["triggers"]
                       if present[t] > 0)
        candidates.append({
            "condition": rule["condition"],
            "score": round(score, 3),
            "severity": rule.get("severity", "info"),
            "evidence": ev,
            "confirm": rule.get("confirm", ""),
            "source": rule.get("source", ""),
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)

    normal_frac = frac.get("normal", 0.0)
    if not candidates and atypical_frac < 0.15 and normal_frac > 0.6:
        summary = "Predominantly normal red-cell morphology; no morphology flag."
    elif not candidates:
        summary = (f"Abnormal morphology present ({atypical_frac*100:.0f}% atypical) "
                   "but no specific pattern matched a rule.")
    else:
        summary = (f"Morphology suggestive of: "
                   + "; ".join(c["condition"] for c in candidates[:3])
                   + f". {atypical_frac*100:.0f}% of cells flagged atypical.")

    return {
        "abnormal_fraction": round(atypical_frac, 3),
        "morphology_profile": {k: round(v, 3) for k, v in sorted(frac.items(), key=lambda kv: -kv[1])},
        "candidates": candidates,
        "summary": summary,
        "disclaimer": ("Research / screening decision-support — NOT a diagnosis. "
                       "Shape morphology cannot establish disease; confirm with CBC "
                       "indices (MCV/MCH/RDW), clinical context and the listed tests."),
    }


def load_rules(rules_list):
    """Replace RULES with a verified ruleset (list of dicts)."""
    global RULES
    RULES = rules_list


def _autoload():
    """Load the web-verified ruleset from condition_rules.json next to this file."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent / "condition_rules.json"
    if p.exists():
        try:
            load_rules(json.loads(p.read_text()))
        except Exception:
            pass


_autoload()
