"""
ml/explain.py
=============
Stage 2 — Student Success Copilot

explain_ml(prediction, profile) -> str
  Turns the result of predict() + student profile
  into a readable text explanation.

explain_ml_short(prediction, profile) -> str
  One-line summary for the explanation card in the UI.
"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════
# 1. TEMPLATES: feature + value → (text, severity)
#    Thresholds taken from label_risk() in generate_dataset.py — for consistency
# ══════════════════════════════════════════════════════════════════════════

def _describe_feature(feature: str, value: float) -> tuple[str, str]:
    """Returns (description, severity) where severity: critical|warning|ok."""

    if feature == "stress_level":
        if value > 7:   return f"very high stress ({int(value)}/10)", "critical"
        if value >= 6:  return f"elevated stress ({int(value)}/10)", "warning"
        return f"stress within normal range ({int(value)}/10)", "ok"

    if feature == "sleep_hours_avg":
        if value < 5.5: return f"acute sleep deprivation ({value:.1f} hrs/night)", "critical"
        if value < 6.5: return f"insufficient sleep ({value:.1f} hrs/night)", "warning"
        return f"sleep within normal range ({value:.1f} hrs/night)", "ok"

    if feature == "past_completion_rate":
        pct = int(value * 100)
        if value < 0.4: return f"low completion rate ({pct}% of tasks)", "critical"
        if value < 0.6: return f"moderate completion rate ({pct}% of tasks)", "warning"
        return f"good completion rate ({pct}% of tasks)", "ok"

    if feature == "days_until_exam":
        if value <= 3:  return f"exam very soon ({int(value)} days)", "critical"
        if value <= 7:  return f"exam this week ({int(value)} days)", "warning"
        return f"{int(value)} days until exam", "ok"

    if feature == "num_pending_tasks":
        if value >= 10: return f"many pending tasks ({int(value)})", "critical"
        if value >= 6:  return f"tasks piling up ({int(value)})", "warning"
        return f"tasks under control ({int(value)})", "ok"

    if feature == "free_hours_per_day":
        if value < 2:   return f"almost no free time ({int(value)} hrs/day)", "warning"
        if value < 4:   return f"limited free time ({int(value)} hrs/day)", "warning"
        return f"sufficient free time ({int(value)} hrs/day)", "ok"

    if feature == "missed_classes_pct":
        pct = int(value * 100)
        if value > 0.35: return f"high absences ({pct}%)", "critical"
        if value > 0.15: return f"some absences ({pct}%)", "warning"
        return f"good attendance ({pct}% missed)", "ok"

    if feature == "extracurricular_hours":
        if value > 12:  return f"high extracurricular load ({int(value)} hrs/week)", "warning"
        return f"moderate extracurricular load ({int(value)} hrs/week)", "ok"

    if feature == "avg_task_duration_hours":
        if value > 4:   return f"tasks are heavy ({value:.1f} hrs/task)", "warning"
        return f"tasks are moderate ({value:.1f} hrs/task)", "ok"

    if feature == "deadline_urgency":
        if value > 0.7: return f"deadlines critically close (urgency={value:.2f})", "critical"
        if value > 0.4: return f"deadlines approaching (urgency={value:.2f})", "warning"
        return f"deadlines not urgent (urgency={value:.2f})", "ok"

    return f"{feature} = {value:.2f}", "ok"


# ══════════════════════════════════════════════════════════════════════════
# 2. ADVICE BY RISK LEVEL
# ══════════════════════════════════════════════════════════════════════════

_ADVICE = {
    "High": [
        "Prioritise immediately: tackle tasks with the nearest deadlines first.",
        "Limit extracurricular activities until the end of the exam period.",
        "Reach out to your academic advisor or student support — it's the right thing to do.",
    ],
    "Medium": [
        "Review your schedule: block focused study periods free of distractions.",
        "Monitor your sleep — at least 7 hours sustains concentration.",
        "Break large tasks into steps to reduce the feeling of overload.",
    ],
    "Low": [
        "You're doing well. Maintain your current rhythm.",
        "Review upcoming deadlines for next week ahead of time.",
    ],
}


# ══════════════════════════════════════════════════════════════════════════
# 3. MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def _normalize_prediction(prediction: dict) -> tuple:
    """
    Normalises prediction keys to a unified format.
    Supports both formats:
      predict():  {"label", "proba", "top_features"}
      copilot:    {"risk_label", "probabilities", "importances"}
    Returns (label, proba_dict, top_features_list).
    """
    label     = prediction.get("label") or prediction.get("risk_label", "Unknown")
    proba     = prediction.get("proba") or prediction.get("probabilities", {})
    top_feats = prediction.get("top_features", [])
    return label, proba, top_feats


def explain_ml(prediction: dict, profile: dict, top_n: int = 3) -> str:
    """
    Parameters:
      prediction  — result of ml.train.predict() or a normalised dict:
                    {"label": str, "proba": dict, "top_features": list}
                    or {"risk_label": str, "probabilities": dict, ...}
      profile     — original student feature dictionary
      top_n       — number of features to include in the explanation

    Returns:
      Multi-line string, ready for print() or notebook output.
    """
    label, proba, top_feats = _normalize_prediction(prediction)
    confidence = proba.get(label, 0.0)

    icons = {"High": "[!]", "Medium": "[~]", "Low": "[OK]"}
    icon  = icons.get(label, "[?]")

    lines = []
    lines.append("─" * 54)
    lines.append(f"  {icon}  ML PREDICTION: {label.upper()} RISK")
    lines.append(f"       Confidence: {confidence*100:.0f}%")

    ordered = ["Low", "Medium", "High"]
    proba_str = "  |  ".join(
        f"{k}: {proba.get(k, 0)*100:.0f}%" for k in ordered
    )
    lines.append(f"       Probabilities: {proba_str}")
    lines.append("─" * 54)
    lines.append("  Key factors (feature importance):")
    lines.append("")

    critical_parts, warning_parts = [], []

    for feat_name, importance in top_feats[:top_n]:
        value = profile.get(feat_name)
        if value is None:
            continue
        desc, severity = _describe_feature(feat_name, value)
        weight = f"{importance*100:.1f}%"

        if severity == "critical":
            prefix = "  [CRITICAL]"
            critical_parts.append(desc)
        elif severity == "warning":
            prefix = "  [warning ]"
            warning_parts.append(desc)
        else:
            prefix = "  [  ok    ]"

        lines.append(f"{prefix}  {desc}  (weight {weight})")

    lines.append("")
    lines.append("  Conclusion:")

    if label == "High":
        factors = (critical_parts + warning_parts)[:2]
        if factors:
            lines.append(f"  High risk caused by: {' and '.join(factors)}.")
        lines.append("  Model detected ≥2 critical signals simultaneously.")
    elif label == "Medium":
        factors = (critical_parts + warning_parts)[:2]
        if factors:
            lines.append(f"  Moderate risk due to: {' and '.join(factors)}.")
        lines.append("  Situation is manageable but requires attention.")
    else:
        lines.append("  No significant risk factors detected.")
        lines.append("  Current study and rest habits are effective.")

    lines.append("")
    lines.append("  Recommendations:")
    for advice in _ADVICE.get(label, []):
        lines.append(f"  • {advice}")
    lines.append("─" * 54)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 4. SHORT VERSION — for explanation card in copilot.py
# ══════════════════════════════════════════════════════════════════════════

def explain_ml_short(prediction: dict, profile: dict) -> str:
    """
    One-line summary. Example:
      "High risk (73%): acute sleep deprivation (4.2 hrs/night) and low completion rate (22% of tasks)"
    """
    label, proba, all_feats = _normalize_prediction(prediction)
    confidence = proba.get(label, 0.0)
    top_feats  = all_feats[:2]

    parts = []
    for feat_name, _ in top_feats:
        value = profile.get(feat_name)
        if value is not None:
            desc, _ = _describe_feature(feat_name, value)
            parts.append(desc)

    reasons = " and ".join(parts) if parts else "several factors"
    return f"{label} risk ({confidence*100:.0f}%): {reasons}"


# ══════════════════════════════════════════════════════════════════════════
# 5. TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ml.train import load_model, predict as ml_predict

    bundle = load_model()

    profile_a = {
        "free_hours_per_day": 5.0, "num_pending_tasks": 3.0,
        "avg_task_duration_hours": 2.0, "deadline_urgency": 0.2,
        "days_until_exam": 14.0, "stress_level": 3.0,
        "sleep_hours_avg": 7.5, "past_completion_rate": 0.88,
        "extracurricular_hours": 5.0, "missed_classes_pct": 0.05,
    }
    profile_b = {
        "free_hours_per_day": 1.0, "num_pending_tasks": 11.0,
        "avg_task_duration_hours": 4.5, "deadline_urgency": 0.92,
        "days_until_exam": 2.0, "stress_level": 9.0,
        "sleep_hours_avg": 4.2, "past_completion_rate": 0.22,
        "extracurricular_hours": 16.0, "missed_classes_pct": 0.42,
    }

    print("\nTest explain_ml — two profiles:\n")

    r_a = ml_predict(profile_a, bundle)
    print("=== Scenario A (Normal User) ===")
    print(explain_ml(r_a, profile_a))

    r_b = ml_predict(profile_b, bundle)
    print("\n=== Scenario B (At-risk User) ===")
    print(explain_ml(r_b, profile_b))

    print(f"\n  Short (A): {explain_ml_short(r_a, profile_a)}")
    print(f"  Short (B): {explain_ml_short(r_b, profile_b)}")

    assert isinstance(explain_ml(r_a, profile_a), str)
    assert isinstance(explain_ml(r_b, profile_b), str)
    assert isinstance(explain_ml_short(r_b, profile_b), str)
    print("\n  Smoke tests: PASSED")
