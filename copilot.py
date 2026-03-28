"""
copilot.py
==========
Stage 5 — Student Success Copilot

Single entry point of the system.

Pipeline:
  1. Backward chaining  → collects missing data
  2. ML predict         → predicts risk level
  3. Forward chaining   → fires rules, generates advice
  4. Planner A* + Greedy → builds schedule
  5. Explanation card   → combines everything into a single report
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field

from rules.backward_chain import run_backward_chain
from rules.forward_chain  import run_forward_chain
from ml.train             import load_model, predict
from ml.explain           import explain_ml
from planner.state        import make_time_slots, make_tasks_from_profile
from planner.astar        import astar_schedule, compare_algorithms
from planner.greedy       import greedy_schedule
from genai.tutor          import generate_study_tips, explain_algorithms, coach_at_risk


# ══════════════════════════════════════════════════════════════════════════
# CopilotResult
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CopilotResult:
    profile:             dict  = field(default_factory=dict)
    # backward
    questions_asked:     list  = field(default_factory=list)
    answers_received:    dict  = field(default_factory=dict)
    # ML (normalised keys for external use)
    risk_label:          str   = "Unknown"
    probabilities:       dict  = field(default_factory=dict)
    ml_explanation:      str   = ""
    # rules
    signals:             list  = field(default_factory=list)
    rules_risk_level:    str   = "Unknown"
    advices:             list  = field(default_factory=list)
    rules_explanation:   str   = ""
    # planner
    astar_result:        dict  = field(default_factory=dict)
    greedy_result:       dict  = field(default_factory=dict)
    comparison_table:    str   = ""
    # genai
    genai_study_tips:    str   = ""
    genai_algo_explain:  str   = ""
    genai_risk_coach:    str   = ""
    # final output
    explanation_card:    str   = ""
    total_time_ms:       float = 0.0

    def print_report(self):
        print(self.explanation_card)


# ══════════════════════════════════════════════════════════════════════════
# NORMALISATION — adapter between predict() and the rest of the system
# ══════════════════════════════════════════════════════════════════════════

def _extract_risk_and_proba(raw: dict) -> tuple[str, dict]:
    """
    predict() may return different keys depending on train.py version:
      Version A: {"label": str, "proba": dict, "top_features": list}
      Version B: {"risk_label": str, "probabilities": dict, "importances": dict}

    Returns (risk_label, probabilities) in a unified format.
    """
    risk_label    = raw.get("risk_label") or raw.get("label", "Unknown")
    probabilities = raw.get("probabilities") or raw.get("proba", {})
    return risk_label, probabilities


# ══════════════════════════════════════════════════════════════════════════
# EXPLANATION CARD — assembles the report from all three components
# ══════════════════════════════════════════════════════════════════════════

def _build_card(profile, raw_pred, fc,
                astar_res, greedy_res,
                comparison, tasks, time_slots,
                genai_tips="", genai_algo="", genai_coach="") -> str:

    risk_label, probabilities = _extract_risk_and_proba(raw_pred)
    confidence = round(probabilities.get(risk_label, 0) * 100)
    icon = {"Low": "[OK]", "Medium": "[!]", "High": "[!!]"}.get(risk_label, "[?]")
    sep  = "─" * 60

    # explain_ml accepts raw predict() output — pass raw_pred
    ml_text = explain_ml(raw_pred, profile)

    lines = [
        "",
        "=" * 60,
        "   STUDENT SUCCESS COPILOT — REPORT",
        "=" * 60,
        "",
        f"  {icon}  RISK: {risk_label.upper()}  (confidence: {confidence}%)",
        sep,
    ]

    for ln in ml_text.strip().splitlines():
        lines.append(f"  {ln}")
    lines.append("")

    # ── Rules ─────────────────────────────────────────────────────────
    lines += ["  EXPERT SYSTEM (forward chaining):", sep]
    for ln in fc.explanation.strip().splitlines():
        lines.append(f"  {ln}")
    lines.append("")

    if fc.advices:
        lines.append("  Advice:")
        for adv in fc.advices[:3]:
            for ln in adv.strip().splitlines():
                lines.append(f"  • {ln}")
        lines.append("")

    # ── Planner ───────────────────────────────────────────────────────
    lines += ["  PLANNER (A* vs Greedy):", sep]

    if not astar_res["unscheduled"]:
        lines.append(
            f"  A* built a complete schedule: "
            f"{len(tasks)} tasks in {len(time_slots)} available slots."
        )
    else:
        placed  = len(astar_res["schedule"])
        missing = astar_res["unscheduled"]
        lines.append(f"  [!] A* placed {placed} of {len(tasks)} tasks.")
        lines.append(f"  Could not fit: {', '.join(missing)}.")
        lines.append(
            f"  With {int(profile.get('free_hours_per_day', 4))} free hrs/day "
            f"× 7 days = {len(time_slots)} slots. "
            f"More tasks than slots."
        )

    if astar_res["schedule"]:
        lines += ["", "  Schedule (A*):"]
        items = list(astar_res["schedule"].items())
        for slot, task in items[:7]:
            lines.append(f"    {slot:<8} → {task}")
        if len(items) > 7:
            lines.append(f"    ... and {len(items) - 7} more tasks")

    lines.append("")
    lines.append("  Algorithm comparison:")
    for ln in comparison.splitlines():
        lines.append(f"  {ln}")

    # ── GenAI Tutor ────────────────────────────────────────────────
    if any([genai_tips, genai_algo, genai_coach]):
        lines += ["  GENERATIVE AI TUTOR:", sep]
        if genai_tips:
            lines.append("  Study tips:")
            for ln in genai_tips.strip().splitlines():
                lines.append(f"  {ln}")
            lines.append("")
        if genai_algo:
            lines.append("  Why A* vs Greedy:")
            for ln in genai_algo.strip().splitlines():
                lines.append(f"  {ln}")
            lines.append("")
        if genai_coach:
            lines.append("  Support:")
            for ln in genai_coach.strip().splitlines():
                lines.append(f"  {ln}")
            lines.append("")

    lines += ["", "=" * 60, ""]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def run_copilot(
    profile:     dict,
    interactive: bool = False,
    io_callback       = None,
) -> CopilotResult:
    """
    Runs the full Student Success Copilot pipeline.

    Arguments:
        profile      : student profile dict (may be partial)
        interactive  : True → backward chain asks via input()
        io_callback  : function(key, question_text) -> value for notebook use

    Returns CopilotResult.

    Example in notebook:
        result = run_copilot(student_profile, io_callback=my_widget_fn)
        result.print_report()
    """
    t0     = time.perf_counter()
    result = CopilotResult(profile=dict(profile))

    # ── 1. Backward chaining ──────────────────────────────────────────
    bc                      = run_backward_chain(profile,
                                                  interactive=interactive,
                                                  io_callback=io_callback)
    full_profile            = bc.profile_complete
    result.profile          = full_profile
    result.questions_asked  = bc.questions_asked
    result.answers_received = bc.answers_received

    # ── 2. ML ─────────────────────────────────────────────────────────
    bundle   = load_model()
    raw_pred = predict(full_profile, bundle)          # raw predict() output
    risk_label, probabilities = _extract_risk_and_proba(raw_pred)

    result.risk_label     = risk_label
    result.probabilities  = probabilities
    result.ml_explanation = explain_ml(raw_pred, full_profile)  # raw → explain_ml

    # ── 3. Forward chaining ───────────────────────────────────────────
    fc                       = run_forward_chain(full_profile)
    result.signals           = fc.signals
    result.rules_risk_level  = fc.risk_level
    result.advices           = fc.advices
    result.rules_explanation = fc.explanation

    # ── 4. Planner ────────────────────────────────────────────────────
    tasks      = make_tasks_from_profile(full_profile)
    free_hours = int(full_profile.get("free_hours_per_day", 4))
    time_slots = make_time_slots(free_hours_per_day=free_hours, days=7)

    astar_res  = astar_schedule(tasks, time_slots)
    greedy_res = greedy_schedule(tasks, time_slots)
    comparison = compare_algorithms(astar_res, greedy_res, tasks, time_slots)

    result.astar_result     = astar_res
    result.greedy_result    = greedy_res
    result.comparison_table = comparison

    # ── 5. GenAI Tutor (optional — requires ANTHROPIC_API_KEY) ───────
    try:
        result.genai_study_tips   = generate_study_tips(result)
        result.genai_algo_explain = explain_algorithms(result)
        result.genai_risk_coach   = coach_at_risk(result)
    except Exception:
        # API key not set or API unavailable — continue without GenAI
        pass

    # ── 6. Explanation card ───────────────────────────────────────────
    result.explanation_card = _build_card(
        full_profile, raw_pred, fc,
        astar_res, greedy_res, comparison,
        tasks, time_slots,
        genai_tips  = result.genai_study_tips,
        genai_algo  = result.genai_algo_explain,
        genai_coach = result.genai_risk_coach,
    )

    result.total_time_ms = round((time.perf_counter() - t0) * 1000, 2)
    return result


# ══════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Scenario A: Normal ────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  SCENARIO A — Normal User")
    print("█" * 60)

    profile_a = {
        "free_hours_per_day":      5,
        "num_pending_tasks":       4,
        "avg_task_duration_hours": 2,
        "deadline_urgency":        0.2,
        "days_until_exam":         14,
        "stress_level":            4,
        "sleep_hours_avg":         7.5,
        "past_completion_rate":    0.85,
        "extracurricular_hours":   6,
        "missed_classes_pct":      0.05,
    }

    result_a = run_copilot(profile_a)
    result_a.print_report()
    print(f"  Total time: {result_a.total_time_ms} ms")

    assert result_a.risk_label       in ("Low", "Medium", "High")
    assert result_a.rules_risk_level == "Low"
    assert result_a.astar_result["unscheduled"] == []
    print("  Smoke tests A: PASSED\n")

    # ── Scenario B: At-risk ───────────────────────────────────────────
    print("█" * 60)
    print("  SCENARIO B — At-risk User (backward chaining demo)")
    print("█" * 60)

    profile_b = {
        "free_hours_per_day":      2,
        "num_pending_tasks":       11,
        "avg_task_duration_hours": 2,
        "deadline_urgency":        0.9,
        "days_until_exam":         2,
        # stress_level intentionally missing → backward chain will ask
        "sleep_hours_avg":         4.5,
        "past_completion_rate":    0.3,
        "extracurricular_hours":   15,
        "missed_classes_pct":      0.4,
    }

    def mock_callback(key, question_text):
        answers = {"stress_level": 8}
        val = answers.get(key, 5)
        print(f"\n  [Backward chaining] {question_text.strip()}")
        print(f"  [Student answer] {key} = {val}")
        return val

    result_b = run_copilot(profile_b, io_callback=mock_callback)
    result_b.print_report()
    print(f"  Total time: {result_b.total_time_ms} ms")

    assert result_b.risk_label       in ("Low", "Medium", "High")
    assert result_b.rules_risk_level == "High"
    assert result_b.questions_asked  == ["stress_level"]
    assert result_b.answers_received == {"stress_level": 8}
    assert len(result_b.astar_result["unscheduled"]) > 0
    print("  Smoke tests B: PASSED")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print(f"  Scenario A: {result_a.total_time_ms} ms")
    print(f"  Scenario B: {result_b.total_time_ms} ms")
    print("=" * 60)
