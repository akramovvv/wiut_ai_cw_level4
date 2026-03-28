"""
rules/backward_chain.py
=======================
Stage 3 — Student Success Copilot

Backward Chaining: goal-driven reasoning.

Logic:
  Goal — compute risk_level.
  To do this we need specific data (stress_level, sleep, etc.).
  Backward chain checks: which data is missing from the profile?
  For each missing key, it asks the student a question.

Why stress_level is the primary candidate:
  - It is a subjective metric that the student doesn't enter in advance
  - Rule R01_high_stress requires it
  - The question sounds natural and demonstrates system interactivity

Two modes:
  interactive=True  — asks questions via input() (for notebook demo)
  interactive=False — returns a list of missing keys (for tests)

Public API:
  run_backward_chain(profile, interactive, io_callback)
      -> BackwardChainResult
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ── goals: these keys are required to reason about risk ───────────────────
REQUIRED_FOR_RISK = [
    "stress_level",          # R01 — most important, often missing
    "sleep_hours_avg",       # R02
    "past_completion_rate",  # R03
    "days_until_exam",       # R04
    "num_pending_tasks",     # R04, R10
]

# ── questions for each key ────────────────────────────────────────────────
QUESTIONS = {
    "stress_level": {
        "text": (
            "\n[Backward Chaining] To assess burnout risk,\n"
            "I need to know your current stress level.\n"
            "How would you rate your stress right now?\n"
            "(1 = calm, 10 = very difficult): "
        ),
        "type": "int", "min": 1, "max": 10,
    },
    "sleep_hours_avg": {
        "text": (
            "\n[Backward Chaining] To check signal R02, I need sleep data.\n"
            "How many hours of sleep do you average per night this week?\n"
            "(e.g. 6.5): "
        ),
        "type": "float", "min": 0.0, "max": 24.0,
    },
    "past_completion_rate": {
        "text": (
            "\n[Backward Chaining] To check task completion history (R03):\n"
            "What fraction of your planned tasks do you usually complete?\n"
            "(0.0 = none, 1.0 = all, e.g. 0.7): "
        ),
        "type": "float", "min": 0.0, "max": 1.0,
    },
    "days_until_exam": {
        "text": (
            "\n[Backward Chaining] To assess urgency (R04):\n"
            "How many days until your nearest exam or major deadline?\n"
            "(e.g. 5): "
        ),
        "type": "int", "min": 0, "max": 90,
    },
    "num_pending_tasks": {
        "text": (
            "\n[Backward Chaining] To check overload (R04, R10):\n"
            "How many unfinished tasks are currently in your queue?\n"
            "(e.g. 8): "
        ),
        "type": "int", "min": 0, "max": 50,
    },
}


@dataclass
class BackwardChainResult:
    profile_complete: dict = field(default_factory=dict)
    questions_asked:  list = field(default_factory=list)   # question keys
    answers_received: dict = field(default_factory=dict)   # key -> value
    was_interactive:  bool = False
    missing_keys:     list = field(default_factory=list)   # keys not collected


def _find_missing(profile: dict) -> list:
    """Returns a list of keys from REQUIRED_FOR_RISK that are absent from the profile."""
    return [k for k in REQUIRED_FOR_RISK if profile.get(k) is None]


def _parse_and_validate(raw: str, key: str):
    """
    Parses and validates the user's answer.
    Returns (value, error_str | None).
    """
    q = QUESTIONS[key]
    try:
        value = int(float(raw.strip())) if q["type"] == "int" else float(raw.strip())
    except (ValueError, AttributeError):
        return None, f"Expected a number between {q['min']} and {q['max']}"

    if not (q["min"] <= value <= q["max"]):
        return None, f"Value out of range [{q['min']}, {q['max']}]"

    return value, None


def run_backward_chain(
    profile: dict,
    interactive: bool = False,
    io_callback=None,
) -> BackwardChainResult:
    """
    Runs backward chaining.

    Arguments:
        profile      : profile dict — some keys may be absent
        interactive  : True → asks via input()
                       False → returns missing_keys without asking
        io_callback  : optional function (key, question_text) -> value
                       used in notebook instead of input()
                       if provided — takes priority over interactive

    Returns BackwardChainResult with profile_complete filled in.
    """
    profile_copy     = dict(profile)
    missing          = _find_missing(profile_copy)
    questions_asked  = []
    answers_received = {}

    if not missing:
        # all data already present — backward chain not needed
        return BackwardChainResult(
            profile_complete = profile_copy,
            questions_asked  = [],
            answers_received = {},
            was_interactive  = False,
            missing_keys     = [],
        )

    was_interactive = False

    if io_callback is not None:
        # notebook mode: callback handles input
        for key in missing:
            if key not in QUESTIONS:
                continue
            q_text = QUESTIONS[key]["text"]
            try:
                raw   = io_callback(key, q_text)
                value, err = _parse_and_validate(str(raw), key)
                if err is None:
                    profile_copy[key]   = value
                    questions_asked.append(key)
                    answers_received[key] = value
            except Exception:
                pass   # callback failed — leave empty
        was_interactive = True

    elif interactive:
        # CLI mode: input() with retry on error
        print(
            "\n[Backward Chaining] System detected missing data.\n"
            f"To complete the analysis, please answer {len(missing)} question(s)."
        )
        for key in missing:
            if key not in QUESTIONS:
                continue
            q_text = QUESTIONS[key]["text"]
            while True:
                try:
                    raw = input(q_text)
                except (EOFError, KeyboardInterrupt):
                    print("\n[skipped]")
                    break

                value, err = _parse_and_validate(raw, key)
                if err:
                    print(f"  Error: {err}. Please try again.")
                    continue

                profile_copy[key]   = value
                questions_asked.append(key)
                answers_received[key] = value
                print(f"  Recorded: {key} = {value}")
                break
        was_interactive = True

    # final list of what is still missing
    still_missing = _find_missing(profile_copy)

    return BackwardChainResult(
        profile_complete = profile_copy,
        questions_asked  = questions_asked,
        answers_received = answers_received,
        was_interactive  = was_interactive,
        missing_keys     = still_missing,
    )


# ══════════════════════════════════════════════════════════════════════════
# TEST (non-interactive mode)
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("BACKWARD CHAINING — test (non-interactive)")
    print("=" * 60)

    # profile without stress_level — the most common case
    incomplete_profile = {
        "free_hours_per_day":      2.0,
        "num_pending_tasks":       11.0,
        "avg_task_duration_hours": 4.0,
        "deadline_urgency":        0.9,
        # "stress_level":          <-- intentionally missing
        "sleep_hours_avg":         4.5,
        "past_completion_rate":    0.3,
        "extracurricular_hours":   15.0,
        "missed_classes_pct":      0.4,
        "days_until_exam":         2.0,
    }

    print("\n--- Test 1: profile without stress_level ---")
    result = run_backward_chain(incomplete_profile, interactive=False)
    print(f"  missing_keys      : {result.missing_keys}")
    print(f"  was_interactive   : {result.was_interactive}")
    print(f"  profile has stress: {'stress_level' in result.profile_complete}")

    assert "stress_level" in result.missing_keys, \
        "stress_level should be in missing_keys"

    print("\n--- Test 2: callback mode (notebook simulation) ---")
    def mock_callback(key, text):
        answers = {"stress_level": "8"}
        print(f"  [mock] question about '{key}' → answering '{answers.get(key, '5')}'")
        return answers.get(key, "5")

    result2 = run_backward_chain(incomplete_profile,
                                  interactive=False,
                                  io_callback=mock_callback)
    print(f"  questions_asked   : {result2.questions_asked}")
    print(f"  answers_received  : {result2.answers_received}")
    print(f"  stress_level now  : {result2.profile_complete.get('stress_level')}")

    assert result2.profile_complete.get("stress_level") == 8

    print("\n--- Test 3: complete profile — backward chain not needed ---")
    full_profile = dict(incomplete_profile)
    full_profile["stress_level"] = 7.0
    result3 = run_backward_chain(full_profile, interactive=False)
    print(f"  questions_asked   : {result3.questions_asked}  (should be empty)")
    assert result3.questions_asked == []

    print("\n  Smoke tests: PASSED")
