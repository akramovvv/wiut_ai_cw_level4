"""
rules/knowledge_base.py
=======================
Stage 3 — Student Success Copilot

Expert system knowledge base.
Rules are stored as a list of dictionaries — this makes them:
  - readable for the examiner
  - easily extendable (add a dict — add a rule)
  - suitable for both types of chaining

IMPORTANT: thresholds deliberately match those in:
  - data/generate_dataset.py  label_risk()
  - ml/explain.py             THRESHOLDS
This is the essence of Explainability: rules explain what the ML model learned.

Rule structure:
  id          — unique identifier
  conditions  — list of lambda functions (profile -> bool)
  signal      — string label of the fired rule
  severity    — "high" | "medium" | "low"
  advice      — advice for the student
  explanation — why this is a problem (for the explanation card)
  requires    — profile keys needed for evaluation
                (backward chaining will ask about missing ones)
"""

RULES = [

    # ══════════════════════════════════════════════
    # HIGH — critical signals (2+ → risk=High)
    # ══════════════════════════════════════════════

    {
        "id": "R01_high_stress",
        "conditions": [
            lambda p: float(p.get("stress_level", 0)) > 7,
        ],
        "signal":      "high_stress",
        "severity":    "high",
        "confidence":  0.85,
        "advice": (
            "Stress level is critically high. "
            "Schedule a full rest day within the next 2 days. "
            "Break large tasks into 25-minute blocks (Pomodoro technique)."
        ),
        "explanation": (
            "Stress above 7/10 reduces productivity by 30-40% "
            "and impairs long-term memory consolidation. "
            "This is one of the five critical signals in the model."
        ),
        "requires": ["stress_level"],
    },

    {
        "id": "R02_critical_sleep",
        "conditions": [
            lambda p: float(p.get("sleep_hours_avg", 10)) < 5.5,
        ],
        "signal":      "critical_sleep_deprivation",
        "severity":    "high",
        "confidence":  0.80,
        "advice": (
            "Less than 5.5 hours of sleep — critical level. "
            "Go to bed before 23:00 tonight. "
            "Remove all tasks after 22:00 from your schedule."
        ),
        "explanation": (
            "Chronic sleep deprivation (<5.5 hrs) is cognitively equivalent "
            "to 1.5 days without sleep. "
            "Studying after midnight under this regime is ineffective."
        ),
        "requires": ["sleep_hours_avg"],
    },

    {
        "id": "R03_low_completion",
        "conditions": [
            lambda p: float(p.get("past_completion_rate", 1)) < 0.4,
        ],
        "signal":      "low_completion_history",
        "severity":    "high",
        "confidence":  0.80,
        "advice": (
            "Task completion rate is below 40%. "
            "Pick 3 tasks with the nearest deadlines — focus only on those. "
            "One task at a time; do not spread yourself thin."
        ),
        "explanation": (
            "Completion rate <40% is the #1 predictor of academic risk "
            "in our model (feature importance: 22%). "
            "The usual cause is an over-broad task list without prioritisation."
        ),
        "requires": ["past_completion_rate"],
    },

    {
        "id": "R04_exam_crunch",
        "conditions": [
            lambda p: float(p.get("days_until_exam", 30)) <= 3,
            lambda p: float(p.get("num_pending_tasks",  0)) >= 6,
        ],
        "signal":      "exam_crunch",
        "severity":    "high",
        "confidence":  0.75,
        "advice": (
            "Exam in 3 days with 6+ tasks in the queue — overload. "
            "Academic triage: split tasks into 'before exam' and 'after exam'. "
            "The planner will help you prioritise."
        ),
        "explanation": (
            "An imminent exam combined with a large task queue creates cognitive overload. "
            "Trying to do everything at once is worse than making a deliberate choice."
        ),
        "requires": ["days_until_exam", "num_pending_tasks"],
    },

    {
        "id": "R05_high_absences",
        "conditions": [
            lambda p: float(p.get("missed_classes_pct", 0)) > 0.35,
        ],
        "signal":      "high_absences",
        "severity":    "high",
        "confidence":  0.75,
        "advice": (
            "More than 35% of classes missed. "
            "This may affect your eligibility to sit exams. "
            "Contact your department office or academic advisor today."
        ),
        "explanation": (
            "A high absence rate correlates with disqualification from exams "
            "and knowledge gaps that are difficult to fill independently."
        ),
        "requires": ["missed_classes_pct"],
    },

    # ══════════════════════════════════════════════
    # MEDIUM — warning signals
    # ══════════════════════════════════════════════

    {
        "id": "R06_elevated_stress",
        "conditions": [
            lambda p: 6 <= float(p.get("stress_level", 0)) <= 7,
        ],
        "signal":      "elevated_stress",
        "severity":    "medium",
        "confidence":  0.65,
        "advice": (
            "Stress is elevated (6-7/10). "
            "Add a 30-minute break every 2 hours of work. "
            "Short physical activity lowers cortisol."
        ),
        "explanation": (
            "Stress at 6-7/10 is a borderline zone. "
            "Without intervention it can escalate to a critical level within 1-2 days."
        ),
        "requires": ["stress_level"],
    },

    {
        "id": "R07_insufficient_sleep",
        "conditions": [
            lambda p: 5.5 <= float(p.get("sleep_hours_avg", 10)) < 6.5,
        ],
        "signal":      "insufficient_sleep",
        "severity":    "medium",
        "confidence":  0.60,
        "advice": (
            "Sleep is slightly below the normal range (5.5-6.5 hrs). "
            "Adding even 30-45 minutes makes a noticeable difference "
            "to concentration and focus."
        ),
        "explanation": (
            "The recommended amount for students is 7-9 hrs. "
            "6-6.5 hrs is not yet critical, but reduces memory consolidation quality."
        ),
        "requires": ["sleep_hours_avg"],
    },

    {
        "id": "R08_medium_completion",
        "conditions": [
            lambda p: 0.4 <= float(p.get("past_completion_rate", 1)) < 0.6,
        ],
        "signal":      "medium_completion",
        "severity":    "medium",
        "confidence":  0.60,
        "advice": (
            "Completion history is 40-60% — a risk zone. "
            "Try the two-tasks-per-day rule: "
            "small wins build momentum."
        ),
        "explanation": (
            "A 50% completion rate means every second task stays unfinished. "
            "The typical cause is underestimating how long tasks take."
        ),
        "requires": ["past_completion_rate"],
    },

    {
        "id": "R09_extracurricular_overload",
        "conditions": [
            lambda p: float(p.get("extracurricular_hours", 0)) > 12,
        ],
        "signal":      "extracurricular_overload",
        "severity":    "medium",
        "confidence":  0.55,
        "advice": (
            "More than 12 hrs/week of extracurricular activities under the current load is high. "
            "Consider temporarily reducing commitments until the end of the exam period."
        ),
        "explanation": (
            "Under academic pressure, every additional hour away from studying "
            "is an hour without recovery."
        ),
        "requires": ["extracurricular_hours"],
    },

    {
        "id": "R10_time_task_mismatch",
        "conditions": [
            lambda p: float(p.get("free_hours_per_day", 10)) < 2,
            lambda p: float(p.get("num_pending_tasks",   0)) >= 5,
        ],
        "signal":      "time_task_mismatch",
        "severity":    "medium",
        "confidence":  0.70,
        "advice": (
            "Too little time (<2 hrs/day) + too many tasks (5+) — the maths doesn't work. "
            "The planner will show the real picture: "
            "which tasks physically cannot fit into the week."
        ),
        "explanation": (
            "With fewer than 2 free hours per day it is impossible to complete 5+ tasks well. "
            "A* will report this honestly, unlike Greedy."
        ),
        "requires": ["free_hours_per_day", "num_pending_tasks"],
    },

    # ══════════════════════════════════════════════
    # LOW — positive confirmation
    # ══════════════════════════════════════════════

    {
        "id": "R11_good_habits",
        "conditions": [
            lambda p: float(p.get("sleep_hours_avg",        0)) >= 7.0,
            lambda p: float(p.get("past_completion_rate",   0)) >= 0.75,
            lambda p: float(p.get("stress_level",          10)) <= 5,
        ],
        "signal":      "good_academic_habits",
        "severity":    "low",
        "confidence":  0.90,
        "advice": (
            "Excellent routine: sleep, task completion, and stress are all within range. "
            "Maintain this rhythm going into the exam period."
        ),
        "explanation": (
            "Three key indicators within range simultaneously — "
            "a strong predictor of successfully completing the semester."
        ),
        "requires": ["sleep_hours_avg", "past_completion_rate", "stress_level"],
    },
]

# quick lookup by id
RULES_BY_ID = {r["id"]: r for r in RULES}

# all keys required by at least one rule
ALL_REQUIRED_KEYS = sorted({
    key for rule in RULES for key in rule["requires"]
})


if __name__ == "__main__":
    print(f"Knowledge base: {len(RULES)} rules\n")
    for r in RULES:
        print(f"  {r['id']:<30}  severity={r['severity']:<6}  "
              f"confidence={r['confidence']:.2f}  "
              f"conditions={len(r['conditions'])}  "
              f"requires={r['requires']}")
    print(f"\nAll required keys: {ALL_REQUIRED_KEYS}")
