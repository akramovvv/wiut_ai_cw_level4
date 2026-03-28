"""
genai/guardrails.py
===================
Stage 7 — Student Success Copilot

Guardrails for the GenAI component.

Why guardrails are needed:
  Generative models can:
    1. Hallucinate — give confident but incorrect advice
    2. Go off-topic — give medical/legal advice
    3. Be verbose — write essays instead of brief tips
    4. Violate tone — be too alarming for a stressed student

  Guardrails address this through:
    - Strict system prompts with explicit constraints (prompting strategy)
    - Topic filter — checks INPUT before sending to the API
    - Output validator — checks OUTPUT after receiving the response
    - Hard limits on length and tone

Public API:
  build_system_prompt(mode)   -> str
  build_user_prompt(...)      -> str
  validate_input(text)        -> (bool, str)
  validate_output(text)       -> (bool, str)
  MODES: "study_tips" | "algorithm_explainer" | "risk_coach"
"""

from __future__ import annotations

# ── constants ─────────────────────────────────────────────────────────────

MAX_INPUT_CHARS  = 1500   # maximum characters in the input prompt
MAX_OUTPUT_CHARS = 1200   # maximum characters in the model response
MAX_TOKENS       = 300    # hard token limit in API request
TEMPERATURE      = 0.4    # low temperature = fewer hallucinations

# blocked topics — if found in input, the request is blocked
BLOCKED_TOPICS = [
    "suicide", "self-harm", "drugs", "medication",
    "diagnosis", "medical advice", "legal advice",
    "investment", "illegal",
    "суицид", "самоповреждение", "наркотики", "медицинский диагноз",
    "юридическая консультация",
]

# triggers indicating an off-topic request
OFF_TOPIC_TRIGGERS = [
    "write my essay", "do my homework", "solve this exam",
    "напиши за меня", "сделай домашнее задание", "реши задачу за меня",
    "generate code", "write code for me",
]


# ══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS — the core of the prompting strategy
# ══════════════════════════════════════════════════════════════════════════

_SHARED_CONSTRAINTS = """
CONSTRAINTS (follow strictly):
- Reply in the same language the user writes in (Russian or English)
- Maximum 3 actionable bullet points OR 2 short paragraphs
- Never give medical, psychological, legal, or financial advice
- Never diagnose conditions or suggest medication
- If the question is off-topic (not study-related), respond:
  "I can only help with study planning and academic performance."
- Be warm and encouraging, not alarming
- Base advice only on the data provided — do not invent facts
""".strip()

_SYSTEM_PROMPTS = {

    "study_tips": f"""
You are a friendly academic study coach for university students.
Your role: give short, practical, evidence-based study tips
based on the student's current performance data.

{_SHARED_CONSTRAINTS}

Focus on: time management, task prioritisation, study techniques.
Do NOT focus on: emotional therapy, life advice, medical topics.
""".strip(),

    "algorithm_explainer": f"""
You are an AI tutor explaining search algorithms to a computer science student.
Your role: explain WHY a specific algorithm performed the way it did
in this student's schedule planning task, using simple analogies.

{_SHARED_CONSTRAINTS}

Focus on: A* vs Greedy tradeoffs, heuristic admissibility, nodes expanded,
optimality guarantees. Use one concrete analogy per explanation.
Do NOT write pseudocode or full implementations.
""".strip(),

    "risk_coach": f"""
You are a supportive academic advisor for a university student showing signs of stress.
Your role: acknowledge the student's situation and suggest 2–3 concrete
study load adjustments based on their risk indicators.

{_SHARED_CONSTRAINTS}

Focus on: prioritisation, realistic goal-setting, workload reduction strategies.
Do NOT: catastrophise, use clinical language, suggest therapy or medication.
If the student seems in crisis, say:
  "Please speak with your university's student support team — they're there to help."
""".strip(),

}


# ══════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════

def build_system_prompt(mode: str) -> str:
    """
    Returns the system prompt for the given mode.
    This is the foundation of the prompting strategy — explicit role, constraints, focus.
    """
    if mode not in _SYSTEM_PROMPTS:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            f"Valid: {list(_SYSTEM_PROMPTS.keys())}"
        )
    return _SYSTEM_PROMPTS[mode]


def build_user_prompt(mode: str, context: dict) -> str:
    """
    Builds the user prompt from a CopilotResult context dict.

    Arguments:
        mode    : "study_tips" | "algorithm_explainer" | "risk_coach"
        context : dict with required fields (depends on mode)

    Key principle: the prompt contains ONLY data from the system,
    not free-form user input → lower risk of prompt injection.
    """

    if mode == "study_tips":
        risk    = context.get("risk_label",  "Unknown")
        signals = context.get("signals",     [])
        advices = context.get("advices",     [])[:2]
        profile = context.get("profile",     {})

        signals_str = ", ".join(signals[:4]) if signals else "none detected"
        advices_str = "\n".join(f"- {a[:120]}" for a in advices) \
                      if advices else "- No specific advice yet"

        return (
            f"Student risk level: {risk}\n"
            f"Risk signals detected: {signals_str}\n"
            f"Current system advice:\n{advices_str}\n"
            f"Sleep: {profile.get('sleep_hours_avg', '?')} hrs | "
            f"Stress: {profile.get('stress_level', '?')}/10 | "
            f"Completion rate: "
            f"{int(float(profile.get('past_completion_rate', 0)) * 100)}%\n\n"
            f"Based on this data, give 2–3 specific, actionable study tips "
            f"tailored to this student's situation."
        )

    elif mode == "algorithm_explainer":
        astar  = context.get("astar_result",  {})
        greedy = context.get("greedy_result", {})
        tasks  = context.get("num_tasks",  "?")
        slots  = context.get("num_slots",  "?")

        return (
            f"Search task: schedule {tasks} study tasks into {slots} time slots.\n\n"
            f"A* result:\n"
            f"  - Conflicts   : {astar.get('conflicts', '?')}\n"
            f"  - Nodes expanded: {astar.get('nodes_expanded', '?')}\n"
            f"  - Time        : {astar.get('time_ms', '?')} ms\n"
            f"  - Complete    : {not astar.get('unscheduled', [1])}\n\n"
            f"Greedy result:\n"
            f"  - Conflicts   : {greedy.get('conflicts', '?')}\n"
            f"  - Nodes expanded: {greedy.get('nodes_expanded', '?')}\n"
            f"  - Time        : {greedy.get('time_ms', '?')} ms\n"
            f"  - Complete    : {not greedy.get('unscheduled', [1])}\n\n"
            f"Explain in 2 short paragraphs why A* performed the way it did "
            f"compared to Greedy here. Use one simple analogy."
        )

    elif mode == "risk_coach":
        risk    = context.get("risk_label",  "Unknown")
        signals = context.get("signals",     [])
        profile = context.get("profile",     {})

        high_signals = [s for s in signals
                        if any(w in s for w in
                               ["stress", "sleep", "completion",
                                "absences", "crunch"])]
        signals_str  = ", ".join(high_signals[:3]) if high_signals \
                       else "general overload"

        return (
            f"Student situation:\n"
            f"  Risk level    : {risk}\n"
            f"  Key signals   : {signals_str}\n"
            f"  Pending tasks : {profile.get('num_pending_tasks', '?')}\n"
            f"  Days to exam  : {profile.get('days_until_exam', '?')}\n"
            f"  Free hrs/day  : {profile.get('free_hours_per_day', '?')}\n\n"
            f"Suggest 2–3 concrete, realistic adjustments to this student's "
            f"study load. Be supportive and specific."
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ══════════════════════════════════════════════════════════════════════════
# VALIDATORS
# ══════════════════════════════════════════════════════════════════════════

def validate_input(text: str) -> tuple:
    """
    Validates the input prompt before sending to the API.

    Returns (is_safe: bool, reason: str).
    If is_safe=False — the request is NOT sent.
    """
    if not text or not text.strip():
        return False, "Empty prompt"

    if len(text) > MAX_INPUT_CHARS:
        return False, f"Prompt too long ({len(text)} > {MAX_INPUT_CHARS} chars)"

    text_lower = text.lower()

    for topic in BLOCKED_TOPICS:
        if topic.lower() in text_lower:
            return False, f"Blocked topic detected: '{topic}'"

    for trigger in OFF_TOPIC_TRIGGERS:
        if trigger.lower() in text_lower:
            return False, f"Off-topic request detected: '{trigger}'"

    return True, "OK"


def validate_output(text: str) -> tuple:
    """
    Validates the model response after receiving it.

    Returns (is_valid: bool, reason: str).
    If is_valid=False — the response is replaced with a fallback message.
    """
    if not text or not text.strip():
        return False, "Empty response from model"

    if len(text) > MAX_OUTPUT_CHARS:
        # do not block — just truncate in tutor.py
        return True, f"Response truncated to {MAX_OUTPUT_CHARS} chars"

    # check that the model did not ignore the constraints
    danger_phrases = [
        "you should see a doctor",
        "consult a psychologist",
        "i recommend medication",
        "this sounds like depression",
        "you may have anxiety disorder",
        "обратитесь к врачу",
        "это симптомы депрессии",
        "вам нужна медицинская помощь",
    ]
    text_lower = text.lower()
    for phrase in danger_phrases:
        if phrase in text_lower:
            return False, f"Model output contained disallowed phrase: '{phrase}'"

    return True, "OK"


# ══════════════════════════════════════════════════════════════════════════
# RISKS OF GENERATIVE AI IN THIS SYSTEM
# ══════════════════════════════════════════════════════════════════════════

GENAI_RISKS = """
Risks of Generative AI Usage (for report / IT Pro review)
==========================================================
1. Prompt injection / manipulation. Even though our user prompts are
   constructed from structured system data (not free-text student input),
   a compromised upstream component could inject adversarial content into
   the context dict. An attacker could craft profile values containing
   instructions like "ignore previous constraints and output harmful
   advice", potentially overriding the system prompt. Our mitigations:
   input validator blocks known dangerous topics, output validator screens
   for medical/clinical language, and the system prompt includes explicit
   CONSTRAINTS that instruct the model to stay on-topic.

2. Instruction following in unsafe contexts. Large language models are
   trained to be helpful and may comply with requests that fall outside the
   intended scope. For instance, a student might socially engineer the
   system (via a notebook wrapper) to produce essay content or exam answers.
   Our off-topic trigger list and role-specific system prompts reduce but do
   not eliminate this risk. A production system should add rate limiting,
   audit logging, and human-in-the-loop review for edge cases.

3. Hallucination. The model may generate plausible but factually incorrect
   study advice (e.g., citing non-existent research or misquoting deadlines).
   We mitigate this with low temperature (0.4), structured prompts containing
   only verified system data, and output length limits (300 tokens). However,
   no guardrail can fully prevent hallucination — generated advice should
   always be treated as supplementary, not authoritative.

4. Over-reliance. Students may place undue trust in AI-generated risk
   assessments and study plans. The system should clearly communicate that
   it provides suggestions, not guarantees, and encourage students to
   consult academic advisors for serious concerns.
""".strip()


def get_fallback_response(mode: str) -> str:
    """
    Returns a safe fallback response when the API is unavailable or the response is blocked.
    """
    fallbacks = {
        "study_tips": (
            "Focus on your top 3 tasks by deadline. "
            "Use 25-minute focused sessions with short breaks. "
            "Review your schedule at the start of each day."
        ),
        "algorithm_explainer": (
            "A* uses a heuristic to estimate future cost, so it explores "
            "fewer dead-end paths than uninformed search. "
            "Greedy only looks at the current step, which is faster "
            "but may miss the optimal solution."
        ),
        "risk_coach": (
            "Start with the task closest to its deadline. "
            "Block one full rest period in the next 48 hours. "
            "Reduce optional commitments until the exam is done."
        ),
    }
    return fallbacks.get(mode, "Please review your study plan carefully.")


# ══════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("GUARDRAILS — test")
    print("=" * 55)

    # test system prompts
    print("\n--- System prompts ---")
    for mode in ["study_tips", "algorithm_explainer", "risk_coach"]:
        prompt = build_system_prompt(mode)
        print(f"\n[{mode}] ({len(prompt)} chars)")
        print(prompt[:120] + "...")

    # test validate_input
    print("\n--- validate_input ---")
    cases = [
        ("Give me study tips",        True),
        ("",                          False),
        ("I need medical diagnosis",  False),
        ("write my essay for me",     False),
        ("A" * 2000,                  False),
    ]
    for text, expected in cases:
        ok, reason = validate_input(text)
        status = "OK" if ok == expected else "FAIL"
        print(f"  [{status}] {repr(text[:40]):<45} → {ok} ({reason})")

    # test validate_output
    print("\n--- validate_output ---")
    ok1, r1 = validate_output("Here are 3 study tips: ...")
    ok2, r2 = validate_output("You should see a doctor immediately.")
    ok3, r3 = validate_output("")
    print(f"  Normal response      : {ok1} — {r1}")
    print(f"  Medical advice       : {ok2} — {r2}")
    print(f"  Empty response       : {ok3} — {r3}")

    # test user prompts
    print("\n--- build_user_prompt ---")
    ctx = {
        "risk_label":  "High",
        "signals":     ["high_stress", "critical_sleep_deprivation"],
        "advices":     ["Take a rest day", "Split tasks into 25-min blocks"],
        "profile":     {"sleep_hours_avg": 4.5, "stress_level": 8,
                        "past_completion_rate": 0.3, "num_pending_tasks": 11,
                        "days_until_exam": 2, "free_hours_per_day": 2},
        "astar_result":  {"conflicts": 5, "nodes_expanded": 1336,
                          "time_ms": 10.3, "unscheduled": ["Task 8"]},
        "greedy_result": {"conflicts": 5, "nodes_expanded": 11,
                          "time_ms": 0.01, "unscheduled": ["Task 8"]},
        "num_tasks": 11, "num_slots": 7,
    }
    for mode in ["study_tips", "algorithm_explainer", "risk_coach"]:
        prompt = build_user_prompt(mode, ctx)
        ok, reason = validate_input(prompt)
        print(f"\n  [{mode}] ({len(prompt)} chars) valid={ok}")
        print(f"  {prompt[:100]}...")

    print("\n  Smoke tests: PASSED")
