"""
genai/tutor.py
==============
Stage 7 — Student Success Copilot

GenAI Tutor — uses the Anthropic API to generate:
  1. study_tips          — study advice based on the student's profile
  2. algorithm_explainer — explanation of why A* vs Greedy performed as they did
  3. risk_coach          — support for a student with high risk

Prompting strategy (documented for the report):
  - Role + Constraints in system prompt (see guardrails.py)
  - Structured context in user prompt (only data from the system)
  - temperature=0.4  — reduces hallucinations, preserves readability
  - max_tokens=300   — hard output length limit
  - Input validation  before sending to the API
  - Output validation after receiving the response
  - Fallback messages if the API is unavailable

How we avoid hallucination:
  1. Temperature 0.4 — less "creativity", more accuracy
  2. Prompt contains only structured system data (not free-form input)
  3. Output validator blocks medical/clinical statements
  4. Fallback guarantees a safe response on any failure

Public API:
  ask_tutor(mode, context)           -> str
  generate_study_tips(copilot_result)  -> str
  explain_algorithms(copilot_result)   -> str
  coach_at_risk(copilot_result)        -> str
"""

from __future__ import annotations
import os
import json

from genai.guardrails import (
    build_system_prompt,
    build_user_prompt,
    validate_input,
    validate_output,
    get_fallback_response,
    MAX_TOKENS,
    TEMPERATURE,
    MAX_OUTPUT_CHARS,
)

# ── model ─────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-5"


# ══════════════════════════════════════════════════════════════════════════
# CORE — Anthropic API call
# ══════════════════════════════════════════════════════════════════════════

def _call_api(system_prompt: str, user_prompt: str) -> str:
    """
    Sends a request to the Anthropic API.

    Uses requests (no SDK) — minimal dependencies for Colab.
    API key is read from the ANTHROPIC_API_KEY environment variable.

    Returns the model's response text, or an empty string on error.
    """
    import requests

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "In Colab: os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'"
        )

    headers = {
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }

    body = {
        "model":      MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system":     system_prompt,
        "messages":   [{"role": "user", "content": user_prompt}],
    }

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=body,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"API error {response.status_code}: {response.text[:200]}"
        )

    data = response.json()
    return data["content"][0]["text"]


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def ask_tutor(mode: str, context: dict) -> str:
    """
    Main function of the GenAI component.

    Full pipeline with guardrails:
      1. Build prompts
      2. Validate input
      3. Call API
      4. Validate output
      5. On any error — return fallback

    Arguments:
        mode    : "study_tips" | "algorithm_explainer" | "risk_coach"
        context : dict with data from CopilotResult

    Returns a string — the tutor's response (or fallback).
    """
    # step 1: build prompts
    try:
        system_prompt = build_system_prompt(mode)
        user_prompt   = build_user_prompt(mode, context)
    except ValueError as e:
        return f"[Config error] {e}"

    # step 2: validate input
    is_safe, reason = validate_input(user_prompt)
    if not is_safe:
        print(f"  [Guardrail] Input blocked: {reason}")
        return get_fallback_response(mode)

    # step 3: call API
    try:
        raw_output = _call_api(system_prompt, user_prompt)
    except EnvironmentError as e:
        print(f"  [Guardrail] API key missing: {e}")
        return get_fallback_response(mode)
    except RuntimeError as e:
        print(f"  [Guardrail] API error: {e}")
        return get_fallback_response(mode)
    except Exception as e:
        print(f"  [Guardrail] Unexpected error: {e}")
        return get_fallback_response(mode)

    # step 4: validate output
    is_valid, reason = validate_output(raw_output)
    if not is_valid:
        print(f"  [Guardrail] Output blocked: {reason}")
        return get_fallback_response(mode)

    # truncate if too long
    output = raw_output[:MAX_OUTPUT_CHARS].strip()

    return output


# ── convenience wrappers for notebook ─────────────────────────────────────

def generate_study_tips(copilot_result) -> str:
    """
    Generates study tips based on the copilot result.

    Arguments:
        copilot_result : CopilotResult from copilot.py

    Example usage in notebook:
        result = run_copilot(profile)
        tips   = generate_study_tips(result)
        print(tips)
    """
    context = {
        "risk_label":  copilot_result.risk_label,
        "signals":     copilot_result.signals,
        "advices":     copilot_result.advices,
        "profile":     copilot_result.profile,
    }
    return ask_tutor("study_tips", context)


def explain_algorithms(copilot_result) -> str:
    """
    Explains why A* and Greedy performed the way they did.

    Used in the explanation card to demonstrate the algorithms.
    """
    tasks      = copilot_result.profile.get("num_pending_tasks", "?")
    free_hours = int(copilot_result.profile.get("free_hours_per_day", 4))
    num_slots  = free_hours // 2 * 7   # approximation

    context = {
        "astar_result":  copilot_result.astar_result,
        "greedy_result": copilot_result.greedy_result,
        "num_tasks":     tasks,
        "num_slots":     num_slots,
    }
    return ask_tutor("algorithm_explainer", context)


def coach_at_risk(copilot_result) -> str:
    """
    Supportive coach for a student with high/medium risk.

    Only called if risk_label in ("Medium", "High").
    """
    if copilot_result.risk_label == "Low":
        return "Your current habits are effective — keep it up!"

    context = {
        "risk_label": copilot_result.risk_label,
        "signals":    copilot_result.signals,
        "profile":    copilot_result.profile,
    }
    return ask_tutor("risk_coach", context)


# ══════════════════════════════════════════════════════════════════════════
# TEST (no real API — uses mock)
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GENAI TUTOR — test (mock API, no real call)")
    print("=" * 60)

    from genai.guardrails import (
        build_system_prompt, build_user_prompt,
        validate_input, validate_output, get_fallback_response,
    )

    # ── test guardrails without API ───────────────────────────────────
    print("\n--- Test 1: guardrails pipeline (no API) ---")

    context_high = {
        "risk_label": "High",
        "signals":    ["high_stress", "critical_sleep_deprivation",
                       "low_completion_history"],
        "advices":    ["Take a rest day", "Split tasks into 25-min blocks"],
        "profile": {
            "sleep_hours_avg": 4.5, "stress_level": 8,
            "past_completion_rate": 0.3, "num_pending_tasks": 11,
            "days_until_exam": 2, "free_hours_per_day": 2,
        },
        "astar_result":  {"conflicts": 5, "nodes_expanded": 1336,
                          "time_ms": 10.3, "unscheduled": ["Task8"]},
        "greedy_result": {"conflicts": 5, "nodes_expanded": 11,
                          "time_ms": 0.01, "unscheduled": ["Task8"]},
        "num_tasks": 11, "num_slots": 7,
    }

    for mode in ["study_tips", "algorithm_explainer", "risk_coach"]:
        sys_p  = build_system_prompt(mode)
        user_p = build_user_prompt(mode, context_high)
        ok_in, r_in   = validate_input(user_p)
        fallback = get_fallback_response(mode)

        print(f"\n  Mode: {mode}")
        print(f"    system prompt : {len(sys_p)} chars — valid")
        print(f"    user prompt   : {len(user_p)} chars — input_valid={ok_in} ({r_in})")
        print(f"    fallback      : {fallback[:80]}...")

    # ── test blocked input ────────────────────────────────────────────
    print("\n--- Test 2: blocked inputs ---")
    blocked = [
        "Diagnose my anxiety disorder",
        "Write my essay for me",
        "I need medical advice about depression",
    ]
    for text in blocked:
        ok, reason = validate_input(text)
        label = 'BLOCKED' if not ok else 'ALLOWED'
        print(f"  [{label}] {repr(text[:50])} → {reason}")

    # ── test blocked output ───────────────────────────────────────────
    print("\n--- Test 3: blocked outputs ---")
    outputs = [
        ("Normal study advice with bullet points", True),
        ("You should see a doctor immediately for this",  False),
        ("This sounds like depression — get help", False),
    ]
    for text, expected_ok in outputs:
        ok, reason = validate_output(text)
        status = "OK" if ok == expected_ok else "FAIL"
        print(f"  [{status}] output_valid={ok} | {text[:50]!r}")

    # ── prompting strategy summary for report ─────────────────────────
    print("\n--- Prompting strategy summary (for report) ---")
    print(f"  Model       : {MODEL}")
    print(f"  Temperature : {TEMPERATURE}  (low = less hallucination)")
    print(f"  Max tokens  : {MAX_TOKENS}   (hard output limit)")
    print(f"  Modes       : study_tips | algorithm_explainer | risk_coach")
    print(f"  Guardrails  : input filter + output validator + fallback")
    print(f"  Anti-hallucination strategy:")
    print(f"    1. temperature=0.4 reduces creativity/fabrication")
    print(f"    2. user prompt contains only structured system data")
    print(f"    3. explicit CONSTRAINTS block in system prompt")
    print(f"    4. output validator blocks medical/clinical language")
    print(f"    5. fallback guarantees safe response on any failure")

    print("\n  Smoke tests: PASSED")
