"""
planner/greedy.py
=================
Stage 4 — Student Success Copilot

Greedy Search — greedy scheduling algorithm.

Why Greedy is fast but suboptimal:
  - Always picks the most urgent task (earliest deadline first)
  - Assigns to the FIRST available slot — with no lookahead
  - No backtracking — decisions are final
  - Complexity O(n log n) — sort + single pass
  - May create deadline conflicts that could have been avoided

Used for COMPARISON with A*:
  Greedy = speed, A* = solution quality.
"""

from __future__ import annotations
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planner.state import StudentScheduleState


def greedy_schedule(tasks: list, time_slots: list) -> dict:
    """
    Builds a schedule using the greedy algorithm.

    Arguments:
        tasks      : list of task dicts [{"name", "duration", "deadline"}, ...]
        time_slots : list of slot labels ["Mon1", "Mon2", ...]

    Returns dict:
        schedule       : {slot -> task_name}
        conflicts      : number of deadline violations
        nodes_expanded : number of tasks processed (= no branching)
        time_ms        : execution time
        unscheduled    : tasks that did not fit
    """
    t_start = time.perf_counter()

    # Step 1: sort by urgency — earliest deadline first
    # This is the "greedy" task selection strategy
    sorted_tasks = sorted(tasks, key=lambda t: t["deadline"])

    state          = StudentScheduleState(tasks, time_slots)
    nodes_expanded = 0

    for task in sorted_tasks:
        available = state.get_available_slots()

        if not available:
            # no more slots — task remains unscheduled
            nodes_expanded += 1
            continue

        # greedy slot choice: simply the first available (no lookahead)
        chosen_slot = available[0]

        state          = state.assign(task, chosen_slot)
        nodes_expanded += 1

    t_ms        = (time.perf_counter() - t_start) * 1000
    conflicts   = state.calculate_conflicts()
    scheduled   = set(state.schedule.values())
    unscheduled = [t["name"] for t in tasks if t["name"] not in scheduled]

    return {
        "schedule":       state.schedule,
        "conflicts":      conflicts,
        "nodes_expanded": nodes_expanded,
        "time_ms":        round(t_ms, 3),
        "unscheduled":    unscheduled,
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from planner.state import make_time_slots, make_tasks_from_profile

    print("=" * 60)
    print("GREEDY SEARCH — test scenarios A and B")
    print("=" * 60)

    # ── Scenario A: Normal ────────────────────────────────────────────
    print("\n--- Scenario A: Normal (4 tasks, 14 slots) ---")
    tasks_a = [
        {"name": "Math Essay",   "duration": 1, "deadline": 3},
        {"name": "ML Lab",       "duration": 1, "deadline": 5},
        {"name": "OS Report",    "duration": 1, "deadline": 9},
        {"name": "DB Project",   "duration": 1, "deadline": 12},
    ]
    slots_a = make_time_slots(free_hours_per_day=4, days=7)  # 14 slots

    result_a = greedy_schedule(tasks_a, slots_a)
    print(f"  Slots          : {len(slots_a)}")
    print(f"  Conflicts      : {result_a['conflicts']}")
    print(f"  Unscheduled    : {result_a['unscheduled']}")
    print(f"  Nodes          : {result_a['nodes_expanded']}")
    print(f"  Time           : {result_a['time_ms']} ms")
    print(f"  Schedule:")
    for slot, task in result_a["schedule"].items():
        print(f"    {slot:<8} → {task}")

    # ── Scenario B: At-risk ───────────────────────────────────────────
    print("\n--- Scenario B: At-risk (11 tasks, 7 slots) ---")
    profile_b = {
        "num_pending_tasks":       11,
        "days_until_exam":         3,
        "avg_task_duration_hours": 2,
        "free_hours_per_day":      2,
    }
    tasks_b = make_tasks_from_profile(profile_b)
    slots_b = make_time_slots(free_hours_per_day=2, days=7)  # 7 slots

    result_b = greedy_schedule(tasks_b, slots_b)
    print(f"  Tasks          : {len(tasks_b)}")
    print(f"  Slots          : {len(slots_b)}")
    print(f"  Conflicts      : {result_b['conflicts']}")
    print(f"  Unscheduled    : {result_b['unscheduled']}")
    print(f"  Nodes          : {result_b['nodes_expanded']}")
    print(f"  Time           : {result_b['time_ms']} ms")
    print(f"  Schedule:")
    for slot, task in result_b["schedule"].items():
        print(f"    {slot:<8} → {task}")

    # smoke tests
    assert result_a["conflicts"] == 0,    "Scenario A: no conflicts expected"
    assert result_a["unscheduled"] == [], "Scenario A: all tasks should be placed"
    assert len(result_b["unscheduled"]) > 0, "Scenario B: some tasks should not fit"

    print("\n  Smoke tests: PASSED")
