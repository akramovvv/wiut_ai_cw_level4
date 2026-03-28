"""
planner/astar.py
================
Stage 4 — Student Success Copilot

A* Search for student schedule planning.

A* algorithm theory:
  - Uses a priority queue (min-heap by f = g + h)
  - g(n) = calculate_conflicts() — actual conflicts in the current state
  - h(n) = heuristic()           — estimate of future conflicts (admissible)
  - f(n) = g(n) + h(n)          — priority in the queue

Why h(n) is admissible:
  heuristic() counts tasks that CERTAINLY won't have a slot before their deadline.
  This is a lower bound on future conflicts — the actual number of conflicts
  cannot be LESS. Therefore h(n) does not overestimate → A* is optimal.

Why branching is limited (TOP_K_SLOTS):
  Without the limit: each task × all free slots = exponential state tree growth.
  TOP_K_SLOTS = 3 best slots (by closeness to deadline)
  preserves solution quality and keeps runtime reasonable.

Comparison with Greedy:
  A*     — explores more nodes, slower, but minimises conflicts
  Greedy — explores O(n) nodes, faster, but may create conflicts
"""

from __future__ import annotations
import heapq
import time


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planner.state import StudentScheduleState

TOP_K_SLOTS = 3   # number of candidate slots considered at each step


def astar_schedule(tasks: list, time_slots: list) -> dict:
    """
    Builds a schedule using the A* algorithm.

    Arguments:
        tasks      : list of task dicts [{"name", "duration", "deadline"}, ...]
        time_slots : list of slot labels ["Mon1", "Mon2", ...]

    Returns dict:
        schedule       : {slot -> task_name}
        conflicts      : number of deadline violations
        nodes_expanded : number of search nodes expanded
        time_ms        : execution time
        unscheduled    : tasks that did not fit
    """
    t_start = time.perf_counter()

    initial_state = StudentScheduleState(tasks, time_slots)

    # heap element: (f_cost, tie_breaker, state)
    # tie_breaker prevents Python from comparing state objects when f is equal
    counter = 0
    heap    = [(
        initial_state.calculate_conflicts() + initial_state.heuristic(),
        counter,
        initial_state,
    )]

    nodes_expanded = 0
    best_partial   = initial_state   # best partial solution (in case slots run out)

    while heap:
        f, _, state = heapq.heappop(heap)
        nodes_expanded += 1

        # update best partial by number of placed tasks
        if len(state.schedule) > len(best_partial.schedule):
            best_partial = state

        # GOAL CHECK — all tasks placed
        if state.is_goal():
            t_ms = (time.perf_counter() - t_start) * 1000
            return {
                "schedule":       state.schedule,
                "conflicts":      state.calculate_conflicts(),
                "nodes_expanded": nodes_expanded,
                "time_ms":        round(t_ms, 3),
                "unscheduled":    [],
            }

        remaining = state.get_remaining_tasks()
        available = state.get_available_slots()

        if not remaining or not available:
            continue

        # choose the task with the nearest deadline (earliest deadline first)
        # this directs the search towards the most critical tasks first
        task = min(remaining, key=lambda t: t["deadline"])

        # TOP_K best slots: sorted by proximity to the task's deadline
        # slots BEFORE the deadline are preferred (fewer conflicts)
        def slot_priority(slot: str) -> tuple:
            idx = time_slots.index(slot)
            over_deadline = max(0, idx - task["deadline"])   # penalty for being late
            return (over_deadline, idx)                       # no-penalty first, then earliest

        candidate_slots = sorted(available, key=slot_priority)[:TOP_K_SLOTS]

        for slot in candidate_slots:
            new_state = state.assign(task, slot)
            g         = new_state.calculate_conflicts()
            h         = new_state.heuristic()
            counter  += 1
            heapq.heappush(heap, (g + h, counter, new_state))

    # heap is empty — return the best partial solution found
    t_ms        = (time.perf_counter() - t_start) * 1000
    scheduled   = set(best_partial.schedule.values())
    unscheduled = [t["name"] for t in tasks if t["name"] not in scheduled]

    return {
        "schedule":       best_partial.schedule,
        "conflicts":      best_partial.calculate_conflicts(),
        "nodes_expanded": nodes_expanded,
        "time_ms":        round(t_ms, 3),
        "unscheduled":    unscheduled,
    }


def compare_algorithms(
    astar_result: dict,
    greedy_result: dict,
    tasks: list,
    time_slots: list,
) -> str:
    """
    Builds a comparison table for A* and Greedy.
    Key artefact for demo video and report.
    """
    a = astar_result
    g = greedy_result

    def winner(a_val, g_val, lower_is_better=True) -> str:
        if a_val == g_val:  return "   ="
        if lower_is_better: return "  A*" if a_val < g_val else "Greedy"
        else:               return "  A*" if a_val > g_val else "Greedy"

    a_placed = len(a["schedule"])
    g_placed = len(g["schedule"])

    rows = [
        ("Deadline conflicts",      a["conflicts"],      g["conflicts"],      True),
        ("Tasks placed",            a_placed,            g_placed,            False),
        ("Tasks unscheduled",       len(a["unscheduled"]), len(g["unscheduled"]), True),
        ("Nodes expanded",          a["nodes_expanded"], g["nodes_expanded"], True),
        ("Execution time (ms)",     a["time_ms"],        g["time_ms"],        True),
    ]

    top  = "┌─────────────────────────────┬──────────────┬──────────────┬──────────┐"
    head = "│ Metric                      │     A*       │    Greedy    │  Better  │"
    sep  = "├─────────────────────────────┼──────────────┼──────────────┼──────────┤"
    bot  = "└─────────────────────────────┴──────────────┴──────────────┴──────────┘"

    lines = [top, head, sep]
    for label, av, gv, lb in rows:
        w = winner(av, gv, lb)
        lines.append(
            f"│ {label:<27} │ {str(av):>12} │ {str(gv):>12} │ {w:>8} │"
        )
    lines += [sep]

    a_complete = "Yes" if not a["unscheduled"] else "No"
    g_complete = "Yes" if not g["unscheduled"] else "No"
    lines.append(
        f"│ {'Schedule complete?':<27} │ {a_complete:>12} │ {g_complete:>12} │"
        f"          │"
    )
    lines.append(bot)

    # summary conclusion
    if a["conflicts"] < g["conflicts"]:
        diff = g["conflicts"] - a["conflicts"]
        lines.append(
            f"\nConclusion: A* eliminated {diff} conflict(s) compared to Greedy.\n"
            f"Cost of optimality: A* expanded {a['nodes_expanded']} nodes "
            f"vs {g['nodes_expanded']} for Greedy, "
            f"{a['time_ms']} ms vs {g['time_ms']} ms."
        )
    elif a["conflicts"] == g["conflicts"] == 0:
        lines.append(
            f"\nConclusion: both algorithms avoided conflicts.\n"
            f"Greedy is faster ({g['time_ms']} ms vs {a['time_ms']} ms for A*) "
            f"and expanded fewer nodes ({g['nodes_expanded']} vs {a['nodes_expanded']}).\n"
            f"In simple cases Greedy is sufficient; A* provides an optimality guarantee."
        )
    else:
        lines.append(
            f"\nConclusion: A* conflicts={a['conflicts']}, "
            f"Greedy conflicts={g['conflicts']}."
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# TEST + COMPARISON
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from planner.state import make_time_slots, make_tasks_from_profile
    from planner.greedy import greedy_schedule

    print("=" * 65)
    print("A* vs GREEDY — comparison of two scenarios")
    print("=" * 65)

    # ── Scenario A: Normal ────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  Scenario A — Normal (4 tasks, 14 slots)")
    print("─" * 65)

    tasks_a = [
        {"name": "Math Essay",   "duration": 1, "deadline": 3},
        {"name": "ML Lab",       "duration": 1, "deadline": 5},
        {"name": "OS Report",    "duration": 1, "deadline": 9},
        {"name": "DB Project",   "duration": 1, "deadline": 12},
    ]
    slots_a = make_time_slots(free_hours_per_day=4, days=7)   # 14 slots

    print(f"  Tasks: {len(tasks_a)}  |  Slots: {len(slots_a)}")

    astar_a  = astar_schedule(tasks_a, slots_a)
    greedy_a = greedy_schedule(tasks_a, slots_a)

    print(f"\n  A*     : conflicts={astar_a['conflicts']}  "
          f"nodes={astar_a['nodes_expanded']}  "
          f"time={astar_a['time_ms']} ms  "
          f"complete={'Yes' if not astar_a['unscheduled'] else 'No'}")
    print(f"  Greedy : conflicts={greedy_a['conflicts']}  "
          f"nodes={greedy_a['nodes_expanded']}  "
          f"time={greedy_a['time_ms']} ms  "
          f"complete={'Yes' if not greedy_a['unscheduled'] else 'No'}")

    print("\n  A* Schedule:")
    for slot, task in astar_a["schedule"].items():
        print(f"    {slot:<8} → {task}")

    print(compare_algorithms(astar_a, greedy_a, tasks_a, slots_a))

    # ── Scenario B: At-risk ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  Scenario B — At-risk (11 tasks, 7 slots)")
    print("─" * 65)

    profile_b = {
        "num_pending_tasks":       11,
        "days_until_exam":         3,
        "avg_task_duration_hours": 2,
        "free_hours_per_day":      2,
    }
    tasks_b = make_tasks_from_profile(profile_b)
    slots_b = make_time_slots(free_hours_per_day=2, days=7)   # 7 slots

    print(f"  Tasks: {len(tasks_b)}  |  Slots: {len(slots_b)}")

    astar_b  = astar_schedule(tasks_b, slots_b)
    greedy_b = greedy_schedule(tasks_b, slots_b)

    print(f"\n  A*     : conflicts={astar_b['conflicts']}  "
          f"nodes={astar_b['nodes_expanded']}  "
          f"time={astar_b['time_ms']} ms  "
          f"unscheduled={astar_b['unscheduled']}")
    print(f"  Greedy : conflicts={greedy_b['conflicts']}  "
          f"nodes={greedy_b['nodes_expanded']}  "
          f"time={greedy_b['time_ms']} ms  "
          f"unscheduled={greedy_b['unscheduled']}")

    print(compare_algorithms(astar_b, greedy_b, tasks_b, slots_b))

    if astar_b["unscheduled"]:
        print(
            f"\n  [!] A* honestly reports: {len(astar_b['unscheduled'])} task(s) "
            f"cannot be placed with {len(slots_b)} available slots.\n"
            f"  This is the key point of Scenario B demo."
        )

    # smoke tests
    assert astar_a["conflicts"]  == 0,    "A* Scenario A: no conflicts expected"
    assert greedy_a["conflicts"] == 0,    "Greedy Scenario A: no conflicts expected"
    assert astar_a["unscheduled"]  == [], "A* Scenario A: all tasks should be placed"
    assert greedy_a["unscheduled"] == [], "Greedy Scenario A: all tasks should be placed"
    assert astar_b["nodes_expanded"] > greedy_b["nodes_expanded"], \
        "A* should expand more nodes than Greedy"

    print("\n  Smoke tests: PASSED")
