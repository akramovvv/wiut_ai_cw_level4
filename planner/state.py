"""
planner/state.py
================
Stage 4 — Student Success Copilot

Formalisation of the search problem (Week 2 syllabus):
  State   : current schedule + list of remaining tasks
  Actions : assign a time slot to a task
  Goal    : all tasks are scheduled (schedule is complete)
  g(n)    : number of deadline conflicts accumulated so far
  h(n)    : estimate of future conflicts (used in A*)
"""

from __future__ import annotations


class StudentScheduleState:
    """
    Planner state.

    tasks      : list of task dicts, each containing:
                   name     (str) — task name
                   duration (int) — duration in slots
                   deadline (int) — index of the last acceptable slot
    time_slots : list of slot labels ["Mon1", "Mon2", ...]
    schedule   : dict {slot_label -> task_name} — current assignments

    Immutability:
      assign() does NOT modify the current object — it returns a NEW state.
      This is required for correct search tree traversal.
    """

    def __init__(
        self,
        tasks: list,
        time_slots: list,
        schedule: dict = None,
    ):
        self.tasks      = tasks                        # all tasks (unchanged)
        self.time_slots = time_slots                   # all slots  (unchanged)
        self.schedule   = schedule if schedule else {} # assignments {slot: name}

    # ── state queries ─────────────────────────────────────────────────────

    def is_goal(self) -> bool:
        """Goal: all tasks are placed in the schedule."""
        scheduled_names = set(self.schedule.values())
        return all(t["name"] in scheduled_names for t in self.tasks)

    def get_remaining_tasks(self) -> list:
        """Returns tasks not yet assigned to any slot."""
        scheduled_names = set(self.schedule.values())
        return [t for t in self.tasks if t["name"] not in scheduled_names]

    def get_available_slots(self) -> list:
        """Returns slots not yet occupied."""
        return [s for s in self.time_slots if s not in self.schedule]

    # ── action ────────────────────────────────────────────────────────────

    def assign(self, task: dict, slot: str) -> "StudentScheduleState":
        """
        Places a task in a slot.
        Returns a NEW state — the current one is not modified.

        This is the central action in the search space:
          state.assign(task, slot) → new_state
        """
        new_schedule = dict(self.schedule)        # copy current schedule
        new_schedule[slot] = task["name"]         # add new assignment
        return StudentScheduleState(
            tasks      = self.tasks,
            time_slots = self.time_slots,
            schedule   = new_schedule,
        )

    # ── metrics ───────────────────────────────────────────────────────────

    def calculate_conflicts(self) -> int:
        """
        Counts the number of deadline conflicts.

        Conflict: a task is assigned to a slot with index > task deadline.
        Example: task deadline=2, but assigned to slot with index 4 → conflict.

        Used as g(n) in A* and as the final quality metric.
        """
        conflicts = 0
        task_map  = {t["name"]: t for t in self.tasks}

        for slot, task_name in self.schedule.items():
            slot_idx = self.time_slots.index(slot)
            task     = task_map.get(task_name)
            if task and slot_idx > task["deadline"]:
                conflicts += 1

        return conflicts

    def heuristic(self) -> int:
        """
        h(n) — admissible heuristic for A*.

        Estimates how many FUTURE conflicts are possible:
        counts tasks that lack an available slot before their deadline.

        Admissible because:
          - we only count tasks WITHOUT a guaranteed slot
          - actual conflicts cannot be FEWER than this estimate
          - therefore h(n) never overestimates → A* remains optimal
        """
        remaining    = self.get_remaining_tasks()
        available    = self.get_available_slots()
        h            = 0

        for task in remaining:
            # slots within the deadline for this task
            slots_in_time = [
                s for s in available
                if self.time_slots.index(s) <= task["deadline"]
            ]
            # if no slot before the deadline exists — conflict is inevitable
            if len(slots_in_time) == 0:
                h += 1

        return h

    # ── debug ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        remaining = len(self.get_remaining_tasks())
        conflicts = self.calculate_conflicts()
        return (
            f"StudentScheduleState("
            f"scheduled={len(self.schedule)}, "
            f"remaining={remaining}, "
            f"conflicts={conflicts})"
        )


# ══════════════════════════════════════════════════════════════════════════
# FACTORY — generate slots and tasks from a student profile
# ══════════════════════════════════════════════════════════════════════════

def make_time_slots(free_hours_per_day: int = 4, days: int = 7) -> list:
    """
    Generates time slot labels for `days` days.
    Each slot = 2 hours of work.
    Example: ["Mon1", "Mon2", "Tue1", ...]
    """
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    slots_per_day = max(1, free_hours_per_day // 2)
    slots = []
    for d in range(days):
        day = day_names[d % 7]
        for s in range(slots_per_day):
            slots.append(f"{day}{s+1}")
    return slots


def make_tasks_from_profile(profile: dict) -> list:
    """
    Generates a task list from a student profile.
    Each task is a dict with name, duration, deadline.
    """
    n_tasks   = int(profile.get("num_pending_tasks", 5))
    days_exam = int(profile.get("days_until_exam", 7))
    duration  = max(1, int(profile.get("avg_task_duration_hours", 2)) // 2)

    subjects = [
        "Algorithms", "ML Lab", "Math", "Networks", "OS",
        "Database", "Software Eng", "AI Ethics", "Calculus", "Stats",
        "Data Structures", "Linear Algebra", "Compiler", "Security", "Cloud",
    ]

    tasks = []
    for i in range(n_tasks):
        # deadlines: earlier tasks are more urgent
        deadline_day = max(0, int(days_exam * (i + 1) / n_tasks) - 1)
        # deadline in slot indices (2 slots per day)
        deadline_slot_idx = deadline_day * 2 + 1
        tasks.append({
            "name":     f"{subjects[i % len(subjects)]} Task {i+1}",
            "duration": duration,
            "deadline": deadline_slot_idx,
        })
    return tasks


# ══════════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("STATE — test")
    print("=" * 55)

    tasks = [
        {"name": "Math Essay",   "duration": 1, "deadline": 2},
        {"name": "ML Lab",       "duration": 1, "deadline": 4},
        {"name": "OS Report",    "duration": 1, "deadline": 6},
        {"name": "DB Project",   "duration": 1, "deadline": 8},
    ]
    slots = make_time_slots(free_hours_per_day=4, days=7)

    print(f"\nSlots ({len(slots)}): {slots}")
    print(f"Tasks ({len(tasks)}):")
    for t in tasks:
        print(f"  {t['name']:<20} deadline=slot[{t['deadline']}]"
              f" ({slots[t['deadline']] if t['deadline'] < len(slots) else 'OOB'})")

    state = StudentScheduleState(tasks, slots)
    print(f"\nInitial state : {state}")
    print(f"is_goal       : {state.is_goal()}")
    print(f"remaining tasks : {[t['name'] for t in state.get_remaining_tasks()]}")
    print(f"available slots : {state.get_available_slots()[:5]}...")
    print(f"h(n)            : {state.heuristic()}")

    # test assign (immutability)
    new_state = state.assign(tasks[0], slots[0])
    print(f"\nAfter assign(Math Essay → Mon1):")
    print(f"  new_state : {new_state}")
    print(f"  old_state : {state}  ← unchanged")
    print(f"  schedule  : {new_state.schedule}")
    print(f"  conflicts : {new_state.calculate_conflicts()}")

    assert state.schedule == {}          # original unchanged
    assert len(new_state.schedule) == 1
    assert new_state.calculate_conflicts() == 0   # slot Mon1 (idx=0) <= deadline=2

    # test conflict: place task after its deadline
    late_state = state.assign(tasks[0], slots[5])   # slot idx=5 > deadline=2
    assert late_state.calculate_conflicts() == 1
    print(f"\nConflict when assigned to a late slot: {late_state.calculate_conflicts()} ✓")

    print("\n  Smoke tests: PASSED")
