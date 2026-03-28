"""
planner/greedy.py
=================
Этап 4 — Student Success Copilot

Greedy Search — жадный алгоритм планирования.

Почему Greedy быстр, но неоптимален:
  - Всегда берёт самую срочную задачу (earliest deadline first)
  - Назначает в ПЕРВЫЙ доступный слот — без оглядки на последствия
  - Не делает backtracking — решение принимается раз и навсегда
  - Сложность O(n log n) — сортировка + один проход
  - Может создавать конфликты дедлайнов, которых можно было избежать

Используется для СРАВНЕНИЯ с A*:
  Greedy = скорость, A* = качество решения.
"""

from __future__ import annotations
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planner.state import StudentScheduleState


def greedy_schedule(tasks: list, time_slots: list) -> dict:
    """
    Строит расписание жадным алгоритмом.

    Аргументы:
        tasks      : список dict-задач [{"name", "duration", "deadline"}, ...]
        time_slots : список меток слотов ["Mon1", "Mon2", ...]

    Возвращает dict:
        schedule       : {slot -> task_name}
        conflicts      : число нарушений дедлайнов
        nodes_expanded : число задач обработано (= нет ветвления)
        time_ms        : время выполнения
        unscheduled    : задачи которые не влезли
    """
    t_start = time.perf_counter()

    # Шаг 1: сортируем по срочности — earliest deadline first
    # Это и есть "жадная" стратегия выбора задачи
    sorted_tasks = sorted(tasks, key=lambda t: t["deadline"])

    state          = StudentScheduleState(tasks, time_slots)
    nodes_expanded = 0

    for task in sorted_tasks:
        available = state.get_available_slots()

        if not available:
            # слотов больше нет — задача остаётся нераспределённой
            nodes_expanded += 1
            continue

        # жадный выбор слота: просто первый доступный (нет оценки последствий)
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
# ТЕСТ
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from planner.state import make_time_slots, make_tasks_from_profile

    print("=" * 60)
    print("GREEDY SEARCH — тест сценариев A и B")
    print("=" * 60)

    # ── Сценарий A: Normal ────────────────────────────────────────────
    print("\n--- Сценарий A: Normal (4 задачи, 14 слотов) ---")
    tasks_a = [
        {"name": "Math Essay",   "duration": 1, "deadline": 3},
        {"name": "ML Lab",       "duration": 1, "deadline": 5},
        {"name": "OS Report",    "duration": 1, "deadline": 9},
        {"name": "DB Project",   "duration": 1, "deadline": 12},
    ]
    slots_a = make_time_slots(free_hours_per_day=4, days=7)  # 14 слотов

    result_a = greedy_schedule(tasks_a, slots_a)
    print(f"  Слотов         : {len(slots_a)}")
    print(f"  Конфликты      : {result_a['conflicts']}")
    print(f"  Нераспределено : {result_a['unscheduled']}")
    print(f"  Узлов (nodes)  : {result_a['nodes_expanded']}")
    print(f"  Время          : {result_a['time_ms']} мс")
    print(f"  Расписание:")
    for slot, task in result_a["schedule"].items():
        print(f"    {slot:<8} → {task}")

    # ── Сценарий B: At-risk ───────────────────────────────────────────
    print("\n--- Сценарий B: At-risk (11 задач, 7 слотов) ---")
    profile_b = {
        "num_pending_tasks":       11,
        "days_until_exam":         3,
        "avg_task_duration_hours": 2,
        "free_hours_per_day":      2,
    }
    tasks_b = make_tasks_from_profile(profile_b)
    slots_b = make_time_slots(free_hours_per_day=2, days=7)  # 7 слотов

    result_b = greedy_schedule(tasks_b, slots_b)
    print(f"  Задач          : {len(tasks_b)}")
    print(f"  Слотов         : {len(slots_b)}")
    print(f"  Конфликты      : {result_b['conflicts']}")
    print(f"  Нераспределено : {result_b['unscheduled']}")
    print(f"  Узлов (nodes)  : {result_b['nodes_expanded']}")
    print(f"  Время          : {result_b['time_ms']} мс")
    print(f"  Расписание:")
    for slot, task in result_b["schedule"].items():
        print(f"    {slot:<8} → {task}")

    # smoke tests
    assert result_a["conflicts"] == 0,    "Сценарий A: не должно быть конфликтов"
    assert result_a["unscheduled"] == [], "Сценарий A: все задачи должны быть размещены"
    assert len(result_b["unscheduled"]) > 0, "Сценарий B: часть задач не влезет"

    print("\n  Smoke tests: PASSED")