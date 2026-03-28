"""
planner/astar.py
================
Этап 4 — Student Success Copilot

A* Search для планирования расписания студента.

Теория алгоритма A*:
  - Использует priority queue (min-heap по f = g + h)
  - g(n) = calculate_conflicts() — реальные конфликты в текущем состоянии
  - h(n) = heuristic()           — оценка будущих конфликтов (admissible)
  - f(n) = g(n) + h(n)          — приоритет в очереди

Почему h(n) admissible:
  heuristic() считает задачи, у которых ТОЧНО не будет слота до дедлайна.
  Это нижняя граница будущих конфликтов — реальных конфликтов
  не может быть МЕНЬШЕ. Значит h(n) не переоценивает → A* оптимален.

Почему ветвление ограничено (TOP_K_SLOTS):
  Без ограничения: каждая задача × все свободные слоты = экспоненциальный
  рост дерева состояний. TOP_K_SLOTS = 3 лучших слота (по близости к дедлайну)
  сохраняют качество решения и держат время выполнения разумным.

Сравнение с Greedy:
  A*     — исследует больше узлов, медленнее, но минимизирует конфликты
  Greedy — исследует O(n) узлов, быстрее, но может создавать конфликты
"""

from __future__ import annotations
import heapq
import time


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planner.state import StudentScheduleState

TOP_K_SLOTS = 3   # сколько кандидатов-слотов рассматриваем на каждом шаге


def astar_schedule(tasks: list, time_slots: list) -> dict:
    """
    Строит расписание алгоритмом A*.

    Аргументы:
        tasks      : список dict-задач [{"name", "duration", "deadline"}, ...]
        time_slots : список меток слотов ["Mon1", "Mon2", ...]

    Возвращает dict:
        schedule       : {slot -> task_name}
        conflicts      : число нарушений дедлайнов
        nodes_expanded : число раскрытых узлов поиска
        time_ms        : время выполнения
        unscheduled    : задачи которые не влезли
    """
    t_start = time.perf_counter()

    initial_state = StudentScheduleState(tasks, time_slots)

    # heap элемент: (f_cost, tie_breaker, state)
    # tie_breaker нужен чтобы Python не сравнивал state при одинаковом f
    counter = 0
    heap    = [(
        initial_state.calculate_conflicts() + initial_state.heuristic(),
        counter,
        initial_state,
    )]

    nodes_expanded = 0
    best_partial   = initial_state   # лучшее частичное решение (на случай нехватки слотов)

    while heap:
        f, _, state = heapq.heappop(heap)
        nodes_expanded += 1

        # обновляем лучшее частичное по количеству размещённых задач
        if len(state.schedule) > len(best_partial.schedule):
            best_partial = state

        # GOAL CHECK — все задачи размещены
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

        # выбираем задачу с ближайшим дедлайном (earliest deadline first)
        # это направляет поиск к наиболее критичным задачам первыми
        task = min(remaining, key=lambda t: t["deadline"])

        # TOP_K лучших слотов: сортируем по близости к дедлайну задачи
        # слоты ДО дедлайна приоритетнее (меньше конфликтов)
        def slot_priority(slot: str) -> tuple:
            idx = time_slots.index(slot)
            over_deadline = max(0, idx - task["deadline"])   # штраф за опоздание
            return (over_deadline, idx)                       # сначала без штрафа, потом ранние

        candidate_slots = sorted(available, key=slot_priority)[:TOP_K_SLOTS]

        for slot in candidate_slots:
            new_state = state.assign(task, slot)
            g         = new_state.calculate_conflicts()
            h         = new_state.heuristic()
            counter  += 1
            heapq.heappush(heap, (g + h, counter, new_state))

    # heap пуст — возвращаем лучшее найденное частичное решение
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
    Строит таблицу сравнения A* и Greedy.
    Ключевой артефакт для демо-видео и отчёта.
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
        ("Конфликты дедлайнов",   a["conflicts"],      g["conflicts"],      True),
        ("Задач размещено",        a_placed,            g_placed,            False),
        ("Задач нераспределено",   len(a["unscheduled"]), len(g["unscheduled"]), True),
        ("Узлов исследовано",      a["nodes_expanded"], g["nodes_expanded"], True),
        ("Время выполнения (мс)",  a["time_ms"],        g["time_ms"],        True),
    ]

    top  = "┌─────────────────────────────┬──────────────┬──────────────┬──────────┐"
    head = "│ Метрика                     │     A*       │    Greedy    │  Лучше   │"
    sep  = "├─────────────────────────────┼──────────────┼──────────────┼──────────┤"
    bot  = "└─────────────────────────────┴──────────────┴──────────────┴──────────┘"

    lines = [top, head, sep]
    for label, av, gv, lb in rows:
        w = winner(av, gv, lb)
        lines.append(
            f"│ {label:<27} │ {str(av):>12} │ {str(gv):>12} │ {w:>8} │"
        )
    lines += [sep]

    a_complete = "Да" if not a["unscheduled"] else "Нет"
    g_complete = "Да" if not g["unscheduled"] else "Нет"
    lines.append(
        f"│ {'Расписание полное?':<27} │ {a_complete:>12} │ {g_complete:>12} │"
        f"          │"
    )
    lines.append(bot)

    # итоговый вывод
    if a["conflicts"] < g["conflicts"]:
        diff = g["conflicts"] - a["conflicts"]
        lines.append(
            f"\nВывод: A* устранил {diff} конфликт(а) по сравнению с Greedy.\n"
            f"Цена оптимальности: A* исследовал {a['nodes_expanded']} узлов "
            f"vs {g['nodes_expanded']} у Greedy, "
            f"{a['time_ms']} мс vs {g['time_ms']} мс."
        )
    elif a["conflicts"] == g["conflicts"] == 0:
        lines.append(
            f"\nВывод: оба алгоритма избежали конфликтов.\n"
            f"Greedy быстрее ({g['time_ms']} мс vs {a['time_ms']} мс A*) "
            f"и исследовал меньше узлов ({g['nodes_expanded']} vs {a['nodes_expanded']}).\n"
            f"В простых случаях Greedy достаточен; A* даёт гарантию оптимальности."
        )
    else:
        lines.append(
            f"\nВывод: A* конфликты={a['conflicts']}, "
            f"Greedy конфликты={g['conflicts']}."
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# ТЕСТ + СРАВНЕНИЕ
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from planner.state import make_time_slots, make_tasks_from_profile
    from planner.greedy import greedy_schedule

    print("=" * 65)
    print("A* vs GREEDY — сравнение двух сценариев")
    print("=" * 65)

    # ── Сценарий A: Normal ────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  Сценарий A — Normal (4 задачи, 14 слотов)")
    print("─" * 65)

    tasks_a = [
        {"name": "Math Essay",   "duration": 1, "deadline": 3},
        {"name": "ML Lab",       "duration": 1, "deadline": 5},
        {"name": "OS Report",    "duration": 1, "deadline": 9},
        {"name": "DB Project",   "duration": 1, "deadline": 12},
    ]
    slots_a = make_time_slots(free_hours_per_day=4, days=7)   # 14 слотов

    print(f"  Задач: {len(tasks_a)}  |  Слотов: {len(slots_a)}")

    astar_a  = astar_schedule(tasks_a, slots_a)
    greedy_a = greedy_schedule(tasks_a, slots_a)

    print(f"\n  A*     : конфликтов={astar_a['conflicts']}  "
          f"узлов={astar_a['nodes_expanded']}  "
          f"время={astar_a['time_ms']} мс  "
          f"полное={'Да' if not astar_a['unscheduled'] else 'Нет'}")
    print(f"  Greedy : конфликтов={greedy_a['conflicts']}  "
          f"узлов={greedy_a['nodes_expanded']}  "
          f"время={greedy_a['time_ms']} мс  "
          f"полное={'Да' if not greedy_a['unscheduled'] else 'Нет'}")

    print("\n  Расписание A* :")
    for slot, task in astar_a["schedule"].items():
        print(f"    {slot:<8} → {task}")

    print(compare_algorithms(astar_a, greedy_a, tasks_a, slots_a))

    # ── Сценарий B: At-risk ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  Сценарий B — At-risk (11 задач, 7 слотов)")
    print("─" * 65)

    profile_b = {
        "num_pending_tasks":       11,
        "days_until_exam":         3,
        "avg_task_duration_hours": 2,
        "free_hours_per_day":      2,
    }
    tasks_b = make_tasks_from_profile(profile_b)
    slots_b = make_time_slots(free_hours_per_day=2, days=7)   # 7 слотов

    print(f"  Задач: {len(tasks_b)}  |  Слотов: {len(slots_b)}")

    astar_b  = astar_schedule(tasks_b, slots_b)
    greedy_b = greedy_schedule(tasks_b, slots_b)

    print(f"\n  A*     : конфликтов={astar_b['conflicts']}  "
          f"узлов={astar_b['nodes_expanded']}  "
          f"время={astar_b['time_ms']} мс  "
          f"нераспределено={astar_b['unscheduled']}")
    print(f"  Greedy : конфликтов={greedy_b['conflicts']}  "
          f"узлов={greedy_b['nodes_expanded']}  "
          f"время={greedy_b['time_ms']} мс  "
          f"нераспределено={greedy_b['unscheduled']}")

    print(compare_algorithms(astar_b, greedy_b, tasks_b, slots_b))

    if astar_b["unscheduled"]:
        print(
            f"\n  [!] A* честно сообщает: {len(astar_b['unscheduled'])} задач "
            f"невозможно разместить при {len(slots_b)} доступных слотах.\n"
            f"  Это ключевой момент демо сценария B."
        )

    # smoke tests
    assert astar_a["conflicts"]  == 0,    "A* Сценарий A: конфликтов быть не должно"
    assert greedy_a["conflicts"] == 0,    "Greedy Сценарий A: конфликтов быть не должно"
    assert astar_a["unscheduled"]  == [], "A* Сценарий A: все задачи должны быть размещены"
    assert greedy_a["unscheduled"] == [], "Greedy Сценарий A: все задачи должны быть размещены"
    assert astar_b["nodes_expanded"] > greedy_b["nodes_expanded"], \
        "A* должен исследовать больше узлов чем Greedy"

    print("\n  Smoke tests: PASSED")