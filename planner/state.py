"""
planner/state.py
================
Этап 4 — Student Success Copilot

Формализация задачи поиска (Week 2 syllabus):
  State   : текущее расписание + список оставшихся задач
  Actions : выделить временной слот под задачу
  Goal    : все задачи распределены (schedule полное)
  g(n)    : количество конфликтов дедлайнов накопленных до сих пор
  h(n)    : оценка будущих конфликтов (используется в A*)
"""

from __future__ import annotations


class StudentScheduleState:
    """
    Состояние планировщика.

    tasks      : список dict-задач, каждая содержит:
                   name     (str) — название
                   duration (int) — длительность в слотах
                   deadline (int) — индекс последнего допустимого слота
    time_slots : список меток слотов ["Mon1", "Mon2", ...]
    schedule   : dict {slot_label -> task_name} — текущие назначения

    Иммутабельность:
      assign() НЕ изменяет текущий объект — возвращает НОВОЕ состояние.
      Это обязательно для корректного поиска по дереву состояний.
    """

    def __init__(
        self,
        tasks: list,
        time_slots: list,
        schedule: dict = None,
    ):
        self.tasks      = tasks                        # все задачи (неизменны)
        self.time_slots = time_slots                   # все слоты  (неизменны)
        self.schedule   = schedule if schedule else {} # назначения {slot: name}

    # ── запросы состояния ─────────────────────────────────────────────────

    def is_goal(self) -> bool:
        """Цель: все задачи размещены в расписании."""
        scheduled_names = set(self.schedule.values())
        return all(t["name"] in scheduled_names for t in self.tasks)

    def get_remaining_tasks(self) -> list:
        """Возвращает задачи, которые ещё не назначены ни в один слот."""
        scheduled_names = set(self.schedule.values())
        return [t for t in self.tasks if t["name"] not in scheduled_names]

    def get_available_slots(self) -> list:
        """Возвращает слоты, которые ещё не заняты."""
        return [s for s in self.time_slots if s not in self.schedule]

    # ── действие (action) ─────────────────────────────────────────────────

    def assign(self, task: dict, slot: str) -> "StudentScheduleState":
        """
        Размещает задачу в слот.
        Возвращает НОВОЕ состояние — текущее не изменяется.

        Это центральное действие пространства поиска:
          state.assign(task, slot) → new_state
        """
        new_schedule = dict(self.schedule)        # копируем текущее расписание
        new_schedule[slot] = task["name"]         # добавляем новое назначение
        return StudentScheduleState(
            tasks      = self.tasks,
            time_slots = self.time_slots,
            schedule   = new_schedule,
        )

    # ── метрики ───────────────────────────────────────────────────────────

    def calculate_conflicts(self) -> int:
        """
        Считает количество конфликтов дедлайнов.

        Конфликт: задача назначена в слот с индексом > deadline задачи.
        Например: task deadline=2, но назначена в слот с индексом 4 → конфликт.

        Используется как g(n) в A* и как итоговая метрика качества.
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
        h(n) — admissible эвристика для A*.

        Оценивает сколько БУДУЩИХ конфликтов возможно:
        считает задачи, у которых не хватает доступных слотов до дедлайна.

        Admissible потому что:
          - мы считаем только задачи БЕЗ гарантированного слота
          - реальных конфликтов не может быть МЕНЬШЕ чем эта оценка
          - значит h(n) никогда не переоценивает → A* остаётся оптимальным
        """
        remaining    = self.get_remaining_tasks()
        available    = self.get_available_slots()
        h            = 0

        for task in remaining:
            # слоты в пределах дедлайна для этой задачи
            slots_in_time = [
                s for s in available
                if self.time_slots.index(s) <= task["deadline"]
            ]
            # если нет ни одного слота до дедлайна — конфликт неизбежен
            if len(slots_in_time) == 0:
                h += 1

        return h

    # ── отладка ───────────────────────────────────────────────────────────

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
# ФАБРИКА — генерация слотов и задач из профиля студента
# ══════════════════════════════════════════════════════════════════════════

def make_time_slots(free_hours_per_day: int = 4, days: int = 7) -> list:
    """
    Генерирует метки временных слотов на `days` дней.
    Каждый слот = 2 часа работы.
    Пример: ["Mon1", "Mon2", "Tue1", ...]
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
    Генерирует список задач из профиля студента.
    Каждая задача — dict с name, duration, deadline.
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
        # дедлайны: первые задачи срочнее
        deadline_day = max(0, int(days_exam * (i + 1) / n_tasks) - 1)
        # deadline в индексах слотов (2 слота в день)
        deadline_slot_idx = deadline_day * 2 + 1
        tasks.append({
            "name":     f"{subjects[i % len(subjects)]} Task {i+1}",
            "duration": duration,
            "deadline": deadline_slot_idx,
        })
    return tasks


# ══════════════════════════════════════════════════════════════════════════
# ТЕСТ
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("STATE — тест")
    print("=" * 55)

    tasks = [
        {"name": "Math Essay",   "duration": 1, "deadline": 2},
        {"name": "ML Lab",       "duration": 1, "deadline": 4},
        {"name": "OS Report",    "duration": 1, "deadline": 6},
        {"name": "DB Project",   "duration": 1, "deadline": 8},
    ]
    slots = make_time_slots(free_hours_per_day=4, days=7)

    print(f"\nСлоты ({len(slots)}): {slots}")
    print(f"Задачи ({len(tasks)}):")
    for t in tasks:
        print(f"  {t['name']:<20} deadline=slot[{t['deadline']}]"
              f" ({slots[t['deadline']] if t['deadline'] < len(slots) else 'OOB'})")

    state = StudentScheduleState(tasks, slots)
    print(f"\nНачальное состояние : {state}")
    print(f"is_goal             : {state.is_goal()}")
    print(f"remaining tasks     : {[t['name'] for t in state.get_remaining_tasks()]}")
    print(f"available slots     : {state.get_available_slots()[:5]}...")
    print(f"h(n)                : {state.heuristic()}")

    # тест assign (иммутабельность)
    new_state = state.assign(tasks[0], slots[0])
    print(f"\nПосле assign(Math Essay → Mon1):")
    print(f"  new_state : {new_state}")
    print(f"  old_state : {state}  ← не изменился")
    print(f"  schedule  : {new_state.schedule}")
    print(f"  conflicts : {new_state.calculate_conflicts()}")

    assert state.schedule == {}          # старое не изменилось
    assert len(new_state.schedule) == 1
    assert new_state.calculate_conflicts() == 0   # слот Mon1 (idx=0) <= deadline=2

    # тест конфликта: ставим задачу позже дедлайна
    late_state = state.assign(tasks[0], slots[5])   # slot idx=5 > deadline=2
    assert late_state.calculate_conflicts() == 1
    print(f"\nКонфликт при назначении в поздний слот: {late_state.calculate_conflicts()} ✓")

    print("\n  Smoke tests: PASSED")