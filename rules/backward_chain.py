"""
rules/backward_chain.py
=======================
Этап 3 — Student Success Copilot

Backward Chaining: goal-driven рассуждение.

Логика:
  Цель — вычислить risk_level.
  Для этого нужны конкретные данные (stress_level, sleep и др.).
  Backward chain смотрит: каких данных нет в профиле?
  Для каждого недостающего ключа задаёт вопрос студенту.

Почему именно stress_level — главный кандидат:
  - Это субъективный показатель, который студент не вводит заранее
  - Правило R01_high_stress его требует
  - Вопрос звучит естественно и демонстрирует интерактивность системы

Два режима:
  interactive=True  — задаёт вопросы через input() (для notebook demo)
  interactive=False — возвращает список missing keys (для тестов)

Публичный API:
  run_backward_chain(profile, interactive, io_callback)
      -> BackwardChainResult
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ── цели: без этих ключей вывод о риске невозможен ───────────────────────
REQUIRED_FOR_RISK = [
    "stress_level",          # R01 — самый важный, часто отсутствует
    "sleep_hours_avg",       # R02
    "past_completion_rate",  # R03
    "days_until_exam",       # R04
    "num_pending_tasks",     # R04, R10
]

# ── вопросы для каждого ключа ─────────────────────────────────────────────
QUESTIONS = {
    "stress_level": {
        "text": (
            "\n[Backward Chaining] Чтобы оценить риск выгорания,\n"
            "мне нужно знать твой уровень стресса.\n"
            "Как ты оцениваешь стресс прямо сейчас?\n"
            "(1 = спокойно, 10 = очень тяжело): "
        ),
        "type": "int", "min": 1, "max": 10,
    },
    "sleep_hours_avg": {
        "text": (
            "\n[Backward Chaining] Для проверки сигнала R02 нужны данные о сне.\n"
            "Сколько часов в среднем ты спишь в сутки на этой неделе?\n"
            "(например: 6.5): "
        ),
        "type": "float", "min": 0.0, "max": 24.0,
    },
    "past_completion_rate": {
        "text": (
            "\n[Backward Chaining] Для проверки истории выполнения задач (R03):\n"
            "Какую долю запланированных задач ты обычно выполняешь?\n"
            "(0.0 = ничего, 1.0 = всё, например: 0.7): "
        ),
        "type": "float", "min": 0.0, "max": 1.0,
    },
    "days_until_exam": {
        "text": (
            "\n[Backward Chaining] Для оценки срочности (R04):\n"
            "Через сколько дней ближайший экзамен или крупная сдача?\n"
            "(например: 5): "
        ),
        "type": "int", "min": 0, "max": 90,
    },
    "num_pending_tasks": {
        "text": (
            "\n[Backward Chaining] Для проверки перегрузки (R04, R10):\n"
            "Сколько незавершённых задач сейчас в очереди?\n"
            "(например: 8): "
        ),
        "type": "int", "min": 0, "max": 50,
    },
}


@dataclass
class BackwardChainResult:
    profile_complete: dict = field(default_factory=dict)
    questions_asked:  list = field(default_factory=list)   # ключи вопросов
    answers_received: dict = field(default_factory=dict)   # key -> value
    was_interactive:  bool = False
    missing_keys:     list = field(default_factory=list)   # что не спросили


def _find_missing(profile: dict) -> list:
    """Возвращает список ключей из REQUIRED_FOR_RISK, которых нет в профиле."""
    return [k for k in REQUIRED_FOR_RISK if profile.get(k) is None]


def _parse_and_validate(raw: str, key: str):
    """
    Парсит и валидирует ответ пользователя.
    Возвращает (value, error_str | None).
    """
    q = QUESTIONS[key]
    try:
        value = int(float(raw.strip())) if q["type"] == "int" else float(raw.strip())
    except (ValueError, AttributeError):
        return None, f"Нужно число от {q['min']} до {q['max']}"

    if not (q["min"] <= value <= q["max"]):
        return None, f"Значение вне диапазона [{q['min']}, {q['max']}]"

    return value, None


def run_backward_chain(
    profile: dict,
    interactive: bool = False,
    io_callback=None,
) -> BackwardChainResult:
    """
    Запускает backward chaining.

    Аргументы:
        profile      : dict профиля — могут отсутствовать некоторые ключи
        interactive  : True → спрашивает через input()
                       False → возвращает missing_keys без вопросов
        io_callback  : необязательная функция (key, question_text) -> value
                       используется в notebook вместо input()
                       если задана — имеет приоритет над interactive

    Возвращает BackwardChainResult с заполненным profile_complete.
    """
    profile_copy     = dict(profile)
    missing          = _find_missing(profile_copy)
    questions_asked  = []
    answers_received = {}

    if not missing:
        # все данные уже есть — backward chain не нужен
        return BackwardChainResult(
            profile_complete = profile_copy,
            questions_asked  = [],
            answers_received = {},
            was_interactive  = False,
            missing_keys     = [],
        )

    was_interactive = False

    if io_callback is not None:
        # режим notebook: callback сам управляет вводом
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
                pass   # callback не смог — оставляем пустым
        was_interactive = True

    elif interactive:
        # режим CLI: input() с повторами при ошибке
        print(
            "\n[Backward Chaining] Система обнаружила недостающие данные.\n"
            f"Для полного анализа нужно ответить на {len(missing)} вопрос(а)."
        )
        for key in missing:
            if key not in QUESTIONS:
                continue
            q_text = QUESTIONS[key]["text"]
            while True:
                try:
                    raw = input(q_text)
                except (EOFError, KeyboardInterrupt):
                    print("\n[пропущено]")
                    break

                value, err = _parse_and_validate(raw, key)
                if err:
                    print(f"  Ошибка: {err}. Попробуй ещё раз.")
                    continue

                profile_copy[key]   = value
                questions_asked.append(key)
                answers_received[key] = value
                print(f"  Записано: {key} = {value}")
                break
        was_interactive = True

    # итоговый список того, что так и осталось пустым
    still_missing = _find_missing(profile_copy)

    return BackwardChainResult(
        profile_complete = profile_copy,
        questions_asked  = questions_asked,
        answers_received = answers_received,
        was_interactive  = was_interactive,
        missing_keys     = still_missing,
    )


# ══════════════════════════════════════════════════════════════════════════
# ТЕСТ (non-interactive режим)
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("BACKWARD CHAINING — тест (non-interactive)")
    print("=" * 60)

    # профиль без stress_level — самый частый случай
    incomplete_profile = {
        "free_hours_per_day":      2.0,
        "num_pending_tasks":       11.0,
        "avg_task_duration_hours": 4.0,
        "deadline_urgency":        0.9,
        # "stress_level":          <-- намеренно отсутствует
        "sleep_hours_avg":         4.5,
        "past_completion_rate":    0.3,
        "extracurricular_hours":   15.0,
        "missed_classes_pct":      0.4,
        "days_until_exam":         2.0,
    }

    print("\n--- Тест 1: профиль без stress_level ---")
    result = run_backward_chain(incomplete_profile, interactive=False)
    print(f"  missing_keys      : {result.missing_keys}")
    print(f"  was_interactive   : {result.was_interactive}")
    print(f"  profile has stress: {'stress_level' in result.profile_complete}")

    assert "stress_level" in result.missing_keys, \
        "stress_level должен быть в missing_keys"

    print("\n--- Тест 2: callback-режим (симуляция notebook) ---")
    def mock_callback(key, text):
        answers = {"stress_level": "8"}
        print(f"  [mock] вопрос о '{key}' → отвечаем '{answers.get(key, '5')}'")
        return answers.get(key, "5")

    result2 = run_backward_chain(incomplete_profile,
                                  interactive=False,
                                  io_callback=mock_callback)
    print(f"  questions_asked   : {result2.questions_asked}")
    print(f"  answers_received  : {result2.answers_received}")
    print(f"  stress_level now  : {result2.profile_complete.get('stress_level')}")

    assert result2.profile_complete.get("stress_level") == 8

    print("\n--- Тест 3: полный профиль — backward chain не нужен ---")
    full_profile = dict(incomplete_profile)
    full_profile["stress_level"] = 7.0
    result3 = run_backward_chain(full_profile, interactive=False)
    print(f"  questions_asked   : {result3.questions_asked}  (должен быть пустой)")
    assert result3.questions_asked == []

    print("\n  Smoke tests: PASSED")