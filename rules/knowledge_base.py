"""
rules/knowledge_base.py
=======================
Этап 3 — Student Success Copilot

База знаний экспертной системы.
Правила хранятся как список словарей — это делает их:
  - читаемыми для экзаменатора
  - легко расширяемыми (добавь строку — добавил правило)
  - пригодными для обоих типов chaining

ВАЖНО: пороги намеренно совпадают с:
  - data/generate_dataset.py  label_risk()
  - ml/explain.py             THRESHOLDS
Это суть Explainability: правила объясняют то же, что выучила ML модель.

Структура правила:
  id          — уникальный идентификатор
  conditions  — список лямбда-функций (profile -> bool)
  signal      — строка-метка сработавшего правила
  severity    — "high" | "medium" | "low"
  advice      — совет студенту
  explanation — почему это проблема (для explanation card)
  requires    — ключи профиля, нужные для проверки
                (backward chaining спросит о недостающих)
"""

RULES = [

    # ══════════════════════════════════════════════
    # HIGH — критические сигналы (2+ → risk=High)
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
            "Уровень стресса критически высок. "
            "Запланируй полный день отдыха в ближайшие 2 дня. "
            "Раздели крупные задачи на 25-минутные блоки (Pomodoro)."
        ),
        "explanation": (
            "Стресс выше 7/10 снижает продуктивность на 30-40% "
            "и ухудшает долгосрочную память. "
            "Это один из пяти критических сигналов модели."
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
            "Менее 5.5 ч сна — критический уровень. "
            "Сегодня ложись до 23:00. "
            "Убери все задачи после 22:00 из расписания."
        ),
        "explanation": (
            "Хроническое недосыпание (<5.5 ч) эквивалентно "
            "1.5 суткам без сна по когнитивному эффекту. "
            "Учёба после полуночи при таком режиме неэффективна."
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
            "История выполнения задач ниже 40%. "
            "Выбери 3 задачи с ближайшими дедлайнами — только они в фокусе. "
            "Одна задача за раз, не распыляйся."
        ),
        "explanation": (
            "Completion rate <40% — предиктор №1 академического риска "
            "в нашей модели (feature importance: 22%). "
            "Причина обычно — слишком широкий список без приоритизации."
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
            "Экзамен через 3 дня, в очереди 6+ задач — перегрузка. "
            "Academic triage: раздели задачи на 'до экзамена' и 'после'. "
            "Плановщик поможет расставить приоритеты."
        ),
        "explanation": (
            "Близкий экзамен + большая очередь задач = когнитивная перегрузка. "
            "Попытка сделать всё одновременно хуже, чем осознанный выбор."
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
            "Пропущено более 35% занятий. "
            "Это может повлиять на допуск к экзаменам. "
            "Свяжись с деканатом или куратором сегодня."
        ),
        "explanation": (
            "Высокий процент пропусков коррелирует с недопуском к сессии "
            "и пробелами в материале, которые сложно компенсировать самостоятельно."
        ),
        "requires": ["missed_classes_pct"],
    },

    # ══════════════════════════════════════════════
    # MEDIUM — предупреждающие сигналы
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
            "Стресс повышен (6-7/10). "
            "Добавь 30-минутный перерыв каждые 2 часа работы. "
            "Короткая физическая активность снижает кортизол."
        ),
        "explanation": (
            "Стресс 6-7/10 — пограничная зона. "
            "Без вмешательства может перейти в критический уровень за 1-2 дня."
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
            "Сна немного меньше нормы (5.5-6.5 ч). "
            "Добавь хотя бы 30-45 минут — "
            "даже небольшое улучшение заметно сказывается на концентрации."
        ),
        "explanation": (
            "Норма для студента 7-9 ч. "
            "6-6.5 ч ещё не критично, но снижает качество консолидации памяти."
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
            "История выполнения 40-60% — зона риска. "
            "Попробуй правило двух задач в день: "
            "небольшие победы накапливают импульс."
        ),
        "explanation": (
            "50% completion rate — каждая вторая задача остаётся незакрытой. "
            "Обычная причина — неверная оценка времени на задачу."
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
            "Более 12 ч/нед внеучебных активностей при текущей нагрузке — много. "
            "Рассмотри временное сокращение до конца сессии."
        ),
        "explanation": (
            "При академическом давлении каждый дополнительный "
            "час вне учёбы — это час без восстановления."
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
            "Мало времени (<2 ч/день) + много задач (5+) — математика не сходится. "
            "Плановщик покажет реальную картину: "
            "какие задачи физически невозможно уместить в неделю."
        ),
        "explanation": (
            "При <2 свободных часах в день невозможно качественно выполнить 5+ задач. "
            "A* честно скажет об этом, в отличие от Greedy."
        ),
        "requires": ["free_hours_per_day", "num_pending_tasks"],
    },

    # ══════════════════════════════════════════════
    # LOW — позитивное подтверждение
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
            "Отличный режим: сон, выполнение задач и стресс — всё в норме. "
            "Поддерживай этот ритм перед сессией."
        ),
        "explanation": (
            "Три ключевых показателя одновременно в норме — "
            "сильный предиктор успешного завершения семестра."
        ),
        "requires": ["sleep_hours_avg", "past_completion_rate", "stress_level"],
    },
]

# быстрый доступ по id
RULES_BY_ID = {r["id"]: r for r in RULES}

# все ключи, которые хоть одно правило требует
ALL_REQUIRED_KEYS = sorted({
    key for rule in RULES for key in rule["requires"]
})


if __name__ == "__main__":
    print(f"База знаний: {len(RULES)} правил\n")
    for r in RULES:
        print(f"  {r['id']:<30}  severity={r['severity']:<6}  "
              f"confidence={r['confidence']:.2f}  "
              f"conditions={len(r['conditions'])}  "
              f"requires={r['requires']}")
    print(f"\nВсе требуемые ключи: {ALL_REQUIRED_KEYS}")