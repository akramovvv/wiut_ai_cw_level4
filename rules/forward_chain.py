"""
rules/forward_chain.py
======================
Этап 3 — Student Success Copilot

Forward Chaining: data-driven рассуждение.

Алгоритм:
  1. Берём текущий профиль студента (факты)
  2. Проходим по ВСЕМ правилам knowledge_base
  3. Если ВСЕ conditions правила выполнены → правило срабатывает
  4. Собираем сработавшие правила → signals, advices, explanation

Почему Forward здесь:
  У нас уже есть данные (или backward chain их собрал).
  Мы движемся от фактов к выводу — это и есть forward chaining.

Публичный API:
  run_forward_chain(profile) -> ForwardChainResult
"""

from __future__ import annotations
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rules.knowledge_base import RULES


@dataclass
class ForwardChainResult:
    signals:     list = field(default_factory=list)
    fired_rules: list = field(default_factory=list)
    risk_level:  str  = "Low"
    advices:     list = field(default_factory=list)
    explanation: str  = ""


def _assess_risk(fired_rules: list) -> str:
    """
    Определяет уровень риска по сработавшим правилам.
    Логика зеркалит generate_dataset.label_risk() — намеренно.
    """
    high   = sum(1 for r in fired_rules if r["severity"] == "high")
    medium = sum(1 for r in fired_rules if r["severity"] == "medium")

    if high >= 2:
        return "High"
    elif high == 1 or medium >= 2:
        return "Medium"
    else:
        return "Low"


def _build_explanation(fired_rules: list, risk_level: str) -> str:
    """
    Строит связный текст объяснения.
    Идёт в explanation card и дополняет explain_ml().
    """
    if not fired_rules:
        return (
            "Экспертная система не обнаружила активных сигналов риска. "
            "Все проверенные показатели в пределах нормы."
        )

    high_rules   = [r for r in fired_rules if r["severity"] == "high"]
    medium_rules = [r for r in fired_rules if r["severity"] == "medium"]
    low_rules    = [r for r in fired_rules if r["severity"] == "low"]

    parts = []

    n = len(fired_rules)
    word = "правило" if n == 1 else "правила" if n < 5 else "правил"
    parts.append(
        f"Экспертная система активировала {n} {word} "
        f"(итоговый риск: {risk_level})."
    )

    if high_rules:
        signals_str = ", ".join(
            f"{r['signal'].replace('_', ' ')} ({r['confidence']:.0%})"
            for r in high_rules
        )
        parts.append(
            f"Критические сигналы ({len(high_rules)}): {signals_str}."
        )
        for r in high_rules[:2]:
            parts.append(f"• {r['explanation']}")

    if medium_rules:
        signals_str = ", ".join(
            f"{r['signal'].replace('_', ' ')} ({r['confidence']:.0%})"
            for r in medium_rules
        )
        parts.append(
            f"Предупреждающие сигналы ({len(medium_rules)}): {signals_str}."
        )

    if low_rules:
        parts.append("Позитивный сигнал: " + low_rules[0]["advice"])

    return " ".join(parts)


def run_forward_chain(profile: dict) -> ForwardChainResult:
    """
    Запускает forward chaining на профиле студента.

    Отсутствующие ключи обрабатываются безопасно:
    правило просто не срабатывает если данных не хватает.
    """
    fired = []

    for rule in RULES:
        try:
            if all(cond(profile) for cond in rule["conditions"]):
                fired.append(rule)
        except (TypeError, KeyError, ValueError):
            continue   # нет данных — правило пропускаем

    # сортировка: high → medium → low
    order = {"high": 0, "medium": 1, "low": 2}
    fired.sort(key=lambda r: order.get(r["severity"], 3))

    risk_level  = _assess_risk(fired)
    explanation = _build_explanation(fired, risk_level)

    return ForwardChainResult(
        signals     = [r["signal"] for r in fired],
        fired_rules = fired,
        risk_level  = risk_level,
        advices     = [r["advice"] for r in fired],
        explanation = explanation,
    )


# ══════════════════════════════════════════════════════════════════════════
# ТЕСТ
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("FORWARD CHAINING — тест сценариев A и B")
    print("=" * 60)

    profile_A = {
        "free_hours_per_day": 5.0, "num_pending_tasks": 3.0,
        "avg_task_duration_hours": 2.0, "deadline_urgency": 0.2,
        "days_until_exam": 14.0, "stress_level": 4.0,
        "sleep_hours_avg": 7.5, "past_completion_rate": 0.85,
        "extracurricular_hours": 6.0, "missed_classes_pct": 0.05,
    }

    profile_B = {
        "free_hours_per_day": 2.0, "num_pending_tasks": 11.0,
        "avg_task_duration_hours": 4.0, "deadline_urgency": 0.9,
        "days_until_exam": 2.0, "stress_level": 8.0,
        "sleep_hours_avg": 4.5, "past_completion_rate": 0.3,
        "extracurricular_hours": 15.0, "missed_classes_pct": 0.4,
    }

    for name, profile in [("A — Normal", profile_A), ("B — At-risk", profile_B)]:
        print(f"\n--- Сценарий {name} ---")
        r = run_forward_chain(profile)
        print(f"  risk_level  : {r.risk_level}")
        print(f"  signals     : {r.signals}")
        print(f"  fired rules : {len(r.fired_rules)}")
        print(f"  explanation :")
        print(f"    {r.explanation}")
        if r.advices:
            print(f"  advice[0]   : {r.advices[0][:80]}...")

    # smoke tests
    assert run_forward_chain(profile_A).risk_level == "Low"
    assert run_forward_chain(profile_B).risk_level == "High"
    print("\n  Smoke tests: PASSED")