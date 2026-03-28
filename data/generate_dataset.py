"""
generate_dataset.py
===================
Stage 1 — Student Success Copilot
Generates a synthetic dataset of 500 rows.

Key principle:
  risk_label is built DETERMINISTICALLY from feature combinations
  (same conditions later used by rules in rules/).
  Controlled noise (~12% of rows) is then added.
  This guarantees consistency between ML, Rules, and Explainability.

Run:
  python data/generate_dataset.py
  → saves data/student_data.csv
"""

import numpy as np
import pandas as pd
import os

# ── reproducibility ────────────────────────────────────────────────────────
SEED = 42
RNG  = np.random.default_rng(SEED)

N    = 500   # number of rows
NOISE_RATE = 0.12   # fraction of rows whose label is randomly flipped


# ══════════════════════════════════════════════════════════════════════════
# 1. FEATURE GENERATION
# ══════════════════════════════════════════════════════════════════════════

def generate_features(n: int) -> pd.DataFrame:
    """
    Generates n rows with all features.
    Ranges are chosen to be realistic for a student over one semester week.
    """

    # --- Search planner features (describe the STATE of the planning problem) ---

    # free hours per day (0–8)
    free_hours_per_day = RNG.integers(0, 9, size=n).astype(float)

    # number of pending tasks (1–15)
    num_pending_tasks = RNG.integers(1, 16, size=n).astype(float)

    # average duration of a single task in hours (1.0–6.0)
    avg_task_duration_hours = np.round(RNG.uniform(1.0, 6.0, size=n), 1)

    # deadline urgency — used as h(n) in A*
    # 0.0 = deadline is far away, 1.0 = deadline is today
    deadline_urgency = np.round(RNG.uniform(0.0, 1.0, size=n), 2)

    # days until the nearest exam (1–30)
    days_until_exam = RNG.integers(1, 31, size=n).astype(float)

    # --- Rule-based features (used as conditions in knowledge_base) ---

    # stress level on a scale of 1–10
    # key feature for backward chaining: if absent — the system asks for it
    stress_level = RNG.integers(1, 11, size=n).astype(float)

    # average sleep hours per night (4.0–9.0)
    sleep_hours_avg = np.round(RNG.uniform(4.0, 9.0, size=n), 1)

    # fraction of tasks completed in previous weeks (0.0–1.0)
    past_completion_rate = np.round(RNG.uniform(0.0, 1.0, size=n), 2)

    # hours per week spent on extracurricular activities (0–20)
    extracurricular_hours = RNG.integers(0, 21, size=n).astype(float)

    # fraction of classes missed (0.0–0.5)
    missed_classes_pct = np.round(RNG.uniform(0.0, 0.5, size=n), 2)

    return pd.DataFrame({
        # planner
        "free_hours_per_day":      free_hours_per_day,
        "num_pending_tasks":       num_pending_tasks,
        "avg_task_duration_hours": avg_task_duration_hours,
        "deadline_urgency":        deadline_urgency,
        "days_until_exam":         days_until_exam,
        # rules / ML
        "stress_level":            stress_level,
        "sleep_hours_avg":         sleep_hours_avg,
        "past_completion_rate":    past_completion_rate,
        "extracurricular_hours":   extracurricular_hours,
        "missed_classes_pct":      missed_classes_pct,
    })


# ══════════════════════════════════════════════════════════════════════════
# 2. DETERMINISTIC RISK LABELLING
#    The same conditions will live in rules/knowledge_base.py
#    → ML and Rules explain the same phenomenon → Explainability is coherent
# ══════════════════════════════════════════════════════════════════════════

def label_risk(row: pd.Series) -> str:
    """
    Labelling rules (3 levels: Low / Medium / High).

    High risk — at least TWO of the critical conditions:
      • stress_level > 7
      • sleep_hours_avg < 5.5
      • past_completion_rate < 0.4
      • days_until_exam <= 3  AND  num_pending_tasks >= 6
      • missed_classes_pct > 0.35

    Medium risk — at least ONE critical condition
      OR two warning conditions:
      • stress_level >= 6
      • sleep_hours_avg < 6.5
      • past_completion_rate < 0.6
      • extracurricular_hours > 12
      • free_hours_per_day < 2  AND  num_pending_tasks >= 5

    Low risk — everything else.
    """

    # critical flags (high-risk signals)
    crit = [
        row["stress_level"] > 7,
        row["sleep_hours_avg"] < 5.5,
        row["past_completion_rate"] < 0.4,
        (row["days_until_exam"] <= 3) and (row["num_pending_tasks"] >= 6),
        row["missed_classes_pct"] > 0.35,
    ]

    # warning flags (medium-risk signals)
    warn = [
        row["stress_level"] >= 6,
        row["sleep_hours_avg"] < 6.5,
        row["past_completion_rate"] < 0.6,
        row["extracurricular_hours"] > 12,
        (row["free_hours_per_day"] < 2) and (row["num_pending_tasks"] >= 5),
    ]

    n_crit = sum(crit)
    n_warn = sum(warn)

    if n_crit >= 2:
        return "High"
    elif n_crit == 1 or n_warn >= 2:
        return "Medium"
    else:
        return "Low"


def add_labels(df: pd.DataFrame, noise_rate: float = NOISE_RATE) -> pd.DataFrame:
    """
    1) Labels each row with the deterministic label_risk() function.
    2) Adds controlled noise: noise_rate% of rows receive a random label.
       This simulates real-world uncertainty and prevents the model from
       perfectly memorising the rules (otherwise RF would give 100% — suspicious).
    """
    df = df.copy()
    df["risk_label"] = df.apply(label_risk, axis=1)

    # noise — randomly flip the label for a subset of rows
    noise_mask = RNG.random(size=len(df)) < noise_rate
    all_labels = ["Low", "Medium", "High"]
    df.loc[noise_mask, "risk_label"] = RNG.choice(all_labels, size=noise_mask.sum())

    return df


# ══════════════════════════════════════════════════════════════════════════
# 3. VALIDATION AND STATISTICS
# ══════════════════════════════════════════════════════════════════════════

def validate_and_report(df: pd.DataFrame) -> None:
    """
    Validates the dataset and prints a brief summary.
    Raises AssertionError if anything is wrong (smoke test).
    """
    # --- basic checks ---
    assert len(df) == N,                          f"Expected {N} rows, got {len(df)}"
    assert df.isnull().sum().sum() == 0,           "Missing values (NaN) detected"
    assert set(df["risk_label"].unique()) <= {"Low", "Medium", "High"}, \
                                                   "Unexpected risk_label values"
    assert df["stress_level"].between(1, 10).all(), "stress_level out of range [1, 10]"
    assert df["sleep_hours_avg"].between(4, 9).all(), "sleep_hours_avg out of range [4, 9]"
    assert df["past_completion_rate"].between(0, 1).all(), "completion_rate out of range [0, 1]"

    # --- statistics ---
    print("=" * 55)
    print("  DATASET GENERATED — STATISTICS")
    print("=" * 55)
    print(f"  Rows:     {len(df)}")
    print(f"  Features: {len(df.columns) - 1}  (+ risk_label)")
    print()
    print("  Class distribution:")
    counts = df["risk_label"].value_counts()
    for label in ["Low", "Medium", "High"]:
        n = counts.get(label, 0)
        pct = n / len(df) * 100
        bar = "█" * int(pct / 3)
        print(f"    {label:<8}  {n:>4} rows  ({pct:5.1f}%)  {bar}")
    print()
    print("  Mean values for key features:")
    for col in ["stress_level", "sleep_hours_avg", "past_completion_rate",
                "num_pending_tasks", "free_hours_per_day"]:
        print(f"    {col:<30}  {df[col].mean():.2f}")
    print()
    print("  Smoke tests: PASSED")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════
# 4. SAVE
# ══════════════════════════════════════════════════════════════════════════

def save_dataset(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n  File saved: {path}")
    print(f"  Size: {os.path.getsize(path) / 1024:.1f} KB")


# ══════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("\nStage 1 — generating dataset...")

    # step 1: features
    df = generate_features(N)

    # step 2: labels
    df = add_labels(df, noise_rate=NOISE_RATE)

    # step 3: validation + report
    validate_and_report(df)

    # step 4: save
    out_path = os.path.join(os.path.dirname(__file__), "student_data.csv")
    save_dataset(df, out_path)

    # step 5: preview first rows
    print("\n  First 5 rows:")
    print(df.head().to_string(index=False))
    print()


if __name__ == "__main__":
    main()
