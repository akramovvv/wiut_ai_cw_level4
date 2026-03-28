"""
ml/train.py
===========
Stage 2 — Student Success Copilot

What it does:
  1. Loads student_data.csv
  2. Train/test split 80/20 (stratified)
  3. Trains RandomForestClassifier
  4. Computes Accuracy and F1-macro
  5. Saves the model to ml/model.pkl
  6. predict(profile, bundle) → {"label", "proba", "top_features"}

Run:
  python ml/train.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml",   "model.pkl")

LABEL_ORDER  = ["Low", "Medium", "High"]
FEATURE_COLS = [
    "free_hours_per_day",
    "num_pending_tasks",
    "avg_task_duration_hours",
    "deadline_urgency",
    "days_until_exam",
    "stress_level",
    "sleep_hours_avg",
    "past_completion_rate",
    "extracurricular_hours",
    "missed_classes_pct",
]


# ══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_data(path: str = DATA_PATH):
    df = pd.read_csv(path)
    assert "risk_label" in df.columns
    assert all(c in df.columns for c in FEATURE_COLS)

    X = df[FEATURE_COLS].values

    le = LabelEncoder()
    le.fit(LABEL_ORDER)            # Low=0, Medium=1, High=2
    y = le.transform(df["risk_label"].values)

    return X, y, le


# ══════════════════════════════════════════════════════════════════════════
# 2. TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Parameters:
      n_estimators=200  — stable feature importance
      max_depth=8       — prevents overfitting on noisy rows
      class_weight='balanced' — compensates for the underrepresented Low class
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


# ══════════════════════════════════════════════════════════════════════════
# 3. EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def evaluate(clf, X_test, y_test, le) -> dict:
    y_pred  = clf.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    f1_mac  = f1_score(y_test, y_pred, average="macro")
    f1_cls  = f1_score(y_test, y_pred, average=None, labels=[0,1,2])

    print("\n" + "="*55)
    print("  MODEL METRICS")
    print("="*55)
    print(f"  Accuracy      : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 (macro)    : {f1_mac:.4f}")
    print()
    for i, lbl in enumerate(LABEL_ORDER):
        print(f"  F1 ({lbl:<6})  : {f1_cls[i]:.4f}")
    print()
    print("  Classification report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_ORDER, digits=3))

    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
    print("  Confusion matrix:")
    print(f"  {'':>12}  {'Pred Low':>10}  {'Pred Med':>10}  {'Pred High':>10}")
    for i, lbl in enumerate(LABEL_ORDER):
        print(f"  {'True '+lbl:<12}  {cm[i][0]:>10}  {cm[i][1]:>10}  {cm[i][2]:>10}")
    print()

    return {"accuracy": acc, "f1_macro": f1_mac,
            "f1_per_class": dict(zip(LABEL_ORDER, f1_cls.tolist()))}


# ══════════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════

def print_feature_importance(clf) -> None:
    pairs = sorted(zip(FEATURE_COLS, clf.feature_importances_),
                   key=lambda x: x[1], reverse=True)
    print("  Feature importances (top-5):")
    for i, (feat, imp) in enumerate(pairs[:5], 1):
        bar = "\u25aa" * int(imp * 80)
        print(f"  {i}. {feat:<30}  {imp:.4f}  {bar}")
    print()


# ══════════════════════════════════════════════════════════════════════════
# 5. SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════

def save_model(clf, le, path: str = MODEL_PATH) -> None:
    bundle = {"clf": clf, "le": le, "feature_cols": FEATURE_COLS}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Model saved: {path}  ({os.path.getsize(path)/1024:.1f} KB)")


def load_model(path: str = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════
# 6. PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def predict(profile: dict, bundle: dict = None) -> dict:
    """
    Prediction for a single student.

    Returns:
      {
        "label":        str   — "Low" | "Medium" | "High"
        "proba":        dict  — {"Low": 0.05, "Medium": 0.17, "High": 0.78}
        "top_features": list  — [("stress_level", 0.24), ("sleep_hours_avg", 0.19), ...]
      }
    """
    if bundle is None:
        bundle = load_model()

    clf  = bundle["clf"]
    le   = bundle["le"]
    cols = bundle["feature_cols"]

    x = np.array([[profile[c] for c in cols]])

    label  = str(le.inverse_transform(clf.predict(x))[0])
    probas = clf.predict_proba(x)[0]

    proba_dict = {
        str(le.inverse_transform([i])[0]): round(float(p), 4)
        for i, p in enumerate(probas)
    }

    top_features = sorted(
        zip(cols, clf.feature_importances_.tolist()),
        key=lambda t: t[1], reverse=True
    )[:5]

    return {"label": label, "proba": proba_dict, "top_features": top_features}


# ══════════════════════════════════════════════════════════════════════════
# 6b. LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════

ML_LIMITATIONS = """
ML Model Limitations
====================
1. Synthetic dataset (500 rows). The data is generated programmatically with
   controlled noise (~12%), not collected from real students. Feature
   distributions and correlations may not reflect genuine academic behaviour.

2. Circular label logic. Risk labels are derived from the same features the
   model learns on (via deterministic rules + noise). This inflates apparent
   accuracy because the model can partially reverse-engineer the labelling
   function rather than discover novel patterns. A real deployment would need
   ground-truth outcomes (e.g., actual exam results, dropout events).

3. No cross-validation. We use a single 80/20 stratified split. k-fold
   cross-validation would give a more robust estimate of generalisation error,
   especially given the small dataset size.

4. Feature set is limited. Real academic risk depends on factors not captured
   here: engagement quality, learning disabilities, financial stress, course
   difficulty, social support, etc. The model's predictions are therefore a
   simplified proxy.

5. RandomForest bias. Tree-based models can overfit to threshold-style
   decision boundaries (which match our rule-generated labels) but may
   underperform on more subtle, continuous relationships found in real data.

6. Class imbalance. Although class_weight='balanced' is used, the Low class
   remains underrepresented after noise injection, which can lower per-class
   recall for minority classes.
""".strip()


# ══════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("\nStage 2 — training ML model...")

    print(f"\n  Loading data from {DATA_PATH}...")
    X, y, le = load_data()
    print(f"  Loaded: {X.shape[0]} rows × {X.shape[1]} features")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_tr)} rows  |  Test: {len(X_te)} rows")

    print("\n  Training RandomForestClassifier...")
    clf = train_model(X_tr, y_tr)
    print("  Done.")

    metrics = evaluate(clf, X_te, y_te, le)
    print_feature_importance(clf)
    save_model(clf, le)

    # smoke test
    print("\n  Test predict() — at-risk student:")
    at_risk = {
        "free_hours_per_day": 1.0, "num_pending_tasks": 11.0,
        "avg_task_duration_hours": 4.5, "deadline_urgency": 0.92,
        "days_until_exam": 2.0, "stress_level": 9.0,
        "sleep_hours_avg": 4.2, "past_completion_rate": 0.22,
        "extracurricular_hours": 16.0, "missed_classes_pct": 0.42,
    }
    bundle = load_model()
    r = predict(at_risk, bundle)
    print(f"    risk_label    : {r['label']}")
    print(f"    probabilities : { {k: f'{v:.2f}' for k,v in r['proba'].items()} }")
    print(f"    top_feature   : {r['top_features'][0][0]}")

    assert r["label"] in LABEL_ORDER
    assert abs(sum(r["proba"].values()) - 1.0) < 1e-3
    assert len(r["top_features"]) == 5
    print("  Smoke tests: PASSED")

    print(f"\n{'='*55}")
    print(ML_LIMITATIONS)
    print(f"{'='*55}")

    print(f"\n  Ready for Stage 3 (Rules).")


if __name__ == "__main__":
    main()
