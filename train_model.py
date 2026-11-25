"""Train and evaluate the EBM model for Framingham 10-year CHD.

Run:  python train_model.py
Steps:
  1. Verify the saved train/test files match the hashes from preprocessing
  2. 5-fold stratified CV on TRAINING data only (accuracy, precision, recall, specificity, F1, ROC-AUC)
  3. Fit the EBM pipeline (median imputer + EBM) on the training data only
  4. Evaluate ONCE on the untouched test set (threshold 0.5)
  5. Save models/framingham_ebm.joblib, reports/metrics.json, classification report and figures
"""

import hashlib
import json
from datetime import datetime

import interpret
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.evaluation import compute_metrics, specificity_score
from src.explain import feature_effects, global_importance
from src.framingham import (
    FEATURES, MODEL_PATH, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET,
)
from src.model import MODEL_ALGORITHM, build_ebm_pipeline
from src.plots import confusion_figure, effects_figure, importance_figure, roc_figure


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def verify_split_files() -> dict:
    report = json.loads((REPORTS_DIR / "preprocessing_report.json").read_text(encoding="utf-8"))
    expected = report["processed_sha256"]
    for name, digest in expected.items():
        actual = sha256(PROCESSED_DIR / name)
        if actual != digest:
            raise RuntimeError(f"{name} changed since preprocessing (hash mismatch). Re-run preprocess.py.")
    return report


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True); REPORTS_DIR.mkdir(exist_ok=True)
    prep = verify_split_files()
    print("1) Split files verified against preprocessing_report.json")

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train, y_train = train[FEATURES], train[TARGET].astype(int)
    X_test, y_test = test[FEATURES], test[TARGET].astype(int)
    print(f"   train={len(X_train)} (positives {int(y_train.sum())}), test={len(X_test)} (positives {int(y_test.sum())})")

    print("2) 5-fold stratified CV on TRAINING data only...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1",
        "roc_auc": "roc_auc", "specificity": make_scorer(specificity_score),
    }
    cvres = cross_validate(build_ebm_pipeline(), X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    cv_summary = {k.replace("test_", ""): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                  for k, v in cvres.items() if k.startswith("test_")}
    for k, v in cv_summary.items():
        print(f"   CV {k:12s} {v['mean']:.4f} +/- {v['std']:.4f}")

    print("3) Fitting EBM on training data only...")
    model = build_ebm_pipeline().fit(X_train, y_train)

    print("4) Evaluating ONCE on the held-out test set (threshold 0.5)...")
    prob = model.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test.values, prob)
    for k in ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc"]:
        print(f"   {k:20s} {m[k]:.4f}")
    print("   confusion matrix:", m["confusion_matrix"])
    cls_report = classification_report(y_test, (prob >= 0.5).astype(int), target_names=["No CHD (0)", "CHD (1)"], digits=4, zero_division=0)
    print(cls_report)

    print("5) Reproducibility check: second fit with same seed...")
    second = build_ebm_pipeline().fit(X_train, y_train)
    reproducible = bool(np.allclose(second.predict_proba(X_test)[:, 1], prob))
    print("   identical test probabilities:", reproducible)

    print("6) Saving model, metrics, explanations and figures...")
    joblib.dump({
        "algorithm": MODEL_ALGORITHM,
        "features": FEATURES,
        "target": TARGET,
        "pipeline": model,
        "metadata": {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "interpret_version": interpret.__version__,
            "scikit_learn_version": sklearn.__version__,
            "random_state": RANDOM_STATE,
            "interactions": 0,
            "imputation": "median (fitted on training data only)",
            "train_sha256": prep["processed_sha256"]["train.csv"],
            "test_sha256": prep["processed_sha256"]["test.csv"],
        },
    }, MODEL_PATH)

    imp = global_importance(model)
    effects = feature_effects(model)
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "primary_model": {"algorithm": MODEL_ALGORITHM, "interactions": 0, "random_state": RANDOM_STATE},
        "library_versions": {"interpret": interpret.__version__, "scikit-learn": sklearn.__version__},
        "data": {"source": prep["source_file"], "source_sha256": prep["source_sha256"],
                 "cleaning": prep["cleaning"], "split": prep["split"]},
        "cross_validation_train_only": cv_summary,
        "test": m,
        "reproducible_across_two_fits": reproducible,
        "global_term_importance": {k: round(float(v), 6) for k, v in imp.items()},
        "intercept_log_odds": float(np.asarray(model.named_steps["ebm"].intercept_).ravel()[0]),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORTS_DIR / "classification_report.txt").write_text(cls_report, encoding="utf-8")
    confusion_figure(m["confusion_matrix"]).savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    roc_figure(m["roc_curve"], m["roc_auc"]).savefig(REPORTS_DIR / "roc_curve.png", dpi=120)
    importance_figure(imp).savefig(REPORTS_DIR / "global_importance.png", dpi=120)
    effects_figure(effects).savefig(REPORTS_DIR / "feature_effects.png", dpi=110)
    print(f"Done. Model: {MODEL_PATH}")
    print(f"      Metrics: {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
