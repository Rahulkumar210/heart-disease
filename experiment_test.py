"""STAGE B: refit the pre-selected model on the full training set and evaluate ONCE on the test set.

Run:  python experiment_test.py   (run only after experiment_select.py)
Checks that selection_decision.json was made on the same train.csv (hash), then:
  - refits every candidate on the full training set (selected model is saved as models/framingham_selected.joblib)
  - reads test.csv ONCE and reports metrics, with bootstrap 95% CIs
  - reports every candidate at its own OOF-chosen threshold, and at 0.5, for transparency only
  - runs a paired bootstrap on TRAINING OOF predictions: LR vs EBM ROC-AUC difference (diagnostic only)
"""

import hashlib
import json
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.experiment import (
    CANDIDATES, bootstrap_ci, metric_recall_at, metric_roc_auc, metric_specificity_at,
    ranking_and_calibration, threshold_metrics, calibration_points,
)
from src.framingham import FEATURES, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, TARGET

EXP_DIR = REPORTS_DIR / "experiment"
SELECTED_MODEL_PATH = MODELS_DIR / "framingham_selected.joblib"


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def paired_bootstrap_auc_diff(y, p_a, p_b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    d = np.array(diffs)
    return {"observed_difference": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "share_of_bootstraps_a_better": float((d > 0).mean()), "n_boot": int(len(d))}


def main() -> None:
    decision_path = EXP_DIR / "selection_decision.json"
    if not decision_path.exists():
        raise SystemExit("selection_decision.json not found. Run experiment_select.py first.")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    train_path, test_path = PROCESSED_DIR / "train.csv", PROCESSED_DIR / "test.csv"
    if sha256(train_path) != decision["train_sha256"]:
        raise SystemExit("train.csv changed after selection. Selection is invalid; re-run experiment_select.py.")
    print("Decision file verified against train.csv hash (selection made before test evaluation).")
    selected, thr = decision["selected_model"], decision["selected_threshold"]
    print(f"Selected model: {selected} | threshold: {thr:.3f}")

    train = pd.read_csv(train_path)
    X, y = train[FEATURES], train[TARGET].astype(int)
    comparison = json.loads((EXP_DIR / "oof_comparison.json").read_text(encoding="utf-8"))

    # Refit all candidates on the full training set (test set not involved)
    fitted = {}
    for name, builder in CANDIDATES.items():
        fitted[name] = builder().fit(X, y)
    joblib.dump({"algorithm": selected, "features": FEATURES, "target": TARGET, "threshold": thr,
                 "pipeline": fitted[selected],
                 "metadata": {"selection_decision": str(decision_path.name), "train_sha256": decision["train_sha256"]}},
                SELECTED_MODEL_PATH)
    print(f"Saved refit {selected} to {SELECTED_MODEL_PATH.name}")

    # ---- single evaluation on the test set ----
    test = pd.read_csv(test_path)
    Xt, yt = test[FEATURES], test[TARGET].astype(int)
    print(f"\nTest set read ONCE: {len(test)} rows, positives {int(yt.sum())}")
    test_probs = {name: fitted[name].predict_proba(Xt)[:, 1] for name in CANDIDATES}
    np.save(EXP_DIR / "test_probabilities.npy", np.column_stack([test_probs[n] for n in CANDIDATES]))

    p_sel = test_probs[selected]
    sel_test = threshold_metrics(yt, p_sel, thr)
    rc_sel = ranking_and_calibration(yt, p_sel)
    ci = {}
    for label, fn, t in [("roc_auc", metric_roc_auc, None), ("recall", metric_recall_at, thr),
                         ("specificity", metric_specificity_at, thr)]:
        lo, hi, nb = bootstrap_ci(yt.values, p_sel, fn, t, n_boot=1000)
        ci[label] = [lo, hi]
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(42); pr_vals = []
    yv = yt.values
    for _ in range(1000):
        idx = rng.integers(0, len(yv), len(yv))
        if yv[idx].min() != yv[idx].max():
            pr_vals.append(average_precision_score(yv[idx], p_sel[idx]))
    ci["pr_auc"] = [float(np.percentile(pr_vals, 2.5)), float(np.percentile(pr_vals, 97.5))]

    print(f"\nTEST RESULTS for selected model {selected} at threshold {thr:.3f}:")
    for k in ["tn", "fp", "fn", "tp"]:
        print(f"  {k.upper()}={sel_test[k]}", end="")
    print()
    for k in ["recall", "specificity", "precision", "f1"]:
        print(f"  {k:12s} {sel_test[k]:.4f}  95% CI {ci.get(k, ['-', '-'])}")
    print(f"  ROC-AUC      {rc_sel['roc_auc']:.4f}  95% CI {[round(v,4) for v in ci['roc_auc']]}")
    print(f"  PR-AUC       {rc_sel['pr_auc']:.4f}  95% CI {[round(v,4) for v in ci['pr_auc']]}")
    print(f"  Brier        {rc_sel['brier']:.4f} | ECE(10 bins) {rc_sel['ece_10bins']:.4f} | "
          f"mean predicted {rc_sel['mean_predicted']:.4f} vs observed {rc_sel['observed_rate']:.4f}")

    # All candidates (reporting only): each at its own OOF threshold and at 0.5
    all_rows = {}
    print("\nAll candidates on the test set (reporting only; decision already fixed):")
    for name in CANDIDATES:
        p = test_probs[name]
        t_own = comparison[name]["youden_threshold_on_oof"]
        rc = ranking_and_calibration(yt, p)
        at_own = threshold_metrics(yt, p, t_own)
        at_05 = threshold_metrics(yt, p, 0.5)
        all_rows[name] = {"test_ranking_and_calibration": rc, "own_oof_threshold": t_own,
                          "test_at_own_oof_threshold": at_own, "test_at_0.5": at_05}
        print(f"  {name:30s} ROC-AUC={rc['roc_auc']:.4f} PR-AUC={rc['pr_auc']:.4f} Brier={rc['brier']:.4f} "
              f"| thr={t_own:.3f} recall={at_own['recall']:.3f} spec={at_own['specificity']:.3f} "
              f"| @0.5 recall={at_05['recall']:.3f} spec={at_05['specificity']:.3f}")

    # Training-only diagnostic: is the LR vs EBM AUC gap larger than bootstrap noise?
    oof = pd.read_csv(EXP_DIR / "oof_predictions.csv")
    diag = paired_bootstrap_auc_diff(oof[TARGET].values, oof["LogisticRegression_balanced"].values,
                                     oof["EBM"].values, n_boot=2000)
    print(f"\nTraining OOF diagnostic, ROC-AUC(LR) - ROC-AUC(EBM): {diag['observed_difference']:+.4f} "
          f"95% CI {[round(v,4) for v in diag['ci95']]}")

    # Calibration plot on the test set
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
    for name in CANDIDATES:
        frac, mp = calibration_points(yt, test_probs[name], n_bins=8)
        ax.plot(mp, frac, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed CHD fraction")
    ax.set_title("Calibration on test set (reporting only)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(EXP_DIR / "calibration_test.png", dpi=120); plt.close(fig)

    out = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_model": selected, "selected_threshold": thr,
        "test_rows": int(len(test)), "test_positives": int(yt.sum()),
        "selected_test_metrics_at_threshold": sel_test,
        "selected_test_ranking_and_calibration": rc_sel,
        "selected_bootstrap_95ci": ci,
        "all_candidates_test_reporting_only": all_rows,
        "training_oof_paired_auc_diagnostic_LR_minus_EBM": diag,
    }
    (EXP_DIR / "test_results.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {EXP_DIR / 'test_results.json'}")


if __name__ == "__main__":
    main()
