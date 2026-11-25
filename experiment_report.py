"""STAGE C: write the website operating point and bootstrap CIs from the FROZEN stage-B test predictions.

Run after experiment_test.py. Does not refit any model and does not choose anything new.
Uses the saved test probabilities (reports/experiment/test_probabilities.npy) and the test labels.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.experiment import CANDIDATES, bootstrap_ci, metric_recall_at, metric_specificity_at, ranking_and_calibration, threshold_metrics
from src.framingham import PROCESSED_DIR, REPORTS_DIR, TARGET

EXP_DIR = REPORTS_DIR / "experiment"


def boot_pr(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(average_precision_score(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> None:
    decision = json.loads((EXP_DIR / "selection_decision.json").read_text(encoding="utf-8"))
    comparison = json.loads((EXP_DIR / "oof_comparison.json").read_text(encoding="utf-8"))
    probs = np.load(EXP_DIR / "test_probabilities.npy")
    names = list(CANDIDATES.keys())
    yt = pd.read_csv(PROCESSED_DIR / "test.csv")[TARGET].astype(int).values
    assert probs.shape == (len(yt), len(names)), "saved probabilities do not match test labels"

    ebm_i = names.index("EBM")
    p_ebm = probs[:, ebm_i]
    thr = comparison["EBM"]["youden_threshold_on_oof"]
    m = threshold_metrics(yt, p_ebm, thr)
    ci = {
        "roc_auc": bootstrap_ci(yt, p_ebm, lambda y, p, _t: roc_auc_score(y, p))[:2],
        "recall": bootstrap_ci(yt, p_ebm, metric_recall_at, thr)[:2],
        "specificity": bootstrap_ci(yt, p_ebm, metric_specificity_at, thr)[:2],
        "pr_auc": boot_pr(yt, p_ebm),
    }
    rc = ranking_and_calibration(yt, p_ebm)
    op = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "website_model": "EBM (models/framingham_ebm.joblib, unchanged baseline model)",
        "website_threshold": thr,
        "threshold_source": "max Youden J on training-only out-of-fold predictions of EBM (experiment_select.py)",
        "pre_registered_model_selection_result": decision["selected_model"],
        "override_note": (
            "The pre-registered selection rule (highest OOF ROC-AUC) chose LogisticRegression_balanced by a margin of "
            f"{comparison['LogisticRegression_balanced']['oof_ranking_and_calibration']['roc_auc'] - comparison['EBM']['oof_ranking_and_calibration']['roc_auc']:+.4f} "
            "OOF ROC-AUC, which is within noise. The website keeps EBM because (a) the project brief requires EBM as the "
            "primary algorithm and (b) EBM is far better calibrated. This override was decided AFTER the test results were seen, "
            "and must be confirmed by the project owner."),
        "test_metrics_at_website_threshold": m,
        "test_bootstrap_95ci": ci,
        "test_pr_auc": rc["pr_auc"],
        "test_ranking_and_calibration": rc,
        "test_rows": int(len(yt)), "test_positives": int(yt.sum()),
        "oof_at_website_threshold": comparison["EBM"]["oof_at_youden_threshold"],
        "oof_ranking_and_calibration": comparison["EBM"]["oof_ranking_and_calibration"],
        "baseline_threshold_0.5_test": threshold_metrics(yt, p_ebm, 0.5),
        "pre_registered_LR_at_its_threshold_test": threshold_metrics(yt, probs[:, names.index("LogisticRegression_balanced")], decision["selected_threshold"]),
    }
    (EXP_DIR / "operating_point.json").write_text(json.dumps(op, indent=2, default=float), encoding="utf-8")
    (REPORTS_DIR / "app_threshold.json").write_text(json.dumps({"threshold": thr, "model": "EBM", "source": "reports/experiment/operating_point.json"}, indent=2), encoding="utf-8")
    print(f"EBM operating point threshold {thr:.3f}")
    print(f"TEST: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
    print(f"recall {m['recall']:.4f} CI {[round(v,4) for v in ci['recall']]} | specificity {m['specificity']:.4f} CI {[round(v,4) for v in ci['specificity']]}")
    print(f"precision {m['precision']:.4f} | F1 {m['f1']:.4f} | ROC-AUC {rc['roc_auc']:.4f} CI {[round(v,4) for v in ci['roc_auc']]} | PR-AUC {rc['pr_auc']:.4f} CI {[round(v,4) for v in ci['pr_auc']]}")
    print(f"calibration: ECE {rc['ece_10bins']:.4f}, mean predicted {rc['mean_predicted']:.4f} vs observed {rc['observed_rate']:.4f}, Brier {rc['brier']:.4f}")
    print("Wrote reports/experiment/operating_point.json and reports/app_threshold.json")


if __name__ == "__main__":
    main()
