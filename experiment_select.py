"""STAGE A (training data only): compare candidates with out-of-fold predictions, select model and threshold.

Run:  python experiment_select.py
Uses ONLY data/processed/train.csv. Never reads test.csv.
Writes reports/experiment/ and reports/experiment/selection_decision.json (read by experiment_test.py).
"""

import hashlib
import json
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment import (
    CANDIDATES, out_of_fold, ranking_and_calibration, select_youden_threshold, threshold_metrics,
    threshold_sweep, calibration_points,
)
from src.framingham import FEATURES, PROCESSED_DIR, REPORTS_DIR, TARGET

EXP_DIR = REPORTS_DIR / "experiment"


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.csv"
    prep = json.loads((REPORTS_DIR / "preprocessing_report.json").read_text(encoding="utf-8"))
    if sha256(train_path) != prep["processed_sha256"]["train.csv"]:
        raise RuntimeError("train.csv does not match preprocessing hash. Stop.")
    train = pd.read_csv(train_path)
    X, y = train[FEATURES], train[TARGET].astype(int)
    print(f"Training data only: {len(train)} rows, positives {int(y.sum())} ({y.mean():.2%})")
    print("Held-out test file is NOT opened in this script.\n")

    oof_frame = pd.DataFrame({"row": np.arange(len(y)), TARGET: y.values})
    comparison = {}
    sweeps = {}
    for name, builder in CANDIDATES.items():
        t0 = time.time()
        oof = out_of_fold(builder, X, y)
        oof_frame[name] = oof
        rc = ranking_and_calibration(y, oof)
        sweep = threshold_sweep(y, oof)
        sweeps[name] = sweep
        t_youden = select_youden_threshold(sweep)
        at_05 = threshold_metrics(y, oof, 0.5)
        at_youden = threshold_metrics(y, oof, t_youden)
        comparison[name] = {
            "oof_ranking_and_calibration": rc,
            "oof_at_threshold_0.5": at_05,
            "youden_threshold_on_oof": t_youden,
            "oof_at_youden_threshold": at_youden,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{name}: OOF ROC-AUC={rc['roc_auc']:.4f} PR-AUC={rc['pr_auc']:.4f} "
              f"Brier={rc['brier']:.4f} ECE={rc['ece_10bins']:.4f} | "
              f"Youden thr={t_youden:.3f} recall={at_youden['recall']:.3f} spec={at_youden['specificity']:.3f} "
              f"({comparison[name]['seconds']}s)")

    oof_frame.to_csv(EXP_DIR / "oof_predictions.csv", index=False)

    # PRE-REGISTERED SELECTION RULE: highest OOF ROC-AUC; tie-break by lower OOF log-loss.
    ranked = sorted(comparison.items(),
                    key=lambda kv: (-round(kv[1]["oof_ranking_and_calibration"]["roc_auc"], 10),
                                    kv[1]["oof_ranking_and_calibration"]["log_loss"]))
    selected = ranked[0][0]
    sweep = sweeps[selected]
    sweep.to_csv(EXP_DIR / f"threshold_sweep_oof_{selected}.csv", index=False)
    thr = select_youden_threshold(sweep)
    sel = threshold_metrics(y, oof_frame[selected].values, thr)

    print(f"\nSelected model (highest OOF ROC-AUC): {selected}")
    print(f"Selected threshold (max Youden J on OOF): {thr:.3f}")
    print(f"OOF at this threshold: recall={sel['recall']:.4f} specificity={sel['specificity']:.4f} "
          f"precision={sel['precision']:.4f} F1={sel['f1']:.4f} J={sel['youden_j']:.4f}")
    print(f"OOF confusion (TN,FP,FN,TP) = ({sel['tn']},{sel['fp']},{sel['fn']},{sel['tp']})")

    # Calibration plots (OOF, training data)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    for name in CANDIDATES:
        frac, mp = calibration_points(y, oof_frame[name].values, n_bins=10)
        ax.plot(mp, frac, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability (quantile bins)")
    ax.set_ylabel("Observed CHD fraction")
    ax.set_title("Out-of-fold calibration (training data)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(EXP_DIR / "calibration_oof.png", dpi=120); plt.close(fig)

    # Trade-off plot (OOF)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(sweep["threshold"], sweep["recall"], label="Recall (sensitivity)")
    ax.plot(sweep["threshold"], sweep["specificity"], label="Specificity")
    ax.plot(sweep["threshold"], sweep["f1"], label="F1")
    ax.axvline(thr, color="black", linestyle="--", label=f"Selected threshold {thr:.3f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Metric value (out-of-fold, training data)")
    ax.set_title(f"Recall-specificity trade-off: {selected}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(EXP_DIR / "tradeoff_oof.png", dpi=120); plt.close(fig)

    decision = {
        "decided_at": datetime.now().isoformat(timespec="seconds"),
        "decided_before_test_evaluation": True,
        "train_sha256": sha256(train_path),
        "data_used": "data/processed/train.csv only (out-of-fold, stratified 5-fold, seed 42)",
        "model_selection_rule": "highest OOF ROC-AUC; tie-break lower OOF log-loss",
        "threshold_rule": "maximise Youden's J = sensitivity + specificity - 1 on OOF predictions of the selected model; ties -> highest threshold; grid 0.005..0.995 step 0.005",
        "selected_model": selected,
        "selected_threshold": thr,
        "ranking_order_by_oof_roc_auc": [k for k, _ in ranked],
        "oof_selected_metrics_at_threshold": sel,
    }
    (EXP_DIR / "selection_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (EXP_DIR / "oof_comparison.json").write_text(json.dumps(comparison, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {EXP_DIR / 'selection_decision.json'}")


if __name__ == "__main__":
    main()
