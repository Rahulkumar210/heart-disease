"""Preprocess the Framingham dataset and create the train/test split.

Run:  python preprocess.py
Writes: data/processed/framingham_clean.csv, train.csv, test.csv,
        reports/preprocessing_report.txt, reports/preprocessing_report.json,
        reports/input_ranges.json

Imputation is NOT done here. It is fitted only on the training data inside the model pipeline
(see train_model.py), so the test set never influences the imputed values.
"""

import hashlib
import json

import pandas as pd

from src.framingham import (
    FEATURES, PROCESSED_DIR, RAW_PATH, REPORTS_DIR, TARGET, check_columns, clean,
    input_ranges, load_raw, make_split, suspicious_values,
)


def sha256_of(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Step 1: load selected CSV:", RAW_PATH.name)
    raw = load_raw()
    print("Step 2: clean column names (done in load_raw); remove unnamed index columns (done).")
    extra = check_columns(raw)  # Step 3 check: expected columns present
    print("Step 3: expected columns present. Unexpected extra columns:", extra or "none")

    df, log = clean(raw)  # Steps 4-9 (placeholders, numeric types, target check, target-missing rows, duplicates)
    print("Steps 4-9 log:", json.dumps({k: v for k, v in log.items() if k != "missing_feature_values_kept_for_imputation"}, indent=1))

    sus = suspicious_values(df)
    sus_text = sus.to_string(index=False) if len(sus) else "none"

    # Step 10-11: separate features and target, stratified 80/20 split with random_state=42
    X_train, X_test, y_train, y_test = make_split(df)
    train = X_train.assign(**{TARGET: y_train})
    test = X_test.assign(**{TARGET: y_test})
    print(f"Step 10-11: train={len(train)} rows, test={len(test)} rows (stratified, random_state=42)")

    # Step 12: save cleaned data and splits
    df.to_csv(PROCESSED_DIR / "framingham_clean.csv", index=False)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    ranges = input_ranges(df)
    (REPORTS_DIR / "input_ranges.json").write_text(json.dumps(ranges, indent=2), encoding="utf-8")

    summary = {
        "source_file": str(RAW_PATH.relative_to(RAW_PATH.parents[2])),
        "source_sha256": sha256_of(RAW_PATH),
        "cleaning": log,
        "split": {
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "train_class_counts": {int(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
            "test_class_counts": {int(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
            "test_size": 0.2, "stratified": True, "random_state": 42,
        },
        "processed_sha256": {
            "framingham_clean.csv": sha256_of(PROCESSED_DIR / "framingham_clean.csv"),
            "train.csv": sha256_of(PROCESSED_DIR / "train.csv"),
            "test.csv": sha256_of(PROCESSED_DIR / "test.csv"),
        },
        "suspicious_values_reported": sus.to_dict(orient="records"),
    }
    (REPORTS_DIR / "preprocessing_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "FRAMINGHAM PREPROCESSING REPORT",
        f"Source: {summary['source_file']}  sha256={summary['source_sha256']}",
        f"Rows loaded: {log['rows_loaded']}",
        f"Values coerced to missing (non-numeric text): {log['values_coerced_to_missing'] or 'none'}",
        f"Rows removed for missing target: {log['target_missing_rows_removed']}",
        f"Exact duplicate rows removed: {log['exact_duplicates_removed']}",
        f"Rows after cleaning: {log['rows_after_cleaning']}",
        f"Rows with at least one missing feature (kept, imputed in pipeline): {log['rows_with_any_missing_feature']}",
        f"Missing feature values kept: {log['missing_feature_values_kept_for_imputation']}",
        f"Train: {len(train)} rows, class counts {summary['split']['train_class_counts']}",
        f"Test:  {len(test)} rows, class counts {summary['split']['test_class_counts']}",
        f"Suspicious values: {sus_text}",
        "Imputation: median, fitted on TRAINING data only (inside the model pipeline).",
        "SMOTE / oversampling: not used.",
    ]
    (REPORTS_DIR / "preprocessing_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
