"""Inspect the Framingham dataset and compare the two Framingham versions.

Run:  python inspect_data.py
Writes: reports/inspection_report.txt, reports/descriptive_stats.csv,
        reports/missing_values.csv, reports/dataset_comparison.txt
Does not modify anything in data/raw/.
"""

from io import StringIO

import numpy as np
import pandas as pd

from src.framingham import (
    EXPECTED_COLUMNS, FEATURES, RAW_PATH, REPORTS_DIR, TARGET,
    check_columns, load_raw, suspicious_values, to_numeric_features,
)

COMPARE_PATH = RAW_PATH.parent / "CHD_preprocessed_reference_not_used.csv"


def inspect(df: pd.DataFrame, buf: StringIO) -> None:
    p = lambda *a: print(*a, file=buf)
    p("=" * 70); p("FRAMINGHAM DATASET INSPECTION"); p("=" * 70)
    p(f"Source file: {RAW_PATH.name}")
    p(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    p(f"Columns: {list(df.columns)}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    p(f"Unexpected extra columns: {extra if extra else 'none'}")
    p("\nData types (as read from CSV):")
    p(df.dtypes.astype(str).to_string())
    p("\nFirst five rows:")
    p(df.head().to_string())
    miss = df.isna().sum()
    p("\nMissing values per column (count, % of rows):")
    p(pd.DataFrame({"missing": miss, "percent": (100 * miss / len(df)).round(2)}).to_string())
    p(f"\nRows with at least one missing value: {int(df.isna().any(axis=1).sum())} "
      f"({100 * df.isna().any(axis=1).mean():.2f}%)")
    p(f"Exact duplicate rows: {int(df.duplicated().sum())}")
    p("\nTarget distribution (TenYearCHD):")
    vc = df[TARGET].value_counts(dropna=False).sort_index()
    for k, v in vc.items():
        p(f"  {k}: {v} ({100 * v / len(df):.2f}%)")
    p("\nDescriptive statistics:")
    desc = df[FEATURES + [TARGET]].describe().T
    p(desc.round(3).to_string())
    p("\nMinimum and maximum:")
    p(pd.DataFrame({"min": df[FEATURES + [TARGET]].min(), "max": df[FEATURES + [TARGET]].max()}).to_string())
    p("\nSuspicious or invalid values (reported, not deleted):")
    sus = suspicious_values(df)
    p(sus.to_string(index=False) if len(sus) else "  none found")
    p("\nNotes:")
    p("  - Missing values are kept for imputation inside the model pipeline.")
    p("  - 'education' is an ordinal code (1-4), not a continuous amount.")
    desc.round(4).to_csv(REPORTS_DIR / "descriptive_stats.csv")
    pd.DataFrame({"missing": miss, "percent": 100 * miss / len(df)}).round(3).to_csv(REPORTS_DIR / "missing_values.csv")


def compare_versions(raw: pd.DataFrame, buf: StringIO) -> None:
    """Compare framingham_heart_disease.csv (4,238 rows) with CHD_preprocessed.csv (4,133 rows)."""
    p = lambda *a: print(*a, file=buf)
    p("\n" + "=" * 70); p("COMPARISON: framingham_heart_disease.csv vs CHD_preprocessed (archive.zip)"); p("=" * 70)
    if not COMPARE_PATH.exists():
        p("Reference file not found; comparison skipped."); return
    other = pd.read_csv(COMPARE_PATH)
    p(f"Original rows: {len(raw)}   Preprocessed rows: {len(other)}   Difference: {len(raw) - len(other)}")
    p(f"Missing education in original: {int(raw['education'].isna().sum())}")
    p(f"Education values  original: {sorted(raw['education'].dropna().unique().tolist())}")
    p(f"Education values  preprocessed: {sorted(other['education'].unique().tolist())}")
    p("Preprocessed file has no missing values: " + str(bool(other.isna().sum().sum() == 0)))

    cols = [c for c in FEATURES if c != "education"]
    a_full = raw[cols].astype(float)
    b = other[cols].astype(float)
    a_complete = a_full.dropna()
    ka = a_complete.round(6).apply(tuple, axis=1).value_counts()
    kb = b.round(6).apply(tuple, axis=1).value_counts()
    matched = int(sum(min(kb[k], ka.get(k, 0)) for k in kb.index))
    p(f"\nRows in preprocessed file that exactly match an original COMPLETE row (ignoring education): {matched} of {len(b)}")
    p(f"Rows in preprocessed file that match no original row (ignoring education): {len(b) - matched}")
    p("Interpretation: the preprocessed file appears to be the original minus rows with missing "
      "education, with other missing values filled by a method that is not documented.")
    p("Its education column is re-encoded (codes 1-2 -> 0, codes 3-4 -> 1), so its meaning differs.")
    p("Decision: use the ORIGINAL file as the primary dataset (explicit missing values, original encoding).")
    dups_orig = int(raw.duplicated().sum()); dups_other = int(other.duplicated().sum())
    p(f"Exact duplicates: original={dups_orig}, preprocessed={dups_other}")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    raw = load_raw()
    extra = check_columns(raw)  # raises if an expected column is missing
    _, coerced = to_numeric_features(raw)
    buf = StringIO()
    inspect(raw, buf)
    p = lambda *a: print(*a, file=buf)
    p(f"\nValues that were not numeric and became missing when converted: {coerced if coerced else 'none'}")
    compare_versions(raw, buf)
    text = buf.getvalue()
    (REPORTS_DIR / "inspection_report.txt").write_text(text, encoding="utf-8")
    (REPORTS_DIR / "dataset_comparison.txt").write_text(text.split("COMPARISON:", 1)[-1] if "COMPARISON:" in text else "", encoding="utf-8")
    print(text)
    print(f"\nSaved: reports/inspection_report.txt, descriptive_stats.csv, missing_values.csv, dataset_comparison.txt")


if __name__ == "__main__":
    main()
