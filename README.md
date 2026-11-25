# Framingham 10-Year CHD Risk Prediction with an Explainable Boosting Machine

An academic research prototype. It estimates the chance of a coronary heart disease (CHD) event within
10 years from 15 measurements in the Framingham Heart Study data, using an **Explainable Boosting Machine
(EBM)** from InterpretML. It is **not** a diagnostic tool, is **not** clinically validated, and must not be
used for medical decisions.

This is a separate project from the Cleveland heart disease EBM project. Nothing is shared or copied between them.

---


## Model status (read this first)

| Item | Value |
|---|---|
| Baseline model | EBM, `models/framingham_ebm.joblib`, threshold **0.50** (original, unchanged) |
| Website model | the same EBM model (not retrained), threshold **0.155** |
| Threshold source | max Youden's J on training-only out-of-fold predictions (no test data used) |
| Pre-registered selection result | Logistic Regression (balanced), saved as `models/framingham_selected.joblib`, **not** used by the website |
| Override | EBM kept for the website (project brief requires EBM; LR calibration poor; AUC gap within noise). Exploratory, made after test results were examined |
| Status | Academic research prototype. Not clinically validated. Not a personal risk score |

## 1. Project objective

Predict `TenYearCHD` (1 = CHD event within 10 years, 0 = no recorded event) and explain each prediction and the
model as a whole with EBM contributions. The website presents the prediction, the per-patient explanation, the
global feature importance, and the test-set performance.

## 2. Dataset source and the version used

Five archives were supplied (file names as uploaded). Only one is used for training:

| File | Contents | Used? |
|---|---|---|
| `framingham_heart_disease.csv` | Framingham, 4,238 rows, 16 columns, original coding | **Yes (primary)** |
| `archive.zip` → `CHD_preprocessed.csv` | Framingham, 4,133 rows, 16 columns, preprocessed | Reference only (`data/raw/CHD_preprocessed_reference_not_used.csv`) |
| `heart_disease_120ec3.zip` (UCI Cleveland, Hungary, Switzerland, VA; byte-identical to the earlier `heart_disease.zip`) | Cleveland project data | No |
| `statlog_heart.zip` | Statlog Heart, 270 rows | No |
| `z_alizadeh_sani.zip` | Z-Alizadeh Sani, 303 rows, 56 columns, Excel | No |

**Why the original Framingham file.** The two Framingham versions differ by exactly 105 rows, which equals the
number of rows with a missing `education` value in the original. Comparing them column by column:
- 3,656 rows of `CHD_preprocessed.csv` match original complete rows exactly (apart from `education`).
- The other 477 rows match original rows that had missing values, which were filled with values from an
  undocumented method. The fill values follow no single rule (for example, `glucose` 82, `heartRate` 86, `BMI` 25.78).
- `education` was re-coded from four levels (1-4) to two levels (0/1). Codes 1-2 became 0 and codes 3-4 became 1.

So the two files cannot be concatenated, and the preprocessed file's imputation may have used statistics from the
full dataset, which would leak test information. The original file keeps explicit missing values, so the imputation
can be fitted on training data only. It is therefore the primary version.

## 3. Features

Target: `TenYearCHD`. Predictors (15):

| Feature | Meaning |
|---|---|
| `male` | Sex (1 = male, 0 = female) |
| `age` | Age at examination (years) |
| `education` | Education, ordinal code 1-4 (code meanings are not in the uploaded files) |
| `currentSmoker` | Current smoker (1 = yes, 0 = no) |
| `cigsPerDay` | Cigarettes smoked per day |
| `BPMeds` | On blood pressure medication (1 = yes, 0 = no) |
| `prevalentStroke` | Previous stroke (1 = yes, 0 = no) |
| `prevalentHyp` | Hypertension (1 = yes, 0 = no) |
| `diabetes` | Diabetes (1 = yes, 0 = no) |
| `totChol` | Total cholesterol (mg/dl) |
| `sysBP` | Systolic blood pressure (mm Hg) |
| `diaBP` | Diastolic blood pressure (mm Hg) |
| `BMI` | Body mass index (kg/m^2) |
| `heartRate` | Heart rate (beats per minute) |
| `glucose` | Blood glucose (mg/dl) |

Binary features are 0/1. `education` is an ordinal code (1-4); its meaning is not documented in the uploaded files.

## 4. Data inspection and cleaning

Run `python inspect_data.py` to produce `reports/inspection_report.txt`. Findings:
- Shape 4,238 × 16; no duplicate rows; no unexpected columns.
- Missing values: glucose 388 (9.16%), education
  105, BPMeds 53,
  totChol 50, cigsPerDay 29,
  BMI 19, heartRate 1.
  582 rows (13.73%) have at least one missing feature.
- Target: 0 = 3,594 (84.80%), 1 = 644 (15.20%). Class imbalance.
- Suspicious-value checks (out-of-range values, systolic ≤ diastolic, smoker/cigarette inconsistency): none found.

Run `python preprocess.py` to produce `data/processed/` and `reports/preprocessing_report.txt`. Steps:
1. Load the CSV and clean column names. 2. Remove unnamed index columns (none present).
3. Check that all expected columns exist. 4. Convert placeholder text (`?`, empty, `NA`, `-` and similar) to missing.
5. Convert feature columns to numbers (no non-numeric values were found). 6. Check that the target contains only 0 and 1.
7. Remove rows with a missing target (0 rows). 8. Remove exact duplicate rows (0 found).
9. Keep missing feature values for imputation. 10. Separate features and target.
11. Stratified 80/20 split with `random_state=42`: train 3390, test 848.
12. Save the cleaned data and the splits. Hashes of each file are stored in `reports/preprocessing_report.json`.

No SMOTE or oversampling is used. Imputation is not done before the split.

## 5. Model architecture

A scikit-learn `Pipeline`:
1. `SimpleImputer(strategy="median")`, fitted on training data only.
2. `ExplainableBoostingClassifier` (InterpretML 0.6.7) with `interactions=0` (main effects only, one feature per term),
   `random_state=42`, and every feature declared `continuous`.

The pipeline is saved as one object, `models/framingham_ebm.joblib`. The website uses the same object, so inputs are
processed exactly as in training.

## 6. Training procedure

1. Verify that the saved train/test files match the hashes from preprocessing (`train_model.py` stops if they changed).
2. 5-fold stratified cross-validation on the **training data only** (the test set is not used).
3. Fit the pipeline on the full training set.
4. Evaluate **once** on the untouched test set at the default threshold of 0.5. No threshold or hyperparameter was tuned on the test set.
5. Refit with the same seed and confirm identical predictions (reproducibility).
6. Save the model, `reports/metrics.json`, `reports/classification_report.txt`, and four figures.

## 7. Evaluation results (actual output of `train_model.py`)

Held-out test set: 848 patients (129 with CHD). Baseline threshold 0.5 (the website uses 0.155, see section 15).

| Metric | Test set |
|---|---|
| Accuracy | 0.853 |
| Precision | 0.667 |
| Recall / sensitivity | 0.062 |
| Specificity | 0.994 |
| F1-score | 0.113 |
| ROC-AUC | 0.692 |

Confusion matrix (test): TN = 715, FP = 4, FN = 121, TP = 8.

5-fold CV on training data (mean ± std):

| Metric | CV mean ± std |
|---|---|
| Accuracy | 0.852 ± 0.002 |
| Precision | 0.648 ± 0.096 |
| Recall | 0.056 ± 0.011 |
| Specificity | 0.994 ± 0.002 |
| F1 | 0.103 ± 0.019 |
| ROC-AUC | 0.730 ± 0.030 |

**Important reading of these results.**
- Accuracy (0.853) is high mainly because most patients have no CHD. Predicting "no CHD" for everyone
  would give 0.848 on this test set.
- At the 0.5 threshold the model finds only 8 of 129 CHD cases (recall 0.062).
  At the baseline threshold of 0.5 the model is very conservative: it rarely predicts CHD because most patients have low estimated risk.
- ROC-AUC (0.692) is the threshold-independent ranking measure, and is the more informative number here.
  Discrimination is modest, which is consistent with the difficulty of 10-year CHD prediction.
- A different threshold trades specificity for recall. Threshold adjustment was done in the v2 experiment (section 15):
  the website threshold of 0.155 was chosen by Youden's J on training-data out-of-fold predictions only, not on the test set.
  The resulting trade-off is shown in section 15.7.

## 8. Explainability methodology

- **Global importance:** `ebm.term_importances()`, the average absolute contribution per feature. Top features:
  age 0.419, prevalentHyp 0.159, cigsPerDay 0.141, male 0.135, sysBP 0.112, education 0.074.
- **Feature-effect (shape) curves:** `ebm.explain_global()`, with confidence bands.
- **Local explanation:** `ebm.explain_local()`. Each feature's contribution is in log-odds. Intercept + sum of
  contributions = model log-odds, checked by the tests (additivity).

**Model contribution is not medical causation.** A large contribution means the model relied on that feature to
produce its score for this data. It does not show that the feature causes CHD, and it cannot tell whether the feature
is a cause, a marker, or a confounder. For example, sex has a large learned effect in this data, which partly reflects
differences in the population and in other risk factors that the model also includes.

**Sparse extreme values.** Some curves are based on very few patients at the extremes (for example total cholesterol
below about 146 mg/dl: 18 training patients). Their bands are wide, and they should not be interpreted.

**Education** has a non-monotonic and small effect. Read it as a model artefact, not a trend.

## 9. Installation and running (Windows PowerShell)

Use Python 3.11 or 3.12. Open PowerShell in the project folder (`Framingham-CHD`).

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Run the pipeline in this order:

```powershell
python inspect_data.py        # inspection report, descriptive statistics, dataset comparison
python preprocess.py          # cleaning, stratified 80/20 split, saved processed files
python train_model.py         # CV, training, test evaluation, saves model and reports
python -m pytest -q           # automated tests (expected: 56 passed)
streamlit run app.py          # website at http://localhost:8501
```

Retraining later: run `preprocess.py` and then `train_model.py`. Both are deterministic for the same data and library versions.

To confirm the input file is the one used, run:

```powershell
Get-FileHash data\raw\framingham_heart_disease.csv -Algorithm SHA256
```

The expected SHA-256 is `40921748dbdf6587045b9822414199a1ae3349358b63b975ab2208aa3cccc45b`.

## 10. Project structure

```
Framingham-CHD/
├── data/
│   ├── raw/                 # original files (read-only copies)
│   └── processed/           # cleaned data and train/test splits (created by preprocess.py)
├── models/                  # framingham_ebm.joblib (created by train_model.py)
├── reports/                 # inspection, preprocessing, metrics, figures
├── src/
│   ├── framingham.py        # loading, validation, cleaning, split, input ranges
│   ├── model.py             # imputer + EBM pipeline
│   ├── evaluation.py        # metrics
│   ├── explain.py           # EBM explanations
│   └── plots.py             # figures
├── tests/                   # dataset and model tests
├── app.py                   # Streamlit website (6 sidebar pages; Risk Prediction appears below Patient Input)
├── inspect_data.py
├── preprocess.py
├── train_model.py
├── requirements.txt
└── README.md
```

## 11. Testing

`python -m pytest -q` runs 56 tests (the original 35 plus 21 for the patient-state and website-flow checks), covering: data loading, required columns, unexpected-column handling,
placeholder conversion, non-numeric coercion, target validation, missing-target removal, missing-feature retention,
duplicate removal and reporting, stratified 80/20 split reproducibility, pipeline configuration, binary predictions,
probabilities between 0 and 1, prediction with missing inputs, save/load identity, explanation additivity, and
training reproducibility.

## 12. Framingham target vs the Cleveland target

| | Framingham (this project) | Cleveland (separate project) |
|---|---|---|
| Question | Will a CHD event occur within 10 years? | Is there angiographic narrowing > 50% now? |
| Type | Future-event risk (prognosis) | Current disease status (diagnosis) |
| Population | Community cohort, many patients without disease | Patients referred for angiography |
| Positive class | 15.2% | 45.9% |

These are different questions and must not be compared as if they measured the same thing.

## 13. Limitations and ethical considerations

- **Academic prototype.** Not clinically validated, not a diagnosis, not approved for use in care.
- **Modest discrimination and low recall at 0.5.** Most positive cases are missed at the default threshold.
- **Single split, no external validation.** Test results come from 848 patients; a different split would give different numbers.
- **Historical and population limits.** The cohort may not represent current or other populations.
- **Missing data.** Glucose is missing for about 9% of rows; imputation with a median assumes values are missing at random.
- **Sparse extremes and unstable shape details.** Some curve regions rest on few patients.
- **Limited calibration evidence.** Calibration was checked on training out-of-fold predictions and on one test set (section 15.8). It has not been checked on external data.
- **Explanations describe the model, not causes.**
- **Ethical risk.** A risk number can cause anxiety or unfair treatment if misread. Keep the disclaimer visible and
  never present a probability as a personal prediction.

---

## 14. Gender (male) input verification

The dataset column `male` is coded **1 = Male, 0 = Female**. The website widget is labelled "Gender (Sex)" with
options Male (sent as 1) and Female (sent as 0). The feature name `male` is kept in the explanation charts and
tables, and the table and chart now also show the patient's input, for example `male = 0 (Female)`.

Verified by tests (`tests/test_gender.py`) and by driving the website with identical values for every other field:

| Case | Widget | DataFrame `male` | Imputed `male` | Predicted probability | Predicted class |
|---|---|---|---|---|---|
| Test A | Female | 0.0 | 0.0 | 0.0793 | 0 |
| Test B | Male | 1.0 | 1.0 | 0.1019 | 0 |

The probabilities differ because the model learned a different gender effect, not because the input was forced.
The verification debug panel used during testing has been removed from the final version.


---

## 15. Model improvement experiment (v2): threshold and model comparison without leakage

### 15.1 Preserved baseline
The original model (`models/framingham_ebm.joblib`, threshold 0.5) and all original reports are unchanged.
A full frozen copy of the project before this experiment is kept in `Framingham-CHD-baseline/`.
The file hashes stayed the same (`models/framingham_ebm.joblib` starts `d40c42a2…`).

### 15.2 Design (fixed before any test result was viewed)
- **Development data:** `data/processed/train.csv` only (3,390 rows, 515 CHD). The test file is read only in stage B, once.
- **Out-of-fold (OOF) predictions:** stratified 5-fold CV, seed 42, same folds for every candidate. Imputation is fitted inside each training fold.
- **Candidates:** EBM (same configuration as baseline); Logistic Regression (median imputer, standard scaler, `class_weight="balanced"`); HistGradientBoosting (median imputer, `class_weight="balanced"`).
- **Model-selection rule:** highest OOF ROC-AUC, ties broken by lower OOF log-loss.
- **Threshold rule:** maximise Youden's J = sensitivity + specificity − 1 on the selected model's OOF predictions; ties go to the highest threshold; grid 0.005 to 0.995 in steps of 0.005.
- **Calibration:** OOF Brier, log-loss, ECE (10 bins), calibration-in-the-large, and calibration plots. These are reported, not used to pick the threshold.
- **Test evaluation:** each model refit on the full training set; the test set is read once. Bootstrap 95% CIs (1,000 resamples).

### 15.3 Stage A: out-of-fold comparison on training data (actual log)
Candidates were compared on the same 5-fold OOF predictions. Each row shows the OOF values, plus the threshold and metrics at that model's own Youden threshold:

| Candidate | OOF ROC-AUC | OOF PR-AUC | OOF Brier | OOF ECE | Youden threshold | Recall | Specificity |
|---|---|---|---|---|---|---|---|
| EBM | 0.7285 | 0.3432 | 0.1162 | 0.0113 | 0.155 | 0.678 | 0.686 |
| LogisticRegression_balanced | 0.7297 | 0.3484 | 0.2076 | 0.2855 | 0.455 | 0.740 | 0.609 |
| HistGradientBoosting_balanced | 0.6700 | 0.2952 | 0.1525 | 0.1175 | 0.195 | 0.678 | 0.574 |

**Pre-registered selection result:** `LogisticRegression_balanced` (highest OOF ROC-AUC). Its margin over EBM was
+0.0012 OOF ROC-AUC.
A paired bootstrap on the training OOF predictions gave a 95% CI of
[-0.0066, +0.0094]
for LR minus EBM. The interval includes zero, so this margin is not reliable evidence that one model is better.

### 15.4 Stage B: single test evaluation (reporting after decisions were fixed)
All candidates on the test set, each at its own OOF threshold:

| Candidate | Test ROC-AUC | Test PR-AUC | Brier | ECE | Threshold | Recall | Specificity | Precision |
|---|---|---|---|---|---|---|---|---|
| EBM | 0.6916 | 0.2930 | 0.1215 | 0.0305 | 0.155 | 0.550 | 0.679 | 0.235 |
| LogisticRegression_balanced | 0.6985 | 0.2954 | 0.2114 | 0.2813 | 0.455 | 0.643 | 0.613 | 0.230 |
| HistGradientBoosting_balanced | 0.6214 | 0.2394 | 0.1694 | 0.1485 | 0.195 | 0.690 | 0.492 | 0.196 |

### 15.5 Website operating point (chosen by the override described below)
The website uses **EBM at threshold 0.155** (the training-only OOF Youden threshold for EBM).

| Test set (n = 848, 129 CHD) | Value | 95% CI |
|---|---|---|
| Recall | 0.550 | 0.462 to 0.638 |
| Specificity | 0.679 | 0.644 to 0.713 |
| Precision | 0.235 | not computed |
| F1 | 0.329 | not computed |
| ROC-AUC | 0.692 | 0.647 to 0.738 |
| PR-AUC | 0.293 | 0.239 to 0.371 |
| Confusion matrix | TN 488, FP 231, FN 58, TP 71 | |
| ECE (10 bins) | 0.0305 | |
| Mean predicted vs observed | 0.148 vs 0.152 | |

Baseline for comparison (same test set, threshold 0.5): recall 0.062, specificity 0.994, precision 0.667, F1 0.113, ROC-AUC 0.692.

### 15.6 Important: override of the pre-registered result
The pre-registered rule selected **Logistic Regression (balanced) at threshold 0.455**. On the test set
it gave recall 0.643, specificity 0.613, ROC-AUC 0.699.
Its probabilities are poorly calibrated (ECE 0.281; mean predicted
0.43 against an observed 0.15).

The website does **not** use it, for three reasons:
1. The project brief requires EBM as the primary algorithm.
2. The OOF ROC-AUC difference is inside the bootstrap noise (above).
3. EBM is much better calibrated.

This is an **exploratory project-level decision made after the test results were examined**. The EBM test results are therefore **not an independent confirmatory estimate**. A new, untouched test set or external validation is required for confirmatory evaluation. The project owner must confirm the decision before submission. The
pre-registered Logistic Regression artefact is saved as `models/framingham_selected.joblib`, but no website page loads it.

### 15.7 Recall-specificity trade-off
Lowering the threshold catches more CHD cases and produces more false alarms. Raising it does the opposite.
At threshold 0.5 the baseline catches only 8 of 129 CHD cases (specificity 0.994). At threshold
0.155 the website model catches 71 of 129 (recall 0.550), but
231 of 719 patients without CHD are flagged (specificity 0.679).
Precision is low (0.235), so most flagged patients will not have an event within 10 years. The
threshold is a policy choice about which error is more costly, and the right choice depends on the use.

### 15.8 Calibration
EBM is well calibrated on both data sets: test ECE 0.030, mean predicted 0.148 against
observed 0.152. The calibration plots are in `reports/experiment/calibration_oof.png` and `calibration_test.png`.

### 15.9 Scope and limitations of this experiment
- **No guarantee of better performance.** The ROC-AUC on test (0.692) is essentially the same as the baseline (0.692).
  The gain is in the operating point, not in ranking ability.
- **One split, one set of OOF folds.** The 95% intervals reflect test-sample noise, not the full range of possible splits.
- **Threshold chosen by Youden's J** treats recall and specificity equally. A clinical use would need a different weighting, agreed with clinicians.
- **Not clinically validated.** This is an academic research prototype.

### 15.10 How to reproduce (Windows PowerShell)
```powershell
python experiment_select.py     # stage A: training data only, writes OOF results and the selection decision
python experiment_test.py       # stage B: refit on training data, read the test set once
python experiment_report.py     # stage C: operating point and bootstrap intervals from the frozen predictions
python -m pytest -q             # tests (includes leakage and decision-integrity checks)
streamlit run app.py
```
Stage B refuses to run if `train.csv` has changed since the decision was written.

---

## 16. Website patient-state architecture (v3)

### 16.1 User flow
1. Open **Patient Input** and fill in the details.
2. Click **Save patient**.
3. The app builds one authoritative saved state, runs the existing EBM prediction (threshold 0.155), and shows the
   prediction directly **below the form** on the same page. There is no separate Risk Prediction sidebar page.
4. Open **Explain This Prediction** to see the explanation for the latest saved patient.

### 16.2 Single authoritative saved state
Streamlit `session_state` holds one object, `saved_patient_state`, plus a counter `patient_version`. The state
(`src/patient_state.py`) contains:
- `version`: incremented on every successful save
- `inputs`: the raw input dictionary
- `df`: the one-row DataFrame sent to the model (training column order, float)
- `gender`: decoded from that DataFrame (`male` 0 = Female, 1 = Male)
- `prediction`: produced by the existing `explain_patient()` from the same `df`, tagged with the same version and a
  fingerprint (SHA-256 of the DataFrame values and column order)

Rules implemented:
- A save builds the whole state in one call. If it fails, the previous state is deleted and an error is shown, so no
  stale result is displayed.
- Changing widgets without clicking **Save patient** does not alter the saved prediction or explanation.
- Before any save, the prediction area shows a message asking for patient details.
- `is_consistent()` rejects any state whose DataFrame, gender or version no longer matches its prediction.
- The Explain page reads only this state, so prediction and explanation always come from the same DataFrame.

### 16.3 Root cause of the stale gender explanation (fixed)
The old Explain page took the **contributions** from a session result that was written only when the Risk page was
visited, but took the **patient labels** from the most recent save. After a Female patient's risk page was viewed and a
Male patient was saved, the page showed the label "Male (male = 1)" with the Female contribution (−0.1190). The
reverse also occurred. Both were reproduced with the old app and are fixed in the new one (Male +0.1564, Female −0.1190).

### 16.4 Tests (executed)
`python -m pytest -q` ran **56 tests, all passed**:
- `tests/test_patient_state.py` (8): Male and Female saved dictionary and DataFrame values, replacement by a new save,
  snapshot behaviour, shared DataFrame fingerprint, tamper and stale-version rejection, invalid input rejection.
- `tests/test_ui_patient_flow.py` (13): sidebar contents, no prediction before the first save, Male and Female UI flows,
  Female then Male with no stale explanation, unsaved changes not affecting the saved result, failed save clearing the
  previous result, and every page loading without an exception.
- The original 35 tests (`test_dataset.py` 10, `test_model.py` 7, `test_gender.py` 8, `test_experiment.py` 10) are unchanged.

### 16.5 Protected artefacts
The following SHA-256 hashes were recorded before and after this version and were identical:
- `models/framingham_ebm.joblib` = `d40c42a2…`
- `models/framingham_selected.joblib` = `c09130a9…`
- `data/raw/framingham_heart_disease.csv` = `40921748…`
- `data/processed/train.csv` = `a6f7dc50…`
- `data/processed/test.csv` = `9c8bec5d…`

No model was retrained, and no threshold, split, preprocessing step or experiment result was changed.
