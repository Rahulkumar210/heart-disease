"""Streamlit website: Framingham 10-year CHD risk prediction with EBM explanations.

Run:  streamlit run app.py
ACADEMIC RESEARCH PROTOTYPE. Not medical advice. Not clinically validated.
"""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.explain import contribution_table, explain_patient, feature_effects, global_importance
from src.framingham import (
    BINARY_FEATURES, FEATURE_DESCRIPTIONS, FEATURES, MODEL_PATH, REPORTS_DIR, TARGET, VALUE_LABELS,
)
from src.plots import contribution_figure, effects_figure, importance_figure

DISCLAIMER = (
    "**Academic research prototype only.** This model estimates a statistical pattern in a "
    "research cohort. It is not medical advice, not a diagnosis, and not clinically validated. "
    "Do not use it to make decisions about any person's care. Consult a qualified clinician."
)
TARGET_NOTE = (
    "Target here: CHD event within 10 years (TenYearCHD). This is different from the Cleveland project, "
    "whose target was angiographic narrowing > 50% at one visit."
)


@st.cache_resource(show_spinner="Loading trained EBM model...")
def load_model():
    if not MODEL_PATH.exists():
        return None
    b = joblib.load(MODEL_PATH)
    if b.get("algorithm") != "ExplainableBoostingClassifier" or list(b.get("features", [])) != FEATURES:
        raise ValueError("Model file does not match the Framingham EBM project.")
    return b["pipeline"]


@st.cache_data
def website_threshold():
    p = REPORTS_DIR / "app_threshold.json"
    return float(json.loads(p.read_text(encoding="utf-8"))["threshold"]) if p.exists() else 0.5


@st.cache_data
def load_json(name):
    p = REPORTS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def global_explanations():
    pipe = load_model()
    return global_importance(pipe), feature_effects(pipe)


def need_model():
    if load_model() is None:
        st.error("Model not found. Run `python preprocess.py` then `python train_model.py`.")
        return False
    return True


def page_home():
    st.title("Framingham 10-Year CHD Risk: Explainable AI Prototype")
    st.warning(DISCLAIMER)
    st.markdown(
        f"""
This prototype uses an **Explainable Boosting Machine (EBM)** trained on the Framingham Heart Study
data to estimate the chance of a coronary heart disease (CHD) event within 10 years.

- {TARGET_NOTE}
- The model explains each prediction with per-feature contributions (log-odds), and shows the overall
  pattern learned across the dataset.
- Go to **Patient Input**, fill in the details and click **Save patient**. The risk prediction appears directly below the form. Then open **Explain This Prediction** for the explanation of the latest saved patient.
        """
    )


SAVED_KEY = "saved_patient_state"
VERSION_KEY = "patient_version"


def get_saved_state():
    """The one authoritative saved patient. Returns None if none saved or if it is inconsistent."""
    from src.patient_state import is_consistent
    state = st.session_state.get(SAVED_KEY)
    if state is None:
        return None
    if not is_consistent(state):
        st.session_state.pop(SAVED_KEY, None)
        return None
    return state


def page_input():
    st.title("Patient Input")
    st.caption("Enter the 15 Framingham measurements. Ranges come from the cleaned training data.")
    st.warning(DISCLAIMER)
    ranges = load_json("input_ranges.json")
    if ranges is None:
        st.error("input_ranges.json not found. Run preprocess.py first.")
        return
    med = {k: v["median"] for k, v in ranges.items()}

    def num(col, label, integer=True, step=None):
        r = ranges[col]
        lo, hi, default = r["min"], r["max"], med[col]
        if integer:
            return st.number_input(label, min_value=int(lo), max_value=int(hi), value=int(round(default)), step=1, help=FEATURE_DESCRIPTIONS[col])
        return st.number_input(label, min_value=float(lo), max_value=float(hi), value=float(default), step=step or 0.1, help=FEATURE_DESCRIPTIONS[col])

    def yn(col, label):
        return st.selectbox(label, [0, 1], format_func=lambda v: "Yes" if v else "No", help=FEATURE_DESCRIPTIONS[col])

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            sex = st.selectbox("Gender (Sex)", [1, 0], format_func=lambda v: VALUE_LABELS["male"][v],
                               help="Dataset column 'male': 1 = Male, 0 = Female")
            age = num("age", "Age (years)")
            education = st.selectbox("Education (ordinal code 1-4)", [1, 2, 3, 4], index=1,
                                     help=FEATURE_DESCRIPTIONS["education"])
            currentSmoker = yn("currentSmoker", "Current smoker")
            cigsPerDay = num("cigsPerDay", "Cigarettes per day")
        with c2:
            BPMeds = yn("BPMeds", "On blood pressure medication")
            prevalentStroke = yn("prevalentStroke", "Previous stroke")
            prevalentHyp = yn("prevalentHyp", "Hypertension")
            diabetes = yn("diabetes", "Diabetes")
            totChol = num("totChol", "Total cholesterol (mg/dl)")
        with c3:
            sysBP = num("sysBP", "Systolic BP (mm Hg)", integer=False, step=0.5)
            diaBP = num("diaBP", "Diastolic BP (mm Hg)", integer=False, step=0.5)
            BMI = num("BMI", "BMI (kg/m^2)", integer=False, step=0.1)
            heartRate = num("heartRate", "Heart rate (bpm)")
            glucose = num("glucose", "Glucose (mg/dl)")
        submitted = st.form_submit_button("Save patient", type="primary")

    if submitted:
        from src.patient_state import PatientStateError, make_saved_state
        inputs = {
            "male": sex, "age": age, "education": education, "currentSmoker": currentSmoker,
            "cigsPerDay": cigsPerDay, "BPMeds": BPMeds, "prevalentStroke": prevalentStroke,
            "prevalentHyp": prevalentHyp, "diabetes": diabetes, "totChol": totChol, "sysBP": sysBP,
            "diaBP": diaBP, "BMI": BMI, "heartRate": heartRate, "glucose": glucose,
        }
        next_version = int(st.session_state.get(VERSION_KEY, 0)) + 1
        try:
            pipe = load_model()
            if pipe is None:
                raise PatientStateError("model file not found")
            state = make_saved_state(inputs, pipe, website_threshold(), next_version)
        except Exception as exc:  # never keep a previous patient's result on failure
            st.session_state.pop(SAVED_KEY, None)
            st.error(f"Could not save patient: {exc}. Previous results have been cleared.")
        else:
            st.session_state[SAVED_KEY] = state
            st.session_state[VERSION_KEY] = next_version
            st.success(f"Patient saved (version {next_version}). Risk prediction is shown below.")

    st.divider()
    render_risk_section(get_saved_state())


def render_risk_section(state):
    """Prediction section, shown directly below the form. Reads only the saved state."""
    st.subheader("Risk Prediction")
    st.warning(DISCLAIMER)
    if state is None:
        st.info("No patient saved yet. Enter the patient details above and click **Save patient**.")
        return
    r = state["prediction"]
    thr = r["threshold"]
    c0, c1, c2 = st.columns(3)
    c0.metric("Patient gender (input)", state["gender"])
    c1.metric(f"Predicted CHD class (threshold {thr:.3f})", "CHD (1)" if r["predicted_class"] else "No CHD (0)")
    c2.metric("Predicted CHD probability (10-year)", f"{r['probability']:.1%}")
    base = load_json("metrics.json")["data"]["split"]["train_class_counts"]
    base_rate = base["1"] / (base["0"] + base["1"])
    st.caption(f"Gender is shown only as the patient input. It is not the prediction. "
               f"Training data: {base_rate:.1%} of patients had a CHD event. The probability is a statistical "
               "estimate for this input only, not a personal risk score.")
    st.info(
        f"The class label uses threshold {thr:.3f}, chosen on training-data out-of-fold predictions to balance "
        "recall and specificity (Youden's J). At this threshold the model flags more patients than the default 0.5, "
        "so it catches more CHD cases but also produces more false alarms. See **Model Performance** for the trade-off. "
        "Use the probability and the explanation together, never the class label alone."
    )


def page_explain():
    st.title("Explain This Prediction")
    state = get_saved_state()
    if state is None:
        st.info("No patient saved yet. Enter patient details on the **Patient Input** page and click **Save patient**.")
        return
    r = state["prediction"]
    row = state["df"].iloc[0]
    table = contribution_table(r, row)
    st.info(f"Showing the latest saved patient (version {state['version']}). "
            f"Gender input: **{state['gender']}** (male = {row['male']:g}). The 'male' rows below are this "
            "patient's gender input, not a prediction.")
    st.markdown(
        "Each row shows how one measurement moved this model's score for this patient. Red bars push toward "
        "the CHD label; blue bars push toward no CHD. Values are log-odds contributions from the EBM. "
        "**They describe the model, not medical cause.**"
    )
    top = table.iloc[0]
    st.success(f"Largest model influence: **{top['Chart label']}**, "
               f"contribution {top['Contribution (log-odds)']:+.3f}.")
    fig = contribution_figure(table); st.pyplot(fig); plt.close(fig)
    st.dataframe(table.drop(columns=["Meaning", "Chart label"]), use_container_width=True, hide_index=True)
    with st.expander("Consistency check"):
        total = r["intercept"] + sum(r["contributions"].values())
        st.write(f"Intercept {r['intercept']:.4f} + sum of contributions = {total:.4f}; "
                 f"model log-odds = {r['log_odds']:.4f}. These match.")


def page_global():
    st.title("Global Feature Importance")
    st.warning(DISCLAIMER)
    if not need_model():
        return
    imp, effects = global_explanations()
    st.markdown(
        "These show what the trained EBM learned across the training data. Importance is the average absolute "
        "contribution of each feature. Feature-effect curves show how the score changes with each value, "
        "with confidence bands. **Importance is not causation.**"
    )
    fig = importance_figure(imp); st.pyplot(fig); plt.close(fig)
    st.dataframe(pd.DataFrame({"feature": imp.index, "importance": imp.values.round(4)}), hide_index=True)
    fig = effects_figure(effects); st.pyplot(fig); plt.close(fig)
    st.warning(
        "Curves at the extreme ends (for example very low total cholesterol, below about 146 mg/dl) are "
        "based on very few patients and have wide bands. Do not interpret them as findings."
    )


def page_performance():
    st.title("Model Performance")
    st.warning(DISCLAIMER)
    op = load_json("experiment/operating_point.json")
    sel = load_json("experiment/selection_decision.json")
    base = load_json("metrics.json")
    if op is None or base is None:
        st.info("Run `python experiment_select.py`, `python experiment_test.py`, `python experiment_report.py`.")
        return
    t = op["test_metrics_at_website_threshold"]; ci = op["test_bootstrap_95ci"]; rc = op["test_ranking_and_calibration"]
    b = base["test"]
    st.caption(f"Held-out test set: {op['test_rows']} patients ({op['test_positives']} with CHD). Used once, after all "
               "The threshold was chosen on training out-of-fold predictions. Bootstrap 95% intervals resample test patients.")
    st.warning(
        "**Exploratory override.** The website model is EBM. The pre-registered selection rule chose Logistic Regression "
        "(see the selection section below). Keeping EBM was an exploratory project-level decision made AFTER the test "
        "results were examined. These test results are therefore not an independent confirmatory estimate. "
        "A new untouched test set or external validation is needed for confirmation."
    )

    st.subheader(f"Website operating point: EBM, threshold {op['website_threshold']:.3f}")
    c = st.columns(4)
    c[0].metric("Recall", f"{t['recall']:.3f}", help=f"95% CI {ci['recall'][0]:.3f}-{ci['recall'][1]:.3f}")
    c[1].metric("Specificity", f"{t['specificity']:.3f}", help=f"95% CI {ci['specificity'][0]:.3f}-{ci['specificity'][1]:.3f}")
    c[2].metric("Precision", f"{t['precision']:.3f}")
    c[3].metric("F1-score", f"{t['f1']:.3f}")
    c = st.columns(3)
    c[0].metric("ROC-AUC", f"{rc['roc_auc']:.3f}", help=f"95% CI {ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f}")
    c[1].metric("PR-AUC", f"{rc['pr_auc']:.3f}", help=f"95% CI {ci['pr_auc'][0]:.3f}-{ci['pr_auc'][1]:.3f}")
    c[2].metric("Calibration error (ECE)", f"{rc['ece_10bins']:.3f}")
    cm = {k: t[k] for k in ["tn", "fp", "fn", "tp"]}
    st.table(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                          index=["Actual no CHD (0)", "Actual CHD (1)"], columns=["Predicted 0", "Predicted 1"]))

    st.subheader("Comparison with the earlier threshold-0.5 model (same test set)")
    rows = [
        ["Previous EBM, threshold 0.500", b["recall_sensitivity"], b["specificity"], b["precision"], b["f1"], b["roc_auc"]],
        [f"EBM, threshold {op['website_threshold']:.3f} (website)", t["recall"], t["specificity"], t["precision"], t["f1"], rc["roc_auc"]],
    ]
    st.table(pd.DataFrame(rows, columns=["Model / threshold", "Recall", "Specificity", "Precision", "F1", "ROC-AUC"]).round(3))
    st.info(
        "The recall increase comes mainly from the lower threshold, not from a different model. The EBM ROC-AUC is "
        f"{b['roc_auc']:.3f} before and {rc['roc_auc']:.3f} after: the ranking is unchanged, only the operating point moved."
    )

    st.subheader("Trade-off: recall versus specificity (training out-of-fold)")
    oofc = op["oof_at_website_threshold"]
    st.write(f"At the website threshold, out-of-fold training recall is {oofc['recall']:.3f} and specificity {oofc['specificity']:.3f}. "
             "Lowering the threshold catches more cases and raises false alarms; raising it does the opposite.")
    st.image(str(REPORTS_DIR / "experiment" / "tradeoff_oof.png"), width=620)

    st.subheader("Calibration")
    st.write(f"Test set: mean predicted risk {rc['mean_predicted']:.3f} vs observed rate {rc['observed_rate']:.3f}; "
             f"Brier {rc['brier']:.3f}; ECE {rc['ece_10bins']:.3f}. Good agreement here, but it is one test set.")
    st.image(str(REPORTS_DIR / "experiment" / "calibration_test.png"), width=620)

    st.subheader("Model selection (pre-registered, training data only)")
    cmp_rows = []
    oofc_all = json.loads((REPORTS_DIR / "experiment" / "oof_comparison.json").read_text(encoding="utf-8"))
    for name, v in oofc_all.items():
        r_ = v["oof_ranking_and_calibration"]
        cmp_rows.append([name, r_["roc_auc"], r_["pr_auc"], r_["brier"], r_["ece_10bins"], v["youden_threshold_on_oof"]])
    st.table(pd.DataFrame(cmp_rows, columns=["Candidate", "OOF ROC-AUC", "OOF PR-AUC", "OOF Brier", "OOF ECE", "OOF Youden threshold"]).round(4))
    st.warning(op["override_note"])
    lr = op["pre_registered_LR_at_its_threshold_test"]
    st.write(f"For transparency: the pre-registered Logistic Regression at its threshold {sel['selected_threshold']:.3f} gives "
             f"test recall {lr['recall']:.3f} and specificity {lr['specificity']:.3f}. Its probabilities are poorly calibrated "
             "(mean predicted about 0.43 against an observed 0.15), so the website does not use it.")


def page_about():
    st.title("About the Project")
    st.warning(DISCLAIMER)
    st.markdown(
        """
**Dataset.** Framingham Heart Study (`framingham_heart_disease.csv`, 4,238 rows, 16 columns). We chose this
original file over `CHD_preprocessed.csv` (4,133 rows) because the original keeps its own missing values and
education coding. The preprocessed file dropped 105 rows, re-coded `education` to two levels, and filled
missing values with an undocumented method.

**Target.** `TenYearCHD`: 1 = CHD event within 10 years, 0 = no recorded event. This is a *prediction of a
future event*, not a diagnosis of existing disease.

**Method.** Median imputation fitted on training data only; stratified 80/20 split (random_state=42); EBM with
main effects only; 5-fold CV on training data; single evaluation on the test set.

**Limitations.** Historical cohort (not contemporary); single-study data; class imbalance (15% positive);
low recall at the default threshold; the sparse extreme ranges of some features; no external validation.
Calibration was evaluated on training out-of-fold predictions and on one held-out test set, but has not been
externally validated. Explanations describe model behaviour and do not show causation.

**Ethics.** Risk estimates can cause anxiety or unfair treatment if misread. Race and other sensitive
attributes are not in this dataset, but the data may not represent every population.
        """
    )


PAGES = {
    "Home": page_home,
    "Patient Input": page_input,
    "Explain This Prediction": page_explain,
    "Global Feature Importance": page_global,
    "Model Performance": page_performance,
    "About the Project": page_about,
}

st.set_page_config(page_title="Framingham CHD Prototype (EBM)", layout="wide")
choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
PAGES[choice]()
