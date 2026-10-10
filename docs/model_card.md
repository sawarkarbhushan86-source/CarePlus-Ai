# CARE-PULSE AI — Model Card: Calibrated XGBoost Readmission Predictor

## 1. Model Details
- **Model Name:** CARE-PULSE Calibrated XGBoost Readmission Classifier
- **Model Version:** 1.0.0
- **Model Type:** Extreme Gradient Boosted Decision Tree Ensemble with Sigmoid Platt Scaling (`CalibratedClassifierCV`)
- **Developer:** CARE-PULSE AI Project Team
- **Serialization:** Joblib pipeline containing Scikit-learn `ColumnTransformer` and calibrated estimator
- **Release Date:** October 2026

---

## 2. Intended Use
- **Primary Use:** Decision-support platform for hospital quality improvement teams, care navigators, and data science researchers to identify inpatient diabetic encounters at high risk of 30-day readmission.
- **Intended User:** Inpatient discharge planners, clinical case managers, healthcare analytics researchers, and technical recruiters.
- **Out-of-Scope Uses:**
  - Autonomous clinical decision-making or discharge denial.
  - Making individual pharmaceutical diagnoses or unverified dosage adjustments.
  - Non-inpatient diabetic settings (e.g. general outpatient checkups).

---

## 3. Quantitative Performance Summary

Evaluation conducted on an unseen, **patient-grouped test cohort** ($N = 3,955$ encounters, positive base rate = 11.25%):

| Performance Dimension | Raw XGBoost | Calibrated XGBoost (Champion) |
|---|---|---|
| **ROC-AUC (Discrimination)** | 0.6565 | **0.6658** |
| **PR-AUC (Average Precision)** | 0.1989 | **0.2097** |
| **Brier Score (Probabilistic Error)** | 0.1120 | **0.0962** |
| **Expected Calibration Error (ECE)** | 0.1116 | **0.0156** (-86% Error) |
| **Sensitivity / Recall (@ threshold=0.15)** | 0.8742 | 0.3281 |
| **Specificity (@ threshold=0.15)** | 0.4352 | 0.8581 |
| **Positive Predictive Value (PPV)** | 0.1578 | **0.2267** |

---

## 4. Demographic Fairness & Disparity Audit

Evaluated at operational decision threshold = 0.15:

| Demographic Stratum | Cohort Category | Sample Size | ROC-AUC | Recall (Sensitivity) | False Positive Rate |
|---|---|---|---|---|---|
| **Gender** | Female | 2,112 | 0.6675 | 32.5% | 14.4% |
| **Gender** | Male | 1,843 | 0.6629 | 33.2% | 14.0% |
| **Race** | Caucasian | 2,950 | 0.6581 | 30.3% | 14.4% |
| **Race** | African American | 760 | 0.6920 | 44.1% | 15.7% |
| **Race** | Hispanic | 75 | 0.6954 | 28.6% | 8.8% |
| **Race** | Other | 50 | 0.6800 | 60.0% | 4.4% |

**Fairness Analysis:** False Positive Rates remain closely bounded (14.0% – 15.7%) across major demographic categories. African American encounters show slightly higher sensitivity (44.1%) due to higher baseline acute utilization in the historical 1999–2008 cohort.

---

## 5. Ethical Considerations & Safety Guardrails
1. **Historical Cohort Caveat:** Data originates from 1999–2008 US hospital encounters; contemporary prescribing patterns (e.g. SGLT2 inhibitors and GLP-1 receptor agonists) are not represented.
2. **Missing Glycemic Warning:** Encounters without an HbA1c lab result are explicitly highlighted so clinicians can order a baseline test prior to discharge.
3. **Alert Fatigue Mitigation:** Operating at threshold 0.15 provides 85.8% specificity, ensuring nurses are not overwhelmed by false positive flags.
