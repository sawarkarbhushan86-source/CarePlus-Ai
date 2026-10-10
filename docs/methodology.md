# CARE-PULSE AI — Scientific & Machine Learning Methodology

## 1. Clinical Problem Formulation

Unplanned 30-day hospital readmission is a benchmark quality indicator in healthcare. Under the CMS Hospital Readmissions Reduction Program (HRRP), hospitals face financial reimbursement penalties for excessive readmission ratios. Patients with diabetes represent a high-risk chronic inpatient cohort characterized by multimorbidity, frequent medication adjustments, and complex care transitions.

**Prediction Task:** Formulate as a supervised binary classification problem:  
Predict the conditional probability that a hospitalized patient will experience an unplanned readmission within 30 days of discharge:

$$P(\text{Readmission} < 30\,\text{days} \mid \mathbf{X}_{\text{encounter}})$$

---

## 2. Cohort Definition & Clinical Exclusion Criteria

Using the authentic **UCI Diabetes 130-US Hospitals (1999–2008)** dataset (101,766 encounters across 71,518 unique patients):

1. **Target Binarization:**
   - Positive class ($Y = 1$): `readmitted == '<30'` (11,357 encounters, ~11.16% baseline prevalence).
   - Negative class ($Y = 0$): `readmitted == '>30'` or `readmitted == 'NO'`.
2. **Clinical Exclusions:**
   - Encounters where the patient expired during admission or was discharged directly to hospice (`discharge_disposition_id` $\in \{11, 19, 20, 21\}$) are excluded, as readmission within 30 days is clinically impossible or censored.
   - Records with invalid biological sex (`Unknown/Invalid`, $N=3$) are excluded.
3. **Data Leakage Mitigation:**
   - Inpatient encounters from the same individual patient (`patient_nbr`) are clustered. A `GroupShuffleSplit` (80% train / 20% test) is enforced, guaranteeing that no patient appearing in the training set has any prior or future encounters in the evaluation set.

---

## 3. Clinical Feature Engineering

### 3.1 ICD-9 Diagnostic Clustering
Raw ICD-9 diagnostic codes (`diag_1`, `diag_2`, `diag_3`) are mapped to 9 high-level clinical categories based on Strack et al. (2014):
- **Circulatory:** 390–459, 785
- **Respiratory:** 460–519, 786
- **Digestive:** 520–579, 787
- **Diabetes:** 250.xx
- **Genitourinary:** 580–629, 788
- **Neoplasms:** 140–239
- **Musculoskeletal:** 710–739
- **Injury:** 800–999
- **Other / Supplemental:** Remaining codes and V/E codes.

### 3.2 Healthcare Utilization Scoring
- `total_prior_visits` = `number_outpatient` + `number_emergency` + `number_inpatient`
- `has_prior_inpatient` = $\mathbb{I}(\text{number\_inpatient} \ge 1)$
- `has_prior_emergency` = $\mathbb{I}(\text{number\_emergency} \ge 1)$

### 3.3 Glycemic Control & Medication Regimen Stability
- `a1c_tested` = $\mathbb{I}(\text{A1Cresult} \ne \text{'None'})$
- `a1c_abnormal` = $\mathbb{I}(\text{A1Cresult} \in \{'>7', '>8'\})$
- `med_changed` = $\mathbb{I}(\text{change} == \text{'Ch'})$
- `num_active_diabetes_meds` = count of active medications prescribed across 11 key agents.
- `on_insulin` = $\mathbb{I}(\text{insulin} \in \{\text{'Steady'}, \text{'Up'}, \text{'Down'}\})$.

---

## 4. Modeling & Probability Calibration

### 4.1 Candidate Architectures
1. **Logistic Regression (Clinical Baseline):** L2 penalty, balanced class weighting, providing interpretable log-odds coefficients.
2. **Random Forest (Bagging Ensemble):** 100 estimators, max depth 12, balanced subsampling.
3. **XGBoost (Gradient Boosted Decision Trees):** 120 estimators, depth 5, learning rate 0.08, subsample 0.8, colsample 0.8, scale_pos_weight 2.5.

### 4.2 Probability Calibration (Platt Scaling)
Raw tree-based ensembles output scores that frequently overestimate or compress probabilities in imbalanced healthcare settings. A 3-fold cross-validated sigmoid calibrator (`CalibratedClassifierCV`) is fitted on top of XGBoost:

$$P(Y = 1 \mid f(\mathbf{x})) = \frac{1}{1 + \exp(A \cdot f(\mathbf{x}) + B)}$$

This reduced Expected Calibration Error (ECE) from **0.1116** down to **0.0156** (an 86% improvement in probabilistic reliability).

---

## 5. Evaluation & Decision Threshold Strategy

In an imbalanced ~11% prevalence setting, evaluating models at a 0.50 threshold yields low sensitivity. CARE-PULSE AI adopts an operational threshold of **0.15**:
- Captures ~33–40% of true 30-day readmissions while flagging ~14–18% of discharged patients.
- Provides high specificity (>85%), minimizing nurse navigator alert fatigue.
- Metrics evaluated: ROC-AUC, PR-AUC (Average Precision), Brier Score, Brier Skill Score, Expected Calibration Error (ECE), and Youden's J statistic.
