# CARE-PULSE AI — Clinical Data Dictionary

Dataset Source: **Diabetes 130-US Hospitals for Years 1999–2008** (OpenML ID: 4541 / UCI ML Repository 296)  
Citation: *Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. BioMed Research International.*

---

## 1. Identifiers & Demographics

| Feature Name | Type | Allowed Values | Clinical Description | Handling & Modeling Role |
|---|---|---|---|---|
| `encounter_id` | Integer | Unique integer | Unique inpatient encounter identifier | **Excluded from model** (Arbitrary ID) |
| `patient_nbr` | Integer | Unique integer | Patient identifier across multiple hospitalizations | **Grouping Variable** (Used for GroupShuffleSplit) |
| `race` | Categorical | `Caucasian`, `AfricanAmerican`, `Hispanic`, `Asian`, `Other`, `?` | Self-reported racial/ethnic category | Imputed with `'Missing/Unknown'`; One-Hot Encoded |
| `gender` | Categorical | `Female`, `Male`, `Unknown/Invalid` | Biological sex | `Unknown/Invalid` ($N=3$) excluded; One-Hot Encoded |
| `age` | Categorical | `[0-10)`, `[10-20)`, ..., `[90-100)` | 10-year age decile band | One-Hot Encoded |
| `weight` | Categorical | `[0-25)`, `[25-50)`, ..., `?` | Recorded body weight in kg | **Excluded from model** (~97% missingness) |

---

## 2. Inpatient Admission & Discharge Attributes

| Feature Name | Type | Allowed Values | Clinical Description | Handling & Modeling Role |
|---|---|---|---|---|
| `admission_type_id` | Integer/Cat | 1 to 8 (1=Emergency, 2=Urgent, 3=Elective, etc.) | Urgency of hospital encounter admission | Converted to string, One-Hot Encoded |
| `discharge_disposition_id` | Integer/Cat | 1 to 30 (1=Home, 3=SNF, 6=Home Health, 11=Expired) | Patient discharge destination | Excluded if $\in \{11, 19, 20, 21\}$; One-Hot Encoded |
| `admission_source_id` | Integer/Cat | 1 to 25 (7=Emergency Room, 1=Physician Referral) | Point of origin preceding hospital entry | Converted to string, One-Hot Encoded |
| `time_in_hospital` | Integer | 1 to 14 days | Total length of inpatient hospital stay | Continuous; Median Imputed & Standardized |
| `payer_code` | Categorical | Blue Cross, Medicare, Medicaid, HMO, etc. | Health insurance category | **Excluded from model** (~40%+ missingness) |
| `medical_specialty` | Categorical | Cardiology, Surgery, Internal Medicine, etc. | Specialty of admitting physician | **Excluded from model** (~49% missingness) |

---

## 3. Clinical Metrics & Prior Healthcare Utilization

| Feature Name | Type | Allowed Values | Clinical Description | Handling & Modeling Role |
|---|---|---|---|---|
| `num_lab_procedures` | Integer | 1 to 132 | Total diagnostic lab tests during encounter | Continuous; Median Imputed & Standardized |
| `num_procedures` | Integer | 0 to 6 | Number of non-lab procedural interventions | Continuous; Median Imputed & Standardized |
| `num_medications` | Integer | 1 to 81 | Distinct medications administered to patient | Continuous; Median Imputed & Standardized |
| `number_outpatient` | Integer | 0 to 42 | Outpatient clinic encounters in prior 12 months | Continuous; Standardized |
| `number_emergency` | Integer | 0 to 76 | Emergency room visits in prior 12 months | Continuous; Standardized |
| `number_inpatient` | Integer | 0 to 21 | Acute inpatient admissions in prior 12 months | Continuous; Standardized (Strongest predictor) |
| `number_diagnoses` | Integer | 1 to 16 | Number of diagnoses recorded on the encounter | Continuous; Standardized |

---

## 4. Diagnostic Codes & Glycemic Lab Tests

| Feature Name | Type | Allowed Values | Clinical Description | Handling & Modeling Role |
|---|---|---|---|---|
| `diag_1` | Categorical | ICD-9 Code (e.g. 250.02, 428) | Primary admitting diagnosis | Mapped to 9 clinical disease groups (`diag_1_group`) |
| `diag_2` | Categorical | ICD-9 Code | Secondary co-occurring diagnosis | Mapped to 9 clinical disease groups (`diag_2_group`) |
| `diag_3` | Categorical | ICD-9 Code | Tertiary co-occurring diagnosis | Mapped to 9 clinical disease groups (`diag_3_group`) |
| `max_glu_serum` | Categorical | `None`, `Norm`, `>200`, `>300` | Glucose serum laboratory test result | Categorical; One-Hot Encoded |
| `A1Cresult` | Categorical | `None`, `Norm`, `>7`, `>8` | Glycated hemoglobin (HbA1c) test result | Categorical; One-Hot Encoded |

---

## 5. Diabetic Medications & Regimen Adjustments

| Feature Name | Type | Allowed Values | Clinical Description | Handling & Modeling Role |
|---|---|---|---|---|
| `metformin` | Categorical | `No`, `Steady`, `Up`, `Down` | Biguanide oral antidiabetic status | Active med count & Categorical |
| `insulin` | Categorical | `No`, `Steady`, `Up`, `Down` | Exogenous insulin administration status | Categorical; One-Hot Encoded |
| `glipizide` | Categorical | `No`, `Steady`, `Up`, `Down` | Sulfonylurea antidiabetic status | Active med count |
| `glyburide` | Categorical | `No`, `Steady`, `Up`, `Down` | Sulfonylurea antidiabetic status | Active med count |
| `pioglitazone` | Categorical | `No`, `Steady`, `Up`, `Down` | Thiazolidinedione status | Active med count |
| `rosiglitazone` | Categorical | `No`, `Steady`, `Up`, `Down` | Thiazolidinedione status | Active med count |
| `change` | Categorical | `Ch`, `No` | Diabetic medication adjusted during encounter | Mapped to binary indicator `med_changed` |
| `diabetesMed` | Categorical | `Yes`, `No` | Any diabetes medication prescribed | Mapped to binary indicator `on_diabetes_med` |

---

## 6. Target Variables

| Feature Name | Type | Allowed Values | Clinical Description | Handling & Modeling Role |
|---|---|---|---|---|
| `readmitted` | Categorical | `NO`, `>30`, `<30` | Original readmission encounter outcome | Preserved in dataset for reference |
| `readmission_30d` | Binary (0/1) | `1` if `<30`, `0` otherwise | Readmission strictly within 30 days of discharge | **Primary Supervised Prediction Target** |
