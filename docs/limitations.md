# CARE-PULSE AI — Limitations, Risk Analysis & Clinical Boundaries

## 1. Historical Dataset Constraints

The underlying dataset comprises 101,766 inpatient encounters collected from 130 US hospitals between **1999 and 2008**. This introduces several critical limitations:
1. **Pharmacological Evolution:** Modern frontline antidiabetic therapies—including SGLT2 inhibitors (e.g. Empagliflozin) and GLP-1 receptor agonists (e.g. Semaglutide)—were not in clinical use during the cohort window.
2. **Coding System Changes:** Diagnoses are recorded in **ICD-9-CM** rather than modern **ICD-10-CM**, requiring translational grouping logic.
3. **Regulatory Shifts:** The data predates the full implementation of the 2010 Affordable Care Act (ACA) and early CMS Hospital Readmissions Reduction Program (HRRP) enforcement, both of which restructured post-discharge care protocols.

---

## 2. Unmeasured Confounders & Social Determinants of Health (SDoH)

A significant proportion of avoidable hospital readmissions is driven by non-clinical social determinants that are absent from historical EHR claims tables:
- **Housing & Food Security:** Inability to refrigerate insulin or maintain consistent nutrition.
- **Health Literacy & Language Barriers:** Misunderstanding complex post-discharge sliding-scale insulin directions.
- **Transportation Access:** Missing follow-up appointments with primary care providers due to transit deficits.
- **Medication Affordability:** Copay abandonment when filling discharge prescriptions at community pharmacies.

---

## 3. Methodological & Survival Censoring Limitations

1. **Out-of-Hospital Mortality Censoring:** While known inpatient deaths (`discharge_disposition_id` 11, 19, 20, 21) are removed, patients who passed away at home within 30 days of discharge without returning to one of the 130 participating hospitals are coded as non-readmissions ($Y=0$). This introduces slight informative right-censoring.
2. **Cross-Hospital Leakage:** Readmissions occurring at health systems outside the 130 participating centers are unrecorded, which is an inherent limitation of regional hospital claims data.

---

## 4. Regulatory & Clinical Safety Boundaries

> **MANDATORY CLINICAL DISCLAIMER:**  
> CARE-PULSE AI is an academic research and portfolio data science project. It has not undergone FDA 510(k) clearance or CE-mark software as a medical device (SaMD) validation.
> 
> - **DO NOT USE** for making real-time autonomous patient discharge decisions.
> - **DO NOT USE** for diagnosing diabetes, ketoacidosis, or adjusting pharmaceutical dosages without physician oversight.
> - **ALWAYS** ensure human-in-the-loop clinical nurse navigation and multidisciplinary team review before allocating post-discharge interventions.
