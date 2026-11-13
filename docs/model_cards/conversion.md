# Model card: MCI-to-dementia conversion

**Task.** Predict time from a baseline visit with mild cognitive impairment (ADNI `DX_bl` in
EMCI, LMCI or MCI) to the first visit with a dementia diagnosis.

**Data.** ADNIMERGE (ADNI 1, GO, 2, 3), run locally under the ADNI Data Use Agreement. The
public repository runs the same code on a synthetic cohort with the ADNIMERGE schema.

**Outcome definition.** Event: first follow-up visit with `DX == "Dementia"` (later reversions
ignored). Censoring: last follow-up visit with a recorded diagnosis. Participants with no
follow-up diagnosis are excluded and counted.

**Features.** Baseline values only: age, MMSE, APOE4 count (baseline model); plus education,
ADAS13, CDR-SB, FAQ, hippocampal volume adjusted for intracranial volume, CSF Aβ42, t-tau,
p-tau (full models). CSF values reported at assay limits (">1700", "<8") are set to the limit
with a flag.

**Models.** Cox (age + MMSE + APOE4); ridge-penalised Cox on all features; random survival
forest; gradient-boosted survival trees. Preprocessing (hippocampus residualisation on ICV,
median imputation with missingness indicators, scaling) is fitted inside each training fold.

**Validation.** 5-fold cross-validation split by participant and stratified by event; metrics on
pooled out-of-fold predictions with 1000-sample participant bootstrap: Harrell's C, Uno's C at
36 months, time-dependent AUC at 24/36/48 months, calibration at 36 months. Horizons at or
beyond the last follow-up are skipped and reported.

**Synthetic check.** Full Cox C = 0.72 (0.68-0.76) vs baseline 0.64 (0.60-0.68); the Cox
coefficients recover the simulated log hazard ratios within ±0.15 (`tests/data`).

**ADNI.** 1,146 participants with MCI and 420 conversions. The full models reach C 0.83
(0.81-0.85), the clinical baseline C 0.69 (0.67-0.72), with good calibration at 36 months.
Data come from ADNIMERGE2 through `data/adnimerge2.py`. Imaging and CSF are taken within
180 days of enrolment.

**Intended use.** Research on prognostic modelling methods. Not a clinical tool.

**Known limitations.** Interval censoring at visit times is treated as exact; competing risk of
death is not modelled; ADNI is a volunteer research cohort, not a clinical population.

**Not published.** Trained ADNI models and per-participant predictions (DUA, Appendix A).
