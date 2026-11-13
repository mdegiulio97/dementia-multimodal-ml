# Results (v0.1)

Each number below comes from a result file in `results/` that records its own provenance
(dataset version or checksum, git commit, configuration, seed). Comparisons with published work
cite only papers whose records were checked (see
[DATA_SOURCES.md, section 6](DATA_SOURCES.md#6-comparison-literature)).

Marks: ✓ consistent with the reference; ⚠ consistent with caveats; ✗ inconsistent.

## Resting-state EEG: AD, FTD and controls (OpenNeuro ds004504)

88 participants, 34,788 four-second epochs. All metrics are per participant and come from
5 × 5-fold CV split by participant, with 95% bootstrap intervals. The intervals resample
participants over pooled out-of-fold predictions, so they do not include the variance from
refitting the models.

| Task | n | Model | AUC | Balanced accuracy |
|---|---|---|---|---|
| AD vs CN | 65 | logistic | 0.91 (0.82-0.97) | 0.81 (0.70-0.91) |
| AD vs CN | 65 | LightGBM | 0.90 (0.81-0.97) | 0.81 (0.70-0.91) |
| FTD vs CN | 52 | logistic | 0.74 (0.59-0.87) | 0.65 (0.53-0.76) |
| FTD vs CN | 52 | LightGBM | 0.83 (0.70-0.94) | 0.73 (0.61-0.84) |
| AD vs FTD | 59 | logistic | 0.61 (0.44-0.76) | 0.57 (0.48-0.66) |
| AD vs FTD | 59 | LightGBM | 0.53 (0.36-0.69) | 0.59 (0.51-0.69) |
| 3-class | 88 | LightGBM | 0.74 (0.66-0.82) (one-vs-rest) | 0.63 (0.54-0.71) |

**Comparison with the literature.** The AHEPA benchmark (Miltiadous et al. 2026,
doi:10.1007/s11571-026-10464-w) averages the 12 subject-level studies on this dataset.

- **AD vs CN ✓.** The benchmark reports 82.1% accuracy, and our balanced accuracy is 0.81.
  Mlinarič et al. 2026 (doi:10.1038/s41598-026-35316-9, leave-one-subject-out) report ROC-AUC
  0.82; we get 0.90-0.91.
- **FTD vs CN ✓.** The benchmark reports 75.2% accuracy; our balanced accuracy is 0.73 with
  LightGBM.
- **AD vs FTD ⚠.** The benchmark reports 71.4% accuracy, but that number disagrees with its own
  table. Mlinarič report AUC 0.65. We are close to chance.

Studies that let epochs of one person cross folds report 9-17 points more. The benchmark names
this leakage as the reason; our splits exclude it by construction and a test checks it.

**Limitations.**
- One small single-centre dataset, with no external test set.
- We used the preprocessing shipped with the dataset.
- In these recordings most power lies below 2 Hz, so relative delta power is about 0.88 in every
  group. The spectrum of two controls was checked and the dominance is a property of the data,
  not a computation error. Phase 2 adds a sensitivity analysis with delta at 1-4 Hz.

### Sensitivity analyses and deep learning (phase 2)

**Delta from 1 Hz instead of 0.5 Hz.** The results do not change.

| Task | Model | AUC, 0.5-45 Hz | AUC, 1-45 Hz |
|---|---|---|---|
| AD vs CN | logistic | 0.91 | 0.90 (0.81-0.97) |
| AD vs CN | LightGBM | 0.90 | 0.90 (0.80-0.97) |
| FTD vs CN | logistic | 0.74 | 0.76 (0.60-0.88) |
| FTD vs CN | LightGBM | 0.83 | 0.83 (0.70-0.94) |

Even from 1 Hz, delta keeps about 80% of the relative power in every group (medians 0.80-0.82).
The slow-activity dominance is a property of these recordings, not of the band edge.

**EEGNet on raw 4-s windows** (125 Hz, z-scored per channel). It uses the same outer folds by
participant, early stopping on an inner split that is also by participant, and one prediction
per participant.

| Task | n | EEGNet AUC | EEGNet balanced accuracy | Best spectral model AUC |
|---|---|---|---|---|
| AD vs CN | 65 (13,282 windows) | 0.81 (0.69-0.90) | 0.63 (0.53-0.73) | 0.91 |
| FTD vs CN | 52 (10,152 windows) | 0.64 (0.48-0.78) | 0.56 (0.46-0.66) | 0.83 |

**With 52-65 participants, the convolutional network does worse than spectral features.** Its
training loss falls from 0.63 to 0.17, while performance on held-out participants stays below the
spectral models. The network fits the participants it has seen. Thousands of windows do not make
up for few people, because the windows of one person are not independent examples. This is the
expected behaviour of deep learning on small clinical cohorts, and it is why the spectral models
remain the primary EEG result.

## Mendelian randomisation: circulating proteins and metabolites on Alzheimer's disease

**Inputs.**
- Exposures: UKB-PPP cis-pQTL (Sun et al. 2023, doi:10.1038/s41586-023-06592-6) and NMR
  metabolite QTL (Karjalainen et al. 2024, doi:10.1038/s41586-024-07148-y).
- Outcome: Bellenguez et al. 2022 (doi:10.1038/s41588-022-01024-z, GCST90027158).

**Instruments.** We excluded the APOE region and pruned to one variant per 500 kb.

| Family | Tested | FDR < 0.05 | Selected for ADNI |
|---|---|---|---|
| Proteins (cis-pQTL) | 1,584 | 44 | 44 |
| Metabolites (NMR) | 233 | 7 | 6 |

**Strongest protein signals.** CR1, TREM2, PILRA/PILRB, CD2AP, GRN, EPHA1, ACE, CD55, CD33 and
others.
- Most of these sit in known AD GWAS loci. That is expected, because a cis-pQTL at a GWAS locus
  gives a strong Wald ratio.
- It is also the main caveat: without colocalisation, the protein and the disease may share a
  locus without sharing a causal variant.
- The full table is `results/open/mr_candidates.csv`, and the list passed to ADNI is
  `results/open/mr_candidates_selected.txt`.

**Metabolites.** The seven FDR-significant traits are lipoprotein composition measures and
glutamine, each with 18-47 instruments, so the sensitivity estimators apply. Six (Gln, L-VLDL-TG%, M-LDL-FC%, M-LDL-PL%, XS-VLDL-FC, XXL-VLDL-CE%)
are passed to ADNI. L-VLDL-C% is held back because its Egger intercept is borderline
(p = 0.049).

**Known-answer check ✓.** For CD33 we estimate +0.032 log-odds of AD per SD (95% CI
0.017-0.047). This is the same direction as the colocalised estimate in Zheng et al. 2020
(doi:10.1038/s41588-020-0682-6, Table S7: +0.100). The magnitudes differ because the studies
use different pQTL sources, outcome GWAS and scaling.

**Sensitivity: an AD GWAS without UK Biobank ✓.** UK Biobank is in both the pQTL study and
Bellenguez 2022, so the screen was repeated with Kunkle et al. 2019
(doi:10.1038/s41588-019-0358-2, GCST007511, no UK Biobank).
- Kunkle has no allele frequencies, so every palindromic instrument is dropped (1,100).
- Of the 50 selected molecules, 45 could be tested:
  - 42 keep the same sign, and 26 of them reach p < 0.05;
  - FES, ARHGAP45 and LRP11 change sign, all with Kunkle p > 0.3;
  - TREM2, CD2AP, F13B, FCRLB and SULT1A1 have no usable instrument.
- Nine proteins pass FDR with Kunkle: CR1, EPHA1, CELA2A, PILRA, PILRB, C1S, C1R, CD33 and
  IL34. All but CELA2A were already selected.
- Across all 1,465 traits tested with both outcomes, the estimates correlate at r = 0.42, mostly
  noise around zero; the selected molecules sit along the diagonal (`figures/open/mr_bellenguez_vs_kunkle.png`).

The sample overlap therefore does not create the signal: the selected molecules keep their
direction in an independent GWAS.

**Colocalisation of the strongest candidates ✓/⚠.** The question is whether the protein and
AD share the same causal variant. If they do not, the MR signal could come from linkage
disequilibrium with a different variant.

**Method.** coloc ABF (Giambartolomei et al. 2014, doi:10.1371/journal.pgen.1004383): priors
p1 = p2 = 1e-4 and p12 = 1e-5, a ±500 kb window around the cis lead variant, UKB-PPP European
discovery regional summary statistics against Bellenguez 2022 (GRCh38). It was run on the nine
candidates that held up with Kunkle 2019.

| Protein | Variants | PP.H4 (shared) | PP.H3 (distinct) |
|---|---|---|---|
| CR1 | 4,261 | 0.996 | 0.004 |
| PILRA | 3,253 | 0.995 | 0.005 |
| PILRB | 3,251 | 0.995 | 0.005 |
| IL34 | 2,629 | 0.970 | 0.025 |
| C1S | 3,958 | 0.961 | 0.033 |
| CD33 | 5,555 | 0.953 | 0.024 |
| C1R | 3,978 | 0.939 | 0.051 |
| EPHA1 | 2,549 | 0.016 | 0.984 |
| ACE | 3,227 | 0.001 | 0.999 |

- **Seven proteins colocalise strongly** (PP.H4 > 0.93): CR1, PILRA/PILRB, IL34, C1S/C1R and
  CD33.
- **CD33 matches the known answer ✓.** Zheng et al. 2020 also report CD33 as colocalised.
- **ACE and EPHA1 do not colocalise under ABF**, which assumes one causal variant per trait.
  The two loci were therefore re-analysed with SuSiE (below).

**SuSiE-based colocalisation for ACE and EPHA1.**

*Method.*
- Fine-mapping with SuSiE-RSS (Zou et al. 2022, doi:10.1371/journal.pgen.1010299; L = 10) for
  each trait.
- The LD comes from UK Biobank (Alkes Price group, `s3://broad-alkesgroup-ukbb-ld`, GRCh37), the
  same population as UKB-PPP.
- Colocalisation is computed for every pair of credible sets (Wallace 2021,
  doi:10.1371/journal.pgen.1009440).
- Implemented in Python in `dementia_ml.mr.susie` and tested on simulated loci with a known
  answer.

| Protein | Variants | Credible sets (protein / AD) | Best PP.H4 | MR, strongest pQTL | MR, colocalising signal |
|---|---|---|---|---|---|
| ACE | 3,072 | 5 / 1 | **0.98** | −0.12 (SE 0.018) | **−0.19 (−0.24 to −0.14), p = 2×10⁻¹⁵** |
| EPHA1 | 2,417 | 10 / 1 | 0.008 | −0.29 (SE 0.043) | none |

- **ACE colocalises ✓.** The ACE locus carries five independent signals for plasma ACE. One of
  them, not the strongest, is shared with AD (PP.H4 = 0.98). The ABF result was an artefact of
  its single-variant assumption.
- **Corrected ACE estimate.** The first MR estimate used the strongest pQTL, which is not the AD
  signal. With the colocalising variant as instrument (17:61548948 C/A, GRCh37) the estimate is
  −0.19 log-odds of AD per SD of plasma ACE, stronger than before and in the same direction.
  It is also the same direction as in ADNI CSF, where higher ACE predicts lower conversion
  (HR 0.76).
- **EPHA1 does not colocalise ✗**, even with multiple signals allowed. Its MR signal most likely
  reflects LD with a different causal variant, so EPHA1 is no longer counted among the credible
  candidates.
- **Caveat ⚠.** Ten credible sets for EPHA1 at n ≈ 34,000 point to sensitivity to small LD
  differences. The reference is UK Biobank, as are the pQTLs, which limits this.

**ACE is the only candidate supported by all three lines of evidence:**
1. MR with a colocalising instrument;
2. colocalisation of a shared variant;
3. the same direction in CSF in an independent cohort (ADNI).

**Correcting a harmonisation bug mattered.** An earlier version flipped the sign of
palindromic instruments (A/T, C/G) whose alleles were listed in the opposite order. This
corrupted the multi-instrument metabolite estimates (0 significant before the fix, 7 after),
while the single-instrument protein results stayed the same. Tests now cover both orderings
and the strand flip.

**Excluding the APOE region mattered.** In the first run, proteins next to APOE (BCAM, NECTIN2,
PVR, CEACAM19) passed FDR. They disappear once APOE-region instruments are dropped, as they
should if APOE drives them.

**Limitations.**
- Almost every protein has one instrument, so Egger and the weighted median cannot be estimated.
- There is no colocalisation.
- UK Biobank is in both the exposure and the outcome GWAS (sample overlap).
- Pruning is by distance, not by LD.

## MCI-to-dementia conversion

### Method check on the synthetic cohort

The synthetic cohort has 1,500 participants with an ADNIMERGE-like structure. Of these, 787
had MCI at baseline with follow-up, and 225 converted. They are generated from a known Weibull
hazard. The metrics are out-of-fold, from 5-fold CV split by participant.

| Model | Harrell's C | Uno's C (36 m) | td-AUC 24 m | td-AUC 36 m |
|---|---|---|---|---|
| Cox, age + MMSE + APOE4 | 0.64 (0.60-0.68) | 0.67 (0.62-0.71) | 0.68 | 0.66 |
| Cox (ridge), all baseline | **0.72 (0.68-0.76)** | 0.74 (0.69-0.78) | 0.76 | 0.76 |
| Random survival forest | 0.70 (0.66-0.73) | 0.71 (0.66-0.75) | 0.73 | 0.74 |
| Gradient-boosted survival | 0.69 (0.65-0.73) | 0.70 (0.65-0.74) | 0.71 | 0.73 |

What should happen does happen:
- The full model beats the baseline, because the simulated hazard uses ADAS13 and hippocampal
  volume.
- The correctly specified Cox model is not beaten by the flexible models.
- Calibration at 36 months follows the diagonal.
- The Cox coefficients recover the simulated log hazard ratios within ±0.15 (test).

The 48-month horizon is skipped because it equals the end of follow-up.

**Reference for the real-data run.** Wang et al. 2023 (doi:10.1016/j.jclinepi.2023.03.009) is
the closest design: ADNI MCI, Cox, 10-fold CV, c-index 0.74 (0.70-0.79). Published values on
ADNI span 0.68 to 0.90, the higher values with richer feature sets.

### Cognitive trajectories (synthetic)

Mixed model with random intercept and slope:
- MMSE declines 1.15 points/year faster in converters (95% CI 1.06-1.24).
- ADAS13 worsens 1.77 points/year faster (1.62-1.92).

## ADNI results

Aggregate results on ADNI (ADNI 1, GO, 2, 3 and 4), published as summary data under the ADNI Data
Use Agreement. Models, per-participant predictions and the ADNI table stay local.

**Data.**
- The ADNIMERGE2 R package, downloaded from LONI IDA on 2026-11-13. ADNIMERGE2 replaces the
  old single `ADNIMERGE.csv`; `pipelines/prepare_adni.py` rebuilds the table from its `ADSL`,
  `ADRS` and `ADQS` tables.
- Hippocampal and intracranial volumes from FreeSurfer, ComBat-harmonised across scanners
  (`ADSP_PHC_T1_FS`).
- CSF Aβ42, t-tau and p-tau181 from Roche Elecsys (UPENN).
- Imaging and CSF are the record closest to enrolment, within 180 days.

**Cohort.**
- 1,271 participants had MCI at baseline. 125 had no follow-up diagnosis and were excluded,
  which leaves **1,146 participants and 420 conversions to dementia**.
- Observed at baseline: hippocampal volume for 1,061, CSF for 716, APOE for 1,135.

| Model | Harrell's C | Uno's C (36 m) | td-AUC 24 m | td-AUC 36 m | td-AUC 48 m |
|---|---|---|---|---|---|
| Cox, age + MMSE + APOE4 | 0.69 (0.67-0.72) | 0.69 (0.66-0.72) | 0.72 | 0.74 | 0.77 |
| Cox (ridge), all baseline | 0.83 (0.81-0.85) | 0.83 (0.81-0.85) | 0.86 | 0.89 | 0.90 |
| Random survival forest | 0.83 (0.81-0.85) | 0.83 (0.81-0.85) | 0.86 | 0.88 | 0.89 |
| Gradient-boosted survival | 0.83 (0.81-0.85) | 0.84 (0.82-0.85) | 0.87 | 0.89 | 0.89 |

**Calibration.** Predicted conversion by 36 months follows the observed (Kaplan-Meier) one
across fifths of risk: 0.05/0.00, 0.10/0.08, 0.20/0.22, 0.44/0.49, 0.77/0.75, with about 230
participants per fifth.

**Comparison ✓.** Wang et al. 2023 (doi:10.1016/j.jclinepi.2023.03.009) report a c-index of
0.74 with clinical variables. Saeed et al. 2024 (doi:10.1371/journal.pone.0314725) report
0.82-0.90 with multimodal random survival forests. Our clinical baseline (0.69) and full
models (0.83) sit inside that range.

The three flexible and linear models reach the same discrimination. A ridge Cox model on
baseline values is as good as the forests here, so it stays the model behind the agent.

**Caveat ⚠.** CDR-SB and FAQ measure daily function, and functional impairment is part of the
clinical definition of dementia. Part of their predictive weight is therefore built into the
outcome. The baseline model without them (0.69) is the honest lower bound.

**Do the MR candidates predict conversion in ADNI? Mostly no.** Each molecule measured at
baseline (within 180 days of enrolment) was tested in its own Cox model of conversion. The
molecule was inverse-normal transformed, so hazard ratios are per SD, and each model was
adjusted for age, sex and APOE4. FDR was computed across all molecules of each platform.

| Platform | MCI participants (conversions) | Molecules tested | FDR < 0.05 | MR candidates measured | Candidates at p < 0.05 | Enrichment vs other molecules | Same direction as MR |
|---|---|---|---|---|---|---|---|
| CSF proteins, SomaScan 7k (CruchagaLab) | 389 (170) | 7,008 aptamers | 81 (78 genes) | 37 genes, 48 aptamers | 4 / 48 (others: 12%) | Mann-Whitney p = 0.49 | 19 / 48 |
| Serum NMR metabolites (ADMC Nightingale) | 786 (342) | 250 | 0 | 6 | 0 / 6 (others: 14%) | p = 0.98 | 4 / 6 |

- **The screen does detect signal.** The CSF proteins most associated with conversion are
  markers of neuronal and synaptic injury: the 14-3-3 family (YWHAG, YWHAZ, YWHAB, YWHAE) and
  neurofilament heavy chain (NEFH), with HR 1.5-1.8 per SD. They act as a positive control for
  the pipeline.
- **The candidates do not stand out.** Taken together, the 48 aptamers of the MR candidates are
  no more associated with conversion than the rest of the proteome. Their direction agrees with
  MR in 19 of 48, which is chance level.
- **ACE is the exception ✓.** Higher CSF ACE predicts lower conversion: HR 0.76 per SD,
  p = 0.0006, which survives Bonferroni across the 48 candidate aptamers. MR points the same
  way: genetically higher ACE means lower AD risk.
- **Two discordant candidates.** CD2AP (HR 0.80) and ACHE (HR 1.28) are nominally associated
  in the direction opposite to MR.

**Why a negative result is plausible ⚠.** Several differences separate the two analyses:
- **Tissue.** MR used plasma protein levels; ADNI measures CSF.
- **Question.** MR asks about lifetime risk of AD; ADNI asks about progression from MCI over a
  few years, a stage where markers of ongoing injury dominate.
- **Instruments.** Most MR hits rest on a single cis variant at a known GWAS locus, without
  colocalisation.
- **Power.** 389 participants detect only fairly large effects.

So the MR candidates are not validated as markers of progression; ACE is the one worth
following up. Full tables: `results/adni_public/omics_proteins.csv` and `omics_metabolites.csv`.

**Multimodal fusion: what do biomarkers add to the clinical prediction?**

**Blocks.**
- Clinical: age, education, APOE4, MMSE, ADAS13, CDR-SB, FAQ.
- MRI: hippocampus and ICV, plus regional FreeSurfer volumes (amygdala, ventricles) and the
  thickness of seven AD-signature regions.
- CSF: Aβ42, t-tau, p-tau.
- CSF proteomics: SomaScan 7k, reduced to 20 principal components.
- Serum metabolomics: NMR, reduced to 10 components.

**Method.**
- Imputation, scaling and PCA are fitted inside each training fold.
- Participants without a block get zeros for it, plus an indicator. This way all 1,146
  participants are kept, although only 389 have proteomics and 785 have metabolomics.
- Differences against the clinical model come from a paired bootstrap on the same
  participants.

| Blocks | C-index, ridge Cox | C-index, boosted | Difference vs clinical (Cox; boosted) |
|---|---|---|---|
| Baseline (age, MMSE, APOE4) | 0.69 (0.67-0.72) | 0.70 (0.67-0.72) | −0.12; −0.13 |
| **Clinical** | 0.82 (0.80-0.83) | 0.82 (0.81-0.84) | — |
| + hippocampus | 0.82 | 0.83 | +0.005 (−0.003, +0.013); +0.004 |
| + regional MRI | 0.83 | 0.83 | +0.010 (−0.001, +0.019); +0.011 (+0.001, +0.019) |
| + CSF | 0.82 | 0.83 | +0.006; +0.006 (both CIs include 0) |
| + proteomics | 0.82 | 0.83 | 0.000; +0.003 (both include 0) |
| + metabolomics | 0.82 | 0.82 | +0.004; +0.001 (both include 0) |
| **All blocks** | 0.83 (0.82-0.85) | 0.84 (0.82-0.86) | **+0.018 (+0.006, +0.030); +0.016 (+0.005, +0.025)** |
| Baseline + biomarkers (no CDR-SB, FAQ) | 0.79 (0.76-0.81) | 0.78 (0.75-0.80) | −0.028; −0.046 |

- **Clinical scales carry most of the information.** All biomarkers together add about 0.02 in
  C-index, which is real but small. Regional MRI is the single block that comes closest to
  adding something on its own.
- **The biology is there, but it overlaps with the clinical scales.** Without CDR-SB and FAQ,
  biomarkers raise the baseline from 0.69 to 0.79. This answers the circularity caveat above:
  the biomarkers predict conversion independently of the functional scales, but they add little
  once those scales are known.
- **Complete-case check ⚠.** Among the 358 participants with every block, all data together
  reach 0.81 (0.77-0.84), the same as the clinical model (0.81). The small gain of the full
  cohort does not show up in this subset.

**Cognitive trajectories** (mixed model, 1,146 participants). Converters decline faster than
stable MCI:
- MMSE: 1.27 points/year faster (1.15-1.39);
- ADAS13: 2.85 points/year faster (2.57-3.14);
- CDR-SB: 1.07 points/year faster (1.00-1.15).

Data used in the preparation of these results were obtained from the Alzheimer's Disease
Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu); see the acknowledgement in the
README.

### ADNI (method notes)

The same code runs locally on ADNIMERGE. ADNI results will be added here as aggregate tables
and figures only, with no per-participant data, as the ADNI Data Use Agreement allows. Trained
ADNI models and per-participant predictions always stay in the gitignored `results/adni/` and
`models/`.

## Agent

Tool selection by the local model `qwen2.5:7b-instruct` (Ollama, CPU) on the 12 golden
questions: **11/12** (run on 2026-11-12; median 40 s per question on CPU).

- **The miss.** "How much faster does MMSE decline in converters?" went to `cohort_summary`
  instead of `get_result("trajectories")`. The model wrote the variable in lower case, the tool
  refused it, and the agent returned the refusal rather than a number.
- **Correct answers.** Risk questions were answered with the tool's probability and bootstrap
  interval, for example "31.5% (28.1%-36.1%) at 3 years" for a synthetic profile. The request
  for a single participant's MMSE was refused.

In CI, a scripted backend checks routing, refusals (small cells, participant identifiers),
provenance and abstention on every run.
