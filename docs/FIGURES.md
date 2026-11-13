# Figure index

Every figure in the repository, what it shows and what to take from it. Figures under
`figures/synthetic/` come from the synthetic ADNIMERGE-shaped cohort (a method check with a
known answer, not a clinical result). Figures under `figures/open/` come from real open data.
ADNI figures are produced locally in `figures/adni/` and are not committed.

## MCI-to-dementia conversion (synthetic cohort)

### `figures/synthetic/conversion_cindex_forest.png`
Harrell's C-index of the four survival models, computed on out-of-fold predictions (5-fold
cross-validation split by participant) with 95% bootstrap intervals. The baseline uses age,
MMSE and APOE4 only. In the synthetic cohort the true hazard depends on APOE4, MMSE, ADAS13 and
ICV-adjusted hippocampal volume, so the full Cox model should beat the baseline and the flexible
models should not do better than the correctly specified Cox model. Both happen
(0.72 vs 0.64), which is the check the figure exists for.

### `figures/synthetic/conversion_km_by_risk_tertile.png`
Kaplan-Meier cumulative conversion for participants grouped by tertile of out-of-fold risk from
the best model. Separation between the curves is what a C-index summarises; the high-risk third
converts several times more often than the low-risk third over four years.

### `figures/synthetic/conversion_calibration_36m.png`
Predicted vs observed (Kaplan-Meier) probability of conversion by 36 months, by fifth of
predicted risk. Points on the diagonal mean the predicted probabilities can be read as
probabilities, not only as rankings.

### `figures/synthetic/trajectories_MMSE.png`
Mean MMSE trajectory with 95% confidence band from a linear mixed model (random intercept and
slope per participant), converters vs stable MCI. The synthetic cohort was generated with a
1.5 points/year faster decline in converters; the model estimates 1.15/year, lower because
"converter" is defined from observed follow-up while the simulated decline uses the latent
conversion time.

### `figures/synthetic/trajectories_ADAS13.png`
Same model for ADAS-Cog 13 (higher is worse): converters worsen about 1.8 points/year faster.

## Resting-state EEG (OpenNeuro ds004504, real data)

### `figures/open/eeg_band_power_by_group.png`
Relative spectral power per band, averaged over the 19 channels, one point per participant.
The known slowing of the EEG in Alzheimer's disease is visible as more theta and less alpha in
AD than in controls; FTD sits in between. Delta dominates every group because these
preprocessed recordings carry most of their power below 2 Hz (see limitations in
[RESULTS.md](RESULTS.md)).

### `figures/open/eeg_topomap_theta_alpha.png`
Scalp maps of the log theta/alpha power ratio per group. Controls keep a posterior alpha
rhythm (blue, low ratio); in AD the ratio is high over the whole scalp; in FTD the posterior
alpha is partly preserved.

### `figures/open/eeg_roc_AD_vs_CN.png`
Participant-level ROC curves for AD vs controls (logistic regression and LightGBM on spectral
features; each participant's prediction is the mean over their epochs, averaged over five
repeats of subject-level 5-fold CV). AUC about 0.90.

### `figures/open/eeg_roc_FTD_vs_CN.png`
Same for FTD vs controls: AUC 0.74 (logistic) to 0.83 (LightGBM), with wide intervals
(23 vs 29 participants).

### `figures/open/eeg_roc_AD_vs_FTD.png`
AD vs FTD: close to chance (AUC 0.53-0.61). Spectral power separates dementia from health
better than it separates the two dementias, which matches the subject-level literature.

## Mendelian randomisation (open summary statistics, real data)

### `figures/open/mr_volcano.png`
Every circulating protein (UKB-PPP cis-pQTL) and NMR metabolite tested against Alzheimer's
disease (Bellenguez et al. 2022): MR estimate vs significance. Large points pass FDR < 0.05
within their family. 44 proteins and 7 metabolites pass.

### `figures/open/mr_top_forest.png`
The 20 most significant MR estimates with 95% confidence intervals (log-odds of AD per SD of
the exposure). Most top hits sit in known AD GWAS loci (CR1, TREM2, CD2AP, EPHA1, ACE, GRN,
CD33); with a single cis instrument and no colocalisation, a hit means "the protein is a
candidate", not "the protein is causal".

## Sensitivity analyses (phase 2)

### `figures/open/eeg_band_power_by_group_delta1.png`
Same as the band-power figure, with the delta band and the total power computed from 1 Hz
instead of 0.5 Hz. Delta still holds about 80% of the power in every group, so the dominance of
slow activity in these recordings does not depend on where the band starts.

### `figures/open/eeg_topomap_theta_alpha_delta1.png`
Theta/alpha scalp maps with the 1-Hz lower edge. Identical pattern: posterior alpha in controls,
diffuse slowing in AD.

### `figures/open/eeg_roc_AD_vs_CN_delta1.png`
AD vs controls with the 1-Hz lower edge: AUC 0.90 (logistic and LightGBM), as in the primary
analysis.

### `figures/open/eeg_roc_FTD_vs_CN_delta1.png`
FTD vs controls with the 1-Hz lower edge: AUC 0.76 (logistic) and 0.83 (LightGBM), as in the
primary analysis.

### `figures/open/eeg_roc_AD_vs_FTD_delta1.png`
AD vs FTD with the 1-Hz lower edge: still near chance.

### `figures/open/mr_bellenguez_vs_kunkle.png`
MR estimate of every protein and metabolite with the primary AD GWAS (Bellenguez 2022, which
includes UK Biobank, x axis) against an AD GWAS without UK Biobank (Kunkle 2019, y axis). Orange
points are the 50 molecules selected for ADNI: 42 of the 45 testable ones keep their sign. The
estimates for all traits correlate at r = 0.42, since many of them are pure noise; the selected
ones sit along the diagonal.

### `figures/open/mr_volcano_kunkle.png`
Volcano plot of the screen with Kunkle 2019 as outcome. Fewer traits pass FDR (9 proteins)
because the GWAS is smaller, and because without allele frequencies every palindromic
instrument has to be dropped.

### `figures/open/mr_top_forest_kunkle.png`
The 20 strongest estimates with Kunkle 2019 as outcome. The nine at FDR < 0.05 are CR1, EPHA1,
CELA2A, PILRA, PILRB, C1S, C1R, CD33 and IL34. All but CELA2A are among the molecules selected
in the primary analysis.

### `figures/open/eeg_roc_AD_vs_CN_eegnet.png`
Participant-level ROC of EEGNet for AD vs controls: AUC 0.81 (0.69-0.90), below the 0.91 of the
spectral logistic model on the same folds.

### `figures/open/eeg_roc_FTD_vs_CN_eegnet.png`
EEGNet for FTD vs controls: AUC 0.64 (0.48-0.78). With 52 participants the network overfits.

## ADNI (aggregate results)

### `figures/adni_public/conversion_km_by_risk_tertile.png`
ADNI, 1,146 participants with MCI. Kaplan-Meier cumulative conversion by tertile of out-of-fold
predicted risk (gradient-boosted survival). The high-risk third converts in most cases within
five years, the low-risk third rarely. Curves stop where fewer than 10 people remain at risk.

### `figures/adni_public/conversion_calibration_36m.png`
Predicted vs observed conversion by 36 months, by fifth of predicted risk (about 230 people
each). The points lie on the diagonal.

### `figures/adni_public/conversion_cindex_forest.png`
C-index of the four models on ADNI. The clinical baseline reaches 0.69; the three full models
reach 0.83 and are indistinguishable from each other.

### `figures/adni_public/trajectories_MMSE.png`
Mixed-model mean MMSE trajectory for converters vs stable MCI on ADNI: 1.27 points/year faster
decline in converters.

### `figures/adni_public/trajectories_ADAS13.png`
Same for ADAS13: 2.85 points/year faster worsening in converters.

### `figures/adni_public/trajectories_CDRSB.png`
Same for CDR-SB: 1.07 points/year faster worsening in converters.

### `figures/adni_public/omics_proteins_qq.png`
QQ plot of p-values from Cox models of MCI-to-dementia conversion, one per CSF aptamer
(SomaScan 7k; 389 participants, 170 conversions). The whole proteome departs from the diagonal
because CSF proteins track ongoing neurodegeneration (top: 14-3-3 proteins, NEFH). The 48
aptamers of the MR candidates (orange) follow the same curve as the rest: no enrichment.

### `figures/adni_public/omics_proteins_selected_forest.png`
Hazard ratio of conversion per SD for each aptamer of the MR-selected proteins, adjusted for
age, sex and APOE4. ACE stands out (HR 0.76, p = 0.0006), in the direction predicted by MR. Most
intervals include 1.

### `figures/adni_public/omics_metabolites_qq.png`
The same QQ plot for 250 serum NMR measures (786 participants, 342 conversions): mild departure
from the diagonal, no measure at FDR < 0.05, and the 6 MR-selected metabolites near the
diagonal.

### `figures/adni_public/fusion_cindex.png`
C-index (out-of-fold, 95% bootstrap CI) for each combination of data blocks, with ridge Cox and
gradient-boosted survival, on all 1,146 ADNI participants with MCI. Adding the clinical scales
lifts the baseline from 0.69 to 0.82. Each biomarker block adds at most about 0.01, and all
blocks together about 0.02. Biomarkers without CDR-SB and FAQ reach 0.78-0.79.
