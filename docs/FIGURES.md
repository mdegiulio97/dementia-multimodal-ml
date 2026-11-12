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
