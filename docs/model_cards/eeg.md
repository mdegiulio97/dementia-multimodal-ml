# Model card: resting-state EEG classification

**Task.** Classify participants as Alzheimer's disease (AD), frontotemporal dementia (FTD) or
healthy control (CN) from eyes-closed resting EEG. Four tasks: AD vs CN, FTD vs CN, AD vs FTD,
three-class.

**Data.** OpenNeuro ds004504 v1.0.9 (CC0): 88 participants (36 AD, 23 FTD, 29 CN), 19
channels, 500 Hz, preprocessed derivatives shipped with the dataset (band-pass 0.5-45 Hz,
ASR, ICA).

**Features.** 4-s epochs with 50% overlap; per channel: relative power in delta (0.5-4),
theta (4-8), alpha (8-13), beta (13-30) and gamma (30-45 Hz), theta/alpha ratio, normalised
spectral entropy; plus the peak alpha frequency. MMSE is not used (all controls score 30).

**Models.** L2 logistic regression; LightGBM.

**Validation.** Participant-level: epochs of one person never appear in both training and test
folds; 5-fold stratified CV repeated 5 times; each participant's prediction is the mean of
their epoch probabilities averaged over repeats; metrics over participants with 1000-sample
bootstrap intervals.

**Results.** AD vs CN AUC 0.91 (0.82-0.97), balanced accuracy 0.81; FTD vs CN AUC 0.83
(0.70-0.94) with LightGBM; AD vs FTD near chance. Comparable to the subject-level studies
summarised by the AHEPA benchmark (doi:10.1007/s11571-026-10464-w).

**Phase 2.** Delta from 1 Hz gives the same results. EEGNet on raw windows (same folds) reaches
AUC 0.81 for AD vs CN and 0.64 for FTD vs CN, below the spectral models, because the network
overfits with about 60 participants.

**Limitations.** One small single-centre dataset; no external test set; preprocessing fixed by
the dataset authors; power below 2 Hz dominates the relative spectra.
