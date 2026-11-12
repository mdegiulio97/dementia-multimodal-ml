# Method card: two-sample Mendelian randomisation screen

**Question.** Which circulating proteins and metabolites have genetic evidence of an effect on
Alzheimer's disease risk? The selected molecules are the ones tested later in ADNI proteomics
and metabolomics.

**Exposures.** UKB-PPP plasma proteins, Supplementary Table 9 of Sun et al. 2023
(doi:10.1038/s41586-023-06592-6), cis-pQTL only, MHC excluded. NMR metabolites, Supplementary
Table 5 of Karjalainen et al. 2024 (doi:10.1038/s41586-024-07148-y).

**Outcome.** Alzheimer's disease GWAS of Bellenguez et al. 2022 (doi:10.1038/s41588-022-01024-z,
GCST90027158, harmonised file).

**Instruments.** p < 5e-8; APOE region (chr19:44.4-46.5 Mb, GRCh37) excluded; one variant per
500-kb window (distance-based pruning, an approximation of LD clumping); harmonised on rsID
with strand and palindrome checks (palindromes with effect-allele frequency 0.42-0.58 dropped).

**Estimators.** Wald ratio (one instrument), multiplicative random-effects IVW (two or more),
MR-Egger and weighted median as sensitivity analyses (three or more). Benjamini-Hochberg FDR
within each exposure family.

**Selection for ADNI.** FDR < 0.05 and no evidence of directional pleiotropy (Egger intercept
p ≥ 0.05 when estimable): 44 proteins and 6 metabolites (Gln, L-VLDL-TG%, M-LDL-FC%, M-LDL-PL%, XS-VLDL-FC, XXL-VLDL-CE%).

**Known-answer check.** CD33: +0.032 log-odds per SD (0.017-0.047), same direction as the
colocalised estimate reported by Zheng et al. 2020 (doi:10.1038/s41588-020-0682-6).

**Limitations.** Nearly all proteins have a single cis instrument, so Egger and weighted median
cannot be estimated and pleiotropy cannot be tested; no colocalisation, so a hit at a GWAS locus
may reflect a nearby causal variant in LD; UK Biobank participants are in both the pQTL study
and the AD GWAS (sample overlap), which biases estimates towards the observational association.
