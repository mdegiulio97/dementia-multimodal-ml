# Data sources

Phase 0 source verification, carried out on 2026-11-12. Every fact below was read from the
page, API response or file named next to it during that session. Status marks:

- ✓ verified: read directly from the primary source (API, file header, official page)
- ⚠ partially verified: something is missing or was inferred indirectly; the note says what
- ✗ not verifiable from public sources

Machine-readable summary: [`configs/sources.yaml`](../configs/sources.yaml). Pipelines read only
from that file.

| Key | Source | Access | Status |
|---|---|---|---|
| `eeg_ds004504` | OpenNeuro ds004504 v1.0.9 | open (CC0) | ✓ |
| `gwas_ad` | Bellenguez et al. 2022, GCST90027158 | open (CC0) | ✓ |
| `pqtl` | UKB-PPP, Sun et al. 2023, Supplementary Table 9 | open (CC BY 4.0) | ✓ (row count ⚠) |
| `mqtl` | Karjalainen et al. 2024, Supplementary Table 5 | open (CC BY 4.0) | ✓ (row count ⚠) |
| `adni` | ADNI via LONI IDA | DUA | ⚠ (DUA read ✓; posting on GitHub not addressed by the DUA) |

---

## 1. EEG: OpenNeuro ds004504 ✓

**What it is.** Resting-state, eyes-closed scalp EEG from 88 people: Alzheimer's disease,
frontotemporal dementia and cognitively normal controls, recorded at the 2nd Department of
Neurology, AHEPA University Hospital, Thessaloniki.

| Field | Value | Status | Where it was read |
|---|---|---|---|
| URL | https://openneuro.org/datasets/ds004504 | ✓ | OpenNeuro GraphQL `https://openneuro.org/crn/graphql` |
| Latest version | 1.0.9 (snapshot created 2026-05-20); earlier tags 1.0.0 to 1.0.8 (2023-02-17 to 2024-10-30) | ✓ | GraphQL `snapshots`, `CHANGES` |
| Dataset DOI | 10.18112/openneuro.ds004504.v1.0.9 | ✓ | `dataset_description.json` |
| Descriptor paper DOI | 10.3390/data8060095 (Miltiadous et al., *Data* 2023, 8(6):95, CC BY 4.0) | ✓ | `dataset_description.json`, Crossref |
| First study on the data | 10.1109/ACCESS.2023.3294618 (DICE-Net, *IEEE Access* 2023) | ✓ | `HowToAcknowledge`, Crossref |
| Benchmark paper | 10.1007/s11571-026-10464-w (AHEPA EEG benchmark, *Cogn Neurodyn* 2026, PMID 42165009) | ✓ | `HowToAcknowledge`, Crossref, PubMed |
| License | CC0 | ✓ | `dataset_description.json` (`"License": "CC0"`) |
| Access | open, no registration | ✓ | anonymous S3 listing and GraphQL worked |
| BIDS version | v1.2.1 | ✓ | `dataset_description.json` |
| Subjects | 88: 36 AD, 29 CN, 23 FTD | ✓ | counted in `participants.tsv`; matches `README` |
| Task | `eyesclosed`; no sessions | ✓ | GraphQL `summary` |
| Sampling rate | 500 Hz | ✓ | `sub-001_task-eyesclosed_eeg.json` (`SamplingFrequency`) and `README` |
| Channels | 19 EEG: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz (µV); no EOG/ECG/EMG | ✓ | `sub-001_task-eyesclosed_channels.tsv`, `eeg.json` |
| Reference | `"EEGReference": "A1 A2"` in the sidecar; the README says the referential montage used Cz as common reference and the derivatives were re-referenced to A1-A2 | ⚠ | the two statements differ; treat the raw reference as uncertain and prefer the derivatives |
| Power-line frequency | 50 Hz | ✓ | `eeg.json` |
| Recording length | about 13.5 min AD (5.1 to 21.3), 12 min FTD (7.9 to 16.9), 13.8 min CN (12.5 to 16.5) | ✓ | `README` (sub-001: `RecordingDuration` 599.8 s) |
| File format | EEGLAB `.set`, data embedded (no `.fdt` files in the listing) | ✓ | S3 listing |
| Preprocessed `derivatives/` | yes: `derivatives/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set` for all 88 subjects. Pipeline per README: Butterworth band-pass 0.5 to 45 Hz, re-reference to A1-A2, ASR (window SD threshold 17), ICA (RunICA) with ICLabel rejection of eye and jaw components. `derivatives/` has no `dataset_description.json` and no sidecars | ✓ | S3 listing, `README` |
| Size | snapshot 1.0.9: 2,829,897,704 bytes (raw only, 269 files). Public S3 prefix: 359 objects, 5,780,997,840 bytes in total, of which `derivatives/` 2,951,099,688 bytes | ✓ | GraphQL `size`; S3 `list-type=2` listing summed |
| sha256 | not published | ✗ | OpenNeuro exposes git-annex object ids, not sha256 |

**`participants.tsv` columns** (read on 2026-11-12, 1,706 bytes): `participant_id`, `Gender`
(M/F), `Age` (years), `Group`, `MMSE`. The `Group` levels from `participants.json`:

| Code in file | Meaning (`participants.json`) | Our label | n |
|---|---|---|---|
| `A` | Alzheimer Disease Group | `AD` | 36 |
| `F` | Frontotemporal Dementia Group | `FTD` | 23 |
| `C` | Healthy Group | `CN` | 29 |

Parsing notes: the `MMSE` cell of `sub-050` has a trailing space (`30 `), so strip strings before
casting. All CN have MMSE 30 (ceiling), so MMSE must never be used as a feature for CN vs patient
tasks (it is the label in disguise).

**Download route (free).** The public bucket answers anonymous requests:

```bash
# whole dataset (about 5.8 GB with derivatives)
aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ./ds004504
# derivatives only (about 2.95 GB) plus the metadata files
aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ./ds004504 \
  --exclude "*" --include "derivatives/*" --include "participants.*" \
  --include "dataset_description.json" --include "README" --include "sub-*/eeg/*.json" \
  --include "sub-*/eeg/*_channels.tsv"
```

Plain HTTPS works too (`https://s3.amazonaws.com/openneuro.org/ds004504/<path>`), which is what
`sources.download` can use without the AWS CLI. The S3 bucket holds the current state of the
dataset, which today is 1.0.9; for a pinned version use DataLad
(`datalad install https://github.com/OpenNeuroDatasets/ds004504.git` then
`git checkout 1.0.9`) or `openneuro-py download --dataset ds004504 --tag 1.0.9`. The DataLad and
openneuro-py routes were not executed in this session ⚠.

**Citation required** (`HowToAcknowledge`): the data descriptor (10.3390/data8060095), the first
study (10.1109/ACCESS.2023.3294618) and the AHEPA benchmark (10.1007/s11571-026-10464-w).

---

## 2. Alzheimer's disease GWAS: Bellenguez et al. 2022 ✓

**What it is.** Stage-I genome-wide meta-analysis of Alzheimer's disease and related dementias
(European ancestry; EADB, GR@CE, EADI, GERAD, PERADES, DemGene, Bonn, RS, CCHS, UK Biobank).

| Field | Value | Status | Where it was read |
|---|---|---|---|
| Paper | Bellenguez C et al. "New insights into the genetic etiology of Alzheimer's disease and related dementias." *Nat Genet* 2022 Apr | ✓ | Crossref, PubMed esummary |
| Paper DOI / PMID / PMCID | 10.1038/s41588-022-01024-z / 35379992 / PMC9005347; article license CC BY 4.0 | ✓ | Crossref, PubMed |
| GWAS Catalog accession | GCST90027158 (`pubmed_id` 35379992, trait MONDO_0004975) | ✓ | `https://www.ebi.ac.uk/gwas/rest/api/v2/studies?pubmed_id=35379992` |
| Sample | 39,106 clinically diagnosed cases, 46,828 proxy cases, 401,577 controls (European); 21,101,114 variants | ✓ | GWAS Catalog v2 study record |
| License (summary stats) | CC0 1.0 (`terms_of_license`: https://creativecommons.org/publicdomain/zero/1.0/) | ✓ | GWAS Catalog v2 study record |
| Access | open, anonymous FTP/HTTPS | ✓ | directory listing |
| Folder | https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90027001-GCST90028000/GCST90027158/ | ✓ | listing |
| Genome build | GRCh38 (`genome_assembly: GRCh38`, 1-based) | ✓ | `*-meta.yaml` |

Files in the folder (sizes from HTTP `Content-Range`, md5 from `md5sum.txt`; no sha256 is
published):

| File | Bytes | md5 | Notes |
|---|---|---|---|
| `harmonised/35379992-GCST90027158-MONDO_0004975.h.tsv.gz` | 1,140,755,970 | `e2c8be73b5aa7698c5e8878ae607fe85` | **chosen**: harmonised to Ensembl release 95, `is_harmonised: true` |
| `harmonised/35379992-GCST90027158-MONDO_0004975-Build38.f.tsv.gz` | 575,146,963 | `cfd56a310f0ec11846354ddf8a9c5152` | formatted, not harmonised |
| `GCST90027158_buildGRCh38.tsv.gz` | 755,201,909 | `9d23b9ba23532da38ab83fb061bab18f` | author file, 2025-02-09, `is_harmonised: false` |

Header of the harmonised file (read from the first 16 KB):

```
hm_variant_id hm_rsid hm_chrom hm_pos hm_other_allele hm_effect_allele hm_beta hm_odds_ratio
hm_ci_lower hm_ci_upper hm_effect_allele_frequency hm_code variant_id p_value chromosome
base_pair_location effect_allele other_allele effect_allele_frequency odds_ratio ci_lower
ci_upper beta standard_error n_cas n_con
```

Column mapping used by `sources.load_outcome_gwas()`:

| Ours | File column |
|---|---|
| `snp` | `hm_rsid` |
| `effect_allele` | `hm_effect_allele` |
| `other_allele` | `hm_other_allele` |
| `eaf` | `hm_effect_allele_frequency` |
| `beta` | `hm_beta` (log-odds) |
| `se` | `standard_error` (no harmonised SE; SE does not change sign under allele flipping) |
| `pval` | `p_value` |

Parsing notes, both seen in the bytes read: the harmonised file has Windows line endings (the
last field reads `n_con\r`), so strip `\r`; for flipped variants `hm_beta` has the opposite sign
of `beta` (example `rs1040232850`: `beta` −0.2494, `hm_beta` 0.2494).

**Caveat for MR: sample overlap.** The cohort list includes UK Biobank (proxy cases). The
pQTL and mQTL sources below are also UK Biobank based, so exposure and outcome samples overlap
in part. Overlap biases two-sample MR towards the confounded observational estimate, with a
size that depends on instrument strength (Burgess, Davies & Thompson, *Genet Epidemiol* 2016,
DOI 10.1002/gepi.21998 ✓). It must be stated, and a UK-Biobank-free outcome should be run as
sensitivity:

**Sensitivity outcome: Kunkle et al. 2019 ⚠.** *Nat Genet*, DOI 10.1038/s41588-019-0358-2,
PMID 30820047 ✓; GWAS Catalog GCST007511 (21,982 cases, 41,944 controls, European, no UK
Biobank), license "EBI terms of use" ✓. File
`https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST007001-GCST008000/GCST007511/Kunkle_etal_Stage1_results.txt`,
569,380,789 bytes, space-separated, header
`Chromosome Position MarkerName Effect_allele Non_Effect_allele Beta SE Pvalue` ✓. It has **no
allele frequency**, so palindromic SNPs cannot be resolved from the file and must be dropped.
Build not stated in the folder (no meta file) ⚠: positions look like GRCh37 but this was not
confirmed. Wightman et al. 2021 (PMID 34493870, GCST013196/GCST013197) also exists in the Catalog
but includes UK Biobank as well, so it does not solve the overlap.

---

## 3. pQTL exposures: UKB-PPP (Sun et al. 2023), Supplementary Table 9 ✓

**Choice and why.** It is the lightest open file that has, for each lead pQTL, rsID, both
alleles, allele frequency, beta, SE, p and a cis/trans flag, on one platform (Olink Explore) and
one cohort, under CC BY 4.0. Per-protein full summary statistics (many GB) are not needed for
lead-variant MR.

| Field | Value | Status |
|---|---|---|
| Paper | Sun BB et al. "Plasma proteomic associations with genetics and health in the UK Biobank." *Nature* 2023 | ✓ Crossref |
| DOI / PMID / PMCID | 10.1038/s41586-023-06592-6 / 37794186 / PMC10567551; CC BY 4.0 | ✓ |
| File | `41586_2023_6592_MOESM3_ESM.xlsx` | ✓ |
| URL | https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06592-6/MediaObjects/41586_2023_6592_MOESM3_ESM.xlsx | ✓ HTTP 200 |
| Size | 34,411,394 bytes | ✓ Content-Length |
| sha256 | not computed (file read with byte-range requests, not saved) | ✗ fill in on first download |
| Sheet | `ST9` "List of significant (p<1.7e-11) pQTLs in discovery cohort"; header on row 5 | ✓ |
| Rows | sheet range A1:AF30748, so about 30.7k pQTL rows | ⚠ from the stored range, not counted |
| Genome build | GRCh37 inside `Variant ID`, GRCh38 in `GENPOS (hg38)` | ✓ checked rs1260326 against Ensembl |

Relevant columns: `Variant ID (CHROM:GENPOS (hg37):A0:A1:imp:v1)`, `GENPOS (hg38)`,
`UKBPPP ProteinID`, `Assay Target`, `Target UniProt`, `rsID`, `A1FREQ (discovery)`,
`BETA (discovery, wrt. A1)`, `SE (discovery)`, `log10(p) (discovery)`, `cis/trans`, `cis gene`.

| Ours | File column |
|---|---|
| `trait` | `Assay Target` |
| `snp` | `rsID` |
| `effect_allele` | A1 = 4th field of `Variant ID` |
| `other_allele` | A0 = 3rd field of `Variant ID` |
| `eaf` | `A1FREQ (discovery)` |
| `beta` | `BETA (discovery, wrt. A1)` (per SD of NPX) |
| `se` | `SE (discovery)` |
| `pval` | `10 ** -log10(p) (discovery)`: despite the name the column holds −log10(p) (first row 79.24) |
| `cis` | `cis/trans == "cis"` |

**Light alternate and test fixture source: Zheng et al. 2020** (*Nat Genet*, "Phenome-wide
Mendelian randomization mapping the influence of the plasma proteome on complex diseases", DOI
10.1038/s41588-020-0682-6, PMID 32895551, PMC7610464 ✓). `41588_2020_682_MOESM3_ESM.xlsx`
(2,990,385 bytes, sha256 `0812c04ca38150e67d6eae1bde3863331ffd61fc0f7634e1120780edd8328a14` ✓),
sheet `TableS1.Instruments`, header row 3, 2,113 instruments for 1,699 proteins (797 cis),
columns `SNP`, `Effect_allele`, `Other_allele`, `Effect_allele_freq`, `Beta`, `SE`, `P_value`,
`Exposure`, `Study`, `Type`, `Cis/Trans`; GRCh37 ✓. Not chosen as primary: instruments are pooled
from five studies (rows from Sun, Emilsson, Suhre, Yao, Folkersen) and the article is under Nature
terms of use, not CC ⚠. Useful as a known-answer check: its Table S7 reports CD33 (cis instrument
rs12459419) against Alzheimer's disease with beta 0.100148, SE 0.018546, colocalised ✓.

**Rejected: deCODE (Ferkingstad et al. 2021, DOI 10.1038/s41588-021-00978-w, PMID 34857953)**:
not open access according to Europe PMC, and the supplementary workbook is 44.4 MB.

---

## 4. mQTL exposures: Karjalainen et al. 2024, Supplementary Table 5 ✓

**Choice and why.** NMR metabolomics on the Nightingale platform, the same technology as part of
the ADNI metabolomics, CC BY 4.0, and the lead-variant table has every field we need.

| Field | Value | Status |
|---|---|---|
| Paper | Karjalainen MK et al. "Genome-wide characterization of circulating metabolic biomarkers." *Nature* 2024 | ✓ Crossref |
| DOI / PMID / PMCID | 10.1038/s41586-024-07148-y / 38448586 / PMC10990933; CC BY 4.0 | ✓ |
| File | `41586_2024_7148_MOESM6_ESM.xlsx` | ✓ |
| URL | https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07148-y/MediaObjects/41586_2024_7148_MOESM6_ESM.xlsx | ✓ HTTP 200 |
| Size | 24,669,233 bytes | ✓ Content-Length |
| sha256 | not computed | ✗ fill in on first download |
| Sheet | `TableS5_all_lead_SNP_assoc`, header row 2; trait definitions in `TableS2_NMR_traits` (233 NMR traits) | ✓ |
| Rows | range A1:BC8797, about 8,795 lead SNP-trait rows | ⚠ from the stored range |
| Genome build | GRCh37 (caption: "Genomic positions refer to hg19. Effects are given for allele 1.") | ✓ |
| Full summary statistics | GWAS Catalog GCST90301941 to GCST90302173, per the data availability statement | ⚠ Catalog not queried |

| Ours | File column |
|---|---|
| `trait` | `Metabolite` (short codes such as `Ala`) |
| `snp` | `Rsid` |
| `effect_allele` | `Allele1` (lower case: upper-case it) |
| `other_allele` | `Allele2` (lower case) |
| `eaf` | `Freq1` |
| `beta` | `Effect` |
| `se` | `StdErr` |
| `pval` | `P-value` |

There is no cis/trans flag (not meaningful for most metabolites), so the cis-only rule of Task 8
applies to proteins only. Whether the meta-analysis includes UK Biobank participants was not
checked ⚠; assume overlap with the outcome GWAS until checked.

**Not used: OpenGWAS.** Most endpoints of https://api.opengwas.io/api/ now need a token ("The
token is required for most endpoints") ✓, so it is not a zero-setup route.

---

## 5. ADNI

**What it is.** Alzheimer's Disease Neuroimaging Initiative: longitudinal multi-site cohort (CN,
MCI, AD) with clinical, cognitive, imaging, CSF/plasma, genetic and omics data, distributed by
LONI IDA after an approved application.

| Field | Value | Status |
|---|---|---|
| URL | https://adni.loni.usc.edu (data through https://ida.loni.usc.edu) | ✓ |
| Access | Data Use Agreement (DUA); the owner's access is active | ✓ |
| Current DUA | https://adni.loni.usc.edu/wp-content/themes/adni_2023/documents/ADNI_Data_Use_Agreement.pdf (the "ADNI Data Use Agreement" link in the site footer; no revision date printed; includes an Appendix A on AI tools) | ✓ |
| Older DUA / policy | https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Data_Use_Agreement.pdf (Rev December 20, 2016); https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_DSP_Policy.pdf (Rev April 3, 2024) | ✓ |
| Citation guide | https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Manuscript_Citations.pdf (Rev January 28, 2016; its funding text is out of date, use the DUA text) | ✓ |
| Tables to download | [`docs/ADNI_DOWNLOAD_LIST.md`](ADNI_DOWNLOAD_LIST.md) | ⚠ file names partly from third-party code |

### 5.1 What the DUA says (quotes from the current DUA unless noted)

**Individual-level data: redistribution is forbidden ✓.**
> "I will not further disclose these data beyond the uses outlined in this agreement and my data
> use application. I understand that redistribution of individual participant-level data in any
> manner is prohibited. Reports and publication of summary (not participant-level) data are
> allowed." (term 3)

> "I will not disclose any participant-level raw or derived datasets beyond the uses outlined in
> this agreement. ... I understand derived datasets should only be shared through the ADNI-LONI
> and/or other NIA-designated data repository." (term 5)

> "I understand that I am not allowed to distribute ADNI data outside of my team, only the ADNI
> study website operated by LONI-IDA can distribute data." (term 6)

**Review by the Data and Publications Committee (DPC) ✓: required for manuscripts.**
> "I will submit all manuscripts to the ADNI Data and Publications Committee (DPC) prior to
> submitting them to a journal. This review will not be a scientific review, but it is intended to
> ensure that the items above are correctly implemented. ... will complete its review within 2
> weeks."

The DSP Policy adds: "The ADNI DPC will not attempt to evaluate the scientific merit of proposed
analyses, nor evaluate proposed abstracts. The DPC will be charged with the administrative review
of manuscripts only." Submission route: IDA "My Account" → "Update" on the ADNI project →
"Publication Update" tab, then e-mail the manuscript to the DPC (address in the DUA).

**Trained models ✓ (Appendix A).**
> "investigators who intend on making the weights of a trained model publicly available should
> consider whether the model could be used to easily reconstruct parts of the ADNI data set. ADNI
> prohibits release of participant level data, even if it is sufficiently processed or altered
> and lack ADNI subject codes."

**AI tools ✓ (term 4, Appendix A).** ADNI data may not be put into third-party AI tools that keep
or reuse inputs, including "inclusion of ADNI data as part of an input or 'prompt' for a
generative language model"; "opt-out" clauses are not "an acceptable safeguard". The exception is
tools "developed and internally trained by individual research groups ... strictly used for the
individual research group" with guaranteed containment.

**Data version ✓.** "I will note the version (download date) of the data I download and provide
this information in any Methods section".

**Small cells ✗.** No minimum cell size is stated in the DUA, the DSP Policy or the citation
guide. The agent's refusal below N = 10 is this project's own rule.

### 5.2 Answer: can ADNI aggregates go on a public GitHub repository without DPC review? ⚠

- **Summary results are explicitly allowed** (term 3: "Reports and publication of summary (not
  participant-level) data are allowed").
- **DPC pre-review is required for "manuscripts" before journal submission.** No clause mentions
  websites, code repositories or software, so the DUA neither requires nor waives DPC review for a
  GitHub README with aggregate tables and figures. This is an inference from silence, not an
  explicit permission.
- **Individual-level data, raw or derived, must never be published**, and this extends to
  anything that could reconstruct it: per-subject out-of-fold predictions and trained model
  files (a random survival forest stores training samples in its leaves).

Conservative route if the owner sets `publish_adni_aggregates: true`: publish only aggregate
metrics and figures (no per-subject points), with the acknowledgement text and the ADNIMERGE
download date in the README; keep `models/adni/` and `conversion_predictions.parquet` local;
optionally notify the DPC through the IDA "Publication Update" route; any later manuscript goes
through the DPC as required.

**Owner decision (2026-11-12): the conservative route.** ADNI conversion results will be
published as aggregate tables and figures only, with the acknowledgement below and the
ADNIMERGE download date; models and per-participant predictions stay local.
`publish_adni_aggregates` is set to `true` in the same commit that adds the first ADNI
aggregates.

### 5.3 Required acknowledgement text ✓ (current DUA)

By-line: "for the Alzheimer's Disease Neuroimaging Initiative*", with the footnote:

> *Data used in preparation of this article were obtained from the Alzheimer's Disease
> Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within
> the ADNI contributed to the design and implementation of ADNI and/or provided data but did not
> participate in the analysis or writing of this report. A complete listing of ADNI investigators
> can be found at:
> http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf

Methods:

> Data used in the preparation of this article were obtained from the Alzheimer's Disease
> Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). The ADNI was launched in 2003 as a
> public-private partnership, led by Principal Investigator Michael W. Weiner, MD. The original
> goal of ADNI was to test whether serial magnetic resonance imaging (MRI), positron emission
> tomography (PET), other biological markers, and clinical and neuropsychological assessment can
> be combined to measure the progression of mild cognitive impairment (MCI) and early Alzheimer's
> disease (AD). The current goals include validating biomarkers for clinical trials, improving the
> generalizability of ADNI data by increasing diversity in the participant cohort, and to provide
> data concerning the diagnosis and progression of Alzheimer's disease to the scientific
> community. For up-to-date information, see adni.loni.usc.edu.

Funding:

> Data collection and sharing for the Alzheimer's Disease Neuroimaging Initiative (ADNI) is funded
> by the National Institute on Aging (National Institutes of Health Grant U19AG024904). The grantee
> organization is the Northern California Institute for Research and Education. In the past, ADNI
> has also received funding from the National Institute of Biomedical Imaging and Bioengineering,
> the Canadian Institutes of Health Research, and private sector contributions through the
> Foundation for the National Institutes of Health (FNIH) including generous contributions from
> the following: AbbVie, Alzheimer's Association; Alzheimer's Drug Discovery Foundation; Araclon
> Biotech; BioClinica, Inc.; Biogen; Bristol-Myers Squibb Company; CereSpir, Inc.; Cogstate; Eisai
> Inc.; Elan Pharmaceuticals, Inc.; Eli Lilly and Company; EuroImmun; F. Hoffmann-La Roche Ltd and
> its affiliated company Genentech, Inc.; Fujirebio; GE Healthcare; IXICO Ltd.; Janssen Alzheimer
> Immunotherapy Research & Development, LLC.; Johnson & Johnson Pharmaceutical Research &
> Development LLC.; Lumosity; Lundbeck; Merck & Co., Inc.; Meso Scale Diagnostics, LLC.; NeuroRx
> Research; Neurotrack Technologies; Novartis Pharmaceuticals Corporation; Pfizer Inc.; Piramal
> Imaging; Servier; Takeda Pharmaceutical Company; and Transition Therapeutics.

If ADMC metabolomics (Nightingale, Biocrates) is used: add the by-line "for the Alzheimer's
Disease Metabolomics Consortium**", whose footnote begins "**Data used in preparation of this
article were generated by the Alzheimer's Disease Metabolomics Consortium (ADMC)." (full text in
the DUA; team list https://sites.duke.edu/adnimetab/team/), and the funding sentence "Data
collection and sharing for this project was funded by the Alzheimer's Disease Metabolomics
Consortium (National Institute on Aging R01AG046171, RF1AG051550 and 3U01AG024904-09S4)."

### 5.4 Consequences for this repository

- ADNI rows are never read by any cloud model, including the coding assistant used to build this
  repo; only the local agent (Ollama) touches ADNI tables, which is the "internally contained"
  case of Appendix A. The MCP server refuses ADNI mode unless explicitly allowed for a local
  client.
- `models/adni/` and per-subject predictions stay in gitignored folders even if aggregates are
  published.
- Provenance blocks record the download date of each ADNI table.
- `DX_bl` also takes the value `MCI` for ADNI3 recruits (see the download list), so the MCI set is
  `{"EMCI", "LMCI", "MCI"}`.

---

## 6. Comparison literature

### 6.1 MCI-to-AD conversion on ADNI with survival models

All three are PubMed records read on 2026-11-12; numbers are quoted from the abstracts.

| Paper | Design | Reported performance | Status |
|---|---|---|---|
| Wang M, Chekouo T, Ismail Z, et al. "Elicited clinician knowledge did not improve dementia risk prediction in individuals with mild cognitive impairment." *J Clin Epidemiol* 2023;158:111-118. DOI 10.1016/j.jclinepi.2023.03.009, PMID 36931477 | ADNI MCI, n = 365, Bayesian Cox, 10-fold CV | c-index 0.74 (95% CI 0.70-0.79), ICI 0.06, IBS 0.17; "similar to the model without expert knowledge" | ✓ |
| Saeed A, Waris A, Fuwad A, et al. "Random survival forest model for early prediction of Alzheimer's disease conversion in early and late Mild cognitive impairment stages." *PLoS One* 2024;19(12):e0314725. DOI 10.1371/journal.pone.0314725, PMID 39671432 | ADNI, 291 EMCI + 546 LMCI; RSF, extra survival trees, GB, Cox-net, CoxPH | multimodal RSF C-index 0.90 (EMCI) and 0.82 (LMCI); IBS 0.10 and 0.16 | ✓ (validation scheme not stated in the abstract ⚠) |
| Zhou B, Nakatani E, Teramukai S, et al. "Risk classification in mild cognitive impairment patients for developing Alzheimer's disease." *J Alzheimers Dis* 2012;30(2):367-375. DOI 10.3233/JAD-2012-112117, PMID 22426014 | ADNI MCI, n = 397, Cox + bootstrap | c-index 0.68 for the ADAS13 + CDR-SB combination | ✓ |

The spread (0.68 to 0.90) is the expected one: simple, internally cross-validated Cox models sit
near 0.70-0.75, and the higher numbers come from richer feature sets. Our baselines should be
read against Wang 2023, which is the closest in design (MCI, Cox, CV, calibration reported).

### 6.2 Classification on ds004504

| Paper | Split | Reported performance | Status |
|---|---|---|---|
| Miltiadous A, Gionanidis E, Tzimourta KD, et al. "DICE-Net: A Novel Convolution-Transformer Architecture for Alzheimer Detection in EEG Signals." *IEEE Access* 2023. DOI 10.1109/ACCESS.2023.3294618 | leave-one-subject-out (subject level) | AD vs CN accuracy 83.28% | ✓ abstract via Semantic Scholar |
| Mlinarič T, Van Den Kerchove A, Barinaga ZI, Van Hulle MM. "EEG-based classification of alzheimer's disease and frontotemporal dementia using functional connectivity." *Sci Rep* 2026;16:4903. DOI 10.1038/s41598-026-35316-9, PMID 41513739 (erratum 10.1038/s41598-026-49348-8) | leave-one-subject-out | stacked model ROC-AUC 81.80% AD vs CN, 71.36% FTD vs CN, 65.10% AD vs FTD | ✓ PubMed abstract |
| Zhang P, Tong W, Jiang L, et al. "Electroencephalography-based Alzheimer's disease detection using multi-scale information retention transform: a subject-level comparison of machine learning classifiers." *Front Neurol* 2026;17:1869211. DOI 10.3389/fneur.2026.1869211, PMID 42876466 | participant-level CV, AD vs CN only (36 vs 29) | random forest accuracy 86.15%, AUC 0.9004 (plain concatenation 84.62%, AUC 0.8956; difference not significant) | ✓ PubMed abstract |
| Miltiadous A, Ntetska A, Aspiotis V, et al. "The AHEPA EEG benchmark: setting the standard for machine learning in dementia diagnosis, a scoping review." *Cogn Neurodyn* 2026;20(1). DOI 10.1007/s11571-026-10464-w, PMID 42165009, PMC13184051 | review of 46 studies in three validity tiers | mean accuracy of the 12 subject-level ("Validity 1") studies: AD vs CN 82.11%, FTD vs CN 75.18%, AD vs FTD 71.44%, 3-class 69.99%; mean over all studies 90.81%, 86.53%, 88.97%, 84.29% | ✓ text; Table 3 disagrees with the text for AD vs FTD and 3-class ⚠ |

The benchmark is the key reference: it names data leakage (epochs of one subject in both train
and test) as the reason why the all-study averages are about 9 to 17 points above the
subject-level ones. Our subject-level results should be compared only with the "Validity 1"
numbers and the LOSO papers above. Its Standardized Reporting Checklist is in the second half of
the article, which was not read in this session ⚠.
