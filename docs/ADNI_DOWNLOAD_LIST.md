# ADNI download list

Tables the owner downloads by hand from LONI IDA (https://ida.loni.usc.edu) into
`DEMENTIA_ML_DATA/adni/` (default `~/data/dementia-ml/adni/`), never
inside the repository. Prepared on 2026-11-12 from public ADNI pages; IDA itself was not opened.
Rules that apply to these files are in [`DATA_SOURCES.md`](DATA_SOURCES.md), section 5.

Status marks: ✓ verified on a public ADNI page or document, ⚠ name or location from secondary
sources (forum, third-party code) and to be confirmed on IDA, ✗ not found.

**For every file:** keep the original file name, write the download date in
`DEMENTIA_ML_DATA/adni/DOWNLOADED.txt` (`<file name> <YYYY-MM-DD>`): the DUA requires the
download date to be reported in Methods, and the pipelines copy it into provenance.

## Where things are in IDA ✓

From the ADNI quick-start guide (https://adni.loni.usc.edu/quick-start-guide-asset/getting_started.html
and `filetour.html`): IDA → **Search and Download** → **Study Files** (also reachable through the
ARC builder). Inside Study Files:

- **Study Info**: "Data and database contains a master data dictionary ... This section also
  includes the necessary files for building the ADNIMERGE package."
- **Biospecimen → Biospecimen Results**: laboratory, proteomics and metabolomics tables.
- **Archive**: "previous versions of many tables".

Older ADNI news posts give the same path as "DOWNLOAD | Biospecimen | Biospecimen Results".

## Phase 1 (needed now)

> **Update 2026-11-13.** IDA no longer offers `ADNIMERGE.csv`. Under Study Info there is only
> the R package **ADNIMERGE2** (`ADNIMERGE2.tar.gz`, ADNI 1/GO/2/3/4), with CDISC-style tables
> (`ADSL`, `ADRS`, `ADQS`, plus the source tables). `pipelines/prepare_adni.py` builds the
> ADNIMERGE-shaped table from it in Python (pyreadr; R is not needed). Also downloaded:
> `UPENNBIOMK_ROCHE_ELECSYS`, `BIOMARK`, the data dictionary `DATADIC`, the CruchagaLab
> SomaScan 7k CSF matrix with its analyte file, and `ADNINIGHTINGALE2` with its dictionary.

| # | File | Where in IDA | Use | Status |
|---|---|---|---|---|
| 1 | `ADNIMERGE.csv` ("ADNIMERGE – Key ADNI tables merged into one table [ADNI1,GO,2,3]") | Study Files → Study Info (Data & Database). If it is not there, look under Study Files → Archive | MCI-to-AD conversion (Task 3, 5), trajectories (Task 6) | ✓ name and existence (ADNI FAQ https://adni.loni.usc.edu/help-faqs/faq/; adni-data group post of 9 Dec 2024 "ADNIMERGE.csv is still available" under "Study Info"); ⚠ a post of 13 Jan 2026 may report a move to Archive, body not readable |
| 2 | `ADNIMERGE_0.0.1.tar.gz` (R package, same data) | Study Files → Study Info | fallback only, if the CSV is gone; data in `data/adnimerge.rdata` must be exported with R | ✓ name from the package site https://adni.bitbucket.io/index.html |
| 3 | Data dictionary (`DATADIC` / master data dictionary) | Study Files → Study Info | column definitions, units, codes | ✓ existence; ⚠ exact file name |
| 4 | `DXSUM_PDXCONV_ADNIALL.csv` (diagnostic summary) | Study Files → Assessments → Diagnosis | cross-check of `DX` and of the conversion date | ⚠ name from ADNI expert Q&A (https://adni.loni.usc.edu/experts-knowledge-base/question?QID=1756) |

Notes on ADNIMERGE:

- It covers ADNI1, GO, 2 and 3, **not ADNI4** ✓ (FAQ tag "[ADNI1,GO,2,3]"). A successor
  "including data from ADNI4" is in development ✓ (expert Q&A QID=2019, 29 Oct 2024). An R
  package called ADNIMERGE2 (v0.1.4, https://atri-biostats.github.io/ADNIMERGE2) is described as
  available on IDA under Study Files → Study Info → Data & Database ⚠; it is not one merged table
  but CDISC-style datasets (`ADSL`, `ADQS`, `ADRS`, ...; subject key `USUBJID`; diagnosis as
  `RSTESTCD == "DX"` with `CN`/`MCI`/`DEM`). Phase 1 uses `ADNIMERGE.csv`; ADNIMERGE2 only if the
  CSV is unavailable, and then `schema.py` changes.
- No official notice that ADNIMERGE is retired was found ⚠.

### Expected ADNIMERGE columns

Source: the ADNIMERGE R package reference, https://adni.bitbucket.io/reference/adnimerge.html
("16378 rows and 115 variables" in that release), and public code that reads `ADNIMERGE.csv`.
To be confirmed on the real file (Task 0, Step 3 gate).

| Column | Meaning (package reference) | Status |
|---|---|---|
| `RID` | participant roster id | ✓ |
| `VISCODE` | visit code (`bl`, `m06`, ...) | ✓ |
| `EXAMDATE` | exam date | ✓ |
| `M` | "Month since baseline" | ✓ (a `Month` column also exists) |
| `DX_bl` | baseline diagnosis; the R package spells it `DX.bl`, the CSV `DX_bl` | ✓ |
| `DX` | diagnosis at the visit | ✓ |
| `AGE` | age at baseline | ✓ |
| `PTGENDER` | sex | ✓ |
| `PTEDUCAT` | years of education | ✓ |
| `APOE4` | number of APOE ε4 alleles | ✓ |
| `MMSE` | Mini-Mental State Examination | ✓ |
| `ADAS13` | ADAS-Cog 13 | ✓ |
| `CDRSB` | CDR sum of boxes | ✓ |
| `FAQ` | Functional Activities Questionnaire | ✓ |
| `Hippocampus` | "UCSF Hippocampus" volume | ✓ |
| `ICV` | "UCSF ICV" | ✓ |
| `ABETA` | CSF Aβ | ✓ |
| `TAU` | CSF total tau | ✓ |
| `PTAU` | CSF p-tau | ✓ |

All 19 are listed (status ✓ = the column name exists). The meanings quoted from the reference are `M`, `Hippocampus`, `ICV` and the "CSF ..." labels of `ABETA`/`TAU`/`PTAU`; the other meanings are the usual reading of the names ⚠, to check in the data dictionary. Other columns seen: `PTID`, `COLPROT`, `ORIGPROT`, `SITE`, `Years_bl`,
`Month_bl` and `*_bl` copies of the baseline values.

Bound strings in `ABETA`/`TAU`/`PTAU` (`">1700"`, `"<80"`, `"<8"`): no public documentation found
⚠. Check on the real file before relying on the loader's parsing.

### Diagnosis codes

| Column | Values | Status / source |
|---|---|---|
| `DX_bl` | `CN`, `SMC`, `EMCI`, `LMCI`, `AD`, **and `MCI`** for ADNI3 recruits | ✓ package vignette https://adni.bitbucket.io/articles/Enrollment.html; adni-data group, 5 May 2021; expert Q&A QID=1016: "DX.bl will now have the following categories (CN, SMC, EMCI, LMCI, MCI, AD; where MCI is E- or L- as recruited in ADNI3.)" |
| `DX` | `CN`, `MCI`, `Dementia` | ✓ same forum thread; vignette https://adni.bitbucket.io/articles/Longitudinal.html |

Consequences for the code:

- The MCI-at-baseline set must be `{"EMCI", "LMCI", "MCI"}`, not `{"EMCI", "LMCI"}` as in the
  current plan (Task 2 `schema.DX_BL_MCI`).
- `DX_bl` is the **screening** diagnosis (expert Q&A QID=304) ✓.
- For ADNI3 rows, `DX` was at one point filled only at `VISCODE == "bl"` (QID=1016) ✓. Check the
  share of missing `DX` per protocol (`COLPROT`) before building conversion times; if follow-up
  `DX` is missing in ADNI3, derive it from `DXSUM_PDXCONV_ADNIALL.csv` (`DXCHANGE` recode
  {1,7,9} = CN, {2,4,8} = MCI, {3,5,6} = Dementia ✓ from the same Q&A).

## Phase 2 (later): proteomics and metabolomics

All under **Study Files → Biospecimen → Biospecimen Results** ✓. Table titles are from the ADNI
biospecimen guide (https://adni.loni.usc.edu/quick-start-guide-asset/bioflood.html) and ADNI news
posts; exact CSV names must be read on IDA.

### Proteomics

| Table (title as on ADNI pages) | Matrix | Size (per guide) | Use | Status |
|---|---|---|---|---|
| CruchagaLab SomaScan 7k CSF proteomics, dataset `CruchagaLab_CSF_SOMAscan7k_Protein_matrix_postQC_20230620` | CSF | 735 participants, 7,008 analytes, ADNI1/GO/2/3 | main test of the MR protein candidates | ✓ title and dataset name (news, 8 Aug 2023, https://adni.loni.usc.edu/tag/biospecimen/page/2/) |
| Biomarker Consortium Plasma QC Multiplex (RBM) | plasma | 566 participants, 190 analytes, ADNI1 | plasma check of candidates | ✓ title; note: ID fields named `rid`/`visit_code` |
| Biomarker Consortium CSF QC Multiplex (RBM) | CSF | 311 participants, 159 analytes, ADNI1 | secondary | ✓ title |
| Emory University CSF Targeted MS ("SET2", 2023-07-12) | CSF | 833 participants, 69 analytes, long format | secondary | ✓ title |
| Biomarkers Consortium CSF Proteomics MRM Data | CSF | 287 participants, 320 analytes, GO/2 | optional | ✓ title |
| FNIH Biomarkers Consortium CSF Proteomics Project | CSF | 198 participants, 278 analytes, longitudinal | optional | ✓ title |
| Hu Lab CSF Inflammatory Proteins | CSF | 386 participants, 14 analytes | optional | ✓ title |
| Plasma SomaScan or Olink | plasma | none found | | ✗ |

Core fluid biomarkers (for adjustment and as reference): "UPENN – CSF Biomarkers Roche Elecsys"
(Aβ42, Aβ40, total tau, p-tau181) ✓; "UPENN – Plasma Biomarkers (AB42, AB40, ptau217, NfL,
GFAP)" (news, 10 Feb 2025) ✓; "ApoE Results" ✓.

### Metabolomics (ADMC; the ADMC by-line and funding text then apply)

| Table | File name (⚠ from third-party code, https://github.com/tomszar/metabo_adni `src/metabo_adni/data/load.py`) | Use | Status |
|---|---|---|---|
| ADMC Nightingale Platform NMR Post-Unblinding and Re-Analysis of Lipoproteins and Metabolites, version 2021-02-22 | `ADNINIGHTINGALE2.csv` | main test of the MR metabolite candidates (same NMR platform as the mQTL source) | ✓ title (ADNI news 2021); ⚠ file name |
| ADMC Nightingale Platform NMR Analysis of Lipoproteins and Metabolites Longitudinal (corrected version posted 7 May 2021; the 2021-01-29 version was withdrawn) | to read on IDA | longitudinal sensitivity | ✓ title; ⚠ file name |
| Biocrates p180, ADNI1 | `ADMCDUKEP180UPLC_01_15_16.csv`, `ADMCDUKEP180FIA_01_15_16.csv` | optional | ⚠ |
| Biocrates p180, ADNI2/GO | `ADMCDUKEP180UPLCADNI2GO.csv`, `ADMCDUKEP180FIAADNI2GO.csv` | optional | ⚠ |
| Fasting status | `BIOMARK.csv` (fields `RID`, `VISCODE2`, `BIFAST`) | covariate for metabolomics | ⚠ |
| Medication classes | `ADMCPATIENTDRUGCLASSES_20170512.csv` | covariate | ⚠ |
| CruchagaLab CSF metabolomics, `CruchagaLab_CSF_metabolomic_matrix_20230620` | to read on IDA | optional | ✓ title |
| ADMC Lipidomics Meikle Lab Longitudinal Data Matrix, version 2021-08-13 | to read on IDA | optional | ✓ title |

## Do not download now

Imaging (MRI/PET), genotypes and other assessments are outside Phase 1 and 2 as planned.
Some volumes for Phase 2 imaging are already in ADNIMERGE (`Hippocampus`, `ICV`, labelled "UCSF"); which FreeSurfer tables to add is decided in Phase 2.
