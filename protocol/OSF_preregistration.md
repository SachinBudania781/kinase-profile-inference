# Pre-registration: Kinase Profile Inference

**Predicting an unmeasured kinase-inhibitor response from two or three measured responses: a pre-registered five-model comparison on two lockboxes (A: seven panels; B: replication panels)**

> **Draft v0.2, prepared 2026-11-13.** v0.1 was never submitted. v0.2 adds the seven design upgrades (win/tie rule, robustness checks, plain baselines, T1/T2/T4 fixes, the 2% justification) and the second lockbox (Lockbox B).
> The only item in `[square brackets]` is set by OSF on submission. Items in `<angle brackets>` are hashes or links produced later. The data, configuration, fold and episode-manifest hashes below were computed on 2026-11-13 with `data_pipeline.py` (`hashes`, `episodes`, `lockbox`) and `sha256sum`; recompute them with the commands in the README before submitting and stop if any differs.
> Suggested OSF template: **OSF Preregistration** (sections below follow its headings: Study Information, Design Plan, Sampling Plan, Variables, Analysis Plan, Other).

---

## 0. Registration record

| Field | Value |
| --- | --- |
| Title | Kinase Profile Inference: predicting an unmeasured inhibitor response from two or three measured responses |
| Authors | Sachin Budania, IIT Bombay (any other contributors are added in the OSF contributors field) |
| Holder of the lockbox response tables | Sachin Budania (the author holds both lockbox tables and runs the final evaluator; there is no independent curator, so the "curator step" named in the code and README is run by the author) |
| GitHub repository | `https://github.com/SachinBudania781/kinase-profile-inference` |
| Protocol revision (full commit hash) | `<PROTOCOL_COMMIT_SHA_40_HEX>` |
| Study design PDF SHA-256 | `1517bcf91c2364fc97d7fb56d1c9016ccc7484fc9957b982b9cdf89272a4dd94` |
| Dataset roles manifest SHA-256 (`DATASETS.md`) | `a121a5ea5da68b2456881a9929afb39d038069eba1ed59422ef95e30182f2319` |
| Panel registry (`configs/panels.json`) SHA-256 | `1a01f3b8a458ba3648dca092adeb124598ef9a94c3ee2e4dbf7d8f6ea5372385` |
| Model settings at registration (`configs/models.json`) SHA-256 | `d4afbf22dae3204f3d8b97a74875af18397d2864d9d8a06af497a6dac2501eaa` |
| Cleaned development data SHA-256 | `7e362a53f43bfcdc825e92411c162e4e994dc9ef291dde8f6d306a79dbc4a7b5` |
| Lockbox A data SHA-256 (hash only; tables stay with the author) | `8c1e0505e2252cbf50ae35c09085aeaf748e6c6e715a906fb77ea9b79090c321` |
| Lockbox B data SHA-256 (hash only; tables stay with the author) | `370b8f8cc14b54781f458f30892aa4368abaa5818aa8143b390348073e2b0d90` |
| Fold assignment table SHA-256 | `8f09dc43a942e4f32cb64f6102e3c468d3643273234fa64a140676613c62541a` |
| Development episode manifest SHA-256 | `509f2338a1c590f9ea2ae95ff4d9b876cf2c5358b0bce585d0023c29c590a94b` |
| Lockbox A episode manifest SHA-256 | `360ac9cc2ea3cb8bf329a506b60a0c9d971d2f18b04476da62461f1d84da3b6f` |
| Lockbox B episode manifest SHA-256 | `6715936870aff7b06cac77825ccd9f52f1da700392f328e9f85751e5a1895031` |
| Registration date | `[set by OSF on submission]` |

---

## 1. Study information

### 1.1 Question

Can two or three measured inhibitor responses predict the response to another inhibitor for the same kinase variant in the same assay?

- **Unit of prediction:** a missing assay response.
- **Unit held out in evaluation:** the residue position (site).

### 1.2 Hypotheses and endpoint hierarchy

**One confirmatory primary hypothesis, on Lockbox A.**

- **H1 (primary, confirmatory).** SER has a positive equal-study-weighted mean relative MAE gain over the matched XGBoost reference across the seven Lockbox A panels (13 panel–budget configurations).
  - *Practical target and evidence criterion:* point estimate above **2%** and a 95% paired residue-site interval above **zero**.
  - *Stronger statement:* an interval entirely above 2% supports a gain above that threshold.
  - *Why 2%.* The threshold is a minimum difference judged worth acting on: an interval above zero alone would accept gains too small to change which method anyone would use. It is a pragmatic floor, not derived from a prior effect size or a power calculation. The 2% value is stated in the design document (hash above). It was fixed before any model was run on a lockbox panel.

**One replication hypothesis, on Lockbox B (does not change the H1 decision).**

- **H1-B (replication).** The same estimand (equal-study-weighted mean relative MAE gain of SER over XGBoost) computed on the three Lockbox B panels (6 panel–budget configurations; section 4.4).
  - *Reading rule, fixed now:* the result is called **supportive** if the point estimate is above zero, and **confirmatory** only if the H1 criterion (above 2% and 95% interval above zero) is met on B. Lockbox B has three panels and 27 residue sites in all, so its interval will be wide; a B result that is not confirmatory is not evidence against H1.
  - A and B are never pooled into one summary. H1 is decided on A only.

**Four secondary model comparisons.**

- **H2.** Each of SAER-7g, AlignedAddAudit, NestedAligned and CovarianceLocalMedian is compared with XGBoost on the same manifests (Lockbox A). If formal claims are made for these four overall contrasts, Holm adjustment is applied to that family of four.

**Headline translational objectives (pre-specified; see section 8).**
These are the biological questions the study is built to answer. They are fully specified in advance, reported in full whatever the outcome, and each is compared against XGBoost on identical episode lists.

- **T1. Hidden-resistance screening.** When two or three measured drugs say a variant is still sensitive, does the predicted response for another drug find the variants that turn out resistant, within a top-5% review list?
- **T2. Inhibitor choice and regret.** For a given variant and observed set, does predicted ranking pick an inhibitor whose measured response is close to the best available one (low regret)?
- **T3. Value of a third measurement.** For the same variant and query, how much does adding a third measured inhibitor reduce error or decision loss compared with two?
- **T4. Binding-mode strata.** Does prediction quality differ when the observed set contains an inhibitor of the same binding mode as the query versus none?

*Status of T1–T4.* The design document fixes one primary summary (H1). T1–T4 are registered as pre-specified secondary assay endpoints with exact rules and denominators, and they are the study's headline translational results. No multiplicity-adjusted confirmatory claim is made for them; they are reported with uncertainty and full denominators.

---

## 2. Disclosures (read before the rest)

1. **No model has been run on any lockbox panel.** The authors state that neither Lockbox A nor Lockbox B has been scored by any model and that no model result on either lockbox has been seen. The 2% threshold and every other rule in this registration were fixed before any lockbox scoring. The only looks at lockbox data are descriptive and are listed in item 2.
2. **What was looked at.** Lockbox B is a second lockbox, built to ask the same question on new studies of the same type. It was never scored by any model. Two descriptive looks were taken. (i) The author read the Lockbox B response tables while cleaning them and computing summary statistics (section 4.4). (ii) To choose panels that resemble Lockbox A, the author computed **descriptive summaries of the Lockbox A response tables** (sizes, censoring share, response median and spread, inter-inhibitor correlation, effective rank) with a script that fits no prediction model. No model was run on Lockbox A or Lockbox B for this purpose.
3. **Variant overlap between Lockbox B and other panels (by name, never the same measurement).** (i) All eight Villa single variants, and all twelve distinct mutations in Takei (G1202R, I1171N and the ten second mutations, as single substitutions), also appear in the development stress-test panel Oh ALK, a different assay on a different scale. (ii) Takei's background and second mutations (G1202R, I1171N, C1156Y, D1203N, E1210K, F1174L) also appear in the Lockbox A panels Zhang ALK and Gainor ALK, measured in other studies. (iii) Peng's G2032R and L2086F also appear in the development panel KiRHub (kinase-level residual activity). No lockbox panel is used to fit any cross-study component, and the pooled routers use aligned support features with no variant identity, so no model can look up a hidden value of these variants. The overlap does mean that A and B are not independent samples of variants: both contain the common ALK resistance mutations. Similar overlaps exist between Lockbox A and development panels (for example the MET and EGFR panels).
4. **Timing of development.** Prototype versions of SAER and SER existed before this registration. This registration fixes the question, datasets and evaluation contract; the **release freeze** (section 10) fixes algorithms, features, hyperparameters, artifacts, evaluator and output schema.
5. **Local co-location of data.** Development and Lockbox A cleaned tables currently sit in one local archive; Lockbox B sits in a separate folder next to the repository. Mitigation: lockbox tables are excluded from the repository and from development checkouts; lockbox episodes are written to a separate git-ignored location; the runner refuses lockbox roles without an explicit flag and a matching integrity record; only hashes are recorded. The author holds all tables (no independent curator), so the protection rests on the protocol, the recorded hashes and the runner checks above.
6. **Development work is exploratory.** Results on development panels, stress tests and deep-learning baselines inform design decisions and are not confirmatory.
7. **No blind dataset of the deep-mutational-scanning type exists for Lockbox B.** Lockbox B panels are small published IC50 tables, because most mutant-by-inhibitor potency data sit in figures or supplements. Lockbox B therefore tests the same estimand on new studies of the same type, not on a new data type.
8. **Suggested OSF "existing data" answer:** *registration following analysis of the data*, with this section as the explanation.

---

## 3. Design plan

| Item | Declaration |
| --- | --- |
| Study type | Computational benchmark on existing published datasets (no new data collection) |
| Blinding | None for development. Lockbox truth is joined to predictions only inside the scoring step, after predictions are saved by episode ID. Lockbox runs are one-shot per lockbox (a marker file blocks a second run) |
| Randomisation / seeds | No random assignment. All seeds fixed: folds and XGBoost 20261112, AlignedAddAudit residual head 20261111, NestedAligned residual head 20261116. Bootstrap seed 20261112 (fixed in the evaluator) |
| Software | Python and package versions locked at the release freeze (`requirements-lock.txt`; XGBoost version locked; one numerical thread) |

---

## 4. Sampling plan: datasets and splits

### 4.1 Archive inventory (59 folders) and Lockbox B

| Role | Count | Permitted use |
| --- | --- | --- |
| Core development panels | 6 | Baselines; development of routing and correction methods |
| Development stress tests | 4 | Small groups, sparse budgets, assay-specific failure modes; may inform decisions |
| **Lockbox A panels** | 7 | Final evaluation only; never used to fit cross-study components |
| Feasibility and data intake | 39 | Outside development and evaluation |
| Literature references | 3 | Literature audit only |
| **Lockbox B panels** (outside the 59 folders) | 3 (section 4.4) | Final evaluation only; never used to fit any component |

Folder-level roles are in `data_manifest/DATASETS.md` (hash above). The panel registry `configs/panels.json` (hash above) declares every role. Which development panels feed cross-study pooled routers and priors is fixed at the release freeze; the pool is development panels only (core development and stress tests).

### 4.2 Core development panels

| Panel | Role | Scale |
| --- | --- | --- |
| Estevam MET | Initial deep mutational scanning benchmark; exon 14 and exon 14-skipped backgrounds kept distinct | Source scale |
| Duong-Ly multiple kinases | Biochemical remaining activity across kinases | Source scale |
| Robichaux EGFR | Cellular inhibitor response ratios (IC50 ratios) | Source scale |
| Nakamura FGFR1/2/3 | Cellular IC50 with source grouping contract; gene-held-out comparisons | Source scale |
| KiRHub multiple kinases | Residual activity at 1 µM; observation-budget comparisons | Source scale |
| OKL multiple kinases | KINOMEscan residual binding at 1 µM | Source scale |

**Development stress tests:** Jung ABL1 (five inhibitors), Chakraborty SRC (four inhibitors), Lee EGFR (three inhibitors, two-observation test), Oh ALK (three inhibitors, two-observation test).

**Estevam MET companion data.** Audit that target and companion backgrounds align by substitution and residue without treating them as the same observation. Companion-background data are a distinct input with their own provenance and are admitted only for methods whose recorded input contract permits them.

### 4.3 Lockbox A: seven panels and 13 configurations

Declared cohort metadata only (no response values).

| Panel | Variants / sites / drugs | Endpoint | Drug order | Budget |
| --- | --- | --- | --- | --- |
| An EGFR | 199 / 131 / 5 | Published log₂ enrichment, Table 5 fast-proliferating cohort | afatinib, erlotinib, gefitinib, icotinib, osimertinib | 2, 3 |
| Abadie EGFR | 370 / 62 / 3 | PC-9 exon 20 day-15 log₂ drug-versus-vehicle contrast | CH7233163, osimertinib, sunvozertinib | 2 |
| Bahcall MET | 8 / 8 / 6 | TPR–MET cellular IC50, Ba/F3 | crizotinib, savolitinib, capmatinib, cabozantinib, glesatinib, merestinib | 2, 3 |
| Fujino MET | 26 / 12 / 7 | MET exon 14-skipping cellular IC50, reported limits retained | crizotinib, capmatinib, tepotinib, savolitinib, cabozantinib, merestinib, glesatinib | 2, 3 |
| Wylie ABL1 | 16 / 15 / 6 | BCR–ABL1 cellular IC50, ATP-site and allosteric inhibitors | imatinib, bosutinib, nilotinib, dasatinib, ponatinib, asciminib | 2, 3 |
| Zhang ALK | 16 / 12 / 4 | EML4–ALK proliferation IC50 | crizotinib, brigatinib, ceritinib, alectinib | 2, 3 |
| Gainor ALK | 10 / 8 / 5 | EML4–ALK phosphorylation IC50 | crizotinib, ceritinib, alectinib, brigatinib, lorlatinib | 2, 3 |

Abadie supports only `k = 2` because one of its three inhibitors is the query. Design total: 13 panel–budget configurations.

**Binding modes (source-defined, for T4):** An: afatinib and osimertinib covalent, erlotinib/gefitinib/icotinib reversible. Abadie: CH7233163 reversible, osimertinib and sunvozertinib covalent. Bahcall and Fujino: crizotinib, savolitinib, capmatinib, tepotinib type I; cabozantinib, glesatinib, merestinib type II. Wylie: asciminib allosteric, the other five ATP-site. Zhang and Gainor: all ATP-site.

**Expected episode counts** (all declared variants as complete profiles, no exclusions; check at manifest generation):

| Panel | k = 2 masks/variant → episodes | k = 3 masks/variant → episodes |
| --- | --- | --- |
| An EGFR | 30 → 5,970 | 20 → 3,980 |
| Abadie EGFR | 3 → 1,110 | not supported |
| Bahcall MET | 60 → 480 | 60 → 480 |
| Fujino MET | 105 → 2,730 | 140 → 3,640 |
| Wylie ABL1 | 60 → 960 | 60 → 960 |
| Zhang ALK | 12 → 192 | 4 → 64 |
| Gainor ALK | 30 → 300 | 20 → 200 |
| **Total** | **11,742** | **9,324** (grand total 21,066) |

Any departure from these counts is reported with its reason.

### 4.4 Lockbox B: replication panels

**Purpose.** The same question on new studies of the same type as Lockbox A (section 2, item 2). Lockbox B is a second, separate lockbox with its own hash, manifest and one-shot run. It is **not** part of the 59 archive folders.

**Selection criteria (written down; applied to every candidate).**

| # | Criterion | How it was checked |
| --- | --- | --- |
| C1 | Published study with openly readable full text; potency values (IC50) for each variant and inhibitor stated in a table of the paper itself, not only in a figure or supplement | Europe PMC full text; table transcribed by hand with cell references kept in `source_IC50.csv` |
| C2 | Not one of the 59 archive folders | Source DOI compared with every archive `study.json`; variant-name search of every archive table, including compressed ones; indirect overlap reported (section 2, item 3) |
| C3 | Same structure as the Lockbox A IC50 panels: one kinase gene (different fusion backgrounds of one gene are allowed, each scaled by its own wild-type row); a wild-type row in the same table; at least 8 variants with every inhibitor measured; at least 4 inhibitors; one assay for all cells | The builder refuses a panel without an exact native row, drops variants with a missing inhibitor value and records each drop with its reason |
| C4 | Same response scale and censoring convention as the Lockbox A IC50 panels | `log10(IC50_variant / IC50_WT)`; `>x` is a lower bound (+1), `<x` an upper bound (−1) |
| C5 | Descriptive similarity to Lockbox A | Six metrics compared with the Lockbox A range widened by 25% of the range: censored share, response median, response IQR, mean inter-inhibitor Spearman correlation, effective rank, first-component share (`lockbox_similarity.py`) |

**How the candidates were found.**

1. *Targeted searches* on resistance-mutation IC50 panels for ALK, ROS1, RET, MET, EGFR, ERBB2, FLT3, BTK and NTRK in Europe PMC open-access full text: 112 papers, 27 tables passed a first filter and were read by hand. This found Villa and Takei.
2. *Systematic sweep* of 744 open-access papers from per-kinase title queries (ALK, ROS1, RET, MET, EGFR, ERBB2, KIT, PDGFRA, FLT3, BRAF, BTK, JAK2, NTRK, FGFR, ABL1, SRC, CSF1R, AXL, DDR1 and others).
   - *Pass 1* (tables with IC50 and variant-style row labels): 5 tables.
   - *Pass 2* (any table with at least 8 rows, at least 5 columns, at least 6 variant-style row labels and mostly numeric cells; 742 papers, 5 not retrievable): 8 tables. One new panel (Peng) was admitted; the others are in the rejection table below.

Every table that passed a filter was read by hand against C1–C4. Most mutant-by-inhibitor potency data sit in figures or supplements, so the number of usable single-table panels is small. The search was done with a keyword and table-shape filter, so it can miss panels; it is not a claim that no other usable panel exists.

**Panels admitted.**

| Panel | Variants / sites / drugs | Endpoint | Drug order | Budget |
| --- | --- | --- | --- | --- |
| Villa ALK (npj Precision Oncology 2024, Table 2; DOI 10.1038/s41698-024-00498-w) | 9 / 8 / 7 | EML4::ALK Ba/F3 proliferation IC50, mean of three or more experiments | crizotinib, brigatinib, ceritinib, alectinib, lorlatinib, zotizalkib, repotrectinib | 2, 3 |
| Takei ALK (Pharmaceutical Research 2025, Table II; DOI 10.1007/s11095-025-03916-1) | 12 / 11 / 4 | EML4–ALK v1 Ba/F3 viability IC50; G1202R or I1171N alone and with a second mutation | alectinib, crizotinib, lorlatinib, brigatinib | 2, 3 |
| Peng ROS1 (Journal of Medicinal Chemistry 2024, Table 6; DOI 10.1021/acs.jmedchem.4c01205) | 14 / 8 / 4 | SLC34A2–ROS1 and CD74–ROS1 Ba/F3 proliferation IC50; each variant scaled by the wild-type row of its own fusion | brigatinib, lorlatinib, repotrectinib, 9o | 2, 3 |

Design total for Lockbox B: **3 panels, 6 panel–budget configurations, 35 variants, 27 residue sites**.

- **Villa.** Dropped: `L1196M + G1202R` (two inhibitors not available; values taken from an earlier paper). Nine of ten non-wild-type rows are admitted. `E1154K + F1174L` is a double mutant and enters under its first listed position (site 1154). Three cells are censored lower bounds. All seven inhibitors are ATP-site. Site rule: first listed position.
- **Takei.** All 12 non-wild-type rows admitted (G1202R, I1171N and ten compound mutants). Zotizalkib, gilteritinib and neladalkib are reported for only a few rows and are not used. Site rule: position of the added (last-listed) mutation, because the compound mutants share their background mutation; this gives 11 sites. Three cells are censored lower bounds (`>3000`). The compound mutants were selected from libraries by resistance to zotizalkib (G1202R background) or gilteritinib (I1171N background), so the panel is resistance-enriched. All four inhibitors are ATP-site.
- **Peng.** Two fusion backgrounds of one kinase domain; variant names carry an `@background` tag so the same substitution on two backgrounds stays two profiles that share one residue site. Site rule: position of the last-listed mutation (8 sites). Nine cells are censored lower bounds (`>2000`). Brigatinib, lorlatinib and repotrectinib are ATP-site; 9o is a proteolysis-targeting degrader and its anti-proliferative IC50 is used as that inhibitor's response; its mode is recorded as `degrader`. **Decision: the degrader is kept** as the fourth inhibitor. It was measured in the same assay on the same variants, and dropping it after the table was read would be a data-dependent choice. T4 reports it as its own stratum.
- **Expected episodes** (all admitted variants are complete profiles): Villa 945 (k = 2) and 1,260 (k = 3); Takei 144 and 48; Peng 168 and 56. Total 1,257 + 1,364 = **2,621**.
- T2 is defined at k = 2 for all three panels and at k = 3 for Villa only (Takei and Peng have one inhibitor left at k = 3). T4: Villa and Takei have one mode only (matching stratum only, no contrast); Peng has a degrader stratum.

**Candidates reviewed and not admitted.**

| Candidate | Reason |
| --- | --- |
| TPX-0131 ALK (PMC9398166, Murray 2021) | already an archive folder (`Murray2021_ALK`) |
| Neuroblastoma ALK review table (PMC8327881, Table 3) | compiled from several papers and assays; IC50 given as ranges with many empty cells |
| Reported lorlatinib-resistant ALK mutations (PMC8554663, Table 1) | a compiled review table of earlier work (not examined further) |
| RET pralsetinib roof mutations (PMC8184971) | one inhibitor only |
| TAS3351 EGFR (PMC13181001) | inhibition constants, not IC50 across inhibitors |
| JND4135 TRK (PMC9570838, Table 1) | three kinases (TRKA, TRKB, TRKC) with different residue numbers for the same structural position; fails C3 (one gene) and would split equivalent sites across folds. The closest alternative; could be added by amendment before the freeze |
| FLT3 activation-loop mutants (PMC5355235, Table 1) | no wild-type row (the reference line is FLT3-ITD); 44% of cells are bounds with several different limits |
| GZD824 FLT3 mutants (PMC7125355, Table 2) | mixed Ba/F3 backgrounds (tel-FLT3, FLT3-ITD), no single native row |
| Tarloxotinib EGFR exon 20 (PMC8107039, Tables 2–4) | no wild-type row; three tables with different inhibitor sets; one inhibitor is the prodrug of another |
| Ibrutinib EGFR isogenic lines (PMC5342513, Table 1) | several backgrounds, parental line as the only reference, many cells above the assay range |
| EGFR variant characterisation (PMC6768113, Table 1) | two or three inhibitors with most cells not determined |
| ROS1 CRISPR cell lines (PMC10910754, Table 1) | area over the curve, not IC50 |
| Zurletrectinib TRK (PMC11300601, Table 1) | three kinases (TRKA, TRKB, TRKC), same reason as JND4135 |
| HQP1351 KIT mutants (PMC6815454, Table 1) | one inhibitor, percent inhibition at two doses |
| Gene-screening table (PMC12738383, Table 2) | not an inhibitor panel (passed the table-shape filter only) |


**Similarity to Lockbox A (descriptive; response-level, no model).**

| Metric | Villa ALK | Takei ALK | Peng ROS1 | Lockbox A range (min–max) |
| --- | --- | --- | --- | --- |
| Censored share | 0.05 | 0.06 | 0.16 | 0.00–0.18 |
| Response median | 0.34 | 1.06 | 1.39 | −2.01–0.96 |
| Response IQR | 0.87 | 1.01 | 1.78 | 0.47–3.41 |
| Mean inter-inhibitor Spearman | **0.79** | 0.49 | **0.68** | 0.25–0.49 |
| Effective rank | 1.46 | 2.14 | 1.67 | 1.65–3.15 |
| First-component share | 0.82 | 0.62 | 0.75 | 0.44–0.76 |
| Inside the widened range on all six | no (correlation) | yes | no (correlation) | |

(Bold = outside the Lockbox A range widened by 25% of the range on each side.) Takei passes all six metrics. Villa and Peng each fail one: their inhibitors move together more than in Lockbox A (mean Spearman 0.79 and 0.68, against at most 0.49). Takei and Peng also have higher median responses than any Lockbox A panel (1.06 and 1.39, against at most 0.96); both are resistance-enriched sets, which is within the widened range. With 9–14 variants, a few strongly resistant variants drive these correlations. **The decision to keep panels that fail one metric was made after the results were seen; the metric screen had no stated exclusion rule beforehand.** It is disclosed and the panels are kept and flagged. A high inter-inhibitor correlation makes the task easier for every method alike and does not favour SER; it does mean Lockbox B is, as a set, more correlated than Lockbox A. **Decision: Villa and Peng are kept.**

**Lockbox B evaluation.** Same frozen code, same models and settings, same evaluator, run once after the release freeze (`--roles lockbox_B --score-lockbox`). Reported: MAE of every method and the gain over XGBoost per panel and budget, the equal-study summary over the three panels, the paired site interval (8 to 11 sites per panel, so the interval is coarse), the robustness table where defined (leave-one-study-out and the medians are computed on three studies only), and T2–T4 where defined. The "5 of 7" count is not reported for B. T1 is undefined for all three B panels (fewer than 20 eligible variants per mask).

---

## 5. Variables and data transformations

### 5.1 Response scale

- **Native-normalised IC50 panels** (Lockbox A: Bahcall, Fujino, Wylie, Zhang, Gainor; Lockbox B: Villa, Takei, Peng): `y_ij = log10(IC50_variant,ij / IC50_native,j)`, with the native reference taken from the declared assay. Reported limits are transformed on the same scale.
- **Source-scale panels:** log₂ enrichment (An), fixed-time log₂ contrast (Abadie), residual activity, residual binding and cellular fitness keep their source-defined scale. Absolute error is always reported in each panel's original units.

### 5.2 Censoring convention

| Flag c | Meaning | Interval loss at prediction p |
| --- | --- | --- |
| 0 | Exact value y | `abs(p − y)` |
| +1 | Lower bound L (response at least L) | `max(L − p, 0)` |
| −1 | Upper bound U (response at most U) | `max(p − U, 0)` |

Boundary error is the distance to the reported numeric value. The primary MAE uses reported boundaries. Interval error enters the SER gate and the evaluation.

### 5.3 Episode definition

`e = (study, i, q, S)`, `q ∉ S`, `|S| = k ∈ {2, 3}`; features `x_e = (q, S, {y_ij : j ∈ S}, permitted annotations)`; target `y_iq`. Masks are enumerated from a versioned inhibitor order, each with a stable episode ID. The same episode IDs, observed responses and hidden targets are used by every method within a panel and budget. The support-selection rule is recorded for any sampled development manifest. Replacing a hidden query response while keeping permitted inputs fixed must leave the prediction unchanged.

---

## 6. Residue grouping, folds and validation

- **Rule:** sort site IDs by a seeded SHA-256 digest of `"{seed}:{site}"` and distribute them cyclically over `min(5, number of sites)` folds. **Seed 20261112.** At least four sites are required. Every substitution at a residue stays in one fold.
- **Broader grouping contracts:** where a source defines a broader group (for example a gene), the whole group stays together. For Nakamura FGFR1/2/3 the minimum is three groups.
- **Outer test / inner validation.** For outer fold *o*, the other folds are outer training. Each inner fold is held out in turn, models are fit on the rest and predict the inner validation episodes. Allowed settings are chosen on aggregated inner validation error, then refit on outer training and used to predict the held-out fold. Predictions are saved by episode ID before truth is joined.
- **Auxiliary training for routed models.** Within the fitting partition, split groups into two halves, fit experts on one half and produce predictions and support audits on the other, then reverse. These supply auxiliary examples for the router and residual correction. Fit groups, auxiliary targets, validation groups and outer test groups stay traceable.
- **Study-specific fitting** uses only the permitted training positions of that study. Cross-study routers and priors use the declared development pool only (development panels other than the one being predicted). Folds with too few groups for a local fit are declared in the release.

---

## 7. Analysis plan

### 7.1 Models

| Model | Registered at design level |
| --- | --- |
| **XGBoost reference** | `reg:absoluteerror`; η = 0.055; L2 leaf penalty 4; seed 20261112; one thread; at most 36,000 episodes per fit (sample without replacement with the fixed seed, keep indices). Grid rows (max depth, trees, min child weight): (3, 200, 15), (5, 260, 15), (7, 300, 15). Row chosen per outer fold and budget by smallest total absolute error over inner validation episodes |
| **SAER-7g** | Experts: prior + residual median; Gaussian conditional transfer; conditional Gaussian mixture; donor transport; pairwise tree (median aggregation); pairwise tree (correlation-weighted aggregation); joint-support tree. Seven experts if the panel has at least 1,000 profiles or 10,000 observed cells, otherwise six. Audit features, softmax router fit by L-BFGS-B, local and pooled routers averaged |
| **AlignedAddAudit** | SAER routers; audit-scaled Laplace-mixture median (34 bisection iterations; multipliers {0.25, 1, 4}; blend τ ∈ {0, 0.25, 0.5, 1}); aligned residual head (histogram gradient boosting, absolute error, 300 iterations, learning rate 0.05, 15 leaves, 40 samples per leaf, L2 1, seed 20261111); λ ∈ {0, 0.25, 0.5, 0.75, 1} |
| **NestedAligned** | Aligned architecture with fit/validation separation; validation labels choose output transformations and blend weights only; seed 20261116; documented fallback when groups are too sparse |
| **CovarianceLocalMedian** | Whitened profile matching (0.85·Cov + 0.15·I), K ∈ {16, 48, 96}, local linear detrending, weighted median; neighbours capped at available training profiles |
| **SER (nominated)** | XGBoost anchor; alternatives: local median, shrunk local median, ridge (κ ∈ {0.1, 1, 10}), query median, pooled median; blends λ ∈ {0.25, 0.5, 0.75, 1}; residue-clustered gate (below); censoring safeguard |

Exact equations are in the protocol PDF at the commit hash above. Every numerical setting of the implementation is in `configs/models.json` (hash above). The settings at registration are those in `configs/models.json` (hash above); any change made before the release freeze is logged in the amendments log (section 11). The **release freeze** fixes final hyperparameters and development-derived artifacts.

**Plain reference baselines (descriptive only; section 7.6).** *QueryMedian*: the median training response of the query inhibitor. *RidgeSupports*: ridge regression of the hidden response on the observed support responses, penalty chosen on inner validation, prediction clipped to the training range. They are not part of the roster, the gate or any hypothesis.

### 7.2 SER gate (registered)

For each alternative and nonzero λ, using the same inner-validation episodes for anchor and blend:

- `Δ_e = |y_e − p0_e| − |y_e − p_{j,λ,e}|`, `T_g = Σ_{e at g} Δ_e`
- `δ̂ = (Σ_g T_g)/N`, `u_g = T_g − n_g δ̂`, `SE = sqrt(G/(G−1)) · sqrt(Σ u_g²) / N`
- `Q = δ̂ − t_{0.90, G−1} · SE`
- Qualify only if `Q > 0` and the total gain stays positive after omitting each site in turn; at least three validation sites are required.
- Choose the qualifying (alternative, λ) with the largest positive `Q`; fixed iteration order breaks ties; if none qualifies, keep XGBoost.
- **Censoring safeguard:** if a validation episode's observed supports contain a limit, the candidate is set to the XGBoost value before gains are computed. `Q_censored = min(Q_boundary, Q_interval, Q_exact)` must be positive, with the leave-one-site-out check on each set. At test time any episode with a censored support keeps the XGBoost prediction.

### 7.3 Primary analysis (H1, Lockbox A)

- `MAE_{p,k,m} = (1/N_{p,k}) Σ_e |y_e − ŷ_{e,m}|`, on all admitted episodes, in the panel's original units, using reported boundaries.
- `G_{p,k,m} = 100 · (MAE_{p,k,XGB} − MAE_{p,k,m}) / MAE_{p,k,XGB}`. Positive favours the evaluated method. If the XGBoost MAE is zero, keep absolute errors and mark the ratio undefined.
- **Equal study weighting (the primary summary; it does not change):** average gains across budgets within each study, then average the seven study means with equal weight: `G_m = (1/7) Σ_{p=1..7} [(1/|K_p|) Σ_{k∈K_p} G_{p,k,m}]`.
- **Uncertainty:** 10,000 draws resampling **residue positions with replacement within each study**. A sampled position contributes all its episodes and paired predictions. The same sampled positions are used for both budgets of a study. Recompute panel-budget gains and the equal-study summary for every draw. 95% interval = 2.5th and 97.5th percentiles, conditional on the seven benchmark studies.
- For every panel and budget report: both MAEs, the gain, number of sites, number of episodes, and the interval.
- A missing required configuration prevents the complete primary summary and is reported with its reason.

### 7.4 Secondary model comparisons (H2)

Each of the four other checkpoints is compared with XGBoost on the same manifests, same bootstrap. For formal claims on those four overall contrasts, apply Holm (smallest p compared with α/4, next with α/3, and so on, stopping at the first failure). `α = 0.05`. The paired resampling test and seed are recorded in the evaluation configuration. Panel and inhibitor breakdowns are descriptive and reported with denominators.

### 7.5 Win/tie rule and SER intervention rate (reported for SER against XGBoost)

- **Win rule.** For a panel and budget, SER **wins** if its MAE is strictly lower than XGBoost's, **ties** if the two MAEs are identical (difference within 1e-12), and **loses** if higher. A tie is never counted as a win. A study's outcome is decided on the mean of its budgets' MAE differences.
- **Intervention rate.** SER starts from XGBoost and changes a prediction only when the gate passes. For every panel, budget and outer fold we report the share of episodes where the SER prediction differs from the XGBoost prediction. A tie at the MAE level can therefore be an intervention rate of zero (SER kept XGBoost everywhere); the tables show both numbers side by side so that "no difference" is not mistaken for "SER helped".

### 7.6 Robustness checks (descriptive; none replaces the primary estimate)

All are computed from the same predictions and the same bootstrap draws.

1. **Leave-one-study-out:** equal-study mean gain and interval with each study removed in turn.
2. **Median study gain:** the median of the seven study gains.
3. **Episode-weighted and site-weighted gains:** studies weighted by their number of episodes, and by their number of residue sites (with intervals). The primary keeps equal study weights.
4. **Studies won (descriptive):** number of the seven Lockbox A studies where SER has a strictly lower mean MAE than XGBoost (ties do not count). A count of at least 5 of 7 is reported as a descriptive statement. Under a coin-flip null the chance of at least 5 of 7 is 29/128 (about 22.7%), so this count is weak evidence by itself and is never a substitute for H1.
5. **Censoring sensitivity** (panels with censored cells): MAE with interval loss instead of boundary loss, and MAE on the subset of fully exact profiles.

### 7.7 Plain baselines (descriptive)

QueryMedian and RidgeSupports (section 7.1) are reported against XGBoost with the same gain definition and bootstrap, per panel and budget and as an equal-study summary. They show whether the five registered models beat something simple. They are not tested and are not part of any claim.

---

## 8. Headline translational endpoints (pre-specified rules)

All four use the same episode lists for every method.

### T1. Screening for hidden resistance

- *Eligible variants* (fixed query and support mask): the observed supports satisfy the assay's sensitivity rule.
  - **Log₂ enrichment-type panels (An, Abadie):** supports below zero; label positive if the hidden response is positive. Abadie's response is the day-15 log2 fold change, the same kind of scale as An's log2 enrichment, so the same rule applies.
  - **IC50-ratio panels:** supports below `log10(3)`; label positive if the hidden response is at least 1 (tenfold native ratio).
  - **Censored responses:** use only labels determined by the compatible interval; record the exclusion count of ambiguous cases.
- *Minimum size.* A (query, support-mask) group is evaluated only if it has **at least 20 eligible variants**. Smaller groups are excluded and counted. With 20 or fewer variants most Lockbox A and all Lockbox B IC50 panels will have no evaluable group, so T1 is expected to be defined mainly for An and Abadie; elsewhere the output states "undefined" and how many groups fell below the minimum.
- *Procedure:* within each group, order eligible variants by decreasing predicted query response; with `n` eligible, review the top `ceil(0.05 n)`; ties broken by stable episode ID.
- *Report:* actual review fraction, hits, precision = hits / reviewed, recall = hits / eligible positives, **average precision** (per group, then averaged), the **base rate** of positives among eligible variants, and the **enrichment factor** (precision divided by base rate), all with denominators. Repeated support sets yield multiple episodes for one variant and query, so distinct (variant, query) pairs are reported separately.

### T2. Inhibitor choice and regret

- For the same variant and observed support set, compare the **remaining** inhibitors on a common endpoint with a declared direction. T2 is defined only when **at least two inhibitors remain** after removing the supports: with `D` inhibitors and budget `k`, `D − k ≥ 2`. Abadie (D = 3, k = 2, one remaining) and Zhang at k = 3 (D = 4, one remaining) are therefore undefined, and the output says so instead of reporting a trivial choice.
- Where lower means greater sensitivity (the case for every Lockbox A and B panel: IC50-ratio panels, and the An and Abadie log2 panels, where a more negative value means more depletion under drug), choose the inhibitor with the lowest prediction. Ties in the predictions go to the lowest inhibitor index.
- `regret = y_chosen − min_{eligible j} y_j`, using exact, comparable responses in the primary summary; sets with any censored remaining response are excluded and counted.
- Report eligible variants and support sets, tie handling, the frequency with which the chosen inhibitor is among the measured best, and the averaging unit: regret is averaged over the support sets of a variant first, and the variant is the unit across the panel.

### T3. Value of a third measurement

- For a fixed variant and query, take each two-drug base support set `S2` and pair it with every admitted three-drug superset `S3`.
- Compute the paired change in absolute error or decision loss; average over supersets within the base episode first, then across sites and studies, so a base episode with more supersets does not carry extra weight.

### T4. Binding-mode strata

- Using source-defined inhibitor modes, stratify episodes by whether the supports include an inhibitor matching the query's mode.
- **Compare within the same query inhibitor.** Because strata differ in which inhibitors are queried, method quality is compared against XGBoost *within each query inhibitor and stratum*, and the gain over XGBoost is reported per query inhibitor and stratum; the stratum summary is the mean of those within-query gains. Raw MAE across strata is not compared.
- Same strata and episode lists for every method. Report observations, sites and missing mode annotations. Interpret strata as assay associations, not mechanisms. Panels with all inhibitors of one mode (Zhang, Gainor, Villa, Takei) have an empty non-matching stratum; the output reports the matching stratum only and no contrast.

**Reporting commitment:** all of T1–T4 are reported for all methods and all panels in which they are defined, including null and unfavourable results.

---

## 9. Exploratory development-only work (not confirmatory)

- **Deep-learning baselines** on development panels (Estevam MET, OKL, and stress tests as needed): masked-attention Set-Transformer (observed supports as unmasked tokens, queries as masked tokens), graph network over inhibitors (nodes = inhibitors, edges = empirical correlation, node features = observed values; GraphSAGE or GCN), 1D-CNN and LSTM. They test whether generic deep learning is competitive at `k ∈ {2, 3}` and track validation MAE at those budgets. They are **not** part of the registered roster and are **never** scored on either lockbox.
- Stress-test panels and diagnostics (Jung, Chakraborty, Lee, Oh) inform development decisions.
- Retain four checkpoints and nominate SER from development evidence; record matched development scores, stability across panels, computation cost and rationale.

---

## 10. Failure handling, exclusions and release freeze

### 10.1 Failure handling

- One prediction **or one explicit failure row** per method and episode. Unsupported configurations (too few sites, too few drugs, unavailable companion profile) emit an explicit failure row with a reason, never a silent drop.
- Where a study lacks a permitted companion profile, use a training-only reference and prior built from per-drug medians with a pooled-median fallback.
- Source-quality exclusions and aggregation are recorded row by row with reasons.

### 10.2 Release freeze (second decision point)

Tag and record on both GitHub and OSF: the five model recipes, XGBoost reference, dependency lock, settings, seeds, cohort and split specifications, artifact hashes, the exact evaluation command, output schema, metrics and failure handling. Save release checks and hashes, cross-link the tagged revision and the OSF record, and keep the tagged bundle, input contract and checksum list together with the lockbox tables.

### 10.3 Release checks

| Check | Acceptance condition |
| --- | --- |
| Residue separation | Fit, validation and test groups respect the split |
| Episode alignment | Every method receives the same manifest; the query is absent from supports |
| Hidden target isolation | With permitted inputs and fits fixed, changing the hidden query leaves predictions identical |
| Model behaviour | Joint support permutation preserves aligned features; censored supports and zero SER weight preserve XGBoost |
| Scoring join | One prediction or explicit failure per method and episode; truth joined at scoring |
| Integrity | Code, environment, manifests and artifacts match the recorded hashes |

### 10.4 Lockbox run protocol

- The runner refuses a lockbox role unless `--allow-lockbox` is given **and** an integrity record that matches the current code, configuration and data hashes is supplied.
- Each lockbox (A and B) is run **once**: a marker file is written at the start and a second run is refused. Changing a model after seeing a lockbox result is outside this protocol and would be an amendment (section 11), recorded with its reason.
- The evaluator joins truth only with `--score-lockbox`, after predictions are saved by episode ID.

---

## 11. Amendments log

Amendments are recorded with their actual dates and the decisions they affect. No outcome-dependent change to H1, the 2% threshold, the seven Lockbox A panels, the budgets or the primary summary is permitted after Lockbox A is scored.

| Date | Section | Change | Reason | Decisions affected |
| --- | --- | --- | --- | --- |
| none at registration | | | | |

---

## 12. Other

- **Data sources and licences:** each panel's source DOI is recorded in the data manifest (Lockbox B: in each panel's `study.json`); source terms apply.
- **Conflicts of interest / funding:** none declared.
- **Code and data availability:** code in the GitHub repository above; the lockbox response tables (A and B) are held by the author. Code, response tables and hashes will all be made public when the study is published; until then only the hashes are public.
