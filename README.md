# Kinase Profile Inference

**Can two or three measured inhibitor responses predict the response to another inhibitor for the same kinase variant, in the same assay?**

This repository holds the code, data recipes, split specifications and evaluator for a pre-registered study. Five predictors are developed on declared development panels and compared with a common XGBoost reference on two lockboxes: **Lockbox A** (seven panels, primary test) and **Lockbox B** (three newly curated panels, replication).

| Item | Value |
| --- | --- |
| OSF registration | https://osf.io/5sbck |
| OSF DOI | https://doi.org/10.17605/OSF.IO/5SBCK |
| Protocol revision registered on OSF (full commit hash) | `06821c682666322dce217f542312dfa032d0baab` |
| Release tag | `freeze-2026-11-24` (commit `db9d8ff2e1ad3fecbc36150368cfca1b154d054e`) |
| Status | Registered on OSF on 13 November 2026. Release freeze recorded on 24 November 2026. Both one-shot lockbox runs were done and scored on 24 November 2026. Development results are in section 10 and lockbox results in section 11 |

## Results so far

| | SER gain over XGBoost (equal-study mean) | Studies | Where |
| --- | --- | --- | --- |
| Development panels (final code) | **4.24%** (95% CI 2.91 to 5.28); wins 9, ties 1, losses 0 | 10 | [`results/dev_final/DEV_RESULTS.md`](results/dev_final/DEV_RESULTS.md) |
| **Lockbox A (primary test, H1)** | **3.52%** (95% CI 1.48 to 5.60); 6 of 7 studies won | 7 | [`results/lockbox_A/LOCKBOX_RESULTS.md`](results/lockbox_A/LOCKBOX_RESULTS.md) |
| Lockbox B (replication, H1-B) | **16.78%** (95% CI 12.71 to 19.53); 3 of 3 studies won | 3 | [`results/lockbox_B/LOCKBOX_RESULTS.md`](results/lockbox_B/LOCKBOX_RESULTS.md) |

- **H1 (Lockbox A):** the registered rule (point estimate above 2% and 95% interval above zero) is met. The interval's lower end is 1.48%, so it does not exclude a true gain of 2%. The episode-weighted gain is 2.39% (95% CI -0.38 to 4.77).
- **H2 (Lockbox A):** none of the four other models beats XGBoost on average (mean gains -1.05% to -11.30%), and none is significant after the Holm adjustment.
- **H1-B (Lockbox B):** the criterion is met, on three studies and 27 sites. Villa2024_ALK holds 84% of the episodes and gains only 3.0%; the episode-weighted gain is 6.22% (95% CI 2.74 to 8.91). All four other models and the plain RidgeSupports baseline (22.88%) also beat XGBoost on these small panels, so this shows SER beats the reference, not that SER is the best predictor. Lockbox A and B are never pooled.
- The first baseline development run D0 (13 Nov, code before amendments A1 to A4) is in [`results/dev/`](results/dev/) and gave the same development headline.

Development numbers are evidence from the panels the models were built on. Only the lockbox runs test the registered hypotheses.

---

## 1. Study question and what a prediction is

- **Unit of prediction:** one missing assay response (variant *i*, query inhibitor *q*).
- **Unit held out during evaluation:** the residue position (site). All substitutions at one site always go to the same fold.
- **Input available for each prediction:** the identities of the query inhibitor and of the observed inhibitors, and the **2 or 3** observed responses of that same variant in that same assay, plus permitted annotations (for example binding mode). Nothing else about the hidden response is ever visible.

An episode is `e = (study, i, q, S)` with `q ∉ S` and `|S| = k ∈ {2, 3}`. For a complete profile of `D` inhibitors there are `D × C(D−1, k)` distinct masks per variant. Every method sees the same episode IDs, the same observed responses and the same hidden targets within a panel and budget.

### The 2–3 inhibitor budget constraint

Only `k = 2` or `k = 3` measured responses are ever available. `Abadie2026_EGFR` has three inhibitors in total, so it supports only `k = 2` (one inhibitor is the query). Lockbox A therefore has **13 panel–budget configurations** (7 panels; 6 with both budgets, 1 with `k = 2` only). Lockbox B has 6 (3 panels, both budgets).

---

## 2. Model roster

Five candidate predictors plus one common reference.

| Model | One-line description |
| --- | --- |
| **XGBoost reference** | `reg:absoluteerror`, η = 0.055, L2 leaf penalty 4, seed 20261112, one thread, at most 36,000 episodes per fit. Depth/trees grid: (3, 200), (5, 260), (7, 300), min child weight 15. Chosen per outer fold and budget by smallest total absolute error on inner validation. Common comparator and anchor for SER. |
| **SAER-7g** | Six or seven experts (prior + residual median, Gaussian conditional transfer, conditional Gaussian mixture, donor transport, pairwise tree ×2, joint-support tree). Each expert audits itself on the observed supports; a softmax router uses those audits to correct and weight the proposals. Seven experts if the panel has at least 1,000 profiles or 10,000 observed cells, otherwise six. |
| **AlignedAddAudit** | SAER routers give a reference; an audit-scaled Laplace-mixture median can replace the router mean when validation supports it; a supervised aligned residual head (histogram boosting, absolute error, seed 20261111) adds a blended correction. |
| **NestedAligned** | Same aligned architecture with a stricter fit/validation separation: validation labels select output settings only. Seed 20261116. |
| **CovarianceLocalMedian** | Whitened (covariance-adjusted) nearest-profile matching on the same support set, local linear detrending, weighted median. Neighbourhood size from {16, 48, 96}. |
| **SER (nominated)** | Starts from the XGBoost prediction and considers a small set of bounded alternatives (local median, shrunk local median, ridge, query median, pooled median) at blends λ ∈ {0.25, 0.5, 0.75, 1}. A correction is applied only if it passes a residue-clustered validation gate (one-sided t quantile 0.90, at least three validation sites, leave-one-site-out check). Any episode with a censored support keeps the XGBoost prediction. |

Four of the five candidates are retained as checkpoints for comparison; SER is the nominated model for the primary contrast. Every numerical setting is in [`configs/models.json`](configs/models.json); full equations are in [`protocol/`](protocol/).

**Plain reference baselines (descriptive only, not part of the roster):** *QueryMedian* (median training response of the query inhibitor) and *RidgeSupports* (ridge on the observed supports). They show whether the roster beats something simple.

**Development-only exploratory baselines (never scored on either lockbox):** Set-Transformer with masked attention, GraphSAGE/GCN over inhibitors, 1D-CNN and LSTM (`dl_baselines.py`).

---

## 3. Data roles

The full archive inventory (59 folders) is in [`data_manifest/DATASETS.md`](data_manifest/DATASETS.md). Roles are declared in [`configs/panels.json`](configs/panels.json). **All 59 archive folders are in [`data/`](data/)** (index with sources: [`data/README.md`](data/README.md)); the seven Lockbox A panels were added after the lockbox runs were scored. The three Lockbox B panels are in [`data_lockbox_B/`](data_lockbox_B/), also added after scoring.

| Role | Folders | Use |
| --- | --- | --- |
| Core development panels | 6 (`Estevam2025_MET`, `DuongLy2016_multi_kinase`, `Robichaux2021_EGFR`, `Nakamura2021_FGFR1_FGFR2_FGFR3`, `KiRHub2026_multi_kinase`, `OKL2026_multi_kinase`) | Baseline comparisons, development of routing and correction methods |
| Development stress tests | 4 (`Jung2025_ABL1`, `Chakraborty2024_SRC`, `Lee2026_EGFR`, `Oh2026_ALK`) | Small groups, sparse budgets, assay-specific failure modes |
| **Lockbox A** | 7 (`An2023_EGFR`, `Abadie2026_EGFR`, `Bahcall2022_MET`, `Fujino2019_MET`, `Wylie2017_ABL1`, `Zhang2016_ALK`, `Gainor2016_ALK`) | Final evaluation only (primary hypothesis) |
| **Lockbox B** (outside the 59 folders) | 3 (`Villa2024_ALK`, `Takei2025_ALK`, `Peng2024_ROS1`) | Final evaluation only (replication) |
| Feasibility and data intake | 39 | Earlier potency-transfer and intake work; outside model development and evaluation |
| Literature references | 3 | Literature audit only |

### Lockbox A panels

| Panel | Variants / sites / drugs | Endpoint | Budgets |
| --- | --- | --- | --- |
| An EGFR | 199 / 131 / 5 | Published log₂ enrichment, Table 5 fast-proliferating cohort | 2, 3 |
| Abadie EGFR | 370 / 62 / 3 | PC-9 exon 20 day-15 log₂ drug-versus-vehicle contrast | 2 |
| Bahcall MET | 8 / 8 / 6 | TPR–MET cellular IC50, Ba/F3 | 2, 3 |
| Fujino MET | 26 / 12 / 7 | MET exon 14-skipping cellular IC50, reported limits retained | 2, 3 |
| Wylie ABL1 | 16 / 15 / 6 | BCR–ABL1 cellular IC50, ATP-site and allosteric inhibitors | 2, 3 |
| Zhang ALK | 16 / 12 / 4 | EML4–ALK proliferation IC50 | 2, 3 |
| Gainor ALK | 10 / 8 / 5 | EML4–ALK phosphorylation IC50 | 2, 3 |

### Lockbox B panels (curated for replication)

Chosen to resemble Lockbox A and never scored by any model. Selection criteria, rejected candidates and the similarity table are in the OSF registration, section 4.4. The transcribed source tables and the built panel folders were kept in `LockboxB_Data/`, next to the repository folder, until the lockbox runs were scored; copies are now in `data_lockbox_B/`.

| Panel | Variants / sites / drugs | Endpoint | Source | Budgets |
| --- | --- | --- | --- | --- |
| Villa ALK | 9 / 8 / 7 | EML4::ALK Ba/F3 proliferation IC50 | npj Precision Oncology 2024, Table 2 (10.1038/s41698-024-00498-w) | 2, 3 |
| Takei ALK | 12 / 11 / 4 | EML4–ALK v1 Ba/F3 viability IC50, G1202R/I1171N compound mutants | Pharmaceutical Research 2025, Table II (10.1007/s11095-025-03916-1) | 2, 3 |
| Peng ROS1 | 14 / 8 / 4 | SLC34A2–/CD74–ROS1 Ba/F3 proliferation IC50, each variant scaled by its own wild type | J Med Chem 2024, Table 6 (10.1021/acs.jmedchem.4c01205) | 2, 3 |

### Lockbox access policy

- Until both lockboxes were scored, the response tables were **not committed**: `.gitignore` blocked them and only SHA-256 checksums were recorded. After the scoring (24 Nov 2026) the tables were added, Lockbox A panels in `data/` and Lockbox B in `data_lockbox_B/`, each with a checksum file. They are byte-identical to the archive folders that were hashed at the freeze, and `python integrity.py verify --archive-root data --lockbox-b-root data_lockbox_B` passes against `protocol/integrity_record.json`.
- Development checkouts may use study IDs, assay descriptions, inhibitor identities, declared cohort sizes and synthetic fixtures.
- Lockbox episodes are written to `lockbox_private/` (git-ignored) and joined to truth only inside the scoring step.
- The runner refuses lockbox roles unless `--allow-lockbox` is given and an integrity record that matches the current code and data is supplied. Each lockbox is run once; a marker file blocks a second run. The evaluator needs `--score-lockbox`.
- **Disclosure (state before the runs):** no model had been run on either lockbox, and no model result on either had been seen. Both runs were done on 24 Nov 2026 (section 11). The only looks at lockbox data before the runs were descriptive: cleaning and summary statistics of Lockbox B, and descriptive summaries of Lockbox A used to choose Lockbox B. See the OSF registration, section 2.

---

## 4. Response scales, censoring and folds

- **Native-normalised IC50 panels:** `y_ij = log10(IC50_variant,ij / IC50_native,j)`. Reported limits are transformed onto the same scale. Enrichment, fixed-time contrasts, residual activity and residual binding keep their source scale.
- **Censoring:** flag `c = 0` exact, `+1` lower bound, `−1` upper bound. Boundary error is the distance to the reported number; interval error is the distance to the set of compatible responses.
- **Residue folds:** sort site IDs by a seeded SHA-256 digest of `"{seed}:{site}"` and deal them cyclically into `min(5, n_sites)` folds (at least 4 sites required). Fold seed: **20261112**. Panels with a broader grouping contract (for example gene-level groups) keep the whole source-defined group together.

---

## 5. Evaluation (summary)

- **Primary (H1, Lockbox A):** per panel and budget, MAE in the panel's own units; relative gain over XGBoost `G = 100 × (MAE_XGB − MAE_m) / MAE_XGB`. Average across budgets within a study, then across the seven studies with equal weight. Uncertainty from **10,000 paired draws** resampling residue positions with replacement within each study. Target for SER: point estimate above **2%** and a 95% interval above zero.
- **Replication (H1-B, Lockbox B):** same estimator on the three B panels. Supportive if the point estimate is above zero; confirmatory only if the H1 criterion is met. A and B are never pooled.
- **Win/tie rule:** strictly lower MAE is a win, identical MAE is a tie, and a tie is never counted as a win. SER's **intervention rate** (share of episodes where it differs from XGBoost) is reported per panel, budget and fold.
- **Robustness (descriptive; the primary stays equal-study weighted):** leave-one-study-out, median study gain, episode-weighted and site-weighted gains, studies won (≥ 5 of 7, ties do not count; chance by coin flip is 29/128 ≈ 22.7%), censoring sensitivity.
- **Plain baselines (descriptive):** QueryMedian and RidgeSupports against XGBoost.
- **Secondary model comparisons:** each of the four other checkpoints against XGBoost, Holm-adjusted if formal claims are made.
- **Pre-specified translational endpoints (headline biological goals):**
  - **T1** hidden-resistance screening: top-5% review list, at least 20 eligible variants per mask, precision against the base rate (enrichment factor), recall, average precision.
  - **T2** inhibitor choice and regret: defined only when at least two inhibitors remain after the supports.
  - **T3** value of a third measurement.
  - **T4** binding-mode strata: compared within the same query inhibitor, gain over XGBoost per stratum.

---

## 6. Repository layout

```
kinase-profile-inference/
├── README.md                    this file
├── requirements.txt             minimum package versions
├── requirements-lock.txt        exact package versions at the freeze
│
├── protocol/                    study design PDF, OSF pre-registration text and amendments, DEVELOPMENT_LOG.md, integrity_record.json
├── data_manifest/               DATASETS.md: the role of each of the 59 archive folders
├── data/                        all 59 archive study folders (52 development, intake, literature; 7 Lockbox A), README.md, CHECKSUMS.sha256
├── data_lockbox_B/              the 3 Lockbox B panels, transcription specs, similarity table, CHECKSUMS.sha256
├── configs/
│   ├── panels.json              panel registry: role, root, format, declared sizes for every panel
│   └── models.json              every numerical model setting (frozen at the release freeze)
├── splits/                      deterministic fold and episode specifications (development only)
├── models/                      MODEL_CARD.md: what each model is, its settings, its development numbers
├── results/                     summary tables from evaluator runs: dev (D0), dev_final, lockbox_A, lockbox_B
├── dev_tools/                   development-only scripts: diagnostics, artificial stress bank, replay experiments, report generator
├── tests/                       unit and end-to-end tests on synthetic panels
│
├── data_pipeline.py             ingestion, scale transform, folds, episodes, hashes
├── core_models.py               XGBoost, SER, CovarianceLocalMedian, SAER-7g, AlignedAddAudit, NestedAligned, plain baselines
├── routers.py                   audit features, softmax routers, Laplace-mixture median (shared by the three routed models)
├── dl_baselines.py              development-only deep-learning baselines
├── run_study.py                 runs methods on panels, writes predictions by episode ID
├── evaluator.py                 scoring, robustness, translational endpoints
├── integrity.py                 SHA-256 records for code, configuration and data
├── build_lockbox_panel.py       turns a transcribed published table into a panel folder (Lockbox B)
└── lockbox_similarity.py        descriptive similarity of Lockbox B to Lockbox A (Lockbox B)
```

The Python modules stay in the repository root on purpose: the integrity record hashes these exact files and the registered code names, so they are not moved before the lockbox runs.

Git-ignored: `runs/` (raw predictions, large and regenerable; their hashes are recorded), `lockbox_private/` (lockbox episode lists), `__pycache__/`.

All 59 archive folders are in `data/` and the three Lockbox B panels in `data_lockbox_B/`, byte-identical to the archive folders that were hashed at the freeze. `--archive-root data --lockbox-b-root data_lockbox_B` therefore works for every run and for `integrity.py verify`. The 24 Nov 2026 freeze and lockbox runs used the original archive folder next to the repository.

---

## 7. How to run (in this order)

```
pip install -r requirements.txt
python -m pytest tests -q

# development data: audit, episodes, hashes (no lockbox)
python data_pipeline.py audit    --archive-root <data>
python data_pipeline.py episodes --archive-root <data> --out splits

# lockbox manifests (episode IDs only; no responses) and bundle hashes
python data_pipeline.py lockbox  --archive-root <data> --lockbox-b-root <LockboxB_Data> --private-dir lockbox_private
python data_pipeline.py hashes   --archive-root <data> --lockbox-b-root <LockboxB_Data>

# development run and evaluation
python run_study.py --archive-root <data> --roles core_dev,stress_test --out runs/dev
python evaluator.py --runs runs/dev --archive-root <data> --roles core_dev,stress_test --out results/dev

# release freeze: record hashes, then run each lockbox once (only you, only after the freeze)
python integrity.py record --archive-root <data> --lockbox-b-root <LockboxB_Data> --out protocol/integrity_record.json
python run_study.py --archive-root <data> --lockbox-b-root <LockboxB_Data> --roles lockbox_A \
    --allow-lockbox --integrity-record protocol/integrity_record.json --out runs/lockbox_A
python run_study.py --archive-root <data> --lockbox-b-root <LockboxB_Data> --roles lockbox_B \
    --allow-lockbox --integrity-record protocol/integrity_record.json --out runs/lockbox_B
python evaluator.py --runs runs/lockbox_A --archive-root <data> --lockbox-b-root <LockboxB_Data> \
    --roles lockbox_A --score-lockbox --out results/lockbox_A
python evaluator.py --runs runs/lockbox_B --archive-root <data> --lockbox-b-root <LockboxB_Data> \
    --roles lockbox_B --score-lockbox --out results/lockbox_B
```

`<data>` is `data` and `<LockboxB_Data>` is `data_lockbox_B` (both in this repository). The 24 Nov 2026 freeze and lockbox runs used the original archive folder next to the repository.

Operational note on the 24 Nov 2026 lockbox runs: to save time, Lockbox A ran as three parallel parts by panel (`--panels`, separate `--out` folders `runs/lockbox_A1`, `A2`, `A3`) and Lockbox B as one part. Each panel and budget was run exactly once, each part passed the integrity gate and wrote its own one-shot marker, and the parts were copied into one folder per lockbox for scoring.

Building a Lockbox B panel from a transcribed table:

```
python build_lockbox_panel.py --spec LockboxB_Data/_specs/Villa2024_ALK.json --out LockboxB_Data
python lockbox_similarity.py --archive-root <data> --lockbox-b-root LockboxB_Data --out LockboxB_Data/_similarity
```

---

## 8. Workflow and commit discipline

1. Create this repository and register the design on OSF **before** fitting the models.
2. Prepare development data, record group assignments, build reproducible episodes.
3. Fit the matched XGBoost reference and develop the five predictors on development data.
4. Fix recipes, models, evaluation rules and software environment for the lockboxes (**release freeze**).
5. Tag the release, record checksums on OSF, keep the lockbox tables private until you publish.

Use separate commits for: study design, development-data recipes, matched reference, candidate implementations, final evaluator. For each development run record the commit, configuration hash, data hashes, episode-manifest hash, seed, environment and output location. Failed runs and decisions are recorded next to the retained settings.

**Two decision points.** The design registration fixes the question, datasets and evaluation contract. The release freeze fixes the exact algorithms, input features, hyperparameters, development-derived artifacts, evaluator and output schema. Amendments are logged with their real dates and the decisions they affect.

---

## 9. Integrity records (filled in as the study proceeds)

| Artifact | SHA-256 |
| --- | --- |
| `protocol/Kinase_Profile_Inference_Study_Design_and_Methods.pdf` (version 0.2, rebuilt 13 Nov) | `1517bcf91c2364fc97d7fb56d1c9016ccc7484fc9957b982b9cdf89272a4dd94` |
| `data_manifest/DATASETS.md` | `a121a5ea5da68b2456881a9929afb39d038069eba1ed59422ef95e30182f2319` |
| `configs/panels.json` | `1a01f3b8a458ba3648dca092adeb124598ef9a94c3ee2e4dbf7d8f6ea5372385` |
| `configs/models.json` at registration (13 Nov) | `d4afbf22dae3204f3d8b97a74875af18397d2864d9d8a06af497a6dac2501eaa` |
| `configs/models.json` after amendment A4 (adds the `revision` block) | `869861a2941b114ca55a0e058296f3f8655fac841088c3b6bc8cdec213c67859` |
| Cleaned development data (all files) | `7e362a53f43bfcdc825e92411c162e4e994dc9ef291dde8f6d306a79dbc4a7b5` |
| Lockbox A data bundle (author-held) | `8c1e0505e2252cbf50ae35c09085aeaf748e6c6e715a906fb77ea9b79090c321` |
| Lockbox B data bundle (author-held) | `370b8f8cc14b54781f458f30892aa4368abaa5818aa8143b390348073e2b0d90` |
| Development episode manifest | `509f2338a1c590f9ea2ae95ff4d9b876cf2c5358b0bce585d0023c29c590a94b` |
| Lockbox A episode manifest | `360ac9cc2ea3cb8bf329a506b60a0c9d971d2f18b04476da62461f1d84da3b6f` |
| Lockbox B episode manifest | `6715936870aff7b06cac77825ccd9f52f1da700392f328e9f85751e5a1895031` |
| Fold assignment table | `8f09dc43a942e4f32cb64f6102e3c468d3643273234fa64a140676613c62541a` |
| Code bundle at release tag `freeze-2026-11-24` | `a74a4b61178ed5baaaf96081af61409105627e3dad05f2a7485c10fb13a5bb58` |
| Configuration bundle (`configs/*.json`) at the freeze | `78827c27563f6a2264d4c07b43d2db2e0ba072614b4acf2012b3a902d7175075` |
| Integrity record | `protocol/integrity_record.json` (commit `db9d8ff`) |
| Dependency lock / environment | `requirements-lock.txt` (committed in `c6bb44f`) |
| Frozen model artifacts | none: every model is refitted inside each run; the settings are in `configs/models.json` (hash above) |
| Evaluator + output schema | part of the code bundle (`evaluator.py`) |
| Lockbox A predictions, part A1 / A2 / A3 | `85b4b2d50c5b040256ab21481b9df3689783ec49050bbae66549676575f3ed5f` / `1b35fd1925f26b91e68f1a1a96beeefb11a655a74f172757ac31a570aad22852` / `e1826e13b1942c7ac6d04dece3f701e877217194914e90e4983d7bd40af3779d` |
| Lockbox B predictions | `67f2ad2b60c1f587e9454e147e54271322ea93c340f1cc263a95263ad6faffb1` |

`integrity.py verify` recomputes these and compares them with `protocol/integrity_record.json`. The freeze hashes were committed and tagged on GitHub before the lockbox runs; the OSF copy of them was not made before the runs (development log D10).

All four lockbox start markers (`runs/lockbox_*/LOCKBOX_RUN_STARTED.json`) carry the same code bundle hash as the record and the same start time, 2026-11-24 12:54:15 UTC (18:24 IST).

---

## 10. Development results (before the freeze)

Final-code run of 24 Nov 2026 on the 10 development panels (code after amendments A1 to A4; this is the code that was frozen). It reproduces the baseline run D0 of 13 Nov (code as registered): SER is 4.24% (95% CI 2.91 to 5.28) in both, and per-study gains moved by at most 0.04 points. Gain G = 100 x (MAE of XGBoost - MAE of SER) / MAE of XGBoost; positive means SER is better. Equal weight per study; a study's gain is the mean of its k = 2 and k = 3 gains.

| Quantity | Value |
| --- | --- |
| Equal-study mean gain | **4.24%** (95% CI 2.91 to 5.28) |
| Median study gain | 3.22% |
| Leave-one-study-out | 3.64% to 4.71% |
| Site-weighted | 3.04% (95% CI 2.65 to 3.40) |
| Episode-weighted | 5.55% (95% CI 4.93 to 6.12) |
| Wins / ties / losses | 9 / 1 / 0 (Nakamura ties: 3 sites, so the gate stays off) |

| Study | Role | Sites | SER gain, k = 2 (%) | SER gain, k = 3 (%) | Study gain (%) |
| --- | --- | --- | --- | --- | --- |
| Estevam2025_MET | core | 287 | 8.00 | 11.16 | 9.58 |
| DuongLy2016_multi_kinase | core | 21 | 9.39 | 7.39 | 8.39 |
| Robichaux2021_EGFR | core | 13 | 5.16 | 8.21 | 6.69 |
| Nakamura2021_FGFR1_FGFR2_FGFR3 | core | 3 | 0.00 | 0.00 | 0.00 |
| KiRHub2026_multi_kinase | core | 48 | 7.97 | 9.34 | 8.65 |
| OKL2026_multi_kinase | core | 11 | 5.72 | 3.78 | 4.75 |
| Jung2025_ABL1 | stress test | 322 | 1.49 | 1.87 | 1.68 |
| Chakraborty2024_SRC | stress test | 246 | 1.24 | 0.92 | 1.08 |
| Lee2026_EGFR | stress test | 286 | 0.65 | - | 0.65 |
| Oh2026_ALK | stress test | 331 | 0.89 | - | 0.89 |

Lee2026_EGFR and Oh2026_ALK have only three inhibitors, so they have k = 2 only.

What the numbers say:

- SER beats XGBoost on 9 of 10 studies and ties on one. The gain is 4.8 to 9.6% on five of the six core panels (Nakamura ties) and 0.65 to 1.7% on the four stress-test panels.
- The development log (D1) traced the small stress-test gains to XGBoost already having plenty of training data per mask type and to noisy replicate data, not to a bug.
- Other models (final code, equal-study mean gain, all 10 studies): CovarianceLocalMedian 3.52%, AlignedAddAudit 2.99%, SAER-7g 2.92%, NestedAligned 2.90%, RidgeSupports 1.72%, QueryMedian -57.83%. In D0 the three routed models completed only 7 studies because of the crash fixed in A1 (D0 means: SAER-7g 2.30%, NestedAligned 2.17%, AlignedAddAudit 2.15%).

Everything else is in:

- [`results/dev_final/DEV_RESULTS.md`](results/dev_final/DEV_RESULTS.md): per-panel and per-budget tables with confidence intervals, all methods, intervention rates (final code)
- [`results/dev_final/panel_budget_table.csv`](results/dev_final/panel_budget_table.csv) and [`results/dev_final/results.json`](results/dev_final/results.json): the raw evaluator output (including the T1 to T4 endpoints)
- [`results/dev/`](results/dev/): the same for the baseline run D0
- [`protocol/DEVELOPMENT_LOG.md`](protocol/DEVELOPMENT_LOG.md): D0 to D11, the diagnosis of the weak panels, the artificial stress bank, the revision experiments, amendments A1 to A4, the freeze and the lockbox runs
- [`dev_tools/results/`](dev_tools/results/): printed output of the revision experiments
- [`models/MODEL_CARD.md`](models/MODEL_CARD.md): what each model is

---

## 11. Lockbox results (one-shot runs, 24 Nov 2026)

Both lockboxes were run once on the frozen code (tag `freeze-2026-11-24`, code bundle `a74a4b61...`) and scored once: 19 panel-budget runs, 23,687 episodes, no failed episodes, no errors. Full reports are in [`results/lockbox_A/LOCKBOX_RESULTS.md`](results/lockbox_A/LOCKBOX_RESULTS.md) and [`results/lockbox_B/LOCKBOX_RESULTS.md`](results/lockbox_B/LOCKBOX_RESULTS.md); the raw evaluator output (`results.json`, `panel_budget_table.csv`, including T1 to T4) sits beside each. Lockbox A and Lockbox B are never pooled.

### Lockbox A (H1, primary test: 7 studies, 248 sites, 21,066 episodes)

| Quantity | Value |
| --- | --- |
| Equal-study mean gain, SER vs XGBoost | **3.52%** (95% CI 1.48 to 5.60) |
| Registered criterion (mean gain above 2% and 95% CI above 0) | **met** |
| 95% CI entirely above 2% | no (lower end 1.48%) |
| Median study gain | 3.51% |
| Leave-one-study-out | 2.51% to 4.46% |
| Site-weighted | 3.51% (95% CI 2.52 to 4.52) |
| Episode-weighted | 2.39% (95% CI -0.38 to 4.77) |
| Studies won / tied / lost | 6 / 0 / 1 |

| Study | Sites | SER gain, k = 2 (%) | SER gain, k = 3 (%) | Study gain (%) |
| --- | --- | --- | --- | --- |
| Abadie2026_EGFR | 62 | 0.98 | - | 0.98 |
| An2023_EGFR | 131 | 3.87 | 4.91 | 4.39 |
| Bahcall2022_MET | 8 | 3.94 | -2.14 | 0.90 |
| Fujino2019_MET | 12 | -7.48 | 3.14 | -2.17 |
| Gainor2016_ALK | 8 | -0.98 | 8.01 | 3.51 |
| Wylie2017_ABL1 | 15 | 7.25 | 7.60 | 7.43 |
| Zhang2016_ALK | 12 | 22.60 | -3.47 | 9.57 |

**H2 (the four other models against XGBoost, Lockbox A):** none beats XGBoost on average, and none is significant after the Holm adjustment (alpha 0.05).

| Model | Mean gain (%) | 95% CI | p | Holm threshold |
| --- | --- | --- | --- | --- |
| CovarianceLocalMedian | -11.30 | -19.83 to -2.33 | 0.0139 | 0.0125 |
| NestedAligned | -3.19 | -9.94 to 1.98 | 0.2285 | 0.0167 |
| AlignedAddAudit | -1.61 | -8.04 to 3.74 | 0.5533 | 0.0250 |
| SAER-7g | -1.05 | -7.22 to 4.53 | 0.7425 | 0.0500 |

Plain baselines, descriptive: RidgeSupports 0.16% (95% CI -4.09 to 3.86), QueryMedian -20.53%.

What to keep in mind when reading Lockbox A:

- Most panels are small (8 to 15 sites in five of the seven), so most per-panel intervals include zero; the interval for the mean is wide (1.48 to 5.60).
- The result is below the development estimate (4.24%) but inside the development interval (2.91 to 5.28).
- `Fujino2019_MET` is the only Lockbox A panel with censor flags, so it is the first real-data test of the censored branch of the gate (amendment A3). SER lost there at k = 2 (-7.48%) and gained at k = 3 (+3.14%); on the episodes with no censored values the gains were -3.01% and -0.88%. A 12-site panel cannot separate a real weakness from noise.

### Lockbox B (H1-B, replication: 3 studies, 27 sites, 2,621 episodes)

| Quantity | Value |
| --- | --- |
| Equal-study mean gain, SER vs XGBoost | **16.78%** (95% CI 12.71 to 19.53) |
| Registered criterion (mean gain above 2% and 95% CI above 0) | **met** (the interval is entirely above 2%) |
| Median study gain | 19.03% |
| Leave-one-study-out | 11.02% to 23.68% |
| Site-weighted | 18.06% (95% CI 13.93 to 20.98) |
| Episode-weighted | 6.22% (95% CI 2.74 to 8.91) |
| Studies won / tied / lost | 3 / 0 / 0 |

| Study | Sites | SER gain, k = 2 (%) | SER gain, k = 3 (%) | Study gain (%) |
| --- | --- | --- | --- | --- |
| Peng2024_ROS1 | 8 | 10.66 | 27.41 | 19.03 |
| Takei2025_ALK | 11 | 7.94 | 48.69 | 28.32 |
| Villa2024_ALK | 8 | -1.07 | 7.07 | 3.00 |

What to keep in mind when reading Lockbox B:

- Three studies and 27 sites make the intervals coarse, and one study can dominate: Villa2024_ALK has 84% of the episodes but gains only 3.0%, while Takei2025_ALK at k = 3 gains 48.69% on 48 episodes.
- All four other models also beat XGBoost on these panels (13.14% to 16.61%), and the plain RidgeSupports baseline does better than SER (22.88%, 95% CI 12.97 to 29.89). This suggests XGBoost is a weak reference on panels with 8 to 11 sites. It supports "SER beats the matched reference", not "SER is the best predictor".
- The Holm family is registered for Lockbox A only (H2); the same table on Lockbox B is descriptive.
- All three Lockbox B panels have censored values. On the episodes with no censored values the SER gains were 5.15, 20.97, 7.00, 50.46, -1.21 and 3.70 (Peng k = 2, 3; Takei k = 2, 3; Villa k = 2, 3). The sign of every gain is unchanged; the size changed most for Peng at k = 2 (10.66% in the full data) and Villa at k = 3 (7.07%).

### What differed from the plan

- **Freeze record.** The registration (section 10.2) asks for the freeze to be recorded on GitHub and OSF before the lockbox runs. The commit, tag and push to GitHub were done by 18:17 IST on 24 Nov (the GitHub timestamps show the exact time); the first lockbox start marker is 18:24 IST. The OSF copy of the freeze hashes was not made before the runs.
- **Artificial stress bank.** The development plan (log D8) included a full pass of the 400-study artificial bank with the final code before the freeze. It was not run. What exists is the first slice of about 60 studies (117 scored runs, no crash, no harm) from before the amendments (D5), and the 47-study revision experiments that tested A1 to A3 (D6, D7). The bank is not part of any hypothesis.
- **Parallel parts.** Lockbox A ran as three parallel parts by panel (see section 7). Panels, budgets and settings were unchanged.

---

## 12. Citation and licence

- Citation: `<TO BE ADDED AT RELEASE>`
- Code licence: `<CHOOSE, for example MIT or Apache-2.0>`
- Source datasets keep their original licences and terms; see `data/README.md` and `data_manifest/`. The licences were not checked folder by folder. Lockbox B source tables are from open-access papers (CC BY).
