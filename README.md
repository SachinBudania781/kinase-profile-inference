# Kinase Profile Inference

**Can two or three measured inhibitor responses predict the response to another inhibitor for the same kinase variant, in the same assay?**

This repository holds the code, data recipes, split specifications and evaluator for a pre-registered study. Five predictors are developed on declared development panels and compared with a common XGBoost reference on two lockboxes: **Lockbox A** (seven panels, primary test) and **Lockbox B** (three newly curated panels, replication).

| Item | Value |
| --- | --- |
| OSF registration | https://osf.io/5sbck |
| OSF DOI | https://doi.org/10.17605/OSF.IO/5SBCK |
| Protocol revision registered on OSF (full commit hash) | `06821c682666322dce217f542312dfa032d0baab` |
| Release tag (set at freeze) | `<RELEASE_TAG>` |
| Status | Registered on OSF on 13 November 2026; development runs and release freeze come next; no model has been run on any lockbox panel |

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

The full archive inventory (59 folders) is in [`data_manifest/DATASETS.md`](data_manifest/DATASETS.md). Roles are declared in [`configs/panels.json`](configs/panels.json).

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

Chosen to resemble Lockbox A and never scored by any model. Selection criteria, rejected candidates and the similarity table are in the OSF registration, section 4.4. The transcribed source tables and the built panel folders live in `LockboxB_Data/`, **next to the repository folder** (not inside it).

| Panel | Variants / sites / drugs | Endpoint | Source | Budgets |
| --- | --- | --- | --- | --- |
| Villa ALK | 9 / 8 / 7 | EML4::ALK Ba/F3 proliferation IC50 | npj Precision Oncology 2024, Table 2 (10.1038/s41698-024-00498-w) | 2, 3 |
| Takei ALK | 12 / 11 / 4 | EML4–ALK v1 Ba/F3 viability IC50, G1202R/I1171N compound mutants | Pharmaceutical Research 2025, Table II (10.1007/s11095-025-03916-1) | 2, 3 |
| Peng ROS1 | 14 / 8 / 4 | SLC34A2–/CD74–ROS1 Ba/F3 proliferation IC50, each variant scaled by its own wild type | J Med Chem 2024, Table 6 (10.1021/acs.jmedchem.4c01205) | 2, 3 |

### Lockbox access policy

- Lockbox response tables are **never committed** to this repository. `.gitignore` blocks them, and only SHA-256 checksums are recorded.
- Development checkouts may use study IDs, assay descriptions, inhibitor identities, declared cohort sizes and synthetic fixtures.
- Lockbox episodes are written to `lockbox_private/` (git-ignored) and joined to truth only inside the scoring step.
- The runner refuses lockbox roles unless `--allow-lockbox` is given and an integrity record that matches the current code and data is supplied. Each lockbox is run once; a marker file blocks a second run. The evaluator needs `--score-lockbox`.
- **Disclosure:** no model has been run on either lockbox, and no model result on either has been seen. The only looks at lockbox data are descriptive: cleaning and summary statistics of Lockbox B, and descriptive summaries of Lockbox A used to choose Lockbox B. See the OSF registration, section 2.

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
├── requirements.txt             minimum package versions (requirements-lock.txt is produced at the freeze)
├── protocol/                    study design PDF, OSF pre-registration text, amendments
├── data_manifest/               dataset roles and sources, checksums
├── configs/
│   ├── panels.json              panel registry: role, root, format, declared sizes for every panel
│   └── models.json              every numerical model setting (frozen at the release freeze)
├── splits/                      deterministic fold and episode specifications (development only)
├── models/                      frozen model artifacts, checkpoint metadata, model cards
├── data_pipeline.py             ingestion, scale transform, folds, episodes, hashes        (Step 1)
├── core_models.py               XGBoost, SAER-7g, AlignedAddAudit, NestedAligned,
├── routers.py                   CovarianceLocalMedian, SER, plain baselines                (Step 3)
├── dl_baselines.py              development-only deep-learning baselines                   (Step 4)
├── run_study.py                 runs methods on panels, writes predictions by episode ID   (Step 4)
├── evaluator.py                 lockbox scoring, robustness, translational endpoints       (Step 5)
├── integrity.py                 SHA-256 records for code, configuration and data           (Step 5)
├── build_lockbox_panel.py       turns a transcribed published table into a panel folder    (Lockbox B)
├── lockbox_similarity.py        descriptive similarity of Lockbox B to Lockbox A           (Lockbox B)
└── tests/                       unit and end-to-end tests on synthetic panels
```

Not in the repository: the study data itself and `LockboxB_Data/`. Keep them in sibling folders and pass their paths to the scripts.

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
| `configs/models.json` | `d4afbf22dae3204f3d8b97a74875af18397d2864d9d8a06af497a6dac2501eaa` |
| Cleaned development data (all files) | `7e362a53f43bfcdc825e92411c162e4e994dc9ef291dde8f6d306a79dbc4a7b5` |
| Lockbox A data bundle (author-held) | `8c1e0505e2252cbf50ae35c09085aeaf748e6c6e715a906fb77ea9b79090c321` |
| Lockbox B data bundle (author-held) | `370b8f8cc14b54781f458f30892aa4368abaa5818aa8143b390348073e2b0d90` |
| Development episode manifest | `509f2338a1c590f9ea2ae95ff4d9b876cf2c5358b0bce585d0023c29c590a94b` |
| Lockbox A episode manifest | `360ac9cc2ea3cb8bf329a506b60a0c9d971d2f18b04476da62461f1d84da3b6f` |
| Lockbox B episode manifest | `6715936870aff7b06cac77825ccd9f52f1da700392f328e9f85751e5a1895031` |
| Fold assignment table | `8f09dc43a942e4f32cb64f6102e3c468d3643273234fa64a140676613c62541a` |
| Code bundle at release tag | `<CODE_BUNDLE_SHA256>` |
| Dependency lock / environment | `<ENV_LOCK_SHA256>` |
| Frozen model artifacts (`models/`) | `<MODEL_ARTIFACTS_SHA256>` |
| Evaluator + output schema | `<EVALUATOR_SHA256>` |

`integrity.py` recomputes these and prints a comparison against the OSF record.

---

## 10. Citation and licence

- Citation: `<TO BE ADDED AT RELEASE>`
- Code licence: `<CHOOSE, for example MIT or Apache-2.0>`
- Source datasets keep their original licences and terms; see `data_manifest/`. Lockbox B source tables are from open-access papers (CC BY).
