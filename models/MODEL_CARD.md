# Model card: the six predictors and the plain baselines

Every model answers the same question: given 2 or 3 measured inhibitor responses of one kinase variant in one assay, predict its response to another inhibitor (the query). All models see the same episodes and write one prediction per episode.

**There are no saved weight files.** Each model is refit inside every run, using nested residue-group folds (outer test fold, inner validation folds). What is frozen is the code, `configs/models.json` and the integrity record `protocol/integrity_record.json` (written at the release freeze). Predictions of a run are written as `<panel>_k<k>_predictions.csv.gz` plus a `<panel>_k<k>_decisions.json` decision log.

All numerical settings are in [`../configs/models.json`](../configs/models.json). Code is in `core_models.py` (models, SER gate, XGBoost) and `routers.py` (audit features, softmax routers, Laplace-mixture median). Seed for everything: **20261112** unless a model lists its own.

## Models

| Model | Role | What it does | Main settings | Code |
| --- | --- | --- | --- | --- |
| **XGBoost reference** | Common comparator and the anchor inside SER | Gradient-boosted trees on the query id, the observed supports and their responses (sparse response vector, support ids, sorted values, mean, range, same-binding-mode mean, compatibility flags). | `reg:absoluteerror`, learning rate 0.055, L2 4, min child weight 15, 1 thread. Depth / trees grid: (3, 200), (5, 260), (7, 300), chosen per outer fold and budget by smallest total absolute error on inner validation. At most 36,000 training episodes per fit. Target standardised (median and SD of the fitting episodes), amendment A2. | `make_xgb`, `fit_xgb`, `StdXGB` |
| **SER** (nominated) | Primary model for H1 and H1-B | Starts from the XGBoost prediction and moves it towards one of five bounded alternatives by a blend weight. The move is made only if it passes the gate; otherwise XGBoost is kept. | Alternatives: local median, shrunk local median, ridge (kappa 0.1, 1, 10), query median, pooled median. Blend weights 0.25, 0.5, 0.75, 1. Gate: score Q = mean validation gain minus t(0.90) times the residue-clustered standard error; at least 3 validation sites; leave-one-site-out check must stay positive; the choice with the largest positive Q wins. Censored panels: Q = min(boundary-error Q, interval-error Q), amendment A3. | `ser_gate`, `ser_apply`, `site_lower_score` |
| **CovarianceLocalMedian** | Checkpoint model | Finds training variants whose profiles look like this variant on the observed supports (whitened, covariance-adjusted distance), removes the local linear trend, and takes the weighted median of their query responses. | Neighbourhood size 16, 48 or 96, chosen on inner validation. Classic whitening (SD and Pearson covariance). | `neighborhood`, `whiten` |
| **SAER-7g** | Checkpoint model | Six or seven experts each propose a value (prior plus residual median, Gaussian conditional transfer, conditional Gaussian mixture, donor transport, two pairwise trees, joint-support tree). Each expert audits itself on the observed supports; a softmax router uses the audits to weight and correct the proposals. Seven experts if the panel has at least 1,000 profiles or 10,000 observed cells, otherwise six. | See the `saer` block in `configs/models.json`. | `ExpertBank`, `routers.py` |
| **AlignedAddAudit** | Checkpoint model | SAER routers give a reference. An audit-scaled Laplace-mixture median can replace the router mean when validation supports it. A supervised residual head (histogram boosting, absolute error) adds a blended correction. | Seed 20261111. See the `aligned` block. | `routed_fold` |
| **NestedAligned** | Checkpoint model | Same aligned design with a stricter split: validation labels select only output settings, and the experts use fixed smaller settings. | Seed 20261116. | `routed_fold` (nested style) |

SER is the only model with a pre-registered success rule (H1: equal-study mean gain over XGBoost above 2% with a 95% interval above 0). The four checkpoint models are compared with XGBoost as secondary contrasts (H2).

## Plain baselines (descriptive only)

| Baseline | What it does |
| --- | --- |
| **QueryMedian** | Predicts the median training response of the query inhibitor. |
| **RidgeSupports** | Ridge regression on the observed supports (kappa chosen on inner validation). |

They show whether the models beat something simple. They are reported descriptively and are never primary.

## Development-only deep-learning baselines (`dl_baselines.py`)

Set-Transformer with masked attention, GraphSAGE/GCN over inhibitors, 1D-CNN and LSTM. They were explored on development panels and are never scored on a lockbox.

## Revisions made before the freeze (amendments A1 to A4, 13 Nov 2026)

| Amendment | Change | Effect on the SER and XGBoost reference |
| --- | --- | --- |
| A1 | Joint-support sampler padded to 4 columns; a failed panel computation is no longer cached as "no records". | Fixes a crash of SAER-7g, AlignedAddAudit and NestedAligned on 3-inhibitor panels. SER and XGBoost were not affected. |
| A2 | XGBoost fitted on a standardised target. | Mean absolute-error change -0.01% across 47 development and artificial studies; makes XGBoost scale-equivariant. |
| A3 | Censoring rule replaced by Q = min(Q_boundary, Q_interval); censored supports are no longer forced back to XGBoost. | No change on the real development panels (they have no censor flags). On censored artificial studies the mean SER gain rose from 3.40 to 6.35. |
| A4 | `configs/models.json` gets a `revision` block that switches A2 and A3 on. | Code defaults keep the registered behaviour so the reference-parity test still passes. |

Details and the experiments behind each decision are in [`../protocol/DEVELOPMENT_LOG.md`](../protocol/DEVELOPMENT_LOG.md).

## Development numbers (SER and the other models against XGBoost)

Gain G = 100 x (MAE of XGBoost - MAE of the model) / MAE of XGBoost, equal weight per study, 10 development panels.

| Model | Studies | Mean gain, baseline run D0 (13 Nov, before A1 to A4) |
| --- | --- | --- |
| SER | 10 | **4.24%** (95% CI 2.91 to 5.28) |
| CovarianceLocalMedian | 10 | 3.51% |
| SAER-7g | 7 | 2.30% |
| NestedAligned | 7 | 2.17% |
| AlignedAddAudit | 7 | 2.15% |
| RidgeSupports | 10 | 1.72% |
| QueryMedian | 10 | -57.85% |

The three routed models completed only 7 of 10 studies in D0 because of the crash fixed in A1 (Lee, Oh, and two Estevam folds), so their means are over fewer studies and are not directly comparable. Per-panel tables, the final-code run and the robustness checks are in [`../results/`](../results/).

## Known limits

- All gains above are development evidence on development panels. The confirmatory result is the one-shot Lockbox A run.
- The four stress-test panels have small gains (about 0.7 to 1.7%) because XGBoost already has plenty of data per mask type and the data are noisy.
- On artificial outlier-contaminated studies SER is not uniformly safe: 3 of 13 were below zero, worst -6.33%.
- The censored branch of the gate has not run on real data; the first real test of it is the lockbox run itself.
