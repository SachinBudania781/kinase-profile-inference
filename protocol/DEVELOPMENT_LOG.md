# Development log (development panels and artificial data through D9; the freeze and the lockbox runs are D10 and D11)

Registered: OSF 5SBCK (DOI 10.17605/OSF.IO/5SBCK), registered commit 06821c68. Entries D0 to D9 are development steps taken **before the release freeze**; D10 and D11 record the freeze and the lockbox runs.
Code changes are also in the amendments table (`protocol/OSF_preregistration.md`, section 11): A1 to A4, dated 2026-11-13.
No lockbox panel was read, run or scored for any entry through D9. Lockbox A and B stayed untouched until the release freeze (D10).

## D0. Baseline dev run (13 Nov 2026, unchanged code)
SER vs XGBoost, equal-study mean gain 4.24% (95% CI 2.91 to 5.28), 10 dev studies. Wins 9, tie 1 (Nakamura: 3 sites, gate off).
Median study gain 3.22; leave-one-study-out 3.64 to 4.71; site-weighted 3.03; episode-weighted 5.53. Development evidence only.

## D1. Why SER gains are small on Jung, Chakraborty, Oh, Lee (stored dev predictions; `dev_tools/diagnostics/diag_weak.py`)
Common profile: fully observed deep-scan panels, 3 to 5 inhibitors, 1,588 to 3,375 variants, 246 to 331 sites. Distinct (query, support) masks are few (3, 3, 12, 30), so 350 to 850 training variants per mask type.
1. XGBoost is already near the ceiling of this expert pool. Hindsight-best blend of XGBoost with the local or ridge expert: Jung 1.6% (SER 1.5), Chakraborty 1.2% (SER 1.2), Oh 1.4% (SER 0.9), Lee 1.3% (SER 0.65). Robichaux 6.0%, Estevam 8.0%.
2. Oh and Lee leave about half the available blend gain unused (small effect: about +0.1 point on the 10-study mean).
3. Replicate noise explains most remaining error on Jung (77 to 95% of the whole error for Nilotinib, Bosutinib, Ponatinib; rough floor).
4. Across the 10 dev studies, gain rises with the number of inhibitors (Spearman 0.69), confounded with size, scale and role.
Conclusion: the four panels are weak because XGBoost has plenty of data per mask type and the data are noisy, not because of a bug.

## D2. Crash in SAER-7g, AlignedAddAudit, NestedAligned (amendment A1)
Symptom: every episode of Lee and Oh (3 inhibitors) and Estevam folds 0 and 1 had no prediction ("operands could not be broadcast together with shapes (20000,3) (20000,2) ()").
Cause 1: `ExpertBank._joint_tree` takes `argsort(...)[:, :4]`; with 3 inhibitors that is 3 columns. Cause 2: `PoolProvider.panel_records` wrote `cache[key] = None` before computing, so a failed panel was silently treated as "no records".
Fix applied in `core_models.py`: pad to 4 columns; cache only after success. Verified on Lee: 4,764 of 4,764 predictions for all three methods (`dev_tools/diagnostics/verify_fix_lee.py`).
Tests added: `test_three_inhibitor_panel_joint_expert_does_not_crash`, `test_pool_provider_does_not_cache_a_failed_computation`. SER and XGBoost outputs are unaffected by this bug.

## D3. Artificial stress bank v1 (`dev_tools/synthetic/`, spec written before running)
400 artificial studies, seed 20261114 (274 matrix_npz, 126 ser_csv with censor flags and modes); all load through the registered loaders. D 3 to 120, n 40 to 3000, sites 4 to 400, rank 1 to 5, noise, outliers, saturation, missing cells, censoring up to 62%, scale 0.03 to 30.
These are not real measurements. They were used only to find failure points and to choose among revisions (registration section 2, item 6a). No model parameter was trained on them.

## D4. Scale check (x0.001, x1, x1000, three artificial studies)
Local, ridge and query-median are exactly scale-equivariant; XGBoost is not. With label SD about 6e-4 its error is up to 10% worse. Degradation starts near label SD 0.01. All real dev panels have SD >= 0.3.

## D5. Artificial bank, first slice (133 runs, about 60 studies)
Crashes 0, harm cases 0 in 117 scored runs. Gain rises with inhibitor count (+0.35) and falls with data per mask type (-0.35), as in real panels.
Gross outliers in 4% of cells cut the local expert's mean gain from 10.7 to 0.7; SER stays positive because the gate holds it back.
Censoring: the dev panels have no censor flags, so the censored gate branch had never run on data. On artificial data SER switches itself off as censoring grows (intervention rate 0.90 / 0.43 / 0.14 at 0 / 35 / 60% censored)
while the local expert's gain was 5.5 / 6.0 / 18.6%.

## D6. Revision experiments (13 Nov 2026)
Method: per-target runner `dev_tools/capture_records.py` stores, per outer fold, the inner-validation records (XGBoost anchor, all five alternatives, censor flags, sites) and the outer-test records; `dev_tools/replay_ser.py` replays gate variants on the stored records with the same `ser_gate` / `ser_apply` code (no refitting).
Printed results: `dev_tools/results/replay_exp*.txt`. All runs are k = 2, nested residue-group folds, seed 20261112.

| Experiment | Targets | Description |
| --- | --- | --- |
| 1 | 19 | 3 real dev panels (Lee, Oh, Robichaux) + 16 artificial (4 censored, 12 uncensored) |
| 2 | 15 | censored artificial studies, 5%, 15%, 35%, 60% censored |
| 3 | 13 | artificial studies with 4% gross outliers and/or heavy tails |

Gate variants: `registered` (registered rule), `nodefer_bi` (A3), `nodefer_all3` (A3 plus Q_exact), `+meansafe` (pick largest mean gain among candidates with Q > 0 instead of largest Q).

| Experiment (mean SER gain over XGBoost, %) | registered | nodefer_bi | nodefer_all3 | registered+meansafe | nodefer_bi+meansafe |
| --- | --- | --- | --- | --- | --- |
| 1, classic whitening | 2.04 | 2.01 | 2.10 | 2.02 | 1.98 |
| 2, censored, classic whitening | 3.92 | 7.70 | 6.17 | 4.19 | 7.88 |
| 3, outliers, classic whitening | 0.53 | 0.53 | 0.53 | -0.11 | -0.11 |
| 1, robust whitening | 1.91 | 1.86 | 1.92 | 1.83 | 1.72 |
| 2, censored, robust whitening | 3.83 | 7.45 | 5.90 | 4.15 | 7.38 |
| 3, outliers, robust whitening | 0.42 | 0.42 | 0.42 | -0.19 | -0.19 |

Real dev panels (SER gain, %): Lee 0.65, Oh 0.89, Robichaux 5.16 under the registered rule, identical under every censoring variant (no censor flags). Harm (site-bootstrap CI entirely below zero): 0 studies in every variant.

### D6.1 Censoring (adopted, A3)
Censored artificial studies (experiments 1 and 2, 19 studies), mean SER gain over XGBoost, registered vs `nodefer_bi`: all 3.40 vs 6.35; by censored share: 5% (4 studies) 2.96 vs 4.30; 15% (7) 4.89 vs 6.36; 35% (4) 2.89 vs 5.27; 60% (4) 1.73 vs 9.45; at or below 15% (11) 4.19 vs 5.61.
The interval-error gain matches the reported-error gain (experiment 2 mean: 7.57 for `nodefer_bi`, 3.92 for `registered`). Four studies got slightly worse under A3: Syn0022 (60 profiles, 8 sites, 60% censored) +0.30 to -2.34; Syn0069 6.29 to 6.03; Syn0333 0.66 to 0.40; Syn0391 5.57 to 5.12.
Reading: the registered safeguard forces XGBoost on every episode with a censored support, so SER cannot use the local expert when much of the data are censored. The gate itself (lower bound on boundary and interval error, at least three sites) already protects against harm. `nodefer_all3` keeps part of that loss, so Q_exact was dropped.
Limits: the 60% level is outside the registered range (0 to 18%); the dev panels have no censor flags, so the first real test of the censored path is the lockbox run itself.

### D6.2 XGBoost standardised target (adopted, A2)
Across 47 development and artificial studies the change in XGBoost's mean absolute error is -0.01% on average (range -0.19% to +0.18%); on Lee, Oh and Robichaux it is identical to four decimals. Scale-equivariance test: `test_standardised_xgboost_is_scale_equivariant`.

### D6.3 Tested and not adopted
- Robust whitening (MAD scale, rank correlation, no clipping) for the local expert: lower mean gain in all three experiments (2.04 to 1.91, 7.70 to 7.45, 0.53 to 0.42); it made the local expert clearly worse on Syn0029 (-1.2 to -18.9) and Syn0337 (-44.6 to -65.8). The code option remains with the registered setting ("classic") as default and in `configs/models.json`.
- Largest-mean-gain selection (`+meansafe`): worse on outlier studies (0.53 to -0.11, worst study -11.9%); better by 0.2 on censored studies; not adopted.
- Winsorised validation gains in the gate (cap at 2, 3, 5 times the median anchor error): outlier experiment mean 0.53 to 0.39 / 0.35 / 0.37, more studies below zero (3 to 6 / 5 / 5); code removed.
- Minimum validation-episode floor: would remove the worst harm cases (120 to 180 episodes) but also the large local-expert win on Syn0181 (180 episodes); not adopted.

### D6.4 Outliers: what is still weak
On outlier-contaminated artificial studies SER stays near XGBoost on average (+0.53 points, 13 studies) but is not uniformly safe: 3 studies below zero, worst Syn0249 (40 profiles, 10 sites, 120 episodes) at -6.33%. The development panels have thousands of episodes (Lee 4,764; Oh 5,796; Robichaux 9,240), so this regime is far from the real panels.
The registered gate keeps SER close to XGBoost, but the local expert's outlier damage was not removed by robust whitening.

## D7. Final state of the code after the pre-freeze revisions
`core_models.py`: A1 (joint-tree padding, no failure caching), A2 (`StdXGB`), A3 (`ser_censor_mode` = `no_defer_bi`). `run_study.py`: development-only `--capture-dir` (refused for lockbox roles).
`configs/models.json`: new `revision` block (A4). Code defaults remain the registered behaviour; the released-reference parity test (`KPI_REFERENCE_SER`) passes with the registered defaults. Tests: 87 passed, 2 skipped.

## D8. Run plan and what was done
1. Dev run with the final code on all 10 dev panels, all roster methods. **Done on 24 Nov 2026** as four parallel parts (seed 20261112): 18 runs (Lee and Oh at k = 2 only), 0 failed episodes, 0 errors. Scored with the evaluator: SER 4.24% (95% CI 2.91 to 5.28), the same as D0; per-study gains moved by at most 0.04 points; the three routed models now complete all 10 studies. Final-code means over the 10 studies: CovarianceLocalMedian 3.52, AlignedAddAudit 2.99, SAER-7g 2.92, NestedAligned 2.90, RidgeSupports 1.72, QueryMedian -57.83. Output in `results/dev_final/`.
2. 400-study artificial bank with the final code: **not run** (see D9).
3. Release freeze, then one run on each lockbox: done (D10, D11).

## D9. Deviation: the full artificial-bank pass was not run on the final code
The plan in D8 was a full pass of the 400-study artificial bank with the final code before the freeze. It was not run. What exists: the first slice of the bank (about 60 studies, 117 scored runs, no crash, no harm), which ran on the code **before** the amendments (D5), and the 47-study revision experiments of D6, which tested A1 to A3 on subsets (real dev panels Lee, Oh and Robichaux plus artificial studies). The bank is not part of any hypothesis; it was used only to find failure points. The freeze was done without the full final-code pass.

## D10. Release freeze (24 Nov 2026)
- The code files frozen are those of commit `367d658` (amendments A1 to A4). Between that commit and the freeze only documentation, data copies, results and the report generator `dev_tools/make_dev_results_md.py` were added; `git diff 367d658 HEAD` on `*.py`, `tests/*.py`, `configs/*.json` and `requirements.txt` shows no other change. Unit tests were last recorded in D7 (87 passed, 2 skipped).
- `requirements-lock.txt` (output of `pip freeze`) was committed in `c6bb44f`.
- `protocol/integrity_record.json` was written with `integrity.py record` using the original archive folder and the original `LockboxB_Data` folder, then checked with `integrity.py verify` (passed). It was committed in `db9d8ff`. Code bundle SHA-256 `a74a4b61178ed5baaaf96081af61409105627e3dad05f2a7485c10fb13a5bb58`; configuration bundle `78827c27563f6a2264d4c07b43d2db2e0ba072614b4acf2012b3a902d7175075`.
- Tag `freeze-2026-11-24` on `db9d8ff` was pushed to GitHub together with the commits, before 18:17 IST on 24 Nov.
- **Deviation.** The registration (section 10.2) asks for the freeze to be recorded on both GitHub and OSF before the lockbox runs. The OSF record was **not** made before the runs. The order of events is shown by the GitHub commit and tag timestamps and by the start markers (code bundle hash, start time 18:24 IST). The OSF note is added afterwards and cross-linked to the tag.

## D11. Lockbox runs and scoring (24 Nov 2026)
- **Start.** All four jobs started at 2026-11-24 12:54:15 UTC (18:24 IST). Each passed the integrity gate and wrote its `LOCKBOX_RUN_STARTED.json` with the same code bundle hash as the record. Lockbox A ran as three parallel parts by panel (A1: An2023_EGFR, Abadie2026_EGFR; A2: Bahcall2022_MET, Fujino2019_MET; A3: Wylie2017_ABL1, Zhang2016_ALK, Gainor2016_ALK), Lockbox B as one part. This split was an operational choice to save time; every panel and budget ran exactly once and no setting differed between parts.
- **End.** The last run finished at about 14:40 UTC (20:10 IST): 13 Lockbox A runs (21,066 episodes) and 6 Lockbox B runs (2,621 episodes), 0 errors, 0 failed episodes. Prediction-folder hashes: A1 `85b4b2d5...`, A2 `1b35fd19...`, A3 `e1826e13...`, B `67f2ad2b...` (full values in the README, section 9).
- **Scoring.** One evaluator call per lockbox with `--score-lockbox`, at about 20:35 IST, on the merged part folders. No model was re-run or retuned after scoring.
- **Results.**
  - Lockbox A (H1): SER 3.52% (95% CI 1.48 to 5.60); registered criterion met; the interval does not exclude 2%; 6 of 7 studies won; episode-weighted 2.39% (95% CI -0.38 to 4.77).
  - H2 (Lockbox A): CovarianceLocalMedian -11.30%, NestedAligned -3.19%, AlignedAddAudit -1.61%, SAER-7g -1.05%; none significant after Holm.
  - Lockbox B (H1-B): SER 16.78% (95% CI 12.71 to 19.53); criterion met; 3 of 3 studies won; episode-weighted 6.22% (95% CI 2.74 to 8.91); plain RidgeSupports 22.88% is higher than SER.
  - Reports: `results/lockbox_A/LOCKBOX_RESULTS.md`, `results/lockbox_B/LOCKBOX_RESULTS.md`.
- **Data release.** After scoring, the seven Lockbox A panels were copied into `data/` and the three Lockbox B panels into `data_lockbox_B/`, with checksums; `integrity.py verify --archive-root data --lockbox-b-root data_lockbox_B` passes against the record.
