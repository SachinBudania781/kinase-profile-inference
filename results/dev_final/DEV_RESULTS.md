# Development results, final code (24 Nov 2026, after amendments A1 to A4)

This is the run on the final code that will be frozen for the lockboxes (seed 20261112, 10 development panels, 18 runs, 0 failed episodes). A1 fixed the crash, so SAER-7g, AlignedAddAudit and NestedAligned now complete on every study. A2 and A3 changed XGBoost and the censored branch of SER slightly. The earlier baseline run D0 is in results/dev.

Gain G = 100 x (MAE of XGBoost - MAE of the method) / MAE of XGBoost. Positive means the method beats XGBoost. Equal weight per study; a study's gain is the mean of its k = 2 and k = 3 gains. 95% intervals come from 10000 residue-position bootstrap draws within study.

## Headline: SER versus XGBoost

| Quantity | Value |
| --- | --- |
| Studies | 10 (6 core, 4 stress test) |
| Equal-study mean gain | **4.24%** (95% CI 2.91 to 5.28) |
| Median study gain | 3.22% |
| Leave-one-study-out mean gain | 3.64% to 4.71% |
| Site-weighted gain | 3.04% (95% CI 2.65 to 3.40) |
| Episode-weighted gain | 5.55% (95% CI 4.93 to 6.12) |
| Wins / ties / losses (studies) | 9 / 1 / 0 |
| Target met (mean gain > 2% and CI above 0) | True (CI entirely above the 2% threshold: True) |

The 2% threshold and the CI rule are the registered H1 rule. These are development numbers on development panels, not the confirmatory result.

## SER gain by panel and budget

| Study | Role | Kinase | k | Sites | Episodes | MAE XGBoost | MAE SER | Gain % | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DuongLy2016_multi_kinase | core | Multiple kinases | 2 | 21 | 9120 | 15.34 | 13.9 | 9.39 | 2.64 to 15.13 |
| DuongLy2016_multi_kinase | core | Multiple kinases | 3 | 21 | 9120 | 15.75 | 14.58 | 7.39 | 0.19 to 14.72 |
| Estevam2025_MET | core | MET | 2 | 287 | 77136 | 0.2875 | 0.2645 | 8.00 | 7.25 to 8.76 |
| Estevam2025_MET | core | MET | 3 | 287 | 77136 | 0.2649 | 0.2353 | 11.16 | 10.13 to 12.21 |
| KiRHub2026_multi_kinase | core | Multiple kinases | 2 | 48 | 18840 | 21.15 | 19.46 | 7.97 | 4.72 to 11.32 |
| KiRHub2026_multi_kinase | core | Multiple kinases | 3 | 48 | 18840 | 21.04 | 19.08 | 9.34 | 6.03 to 12.33 |
| Nakamura2021_FGFR1_FGFR2_FGFR3 | core | FGFR1, FGFR2, FGFR3 | 2 | 3 | 10791 | 0.4507 | 0.4507 | 0.00 | 0.00 to 0.00 |
| Nakamura2021_FGFR1_FGFR2_FGFR3 | core | FGFR1, FGFR2, FGFR3 | 3 | 3 | 17280 | 0.4581 | 0.4581 | 0.00 | 0.00 to 0.00 |
| OKL2026_multi_kinase | core | Multiple kinases | 2 | 11 | 7080 | 28.07 | 26.46 | 5.72 | -3.59 to 11.21 |
| OKL2026_multi_kinase | core | Multiple kinases | 3 | 11 | 7080 | 27.82 | 26.77 | 3.78 | -7.72 to 11.63 |
| Robichaux2021_EGFR | core | EGFR | 2 | 13 | 9240 | 0.7752 | 0.7352 | 5.16 | 2.25 to 6.67 |
| Robichaux2021_EGFR | core | EGFR | 3 | 13 | 9240 | 0.7548 | 0.6928 | 8.21 | 3.99 to 10.92 |
| Chakraborty2024_SRC | stress test | SRC | 2 | 246 | 40500 | 0.1844 | 0.1821 | 1.24 | 0.72 to 1.86 |
| Chakraborty2024_SRC | stress test | SRC | 3 | 246 | 13500 | 0.1746 | 0.173 | 0.92 | 0.37 to 1.55 |
| Jung2025_ABL1 | stress test | ABL1 | 2 | 322 | 55080 | 1.068 | 1.052 | 1.49 | 1.18 to 1.81 |
| Jung2025_ABL1 | stress test | ABL1 | 3 | 322 | 36720 | 0.9825 | 0.9641 | 1.87 | 1.42 to 2.30 |
| Lee2026_EGFR | stress test | EGFR | 2 | 286 | 4764 | 0.7822 | 0.7771 | 0.65 | -0.82 to 1.80 |
| Oh2026_ALK | stress test | ALK | 2 | 331 | 5796 | 0.5907 | 0.5854 | 0.89 | 0.19 to 1.58 |

Per-study gain (mean of its budgets): Chakraborty2024_SRC 1.08; DuongLy2016_multi_kinase 8.39; Estevam2025_MET 9.58; Jung2025_ABL1 1.68; KiRHub2026_multi_kinase 8.65; Lee2026_EGFR 0.65; Nakamura2021_FGFR1_FGFR2_FGFR3 0.00; OKL2026_multi_kinase 4.75; Oh2026_ALK 0.89; Robichaux2021_EGFR 6.69

## All methods against XGBoost

Equal-study mean gain over the studies where the method produced every prediction (no failed episodes). A method with fewer than 10 studies could not be run on every panel in that run.

| Method | Studies used | Mean gain % | Studies skipped (failed episodes or undefined) |
| --- | --- | --- | --- |
| SER | 10 | 4.24 | none |
| CovarianceLocalMedian | 10 | 3.52 | none |
| NestedAligned | 10 | 2.90 | none |
| AlignedAddAudit | 10 | 2.99 | none |
| SAER-7g | 10 | 2.92 | none |
| RidgeSupports | 10 | 1.72 | none |
| QueryMedian | 10 | -57.83 | none |

## How often SER changes the XGBoost prediction

Intervention rate = share of test episodes where SER's prediction differs from XGBoost's.

| Study | k | Episodes | Intervention rate |
| --- | --- | --- | --- |
| Chakraborty2024_SRC | 2 | 40500 | 1.000 |
| Chakraborty2024_SRC | 3 | 13500 | 1.000 |
| DuongLy2016_multi_kinase | 2 | 9120 | 1.000 |
| DuongLy2016_multi_kinase | 3 | 9120 | 1.000 |
| Estevam2025_MET | 2 | 77136 | 1.000 |
| Estevam2025_MET | 3 | 77136 | 1.000 |
| Jung2025_ABL1 | 2 | 55080 | 1.000 |
| Jung2025_ABL1 | 3 | 36720 | 1.000 |
| KiRHub2026_multi_kinase | 2 | 18840 | 1.000 |
| KiRHub2026_multi_kinase | 3 | 18840 | 1.000 |
| Lee2026_EGFR | 2 | 4764 | 1.000 |
| Nakamura2021_FGFR1_FGFR2_FGFR3 | 2 | 10791 | 0.000 |
| Nakamura2021_FGFR1_FGFR2_FGFR3 | 3 | 17280 | 0.000 |
| OKL2026_multi_kinase | 2 | 7080 | 1.000 |
| OKL2026_multi_kinase | 3 | 7080 | 1.000 |
| Oh2026_ALK | 2 | 5796 | 1.000 |
| Robichaux2021_EGFR | 2 | 9240 | 1.000 |
| Robichaux2021_EGFR | 3 | 9240 | 1.000 |

