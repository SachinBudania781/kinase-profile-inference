# Lockbox B results (H1-B, replication), one-shot run on the frozen code

Run once on 24 Nov 2026 after the release freeze (tag freeze-2026-11-24, commit db9d8ff), three panels, 6 panel-budget runs, 2,621 episodes, 0 failed episodes.

Gain G = 100 x (MAE of XGBoost - MAE of the method) / MAE of XGBoost. Positive means the method beats XGBoost. Equal weight per study; a study's gain is the mean of its budgets (k = 2 and k = 3). 95% intervals come from 10000 residue-position bootstrap draws within study. Lockbox A and Lockbox B are never pooled.

## Headline: SER versus XGBoost (H1-B)

| Quantity | Value |
| --- | --- |
| Studies | 3 |
| Sites (summed over studies) | 27 |
| Episodes (summed over panels and budgets) | 2621 |
| Equal-study mean gain | **16.78%** (95% CI 12.71 to 19.53) |
| Registered criterion (mean gain above 2% and 95% CI above 0) | **met** |
| 95% CI entirely above 2% | True (lower end 12.71%) |
| Median study gain | 19.03% |
| Leave-one-study-out mean gain | 11.02% to 23.68% |
| Site-weighted gain | 18.06% (95% CI 13.93 to 20.98) |
| Episode-weighted gain | 6.22% (95% CI 2.74 to 8.91) |
| Studies won / tied / lost | 3 / 0 / 0 of 3 (the 5-of-7 count is not reported for Lockbox B) |

Lockbox B has three studies and 27 sites, so intervals are coarse and one study can dominate. The episode-weighted gain gives most of the weight to Villa2024_ALK.

## SER gain by panel and budget

| Study | Kinase | k | Sites | Episodes | MAE XGBoost | MAE SER | Gain % | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Peng2024_ROS1 | ROS1 | 2 | 8 | 168 | 0.7451 | 0.6657 | 10.66 | -1.36 to 17.80 |
| Peng2024_ROS1 | ROS1 | 3 | 8 | 56 | 0.9661 | 0.7013 | 27.41 | 14.40 to 33.70 |
| Takei2025_ALK | ALK | 2 | 11 | 144 | 0.3662 | 0.3371 | 7.94 | -1.29 to 14.46 |
| Takei2025_ALK | ALK | 3 | 11 | 48 | 0.6137 | 0.3149 | 48.69 | 42.63 to 55.52 |
| Villa2024_ALK | ALK | 2 | 8 | 945 | 0.3747 | 0.3787 | -1.07 | -2.81 to -0.11 |
| Villa2024_ALK | ALK | 3 | 8 | 1260 | 0.3986 | 0.3704 | 7.07 | 0.82 to 12.85 |

Per-study gain (mean of its budgets): Peng2024_ROS1 19.03; Takei2025_ALK 28.32; Villa2024_ALK 3.00

## All methods against XGBoost (equal-study mean gain)

| Method | Mean gain % | 95% CI | Role |
| --- | --- | --- | --- |
| SER | 16.78 | 12.71 to 19.53 | primary (H1-B) |
| CovarianceLocalMedian | 15.05 | 2.56 to 23.63 | secondary (H2 is registered for Lockbox A; descriptive here) |
| NestedAligned | 14.90 | 0.69 to 24.39 | secondary (H2 is registered for Lockbox A; descriptive here) |
| AlignedAddAudit | 13.14 | 0.33 to 22.61 | secondary (H2 is registered for Lockbox A; descriptive here) |
| SAER-7g | 16.61 | 3.58 to 25.63 | secondary (H2 is registered for Lockbox A; descriptive here) |
| RidgeSupports | 22.88 | 12.97 to 29.89 | plain baseline (descriptive) |
| QueryMedian | -16.39 | -49.83 to 6.91 | plain baseline (descriptive) |

## Per-study gain by method

| Method | Peng2024_ROS1 | Takei2025_ALK | Villa2024_ALK |
| --- | --- | --- | --- |
| SER | 19.03 | 28.32 | 3.00 |
| CovarianceLocalMedian | 15.50 | 26.93 | 2.73 |
| NestedAligned | 16.75 | 16.31 | 11.66 |
| AlignedAddAudit | 18.60 | 8.24 | 12.57 |
| SAER-7g | 21.76 | 15.33 | 12.73 |
| RidgeSupports | 23.75 | 28.91 | 15.96 |
| QueryMedian | 0.16 | 11.32 | -60.64 |

## Holm-adjusted comparison of the four secondary models with XGBoost (descriptive on Lockbox B)

alpha = 0.05. Smallest p is compared with alpha/4, the next with alpha/3, and so on, stopping at the first failure. The p values are two-sided, so a rejection would mean a difference from XGBoost in either direction; check the sign of the mean gain above.

| Method | p | Holm threshold | Rejected |
| --- | --- | --- | --- |
| SAER-7g | 0.0123 | 0.0125 | True |
| CovarianceLocalMedian | 0.0141 | 0.0167 | True |
| NestedAligned | 0.0389 | 0.0250 | False |
| AlignedAddAudit | 0.0425 | 0.0500 | False |

## How often SER changes the XGBoost prediction

Intervention rate = share of test episodes where SER's prediction differs from XGBoost's.

| Study | k | Episodes | Intervention rate |
| --- | --- | --- | --- |
| Peng2024_ROS1 | 2 | 168 | 0.929 |
| Peng2024_ROS1 | 3 | 56 | 1.000 |
| Takei2025_ALK | 2 | 144 | 1.000 |
| Takei2025_ALK | 3 | 48 | 1.000 |
| Villa2024_ALK | 2 | 945 | 0.333 |
| Villa2024_ALK | 3 | 1260 | 0.556 |

## Secondary endpoints

T1 to T4 (hidden-resistance screening, inhibitor choice and regret, value of a third measurement, binding-mode strata) are in `results.json` under their own keys. They are descriptive and are not summarised here.

