# Lockbox A results (H1, primary test), one-shot run on the frozen code

Run once on 24 Nov 2026 after the release freeze (tag freeze-2026-11-24, commit db9d8ff), seven panels, 13 panel-budget runs, 21,066 episodes, 0 failed episodes. The run was executed as three parallel parts by panel; every panel and budget was run exactly once.

Gain G = 100 x (MAE of XGBoost - MAE of the method) / MAE of XGBoost. Positive means the method beats XGBoost. Equal weight per study; a study's gain is the mean of its budgets (k = 2 and k = 3). 95% intervals come from 10000 residue-position bootstrap draws within study. Lockbox A and Lockbox B are never pooled.

## Headline: SER versus XGBoost (H1)

| Quantity | Value |
| --- | --- |
| Studies | 7 |
| Sites (summed over studies) | 248 |
| Episodes (summed over panels and budgets) | 21066 |
| Equal-study mean gain | **3.52%** (95% CI 1.48 to 5.60) |
| Registered criterion (mean gain above 2% and 95% CI above 0) | **met** |
| 95% CI entirely above 2% | False (lower end 1.48%) |
| Median study gain | 3.51% |
| Leave-one-study-out mean gain | 2.51% to 4.46% |
| Site-weighted gain | 3.51% (95% CI 2.52 to 4.52) |
| Episode-weighted gain | 2.39% (95% CI -0.38 to 4.77) |
| Studies won / tied / lost | 6 / 0 / 1 of 7 (ties do not count as wins; chance of at least 5 of 7 by coin flip is 29/128) |

## SER gain by panel and budget

| Study | Kinase | k | Sites | Episodes | MAE XGBoost | MAE SER | Gain % | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Abadie2026_EGFR | EGFR | 2 | 62 | 1110 | 0.3195 | 0.3163 | 0.98 | -0.28 to 2.21 |
| An2023_EGFR | EGFR | 2 | 131 | 5970 | 0.8385 | 0.8061 | 3.87 | 2.74 to 4.95 |
| An2023_EGFR | EGFR | 3 | 131 | 3980 | 0.7751 | 0.7371 | 4.91 | 3.07 to 6.73 |
| Bahcall2022_MET | MET | 2 | 8 | 480 | 0.756 | 0.7262 | 3.94 | -5.00 to 11.58 |
| Bahcall2022_MET | MET | 3 | 8 | 480 | 0.6851 | 0.6998 | -2.14 | -5.63 to 0.00 |
| Fujino2019_MET | MET | 2 | 12 | 2730 | 0.5182 | 0.557 | -7.48 | -18.83 to 0.70 |
| Fujino2019_MET | MET | 3 | 12 | 3640 | 0.4363 | 0.4226 | 3.14 | -6.73 to 10.64 |
| Gainor2016_ALK | ALK | 2 | 8 | 300 | 0.5122 | 0.5173 | -0.98 | -10.59 to 9.18 |
| Gainor2016_ALK | ALK | 3 | 8 | 200 | 0.5091 | 0.4683 | 8.01 | -0.08 to 17.83 |
| Wylie2017_ABL1 | ABL1 | 2 | 15 | 960 | 0.3592 | 0.3332 | 7.25 | 4.23 to 11.53 |
| Wylie2017_ABL1 | ABL1 | 3 | 15 | 960 | 0.3595 | 0.3322 | 7.60 | 2.19 to 13.54 |
| Zhang2016_ALK | ALK | 2 | 12 | 192 | 0.4373 | 0.3385 | 22.60 | 17.30 to 27.32 |
| Zhang2016_ALK | ALK | 3 | 12 | 64 | 0.367 | 0.3797 | -3.47 | -7.30 to 0.26 |

Per-study gain (mean of its budgets): Abadie2026_EGFR 0.98; An2023_EGFR 4.39; Bahcall2022_MET 0.90; Fujino2019_MET -2.17; Gainor2016_ALK 3.51; Wylie2017_ABL1 7.43; Zhang2016_ALK 9.57

## All methods against XGBoost (equal-study mean gain)

| Method | Mean gain % | 95% CI | Role |
| --- | --- | --- | --- |
| SER | 3.52 | 1.48 to 5.60 | primary (H1) |
| CovarianceLocalMedian | -11.30 | -19.83 to -2.33 | H2 secondary |
| NestedAligned | -3.19 | -9.94 to 1.98 | H2 secondary |
| AlignedAddAudit | -1.61 | -8.04 to 3.74 | H2 secondary |
| SAER-7g | -1.05 | -7.22 to 4.53 | H2 secondary |
| RidgeSupports | 0.16 | -4.09 to 3.86 | plain baseline (descriptive) |
| QueryMedian | -20.53 | -28.11 to -12.85 | plain baseline (descriptive) |

## Per-study gain by method

| Method | Abadie2026_EGFR | An2023_EGFR | Bahcall2022_MET | Fujino2019_MET | Gainor2016_ALK | Wylie2017_ABL1 | Zhang2016_ALK |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SER | 0.98 | 4.39 | 0.90 | -2.17 | 3.51 | 7.43 | 9.57 |
| CovarianceLocalMedian | 1.71 | 4.11 | 9.58 | 3.66 | -51.02 | -9.08 | -38.08 |
| NestedAligned | 1.22 | 2.96 | 0.76 | -10.01 | -11.85 | -3.78 | -1.64 |
| AlignedAddAudit | 1.07 | 2.95 | 7.32 | -8.51 | -9.61 | -0.78 | -3.72 |
| SAER-7g | 0.99 | 2.46 | 10.65 | -13.50 | -7.08 | 0.58 | -1.46 |
| RidgeSupports | -1.05 | 4.46 | 7.66 | 1.02 | -5.21 | 3.77 | -9.56 |
| QueryMedian | -2.21 | -34.55 | -14.93 | -67.15 | -3.60 | -32.71 | 11.44 |

## H2: Holm-adjusted comparison of the four secondary models with XGBoost

alpha = 0.05. Smallest p is compared with alpha/4, the next with alpha/3, and so on, stopping at the first failure. The p values are two-sided, so a rejection would mean a difference from XGBoost in either direction; check the sign of the mean gain above.

| Method | p | Holm threshold | Rejected |
| --- | --- | --- | --- |
| CovarianceLocalMedian | 0.0139 | 0.0125 | False |
| NestedAligned | 0.2285 | 0.0167 | False |
| AlignedAddAudit | 0.5533 | 0.0250 | False |
| SAER-7g | 0.7425 | 0.0500 | False |

## How often SER changes the XGBoost prediction

Intervention rate = share of test episodes where SER's prediction differs from XGBoost's.

| Study | k | Episodes | Intervention rate |
| --- | --- | --- | --- |
| Abadie2026_EGFR | 2 | 1110 | 1.000 |
| An2023_EGFR | 2 | 5970 | 1.000 |
| An2023_EGFR | 3 | 3980 | 1.000 |
| Bahcall2022_MET | 2 | 480 | 0.750 |
| Bahcall2022_MET | 3 | 480 | 0.250 |
| Fujino2019_MET | 2 | 2730 | 0.654 |
| Fujino2019_MET | 3 | 3640 | 1.000 |
| Gainor2016_ALK | 2 | 300 | 1.000 |
| Gainor2016_ALK | 3 | 200 | 1.000 |
| Wylie2017_ABL1 | 2 | 960 | 1.000 |
| Wylie2017_ABL1 | 3 | 960 | 1.000 |
| Zhang2016_ALK | 2 | 192 | 1.000 |
| Zhang2016_ALK | 3 | 64 | 0.438 |

## Secondary endpoints

T1 to T4 (hidden-resistance screening, inhibitor choice and regret, value of a third measurement, binding-mode strata) are in `results.json` under their own keys. They are descriptive and are not summarised here.

