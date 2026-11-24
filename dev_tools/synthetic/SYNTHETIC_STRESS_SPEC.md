# Synthetic stress-test bank: specification (written before any model was run on it)

Status: development tool only. Date: 13 November 2026.

## What it is
A bank of artificial kinase-variant response studies, made by a random generator (`synth_bank.py`).
The data are NOT real and are never used to train or tune a model for the final claim.
They are used to find failure points of SER, the XGBoost reference and the roster methods.

## What it is not
- It is not built from any lockbox panel. The generator uses only (a) general assay behaviour
  (latent drug-response factors, site effects, noise, saturation, censoring, missing cells) and
  (b) the general shape of the DEVELOPMENT panels (number of inhibitors, number of variants, sites).
  No lockbox response value, correlation or panel-specific feature was used to choose any range.
- It is not used to pick hyperparameters by maximising gain. A change is kept only if it also helps
  on the real development studies (leave-one-study-out) and does not hurt on the stress bank.

## Axes that vary across studies (each drawn independently, wide ranges on purpose)
1. Number of inhibitors D: 3 to 120.  2. Number of variants n: 40 to 3000.
3. Variants per site (site count 4 to ~400).  4. Latent rank 1 to 5.
5. Drug families and strength of drug-drug correlation (about 0.1 to 0.95); a few negatively correlated drugs.
6. Share of variance that is site-level (not transferable across held-out sites).
7. Fraction of "resistant" variants (heavy right tail).  8. Saturating (sigmoid) links, floor/ceiling effects.
9. Noise level per drug (signal-to-noise from about 0.7 to 10), heteroscedastic noise, heavy-tailed noise, outliers.
10. Site-by-drug batch offsets.  11. Study scale from 0.03 to 30 (tests scale assumptions) and drug-specific scales.
12. Missing cells (matrix format only): 0 to 50 percent, random or depending on the signal.
13. Censoring (complete-profile CSV format only): 0 to 60 percent of cells with flags +1 / -1.
14. Binding-mode labels (CSV format): two or three modes tied to drug families.
15. Planted adversarial features, each with a small probability: duplicate inhibitor, constant inhibitor,
    inhibitor 500 times larger in scale, pure-noise inhibitor, context-dependent sign flip, 70 percent missing inhibitor.

## Two file formats (so both loaders are exercised)
- `matrix_npz`: the development format (missing cells allowed, no censor flags).
- `ser_csv`: the complete-profile CSV format with censor flags and modes. (Code format only; no lockbox data involved.)

## Outputs and checks
For each study and budget k in {2, 3}: MAE of XGBoost, SER, CovarianceLocalMedian, RidgeSupports, QueryMedian;
SER gain over XGBoost with a site-clustered bootstrap CI; number of crashed or missing predictions;
runtime. Flags raised: crash; non-finite prediction; SER worse than XGBoost with CI below 0 ("harm");
SER gate switched on where it hurts; scale-dependent behaviour (same study at x0.001 and x1000 must give the same relative gain).
Then a regression of gain on the generator parameters to see which condition removes (or creates) SER's advantage.

## Disclosure
Final report and OSF/GitHub notes will state: "Synthetic stress tests (artificial data, no lockbox information)
were used during development to find failure points; they were not used for the confirmatory analysis."
