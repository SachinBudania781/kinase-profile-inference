# Results

Summary tables written by `evaluator.py`. Raw predictions (`runs/`) are not committed because they are large and can be regenerated; their hashes are recorded.

| Folder | What it is | Status |
| --- | --- | --- |
| `dev/` | Baseline development run D0 (13 Nov 2026) on the 10 development panels, with the code **before** amendments A1 to A4. `DEV_RESULTS.md` is the readable version; `results.json` and `panel_budget_table.csv` are the raw evaluator output. | Done |
| `dev_final/` | The same 10 development panels with the final code (after A1 to A4), all roster methods, 18 runs, no failed episodes. SER 4.24% (95% CI 2.91 to 5.28). `DEV_RESULTS.md` is the readable version. | Done (24 Nov 2026) |
| `lockbox_A/` | One-shot scored run on the seven Lockbox A panels (H1, primary test). SER 3.52% (95% CI 1.48 to 5.60). `LOCKBOX_RESULTS.md` is the readable version. | Done (24 Nov 2026) |
| `lockbox_B/` | One-shot scored run on the three Lockbox B panels (H1-B, replication). SER 16.78% (95% CI 12.71 to 19.53). `LOCKBOX_RESULTS.md` is the readable version. | Done (24 Nov 2026) |

Each `DEV_RESULTS.md` is produced from a `results.json` and `panel_budget_table.csv` by `dev_tools/make_dev_results_md.py`:

```
python dev_tools/make_dev_results_md.py --results results/dev_final --data data --out results/dev_final/DEV_RESULTS.md --title "Development results, final code"
```

Reading the numbers: gain G = 100 x (MAE of XGBoost - MAE of the method) / MAE of XGBoost, so positive means the method beats XGBoost. Equal weight per study.

Lockbox reports are produced the same way from the evaluator output:

```
python dev_tools/make_lockbox_results_md.py --results results/lockbox_A --lockbox A --study-roots data --out results/lockbox_A/LOCKBOX_RESULTS.md
python dev_tools/make_lockbox_results_md.py --results results/lockbox_B --lockbox B --study-roots data_lockbox_B --out results/lockbox_B/LOCKBOX_RESULTS.md
```
