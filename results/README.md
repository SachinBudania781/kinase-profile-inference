# Results

Summary tables written by `evaluator.py`. Raw predictions (`runs/`) are not committed because they are large and can be regenerated; their hashes are recorded.

| Folder | What it is | Status |
| --- | --- | --- |
| `dev/` | Baseline development run D0 (13 Nov 2026) on the 10 development panels, with the code **before** amendments A1 to A4. `DEV_RESULTS.md` is the readable version; `results.json` and `panel_budget_table.csv` are the raw evaluator output. | Done |
| `dev_final/` | The same 10 development panels with the final code (after A1 to A4), all five roster methods. | To be added when the run finishes |
| `lockbox_A/` | One-shot scored run on the seven Lockbox A panels (H1, primary test). | After the release freeze |
| `lockbox_B/` | One-shot scored run on the three Lockbox B panels (H1-B, replication). | After the release freeze |

Each `DEV_RESULTS.md` is produced from a `results.json` and `panel_budget_table.csv` by `dev_tools/make_dev_results_md.py`:

```
python dev_tools/make_dev_results_md.py --results results/dev_final --data data --out results/dev_final/DEV_RESULTS.md --title "Development results, final code"
```

Reading the numbers: gain G = 100 x (MAE of XGBoost - MAE of the method) / MAE of XGBoost, so positive means the method beats XGBoost. Equal weight per study.
