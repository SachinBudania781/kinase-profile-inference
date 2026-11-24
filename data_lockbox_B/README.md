# Lockbox B panels

Three panels curated from open-access publications for the replication test (H1-B). The source tables were transcribed by hand from the published tables; the transcription specs are in `_specs/`, and the descriptive similarity of these panels to Lockbox A is in `_similarity/`. No model was run on these panels before the release freeze; the one-shot run and its scoring were done on 24 Nov 2026 (see the main README, section 11). The folders were kept outside the repository until then and were added afterwards.

| Folder | Kinase | Source |
| --- | --- | --- |
| `Villa2024_ALK` | ALK | Villa et al. 2024, New pan-ALK inhibitor-resistant EML4::ALK mutations detected by liquid biopsy in lung cancer patients. npj Precision Oncology <https://doi.org/10.1038/s41698-024-00498-w> |
| `Takei2025_ALK` | ALK | Takei et al. 2025, Preclinical Prediction of Resistance Mutations and Proposal of Sequential Treatment Strategies for ALK-positive Lung Cancer Using Next-generation ALK Inhibitors. Pharmaceutical Research <https://doi.org/10.1007/s11095-025-03916-1> |
| `Peng2024_ROS1` | ROS1 | Peng et al. 2024, Discovery of Oral Degraders of the ROS1 Fusion Protein with Potent Activity against Secondary Resistance Mutations. Journal of Medicinal Chemistry <https://doi.org/10.1021/acs.jmedchem.4c01205> |

Each panel folder holds `SER_responses.csv` (the table the pipeline reads), `source_IC50.csv` (the transcribed values), `EDA.json`, `study.json` and `dropped_variants.json`.

## Check that a copy is intact

```
cd data_lockbox_B
sha256sum -c CHECKSUMS.sha256 --quiet
```

`CHECKSUMS.sha256` lists a SHA-256 for every file in this folder except itself and this README. The source papers are open access (CC BY); check each source before reusing a table outside this study.
