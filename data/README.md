# Study data

Cleaned study tables for all 59 archive folders, copied **byte for byte** from the study archive (the seven Lockbox A panels were added on 24 Nov 2026, after the lockbox run was scored). They are the inputs read by `data_pipeline.py`.
Run any script with `--archive-root data` to use them. The layout is `data/<study>/...` with a `study.json` (kinase, source) in each folder.

## Lockbox panels

The seven Lockbox A panels are in this folder (table below). They were held back by the pre-registered lockbox policy until the one-shot lockbox runs were done and scored, and were added on 24 Nov 2026. The three Lockbox B panels are in [`../data_lockbox_B/`](../data_lockbox_B/).

## Lockbox A panels (7)

| Folder | Kinase | Source | Size (KB) | Note |
| --- | --- | --- | --- | --- |
| `An2023_EGFR` | EGFR | <https://doi.org/10.1016/j.trsl.2022.11.002> | 35 | Lockbox A panel (primary test), added after scoring |
| `Abadie2026_EGFR` | EGFR | <https://doi.org/10.1016/j.xgen.2026.101167> | 123 | Lockbox A panel (primary test), added after scoring |
| `Bahcall2022_MET` | MET | <https://doi.org/10.1158/1535-7163.MCT-21-0344> | 8 | Lockbox A panel (primary test), added after scoring |
| `Fujino2019_MET` | MET | <https://doi.org/10.1016/j.jtho.2019.06.023> | 20 | Lockbox A panel (primary test), added after scoring |
| `Wylie2017_ABL1` | ABL1 | <https://doi.org/10.1038/nature21702> | 16 | Lockbox A panel (primary test), added after scoring |
| `Zhang2016_ALK` | ALK | <https://doi.org/10.1158/1078-0432.CCR-16-0569> | 9 | Lockbox A panel (primary test), added after scoring |
| `Gainor2016_ALK` | ALK | <https://doi.org/10.1158/2159-8290.CD-16-0596> | 9 | Lockbox A panel (primary test), added after scoring |

## Check that a copy is intact

`CHECKSUMS.sha256` lists a SHA-256 for every copied study file (234 files; this README and the checksum file itself are not listed).

```
cd data
sha256sum -c CHECKSUMS.sha256 --quiet
```

On Windows PowerShell: `Get-FileHash -Algorithm SHA256 <file>` and compare with the line in `CHECKSUMS.sha256`.

## Sources and licences

Each `study.json` names the source (paper DOI or database URL). The tables are cleaned and reformatted values from those sources. **The licence of each source applies** (for example ChEMBL is CC BY-SA 3.0; open-access paper supplements are usually CC BY; others follow the publisher's terms). The licences were not checked folder by folder, so check the source before reusing a table outside this study.

## Core development panels (6)

| Folder | Kinase | Source | Size (KB) | Note |
| --- | --- | --- | --- | --- |
| `Estevam2025_MET` | MET | <https://doi.org/10.7554/eLife.101882> | 7,274 | core development panel |
| `DuongLy2016_multi_kinase` | Multiple kinases | <https://doi.org/10.1016/j.celrep.2015.12.080> | 1,256 | core development panel |
| `Robichaux2021_EGFR` | EGFR | <https://doi.org/10.1038/s41586-021-03898-1> | 264 | core development panel |
| `Nakamura2021_FGFR1_FGFR2_FGFR3` | FGFR1, FGFR2, FGFR3 | <https://doi.org/10.1038/s41698-021-00204-0> | 61 | core development panel |
| `KiRHub2026_multi_kinase` | Multiple kinases | <https://doi.org/10.1038/s41587-026-03090-8> | 5,723 | core development panel |
| `OKL2026_multi_kinase` | Multiple kinases | <https://doi.org/10.64898/2026.03.17.711623> | 1,576 | core development panel |

## Development stress tests (4)

| Folder | Kinase | Source | Size (KB) | Note |
| --- | --- | --- | --- | --- |
| `Jung2025_ABL1` | ABL1 | <https://doi.org/10.1038/s41551-025-01531-4> | 1,141 | stress-test panel |
| `Chakraborty2024_SRC` | SRC | <https://pmc.ncbi.nlm.nih.gov/articles/PMC10902203/> | 435 | stress-test panel |
| `Lee2026_EGFR` | EGFR | <https://www.nature.com/articles/s41698-026-01726-1> | 109 | stress-test panel |
| `Oh2026_ALK` | ALK | <https://doi.org/10.1186/s13059-026-03977-4> | 129 | stress-test panel |

## Feasibility and data intake (39)

Earlier potency-transfer, assay-feasibility and data-intake work. Outside model development and evaluation.

| Folder | Kinase | Source | Size (KB) | Role |
| --- | --- | --- | --- | --- |
| `AsciminibDMS_ABL1` | ABL1 | <https://github.com/TheBioinformatician/Dual-Functional-Genomics-Asciminib> | 251 | Asciminib resistance feasibility |
| `CellEKT2025_multi_kinase` | Multiple kinases | <https://doi.org/10.1016/j.mcpro.2025.100961> | 21 | Cellular target-engagement feasibility |
| `ChEMBL37_multi_target` | Multiple targets | <https://www.ebi.ac.uk/chembl/> | 5,406 | Binding benchmark and prior-pair overlap audit |
| `Davis2011_multi_kinase` | Multiple kinases | <https://doi.org/10.1038/nbt.1990> | 1,654 | Construct comparisons, affinity evaluation and censoring audit |
| `DrugMap_multi_protein` | Multiple proteins including kinases | <https://doi.org/10.1016/j.cell.2024.03.027> | 60,851 | Cysteine ligandability feasibility |
| `M4K_2026_ALK2` | ACVR1_ALK2 | <https://doi.org/10.5281/zenodo.21164806> | 19 | ALK2 potency translation and learning-curve analysis |
| `Masuzawa2017_EGFR` | EGFR | <https://doi.org/10.18632/oncotarget.22297> | 4 | Cellular IC50 intake and quality checks |
| `MdrDB2023_multi_kinase` | Multiple kinases | <https://doi.org/10.1038/s42004-023-00920-7> | 10 | Intake and assay heterogeneity audit |
| `Murray2021_ALK` | ALK | <https://doi.org/10.1158/1535-7163.MCT-21-0221> | 5 | Cellular IC50 intake for TPX-0131 and comparator inhibitors |
| `Pendyala2026_MAPK_pathway` | MAPK pathway kinases | <https://doi.org/10.5281/zenodo.23121367> | 3,023 | Gene-held-out activity-from-abundance analysis |
| `Pritchard2025_ABL1` | ABL1 | <https://doi.org/10.1101/2025.10.31.685392> | 913 | Imatinib dose-dependent growth and untreated-control intake |
| `acs_jmedchem_0c00395_ACVR1_ALK2` | ACVR1_ALK2 | <https://doi.org/10.1021/acs.jmedchem.0c00395> | 3 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_0c01199_ACVR1_ALK2` | ACVR1_ALK2 | <https://doi.org/10.1021/acs.jmedchem.0c01199> | 3 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_0c02274_CHEMBL5284` | CHEMBL5284 | <https://doi.org/10.1021/acs.jmedchem.0c02274> | 3 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_1c00440_BMP2K` | BMP2K | <https://doi.org/10.1021/acs.jmedchem.1c00440> | 4 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_1c00440_CSF1R` | CSF1R | <https://doi.org/10.1021/acs.jmedchem.1c00440> | 4 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_1c00440_MYLK2` | MYLK2 | <https://doi.org/10.1021/acs.jmedchem.1c00440> | 3 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_1c00440_NUAK1` | NUAK1 | <https://doi.org/10.1021/acs.jmedchem.1c00440> | 3 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_1c00440_STK17A` | STK17A | <https://doi.org/10.1021/acs.jmedchem.1c00440> | 4 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_2c00697_PIKFYVE` | PIKFYVE | <https://doi.org/10.1021/acs.jmedchem.2c00697> | 2 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_2c00751_LIMK1` | LIMK1 | <https://doi.org/10.1021/acs.jmedchem.2c00751> | 7 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_2c00751_LIMK2` | LIMK2 | <https://doi.org/10.1021/acs.jmedchem.2c00751> | 7 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_3c01927_CHEMBL5251` | CHEMBL5251 | <https://doi.org/10.1021/acs.jmedchem.3c01927> | 2 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_3c02245_CHEMBL2409` | CHEMBL2409 | <https://doi.org/10.1021/acs.jmedchem.3c02245> | 2 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_3c02246_CHEMBL5699` | CHEMBL5699 | <https://doi.org/10.1021/acs.jmedchem.3c02246> | 2 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_3c02246_CHEMBL6149` | CHEMBL6149 | <https://doi.org/10.1021/acs.jmedchem.3c02246> | 2 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_4c00962_CHEMBL4070` | CHEMBL4070 | <https://doi.org/10.1021/acs.jmedchem.4c00962> | 2 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_4c01337_CHEMBL4295801` | CHEMBL4295801 | <https://doi.org/10.1021/acs.jmedchem.4c01337> | 2 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_4c01337_MLLT1` | MLLT1 | <https://doi.org/10.1021/acs.jmedchem.4c01337> | 3 | Matched IC50 potency-transfer comparison |
| `acs_jmedchem_4c02531_PIKFYVE` | PIKFYVE | <https://doi.org/10.1021/acs.jmedchem.4c02531> | 8 | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_5c00097_CHEMBL260` | CHEMBL260 | <https://doi.org/10.1021/acs.jmedchem.5c00097> | 2 | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_2c00523_CHEMBL5251` | CHEMBL5251 | <https://doi.org/10.1021/acsmedchemlett.2c00523> | 2 | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_2c00530_CHEMBL3629` | CHEMBL3629 | <https://doi.org/10.1021/acsmedchemlett.2c00530> | 2 | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_4c00219_BMP2K` | BMP2K | <https://doi.org/10.1021/acsmedchemlett.4c00219> | 2 | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_4c00219_CHEMBL3830` | CHEMBL3830 | <https://doi.org/10.1021/acsmedchemlett.4c00219> | 2 | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_4c00219_CHEMBL5728` | CHEMBL5728 | <https://doi.org/10.1021/acsmedchemlett.4c00219> | 2 | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_4c00269_CHEMBL3778` | CHEMBL3778 | <https://doi.org/10.1021/acsmedchemlett.4c00269> | 2 | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_4c00579_NUAK1` | NUAK1 | <https://doi.org/10.1021/acsmedchemlett.4c00579> | 7 | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_5c00147_CHEMBL3778` | CHEMBL3778 | <https://doi.org/10.1021/acsmedchemlett.5c00147> | 2 | Biochemical-to-cellular potency comparison |

## Literature references (3)

Review-derived records for the literature audit. Outside model development and evaluation.

| Folder | Kinase | Source | Size (KB) | Role |
| --- | --- | --- | --- | --- |
| `Blood501569_ABL1_review` | ABL1 | <https://doi.org/10.1182/blood-2013-05-501569> | 5 | ABL1 IC50 literature audit |
| `SeminOncol01016_KIT_review` | KIT | <https://doi.org/10.1053/j.seminoncol.2011.01.016> | 1 | KIT IC50 literature audit |
| `TherAdv17588359221101691_RET_review` | RET | <https://doi.org/10.1177/17588359221101691> | 3 | RET IC50 literature audit |

The full inventory with roles for all 59 archive folders is in [`../data_manifest/DATASETS.md`](../data_manifest/DATASETS.md).
