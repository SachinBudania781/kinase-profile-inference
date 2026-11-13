# Dataset roles

The study asks whether two or three measured inhibitor responses can predict another response for the same kinase variant and assay. This inventory records the role of each of the 59 folders in the study archive. A source publication can contribute more than one folder.

## Core development panels

These six panels supported baseline comparisons and development of the expert-routing and correction methods. Each panel retained its assay scale and grouping rules.

| Archive folder | Role |
| --- | --- |
| `Estevam2025_MET` | Initial MET deep mutational scanning benchmark. The exon 14 and exon 14-skipped backgrounds were tracked separately. |
| `DuongLy2016_multi_kinase` | Biochemical remaining-activity benchmark across kinases. |
| `Robichaux2021_EGFR` | Cellular EGFR IC50-ratio benchmark. |
| `Nakamura2021_FGFR1_FGFR2_FGFR3` | Cellular FGFR IC50 benchmark with gene-held-out comparisons. |
| `KiRHub2026_multi_kinase` | Cross-kinase residual activity at 1 µM and observation-budget comparisons. |
| `OKL2026_multi_kinase` | Cross-kinase KINOMEscan residual binding at 1 µM. |

## Additional tests and follow-up

These four panels were used in separate tests or follow-up analyses. Their uses varied by analysis.

| Archive folder | Role |
| --- | --- |
| `Jung2025_ABL1` | Separate five-inhibitor ABL1 evaluation of an earlier model, followed by residual and covariance analyses. |
| `Chakraborty2024_SRC` | Four-inhibitor SRC test, followed by resistance and untreated-activity analyses. |
| `Lee2026_EGFR` | Two-observation test of a residual model in the three-inhibitor 11-18 cell assay. |
| `Oh2026_ALK` | Two-observation test in the three-inhibitor H3122 cell assay, followed by an untreated-fitness analysis. |

## Seven-panel benchmark

An earlier SAER model was evaluated on these panels. Diagnostics from that evaluation informed the final SER safeguards. The final SER analysis reused the benchmark episodes. The two budgets specify how many inhibitor responses are available for each prediction. Abadie has three inhibitors total, so it supports only two observations and one hidden response. Together the panels yield 13 panel–budget combinations.

| Archive folder | Assay endpoint | Observed inhibitors |
| --- | --- | --- |
| `An2023_EGFR` | Published log₂ enrichment for the Table 5 fast-proliferating cohort | 2, 3 |
| `Abadie2026_EGFR` | PC-9 exon 20 day-15 log₂ drug-versus-vehicle contrast | 2 |
| `Bahcall2022_MET` | TPR–MET cellular IC50 in Ba/F3 cells | 2, 3 |
| `Fujino2019_MET` | MET exon 14-skipping cellular IC50 with censor limits retained | 2, 3 |
| `Wylie2017_ABL1` | BCR–ABL1 cellular IC50 for ATP-site and allosteric inhibitors | 2, 3 |
| `Zhang2016_ALK` | EML4–ALK proliferation IC50 | 2, 3 |
| `Gainor2016_ALK` | EML4–ALK phosphorylation IC50 | 2, 3 |

## Feasibility and data intake

These 39 folders document earlier potency-transfer, assay-feasibility and data-intake work. They sit outside the six-panel development comparison and the seven-panel benchmark above.

| Archive folder | Role |
| --- | --- |
| `AsciminibDMS_ABL1` | Asciminib resistance feasibility |
| `CellEKT2025_multi_kinase` | Cellular target-engagement feasibility |
| `ChEMBL37_multi_target` | Binding benchmark and prior-pair overlap audit |
| `Davis2011_multi_kinase` | Construct comparisons, affinity evaluation and censoring audit |
| `DrugMap_multi_protein` | Cysteine ligandability feasibility |
| `M4K_2026_ALK2` | ALK2 potency translation and learning-curve analysis |
| `Masuzawa2017_EGFR` | Cellular IC50 intake and quality checks |
| `MdrDB2023_multi_kinase` | Intake and assay heterogeneity audit |
| `Murray2021_ALK` | Cellular IC50 intake for TPX-0131 and comparator inhibitors |
| `Pendyala2026_MAPK_pathway` | Gene-held-out activity-from-abundance analysis |
| `Pritchard2025_ABL1` | Imatinib dose-dependent growth and untreated-control intake |
| `acs_jmedchem_0c00395_ACVR1_ALK2` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_0c01199_ACVR1_ALK2` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_0c02274_CHEMBL5284` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_1c00440_BMP2K` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_1c00440_CSF1R` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_1c00440_MYLK2` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_1c00440_NUAK1` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_1c00440_STK17A` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_2c00697_PIKFYVE` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_2c00751_LIMK1` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_2c00751_LIMK2` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_3c01927_CHEMBL5251` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_3c02245_CHEMBL2409` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_3c02246_CHEMBL5699` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_3c02246_CHEMBL6149` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_4c00962_CHEMBL4070` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_4c01337_CHEMBL4295801` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_4c01337_MLLT1` | Matched IC50 potency-transfer comparison |
| `acs_jmedchem_4c02531_PIKFYVE` | Biochemical-to-cellular potency comparison |
| `acs_jmedchem_5c00097_CHEMBL260` | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_2c00523_CHEMBL5251` | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_2c00530_CHEMBL3629` | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_4c00219_BMP2K` | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_4c00219_CHEMBL3830` | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_4c00219_CHEMBL5728` | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_4c00269_CHEMBL3778` | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_4c00579_NUAK1` | Biochemical-to-cellular potency comparison |
| `acsmedchemlett_5c00147_CHEMBL3778` | Biochemical-to-cellular potency comparison |

## Literature references

These three folders contain review-derived records used for the literature audit. They sit outside model development and evaluation.

| Archive folder | Role |
| --- | --- |
| `Blood501569_ABL1_review` | ABL1 IC50 literature audit |
| `SeminOncol01016_KIT_review` | KIT IC50 literature audit |
| `TherAdv17588359221101691_RET_review` | RET IC50 literature audit |
