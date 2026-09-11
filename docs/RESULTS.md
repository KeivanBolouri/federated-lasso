# Result definitions

The main study uses 200 seeds per design: 5000–5066, 6000–6066, and 7000–7065. One row is one method/schedule applied to one replicate. The main raw file contains 29,400 rows (49 per replicate). A coefficient is active above absolute value 1e-4.

- `precision`, `recall`, `F1`, `jaccard`: compare selected and true support in the main simulations. Precision and F1 are zero when no variables are selected; no empty-model replicates are omitted from means.
- `coef_err`: Euclidean coefficient-estimation error.
- `test_mse`: mean squared error on 2,000 independent test observations from the explicitly specified target distribution.
- `rounds`, `local_epochs`, `vector_kib`: all federated candidate fitting required for a returned estimate. For FedAvg-P these sum all five threshold candidates.
- `selected_rounds`, `selected_local_epochs`, `selected_vector_kib`: cost of the selected trajectory alone.
- `tuning_*`: cost of the discarded fitted candidate trajectories.
- `scalar_kib`, `scalar_rounds`: additional validation summaries and exchanges.
- `upload_kib`, `total_rounds`: fitting plus scalar validation. A KiB is 1,024 bytes; values are double precision (8 bytes).
- Convergence columns record the applicable update, residual or dual-gap checks. A fixed-point update check is not a pooled-Lasso KKT condition.

The main study's upload convention excludes downloads, network headers, initial local penalty-search computation and setup time. CD sweeps are a local-work measure within the CD family, not a wall-clock comparison with ADMM. The `ADMM-CV` paths are evaluated centrally using the mathematically equivalent pooled objective; their unmeasured federated tuning costs are left missing, not treated as zero.

`paired_epoch_results.csv` reports within-replicate E=20 minus E=1 F1 changes and paired Monte Carlo standard errors. Summary MCSEs quantify numerical uncertainty across generated datasets, not statistical uncertainty for one fitted dataset.

The separate budget study uses the same common penalty at every client for both algorithm families. Its selected-fit curves condition on the selected learning-rate path. `budget_candidates.csv` and `budget_cost_ledger.csv` expose the extra learning-rate candidates and setup communication. Conditional equal training budgets are not equal total tuning budgets.

For the budget study, one vector-upload round is `3 × 600 × 8 = 14,400` bytes. At 100 selected-fit rounds:

| Quantity | Bytes | KiB |
|---|---:|---:|
| Either selected vector path | 1,440,000 | 1,406.25 |
| CD-common, including its three local-penalty scalars | 1,440,024 | 1,406.2734375 |
| Adapted FedDualAvg, all three paths plus six setup and nine validation scalars | 4,320,120 | 4,218.8671875 |

The shared whole-grid ledger records 12,960,000 DA vector bytes, 4,320,000 CD vector bytes, 1,080 DA checkpoint-validation scalar bytes and 48 shared setup bytes per replicate. Summing standalone checkpoint totals would count reused prefixes and setup repeatedly.

`budget_total_cap_summary.csv` and `budget_total_cap_paired.csv` report a separate sensitivity analysis under one common **total-upload cap** of 1,440,120 bytes. The largest affordable archived checkpoint is selected using cost alone: CD at 100 rounds and DA at 20 rounds, with all three DA candidate paths counted. The coarse checkpoint grid leaves unused budget, so this is a common-cap comparison with unequal actual spending. [BUDGET_TOTAL_CAP.md](BUDGET_TOTAL_CAP.md) gives the selection rule, actual costs and paired differences. ST and P are not included in either budget comparison.

The archived public diabetes study, excluded from the revised article, repeats partitions of one finite dataset. Its reported variability is standard deviation across splits, not an independent-dataset MCSE. Pooled-support agreement is not scientific correctness; selection frequency and pairwise Jaccard measure stability across the stated splits. Capped returned iterates remain in all summaries and are explicitly counted. The original fixed synthetic node benchmark also has unknown true support, so its F1 denotes agreement with its pooled reference.
