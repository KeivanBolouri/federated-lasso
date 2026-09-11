# Common total-upload-cap sensitivity analysis

This archived-results analysis applies one cap of 1,440,120 bytes (1406.37 KiB) to each method, design and local-step setting. The cap is the 100-round selected-vector budget plus 120 bytes of DA setup and validation scalars. The selection rule uses cost only: choose the largest affordable checkpoint in {5, 10, 20, 50, 100}. CD therefore uses 100 rounds, and DA uses 20 rounds for each of its three learning-rate candidates. Candidate selection still uses validation MSE. No observations are rerun or interpolated.

| Method | Selected rounds | Actual total upload (KiB) | Unused cap (bytes) |
|---|---:|---:|---:|
| CD-common | 100 | 1406.2734375 | 96 |
| Adapted FedDualAvg | 20 | 843.8671875 | 576000 |

Both procedures obey the cap, but their actual spending differs because only the recorded checkpoints are available. The analysis does not establish an equal-spending or wall-clock comparison. Local gradient evaluations and CD sweeps also differ.

The supplementary table and `budget_total_cap_summary.csv` contain method-specific means and MCSEs. Below, differences are adapted FedDualAvg minus CD-common within each paired seed; parentheses give paired MCSEs.

| Design | E | F1 difference (MCSE) | Objective-gap difference (MCSE) | Test-MSE difference (MCSE) |
|---|---:|---:|---:|---:|
| Independent | 1 | +0.4388 (0.0188) | -1.6626 (0.0499) | -0.8588 (0.1420) |
| Independent | 5 | +0.4637 (0.0207) | -1.6070 (0.0538) | -2.1076 (0.1981) |
| Independent | 20 | +0.4618 (0.0205) | -1.6321 (0.0563) | -2.1602 (0.2005) |
| Correlated | 1 | +0.3765 (0.0095) | -1.5427 (0.0360) | -0.0998 (0.0873) |
| Correlated | 5 | +0.4640 (0.0087) | -1.5253 (0.0341) | -1.9712 (0.1205) |
| Correlated | 20 | +0.4670 (0.0088) | -1.5379 (0.0349) | -1.9471 (0.1256) |
| Heterogeneous | 1 | +0.2430 (0.0247) | -3.0025 (0.0859) | -0.7053 (0.2080) |
| Heterogeneous | 5 | +0.2873 (0.0306) | -2.3796 (0.0641) | -1.7145 (0.2697) |
| Heterogeneous | 20 | +0.2887 (0.0323) | -2.3816 (0.0698) | -1.7519 (0.2796) |

All 50 seeds per design are retained. Positive F1 differences favour DA; negative objective-gap and test-MSE differences favour DA. ST and P are absent from this experiment, so these results do not rank those retrofit procedures against DA.
