# Public diabetes regression data

This folder contains the publicly distributed diabetes regression dataset used
by Efron, Hastie, Johnstone and Tibshirani (2004), *Least Angle Regression*,
Annals of Statistics 32(2), 407–499,
[doi:10.1214/009053604000000067](https://doi.org/10.1214/009053604000000067).
It is separate from the generated `node*_*.csv` benchmark at repository root.

Primary documentation:

- [Original-data description and source links](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)
- [scikit-learn loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)

The 442 rows represent patients with diabetes. The ten baseline predictors are
age, sex, body mass index, average blood pressure, and six serum measurements
(`s1`–`s6`). The outcome `progression` is a quantitative measure of disease
progression one year after baseline. Predictor names preserve the source
convention. The study does not interpret selected variables as causal effects,
validated biomarkers, or known nonzero population coefficients.

`diabetes_raw.csv` is extracted by
`sklearn.datasets.load_diabetes(scaled=False)`. Thus this file does **not** use
the loader's default transformation calculated from all 442 observations.
`source_metadata.json` records the loader/package version and SHA-256 checksum
of the archived CSV. The extraction script requires no network download.

Run from repository root:

```bash
python ms/code/diabetes_study.py --replicates 100 --seed 20260911
```

To rebuild the table, selection frequencies, and summaries from the archived
fit outputs without rerunning the experiment:

```bash
python ms/code/diabetes_study.py --from-results
```

For split `r=0,...,99`, NumPy's `default_rng(20260911+r).permutation(442)`
assigns its first 260 rows to training, its next 80 to validation, and its last
102 to test. Training site sizes are `(130,78,52)` and validation site sizes
are `(40,24,16)`, taken consecutively from those permutations. Sites are
simulated partitions, not distinct hospitals. `split_assignments.csv` records
every assignment using zero-based source row indices, along with split and seed.

Within each split, each feature is centered and divided by its population
standard deviation (`ddof=0`) calculated from the 260 training rows only.
The response is centered at the training mean; its scale is unchanged.
These same transformations are applied to validation and test rows. A shared
training scale and intercept can be obtained from site-level counts, feature
sums, feature squared sums, and response sums. The largest-site comparators
also use this shared preprocessing; their coefficient fitting and penalty
tuning use only the largest site's training and validation rows.

Local penalties use minimum validation MSE on the main study's 30-point grid.
Pooled and largest-site Lasso fits are reported under minimum-MSE and
one-standard-error rules. The within-split reference support is the pooled
one-standard-error fit. ADMM uses the weighted mean of local penalties.
FedAvg, post-fit thresholding (ST), and per-round thresholding (P) use
`E in {1,3,20}` and the main study's threshold grids. All five full P fits
are counted in its communication/work outputs. Averaging methods stop at
update norm below `1e-5` or at 500 rounds. Nonconverged capped iterates and every
candidate threshold remain in the summaries; selected/candidate cap counts and
diagnostics are recorded. ADMM is allowed 20,000 iterations and must satisfy
its residual criterion. A preliminary run identified a nonconvergent P candidate;
the final protocol retains all 100 splits and uses the same 500-round averaging
cap as the main simulation instead of removing problematic splits or candidates.
The held-out test rows are used only for final prediction-error evaluation.

Outputs in `ms/code/` are `diabetes_results.csv` (all fitted estimates and
metrics), `diabetes_summary.csv`, `diabetes_selection_frequencies.csv`, and
`diabetes_study_metadata.json`. The manuscript table is
`ms/tables/diabetes_results.tex`. A coordinate is selected when its absolute
coefficient on the common training-standardized feature scale exceeds `1e-4`.
When both compared supports are empty, F1 and Jaccard are defined as one.

The experiment reuses one fixed dataset in 100 random splits. Reported standard
deviations describe split variability; they are not independent-sample standard
errors or population confidence intervals. Pairwise support overlap and feature
selection frequencies are descriptive and cannot establish true support.
