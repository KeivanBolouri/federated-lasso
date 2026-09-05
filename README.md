# How Many Local Epochs? Communication Cost and Variable Selection in Federated Lasso

One-page abstract, analysis notebook, and node-level data for a synchronous federated coordinate-descent Lasso.

## Dataset

A high-dimensional continuous-response regression sample (\(n = 800\) training observations, \(p = 600\) predictors). The data-generating process and true coefficient vector were not disclosed, so a centralized Lasso on the pooled sample is the reference support rather than a known truth. The training sample is computationally partitioned into \(K = 3\) disjoint sites with \(m_j = 400, 240, 160\); variation across sites comes from that partition, not from distinct data sources.

| Partition | Train \(n\) | Validation \(n\) | Predictors \(p\) |
|-----------|-------------|------------------|------------------|
| Node 1 | 400 | 100 | 600 |
| Node 2 | 240 | 60 | 600 |
| Node 3 | 160 | 40 | 600 |
| **Pooled** | **800** | **200** | **600** |

The CSV files are named `node1`, `node2`, `node3` in reverse size order (160, 240, 400). The notebook re-labels them so Node 1 is the largest site (weight 0.50). Each site selects its penalty on its own validation split; no row-level data are exchanged during federated fitting.

## Files

- `Bolouri_Keivan_abs.pdf` / `Bolouri_Keivan_abs.tex` — one-page abstract
- `FederatedLasso.Rmd` — analysis notebook (full local-epoch grid \(E\in\{1,2,3,5,8,10,15,20\}\))
- `node*_X_*.csv`, `node*_y_*.csv` — site-level design matrices and responses (all 12 files)
- `data/` — simulation and pilot CSVs (raw draws, summaries, and `data/sim/` replication files)

To rebuild the abstract PDF (TinyTeX / pdfLaTeX):

```text
pdflatex Bolouri_Keivan_abs.tex
```
