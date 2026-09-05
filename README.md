# How Many Local Epochs? Communication Cost and Variable Selection in Federated Lasso

One-page abstract, analysis notebook, and STAT 102B node data for a synchronous federated coordinate-descent Lasso.

## Dataset

This is **not** a named public repository dataset. The files are a **high-dimensional continuous-response regression dataset provided for the STAT 102B (Spring 2026) final project**. Predictors are unlabeled (columns `0`–`599`) and look approximately standard-scale.

After the course-supplied preprocessing and the given train/validation splits:

| Partition | Train \(n\) | Validation \(n\) | Predictors \(p\) |
|-----------|-------------|------------------|------------------|
| Node 1 (file `node1_*`; smallest site in the raw files) | 160 | 40 | 600 |
| Node 2 | 240 | 60 | 600 |
| Node 3 (largest site in the raw files) | 400 | 100 | 600 |
| **Pooled training** | **800** | **200** | **600** |

The abstract and the course specification label sites by sample size as \(m_j = 400, 240, 160\) (computationally partitioned into \(K=3\) disjoint sites). The analysis notebook re-labels the CSVs so the largest file is Node 1 with weight 0.50.

Because the true coefficient support is unknown, a centralized Lasso on the pooled sample is the reference.

## Files

- `Bolouri_Keivan_abs.pdf` / `Bolouri_Keivan_abs.tex` — one-page abstract
- `STAT102B_Final_Project.Rmd` — course analysis notebook
- `node*_X_*.csv`, `node*_y_*.csv` — site-level design matrices and responses

To rebuild the abstract PDF (TinyTeX / pdfLaTeX):

```text
pdflatex Bolouri_Keivan_abs.tex
```
