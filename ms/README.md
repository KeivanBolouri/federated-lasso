# Manuscript and replication package

The current source is `main.tex`; the compiled manuscript is `federated_lasso_manuscript.pdf`. Supplement S1 is `supplementary_results.pdf`, built from `supplement.tex`, and contains the full simulation grid and the common total-upload-cap analysis. Numerical macros, tables and plots are rebuilt from committed result files. The removed diabetes illustration remains an archived experiment in the repository and is excluded from the submission source archive.

From the repository root:

```bash
python reproduce.py             # all tables/plots from stored results
python reproduce.py --full      # rerun every experiment first
python reproduce.py --build     # tables/plots, both PDFs, LaTeX ZIP
```

Individual computations:

```bash
python ms/code/run_study.py --workers 4
python ms/code/real_data.py
python ms/code/eps_sens.py
python ms/code/diabetes_study.py --replicates 100
python ms/code/budget_study.py --reps 50 --workers 2
python ms/code/make_outputs.py
```

Rebuild only additional study summaries/plots:

```bash
python ms/code/diabetes_study.py --from-results
python ms/code/budget_study.py --from-results
```

Numerical checks:

```bash
python -m unittest discover -s ms/code/tests
python -m unittest discover -s tests
```

The main simulation runner can resume checkpoints with `--checkpoint-dir PATH`. Its manifest rejects checkpoints created by different source code or a different seed schedule. Output replacement is atomic after complete successful runs.

Direct LaTeX build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The manuscript has an unstructured 150-word abstract, six keywords, a named corresponding author, and funding, disclosure, data-availability and supplementary-material statements. `latex_source.zip` contains the editable source, bibliography, tables and figures required by the two documents. Build with `latexmk -pdf main.tex` and `latexmk -pdf supplement.tex` after extracting it. The build records an approximate inclusive word count separately; the title page does not display it. See `../docs/REVISION_NOTES.md` for the revision history.
