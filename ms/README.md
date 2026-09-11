# Manuscript and replication package

The current source is `main.tex`; the compiled author-review PDF is `federated_lasso_manuscript.pdf`. The full simulation grid appears in `supplementary_results.pdf`, built from `supplement.tex`. All main-section numerical macros, tables and plots are rebuilt from committed result files. New comparison and public-data outputs have separate documented generators.

From the repository root:

```bash
python reproduce.py             # all tables/plots from stored results
python reproduce.py --full      # rerun every experiment first
python reproduce.py --build     # tables/plots, PDF, word count, LaTeX ZIP
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

The revised manuscript follows the supplied JSCS instructions: unstructured 150-word abstract, fewer than 10 keywords, named corresponding author, funding, competing-interest, data-availability and AI-use statements. Initial submissions permit a scholarly format with embedded figures and tables. `latex_source.zip` contains the editable source, bibliography and referenced figures. The manuscript and publisher declarations still require independent author review; see `../docs/REVISION_NOTES.md`.
