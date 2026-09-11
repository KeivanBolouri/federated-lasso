# Local Epochs, Aggregation and Variable Selection in Federated Coordinate-Descent Lasso

Research code, manuscript source, datasets and reproducible results for Keivan Bolouri's study of local coordinate-descent epochs in federated Lasso.

**[Read the revised manuscript](ms/federated_lasso_manuscript.pdf)** · **[Supplementary results](ms/supplementary_results.pdf)** · **[LaTeX source archive](ms/latex_source.zip)** · **[Revision and author-review record](docs/REVISION_NOTES.md)**

The study separates three questions: which variables are selected, how accurately the specified Lasso objective is solved, and how much communication and local computation are required. It compares local and pooled tuning, averaging, server thresholding, consensus ADMM, and an explicitly documented deterministic specialization of FedDualAvg. Results should be interpreted within the tested designs; this repository makes no claim of guaranteed journal acceptance or formal privacy.

## Reproduce

Python 3.11 or newer and the versions in `requirements.txt` are used. Install the dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Regenerate all tables, figures and summaries from the committed result files:

```bash
python reproduce.py
```

Recompute every experiment, then regenerate the outputs:

```bash
python reproduce.py --full --workers 4
```

Build the manuscript and submission source archive:

```bash
python reproduce.py --build
```

The build requires `pdflatex`, `bibtex`, `latexmk` and Poppler's `pdftotext`, with the standard LaTeX article, mathematics, graphics, bibliography and table packages. The source uses ordinary numbered algorithm steps and does not require `algorithm` or `algpseudocode` packages. `--build` also refreshes the displayed word count and produces the supplementary PDF. See [manuscript instructions](ms/README.md) for individual commands and [result definitions](docs/RESULTS.md) for metrics and cost conventions.

## Experiments and files

| Location | Contents |
|---|---|
| `ms/main.tex`, `ms/sections/`, `ms/refs.bib` | Manuscript and bibliography |
| `ms/code/fedlasso.py` | Coordinate-descent, thresholding, validation and ADMM implementations |
| `ms/code/run_sim.py`, `run_study.py` | Main simulations: 200 replicates in each of three designs |
| `ms/code/feddualavg.py`, `budget_study.py` | Published-algorithm specialization and communication-budget comparison: 50 replicates per design |
| `ms/code/diabetes_study.py` | Public diabetes example: 100 prespecified data splits |
| `ms/code/real_data.py`, `eps_sens.py` | Fixed synthetic node-data benchmark and activity-threshold sensitivity |
| `ms/code/*results*.csv`, `*raw.csv`, `*summary*.csv` | Committed numerical results |
| `ms/code/*metadata.json`, `data/diabetes/` | Reproduction metadata and public-data provenance |
| `ms/tables/`, `ms/figures/` | Generated manuscript tables and figures |
| `tests/`, `ms/code/tests/` | Numerical and accounting checks |
| `node*_*.csv` | Original fixed synthetic benchmark, preserved without modification |
| `FederatedLasso.Rmd`, `Bolouri_Keivan_abs.*`, `figure1.pdf`, `data/pilot*`, `data/sim*` | Earlier exploratory code, abstract and results; retained for history and not used to generate the revised paper |

## Data provenance

The author confirmed that the original node CSVs are **synthetic**. Their original generation procedure, seed and coefficient vector were not supplied with this revision. They are treated as a fixed numerical benchmark, never as an observational application or a known-support simulation. The current main simulations have explicit generators and recorded seeds.

The public diabetes example uses the 442-observation, 10-predictor dataset described by Efron et al. (2004), distributed with scikit-learn. Its source, variables, raw values, hashes and split assignments are documented in [data/diabetes](data/diabetes/). All preprocessing uses training rows only.

## Status

This is a revised research manuscript for author review. The revision corrects numerical reporting and adds experiments, but new scientific material and AI-use declarations require the author's independent review before journal submission. See the [revision record](docs/REVISION_NOTES.md). Funding was confirmed by the author as no dedicated funding.
