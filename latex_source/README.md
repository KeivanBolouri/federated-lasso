# Local Epochs, Aggregation and Variable Selection in Federated Coordinate-Descent Lasso

This archive contains the revised manuscript, separate supplement, and the inputs needed to reproduce their numerical results. `main.tex` builds the manuscript; `supplement.tex` builds Supplement S1. Both use standard LaTeX packages and the `apalike` bibliography style.

## Build the PDFs directly

Run these commands from the extracted archive directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build supplement.tex
```

The resulting files are `build/main.pdf` and `build/supplement.pdf`. All included figures are vector PDFs and should be kept in that format.

## Reproduce the numerical results

Python 3.11 or newer is required. The archived dependency versions are in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --build
```

This verifies the 12 fixed-data checksums, regenerates the main and budget tables from the included replicate-level results, reruns the small orthogonal experiment, and builds both PDFs. It does not rerun the large simulations unless `--full` is supplied:

```bash
python reproduce.py --full --workers 4 --build
```

The main experiment uses the same 200 seeds in every design: 5000–5066 (67), 6000–6066 (67), and 7000–7065 (66). The budget experiment uses seeds 5000–5049 in every design. The orthogonal experiment records its own seed schedule and implementation metadata in `code/orthogonal_metadata.json`.

| Location | Contents |
|---|---|
| `main.tex`, `sections/`, `numbers.tex`, `refs.bib` | Main manuscript and complete bibliography metadata |
| `supplement.tex`, `tables/`, `figures/` | Supplement and generated tables/vector figures |
| `code/fedlasso.py`, `code/feddualavg.py` | Numerical solvers |
| `code/run_study.py`, `code/run_sim.py` | Main simulation generator and recorded seed schedule |
| `code/budget_study.py`, `code/budget_*.csv` | Budget experiment, candidate results and complete cost summaries |
| `code/budget_total_cap_paired.csv` | Paired differences and MCSEs underlying Table S2 |
| `code/real_data.py`, `code/real_data_results.csv` | Fixed synthetic benchmark, including the complete E grid |
| `code/real_data_metadata.json`, `data/` | Original fixed inputs and their recorded SHA-256 hashes |
| `code/orthogonal_check.py`, `code/orthogonal_*.csv` | Reproducible orthogonal-design illustration for Table S3 |
| `code/tests/`, `docs/` | Existing solver checks, result definitions and revision verification |

Run the solver checks with `python -m unittest discover -s code/tests`.

The fixed node files are synthetic. Their original generation procedure, seed and true coefficients were not supplied. They support repeatable analysis of these fixed inputs; they do not support recovery claims about an unknown true model. Main-study and orthogonal-study truth are known by construction. See `docs/RESULTS.md` for metric and cost definitions.

Bibliographic author lists are complete, including Kairouz and Pedregosa. Zhu and Han (2020) is cited as the Springer book chapter, with its correct DOI. Verified available DOIs are printed in the reference list. The original manuscript's title, author, abstract, mathematical results and existing numerical findings are retained. An optional editable cover-letter draft is provided in `cover_letter.txt`.
