"""Single entry point for reproducing numerical outputs and manuscript files."""
from pathlib import Path
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "ms" / "code"
MS = ROOT / "ms"


def call(*args, cwd=ROOT):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def python_script(name, *args):
    call(sys.executable, CODE / name, *args)


def build():
    for tool in ("latexmk", "pdflatex", "bibtex", "pdftotext"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Missing build dependency: {tool}")
    command = ("latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
               "-outdir=build", "main.tex")
    call(*command, cwd=MS)
    pdf = MS / "build" / "main.pdf"
    extracted = subprocess.run(["pdftotext", str(pdf), "-"], check=True,
                               text=True, capture_output=True).stdout
    # Inclusive approximation, including tables, captions, references and page labels.
    count = len(extracted.split())
    if count > 10000:
        raise SystemExit(f"Inclusive word count {count} exceeds the journal's 10,000-word guide")
    (MS / "word_count.tex").write_text(
        "\\newcommand{\\ManuscriptWordCount}{" + str(count) + "}\n")
    call(*command, cwd=MS)
    shutil.copy2(pdf, MS / "federated_lasso_manuscript.pdf")
    call("latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=build", "supplement.tex", cwd=MS)
    shutil.copy2(MS / "build" / "supplement.pdf", MS / "supplementary_results.pdf")
    files = [MS / n for n in ("main.tex", "supplement.tex", "numbers.tex", "word_count.tex", "refs.bib")]
    files += sorted((MS / "sections").glob("*.tex"))
    files += sorted((MS / "tables").glob("*.tex"))
    files += sorted((MS / "figures").glob("*.pdf"))
    with zipfile.ZipFile(MS / "latex_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(MS).as_posix(), date_time=(2026, 9, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    print(f"Built manuscript ({count:,} inclusive words), supplementary results, and LaTeX source archive.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Rerun all experiments before rendering")
    parser.add_argument("--workers", type=int, default=4, help="Workers for the main study")
    parser.add_argument("--build", action="store_true", help="Also compile and package the article")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if args.full:
        python_script("run_study.py", "--workers", str(args.workers))
        python_script("real_data.py")
        python_script("eps_sens.py")
        python_script("diabetes_study.py", "--replicates", "100", "--seed", "20260911")
        python_script("budget_study.py", "--reps", "50", "--workers", str(min(args.workers, 2)))
    python_script("make_outputs.py")
    python_script("diabetes_study.py", "--from-results")
    python_script("budget_study.py", "--from-results")
    abstract = (MS / "sections" / "abstract.tex").read_text().split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    if len(abstract.split()) != 150:
        raise SystemExit("The journal abstract must contain 150 words")
    if args.build:
        build()


if __name__ == "__main__":
    main()
