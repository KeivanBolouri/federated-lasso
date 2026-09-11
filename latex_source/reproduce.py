"""Regenerate the submitted tables and PDFs from this self-contained archive."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"


def run(*args):
    subprocess.run([str(x) for x in args], cwd=ROOT, check=True)


def script(name, *args):
    run(sys.executable, CODE / name, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Rerun the main and budget simulations and fixed-data analyses")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--build", action="store_true", help="Also compile both PDFs")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    metadata = json.loads((CODE / "real_data_metadata.json").read_text())
    for name, expected in metadata["source_file_sha256"].items():
        actual = hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Fixed-input checksum mismatch: {name}")
    if args.full:
        script("run_study.py", "--workers", args.workers)
        script("real_data.py")
        script("eps_sens.py")
        script("budget_study.py", "--reps", 50, "--workers", min(args.workers, 2))
    script("make_outputs.py")
    script("budget_study.py", "--from-results")
    # The small orthogonal experiment is deliberately regenerated on each run.
    script("orthogonal_check.py")
    if args.build:
        for document in ("main.tex", "supplement.tex"):
            run("latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                "-outdir=build", document)
        text = subprocess.check_output(
            ["pdftotext", str(ROOT / "build/main.pdf"), "-"], text=True)
        count = len(text.split())
        (ROOT / "docs/word_count.txt").write_text(
            f"Inclusive PDF text word count: {count}\n"
            "Method: whitespace tokens in pdftotext output; includes references, tables, captions and page numbers.\n")
        if count > 10000:
            raise SystemExit(f"Inclusive word count is {count}; review manuscript length")
        print(f"Built build/main.pdf and build/supplement.pdf; inclusive word count {count}.")


if __name__ == "__main__":
    main()
