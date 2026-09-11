"""Reproduce all 600 Monte Carlo replicates with the recorded seed sequence.

Example (from any working directory):
    python ms/code/run_study.py --workers 6
Output is replaced atomically only after every replicate succeeds. Optional
checkpoints permit resumption and are tied to a hash of the simulation code.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

# Set these before importing NumPy in either the parent or spawned workers.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import pandas as pd
from run_sim import one_rep

SCENARIOS = ("IID", "Correlated", "Heterogeneous")
SEEDS = tuple(range(5000, 5067)) + tuple(range(6000, 6067)) + tuple(range(7000, 7066))


def code_hash():
    root = Path(__file__).resolve().parent
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in
                                  ("fedlasso.py", "run_sim.py", "run_study.py"))).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "sim_all_raw.csv")
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    jobs = [(scenario, seed) for scenario in SCENARIOS for seed in SEEDS]
    checkpoints = args.checkpoint_dir
    if checkpoints:
        checkpoints.mkdir(parents=True, exist_ok=True)
        manifest = checkpoints / "manifest.json"
        expected = {"code_sha256": code_hash(), "seeds": list(SEEDS),
                    "scenarios": list(SCENARIOS)}
        if manifest.exists() and json.loads(manifest.read_text()) != expected:
            raise RuntimeError("Checkpoint manifest differs from this code/seed sequence")
        manifest.write_text(json.dumps(expected, indent=2) + "\n")
    completed, pending = {}, []
    for scenario, seed in jobs:
        path = checkpoints / f"{scenario}_{seed}.csv" if checkpoints else None
        if path and path.exists():
            completed[(scenario, seed)] = pd.read_csv(path)
        else:
            pending.append((scenario, seed))
    start = time.monotonic()
    print(f"{len(jobs)} replicates, {len(completed)} resumed, {args.workers} workers; BLAS=1", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one_rep, scenario, seed): (scenario, seed)
                   for scenario, seed in pending}
        for future in as_completed(futures):
            scenario, seed = futures[future]
            frame = pd.DataFrame(future.result())
            if checkpoints:
                path = checkpoints / f"{scenario}_{seed}.csv"
                temporary = path.with_suffix(".tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(path)
            completed[(scenario, seed)] = frame
            if len(completed) % 20 == 0 or len(completed) == len(jobs):
                print(f"Completed {len(completed)}/{len(jobs)} in {time.monotonic()-start:.1f}s", flush=True)
    combined = pd.concat([completed[job] for job in jobs], ignore_index=True)
    if combined.duplicated(["scenario", "rep", "method", "E"]).any():
        raise RuntimeError("Duplicate simulation keys")
    counts = combined.groupby(["scenario", "method", "E"], dropna=False).size()
    if not counts.eq(len(SEEDS)).all():
        raise RuntimeError("Incomplete replication groups")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(args.output)
    print(f"Wrote {len(combined)} rows to {args.output} in {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
