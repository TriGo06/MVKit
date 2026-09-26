"""Compare complete recorded trajectories from two installed MVKit revisions.

Run with --baseline-python PATH and --candidate-python PATH. Each interpreter
must have MVKit and NumPy installed. Workers use independent processes and
fixed Rayon thread counts. This is a compatibility check, not a timing study.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def worker(output):
    import mvkit._core as core
    from mvkit import (
        simulate_cucker_smale,
        simulate_kuramoto,
        simulate_linear_quadratic,
        simulate_mean_field_cir,
    )

    n, steps = 41, 19
    cases = [
        ("linear", simulate_linear_quadratic, 1, dict(a=-0.5, b=0.3, sigma=0.25)),
        ("cir", simulate_mean_field_cir, 1, dict(kappa=1.5, theta=0.8, b=0.3, sigma=0.25)),
        ("kuramoto", simulate_kuramoto, 1,
         dict(coupling_k=0.8, omegas=np.linspace(-0.4, 0.4, n), sigma=0.25)),
    ]
    cases += [
        (f"cucker-{d}", simulate_cucker_smale, 2 * d,
         dict(spatial_dim=d, beta=0.4, sigma=0.25)) for d in (1, 2, 3, 5)
    ]
    traces = {}
    for name, solve, width, parameters in cases:
        for seed in (7, 19):
            rng = np.random.default_rng(seed)
            x0 = rng.normal(size=(n, width))
            if name == "cir":
                x0 = np.exp(0.2 * x0)
            noise = rng.normal(size=(steps, n, width))
            for scheme in ("euler", "milstein"):
                for supplied in (False, True):
                    key = f"{name}-{seed}-{scheme}-{'supplied' if supplied else 'generated'}"
                    traces[key] = solve(
                        x0, 0.25, steps, record_every=4, seed=seed, scheme=scheme,
                        increments=noise if supplied else None, **parameters,
                    )
    np.savez(output, **traces)
    output.with_suffix(".json").write_text(json.dumps({
        "python": sys.version.split()[0], "numpy": np.__version__,
        "core_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python")
    parser.add_argument("--candidate-python")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
        return
    if not args.baseline_python or not args.candidate_python:
        parser.error("both Python interpreters are required")
    report = {"protocol": "bitwise recorded-trajectory comparison", "workers": [], "checks": []}
    with tempfile.TemporaryDirectory(prefix="mvkit-compatibility-") as temp:
        paths = {}
        for revision, python in (("baseline", args.baseline_python), ("candidate", args.candidate_python)):
            for threads in (1, 4):
                path = Path(temp) / f"{revision}-{threads}.npz"
                env = dict(os.environ, RAYON_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1")
                subprocess.run([python, str(Path(__file__).resolve()), "--worker", str(path)],
                               check=True, env=env)
                paths[revision, threads] = path
                report["workers"].append(dict(revision=revision, threads=threads,
                                             **json.loads(path.with_suffix(".json").read_text())))
        with np.load(paths["baseline", 1]) as reference:
            for (revision, threads), path in paths.items():
                if (revision, threads) == ("baseline", 1):
                    continue
                with np.load(path) as actual:
                    if set(actual.files) != set(reference.files):
                        raise AssertionError("Different trajectory sets")
                    for name in reference.files:
                        expected, observed = reference[name], actual[name]
                        identical = (observed.shape == expected.shape
                                     and observed.dtype == expected.dtype
                                     and observed.tobytes() == expected.tobytes())
                        report["checks"].append({"revision": revision, "threads": threads,
                                                 "case": name, "bitwise_equal": identical})
    report["passed"] = all(c["bitwise_equal"] for c in report["checks"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{sum(c['bitwise_equal'] for c in report['checks'])}/{len(report['checks'])} "
          "recorded trajectories bitwise identical to the single-thread baseline")
    if not report["passed"]:
        raise SystemExit("Trajectory compatibility check failed")


if __name__ == "__main__":
    main()
