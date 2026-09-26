"""Run isolated CPU workers and retain raw timings, validation, and provenance."""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil

from .cases import Case, coarsen, inputs, numpy_solve

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("mvkit-core", "mvkit-python", "numpy", "numba", "diffrax", "sciml")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    files = sorted((ROOT / "benchmarks/particle_suite").glob("*.py"))
    files += sorted((ROOT / "crates/mvkit-core/src").glob("*.rs"))
    files += [ROOT / "benchmarks/sciml/worker.jl", ROOT / "crates/mvkit-core/examples/particle_benchmark.rs"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((s.split(":", 1)[1].strip() for s in cpuinfo.read_text().splitlines()
                    if s.startswith("model name")), cpu)
    return {
        "git_revision": git("rev-parse", "HEAD"), "working_tree_dirty": bool(git("status", "--porcelain")),
        "source_sha256": digest.hexdigest(), "python": platform.python_version(),
        "system": platform.system(), "machine": platform.machine(), "cpu": cpu,
        "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
        "available_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else None,
        "memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip() if Path("/sys/fs/cgroup/memory.max").exists() else None,
        "packages": package_versions(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip() if shutil.which("rustc") else None,
    }


def package_versions():
    from importlib.metadata import distributions
    return sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())


def command(backend, directory, repeats, julia):
    if backend == "mvkit-core":
        suffix = ".exe" if os.name == "nt" else ""
        binary = ROOT / f"target/release/examples/particle_benchmark{suffix}"
        if not binary.exists():
            raise FileNotFoundError("Build the Rust worker: cargo build --release -p mvkit-core --example particle_benchmark")
        return [str(binary), str(directory), str(repeats)]
    if backend == "sciml":
        executable = shutil.which(julia)
        if executable is None:
            raise FileNotFoundError("Julia is required for the sciml backend; pass --julia PATH")
        return [executable, "--startup-file=no", f"--project={ROOT / 'benchmarks/sciml'}",
                str(ROOT / "benchmarks/sciml/worker.jl"), str(directory), str(repeats)]
    return [sys.executable, "-m", "particle_suite.worker", backend, str(directory), str(repeats)]


def run_worker(backend, case, x0, z, *, threads, repeats, julia, timeout):
    if backend == "mvkit-python" and case.name == "multiplicative":
        return {"status": "unsupported", "reason": "Custom multidimensional model requires the Rust API"}, None
    with tempfile.TemporaryDirectory(prefix="mvkit-benchmark-") as tmp:
        directory = Path(tmp)
        (directory / "case.json").write_text(json.dumps(case.to_dict()))
        x0.astype("<f8", copy=False).tofile(directory / "x0.bin")
        z.astype("<f8", copy=False).tofile(directory / "z.bin")
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(ROOT / "benchmarks"), "RAYON_NUM_THREADS": str(threads),
                    "NUMBA_NUM_THREADS": str(threads), "JULIA_NUM_THREADS": str(threads),
                    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": str(threads),
                    "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "true", "PYTHONHASHSEED": "0"})
        cmd = command(backend, directory, repeats, julia)
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        if affinity is not None and threads > len(affinity):
            raise ValueError(f"Requested {threads} CPUs but only {len(affinity)} are available")
        start = time.perf_counter()
        peak_rss = None
        with (directory / "worker.log").open("w") as log:
            try:
                if affinity is not None:
                    os.sched_setaffinity(0, affinity[:threads])
                process = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=log, stderr=log)
            finally:
                if affinity is not None:
                    os.sched_setaffinity(0, affinity)
            try:
                monitor = psutil.Process(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Some containers cannot inspect subprocesses through /proc.
                monitor = None
            while process.poll() is None:
                try:
                    if monitor is not None:
                        peak_rss = max(peak_rss or 0, monitor.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if time.perf_counter() - start > timeout:
                    if monitor is not None:
                        for child in monitor.children(recursive=True):
                            child.kill()
                    process.kill()
                    process.wait()
                    raise TimeoutError(f"{backend} exceeded {timeout}s")
                time.sleep(0.005)
        total = time.perf_counter() - start
        if process.returncode:
            raise RuntimeError((directory / "worker.log").read_text()[-6000:])
        metrics = json.loads((directory / "timing.json").read_text())
        result = np.fromfile(directory / "result.bin", dtype="<f8").reshape(2, case.n, case.width)
        metrics.update({"status": "ok", "process_wall_s": total, "sampled_peak_rss_bytes": peak_rss,
                        "cpu_budget": threads, "affinity_enforced": affinity is not None})
        return metrics, result


def validate(result, expected, reference):
    if result.shape != expected.shape or not np.isfinite(result).all():
        raise ValueError("Backend returned invalid shape or nonfinite values")
    delta = np.abs(result - expected)
    max_error = float(delta.max())
    if not np.allclose(result, expected, atol=3e-11, rtol=3e-11):
        raise ValueError(f"Shared-increment endpoint mismatch: {max_error:.6g}")
    error = result[-1] - reference[-1]
    return {
        "agreement_max_abs": max_error,
        "reference_rmse": float(np.sqrt(np.mean(error**2))),
        "reference_mean_abs_error": float(np.max(np.abs(error.mean(axis=0)))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["mvkit-core", "numpy"])
    parser.add_argument("--cases", nargs="+", choices=["moments", "pairwise", "multiplicative"],
                        default=["moments", "pairwise", "multiplicative"])
    parser.add_argument("--steps", nargs="+", type=positive_int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 41])
    parser.add_argument("--threads", nargs="+", type=positive_int, default=[1])
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--reference-factor", type=positive_int, default=4)
    parser.add_argument("--timeout", type=positive_int, default=600)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reference_factor < 2:
        parser.error("--reference-factor must be at least 2")
    steps = sorted(set(args.steps or ([8, 16] if args.profile == "smoke" else [64, 128])))
    fine_steps = max(steps) * args.reference_factor
    if any(fine_steps % s for s in steps):
        parser.error("Every step count must divide max(steps) * reference-factor")
    if fine_steps % 2:
        parser.error("The reference grid must have an even number of steps")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be nonnegative")
    sizes = {"moments": (32, 1), "pairwise": (12, 2), "multiplicative": (24, 4)} if args.profile == "smoke" else {
        "moments": (4096, 1), "pairwise": (128, 2), "multiplicative": (2048, 4),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "results.jsonl"
    if rows_path.exists():
        parser.error("Output already contains results.jsonl; choose a new directory")
    metadata = {"schema_version": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "protocol": "fixed-grid Euler, shared standard normals, float64, initial+final output",
                "reference": "NumPy Euler on a coupled finer grid, not an exact solution",
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("output", "julia")},
                "environment": provenance()}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    failures = 0
    with rows_path.open("w") as stream:
        for name in args.cases:
            n, dim = sizes[name]
            fine = Case(name, n, dim, fine_steps)
            for seed in args.seeds:
                x0, z_fine = inputs(fine, seed)
                reference = numpy_solve(fine, x0, z_fine)
                # A second coupled grid exposes sensitivity to reference resolution.
                half_ref = numpy_solve(replace(fine, steps=fine_steps // 2), x0, coarsen(z_fine, 2))
                reference_gap = float(np.sqrt(np.mean((reference[-1] - half_ref[-1])**2)))
                for count in steps:
                    case = replace(fine, steps=count)
                    z = coarsen(z_fine, fine_steps // count)
                    expected = numpy_solve(case, x0, z)
                    for threads in args.threads:
                        # Rotate backend order reproducibly to reduce ordering bias.
                        order = list(args.backends)
                        np.random.default_rng(seed + count + threads).shuffle(order)
                        for backend in order:
                            row = {"case": case.to_dict(), "seed": seed, "backend": backend,
                                   "cpu_budget": threads, "reference_steps": fine_steps,
                                   "reference_refinement_rmse": reference_gap}
                            try:
                                metrics, result = run_worker(backend, case, x0, z, threads=threads,
                                    repeats=args.repeats, julia=args.julia, timeout=args.timeout)
                                row.update(metrics)
                                if result is not None:
                                    row.update(validate(result, expected, reference))
                                    row["median_s"] = float(np.median(row["warm_s"]))
                                    row["particle_steps_per_second"] = n * count / row["median_s"]
                            except Exception as exc:
                                failures += 1
                                row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                            timing = f" {row['median_s']:.6f}s" if row.get("status") == "ok" else ""
                            print(f"{name} N={n} steps={count} seed={seed} cpu={threads} {backend}: {row['status']}{timing}", flush=True)
    if failures:
        raise SystemExit(f"{failures} benchmark configurations failed; see results.jsonl")


if __name__ == "__main__":
    main()
