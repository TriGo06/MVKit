"""Summarize raw benchmark rows without hiding failed configurations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def render(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    problems = []
    for row in rows:
        c = row["case"]
        if row["status"] != "ok":
            problems.append(f"- {c['name']}, {row['backend']}, steps={c['steps']}, seed={row['seed']}, "
                            f"CPU budget={row['cpu_budget']}: {row['status']}")
            continue
        groups[(c["name"], c["n"], c["dim"], c["steps"], row["cpu_budget"], row["backend"])].append(row)
    env = metadata["environment"]
    lines = ["# Particle benchmark results", "", f"Generated: {metadata['created_utc']}", "",
             f"CPU: {env['cpu']}; {env['system']} {env['machine']}.", "",
             f"Source revision: `{env['git_revision']}` (working tree dirty: {env['working_tree_dirty']}).",
             f"Source fingerprint: `{env['source_sha256']}`.", "",
             "Fixed-grid Euler, float64, shared precomputed normals, initial and final states saved.",
             "Warm timings exclude imports, compilation, input preparation and output serialization.",
             "Reference errors compare coupled finite-particle paths with a finer NumPy Euler grid.",
             "Peak RSS covers the whole worker process, including runtime, inputs and compilation; kernel high-water marks are preferred over sampled peaks. n/a means unavailable.", "",
             "Each time is the median of per-seed warm medians. The range spans those medians.",
             "CPU budget denotes allowed logical CPUs, not the number of threads actually used by every backend.", "",
             "| Case | N | d | Steps | CPU budget | Backend | Seeds | Warm ms (range) | First call ms | Peak RSS MiB | Reference RMSE | Reference refinement RMSE |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for key, values in sorted(groups.items()):
        name, n, dim, steps, cpus, backend = key
        medians = [r["median_s"] * 1000 for r in values]
        first = statistics.median(r["first_call_s"] * 1000 for r in values)
        peaks = [(r.get("self_peak_rss_bytes") or r.get("sampled_peak_rss_bytes")) for r in values]
        peaks = [p / 2**20 for p in peaks if p is not None]
        peak = f"{max(peaks):.1f}" if peaks else "n/a"
        error = (statistics.mean(r["reference_rmse"]**2 for r in values))**0.5
        ref_gap = (statistics.mean(r["reference_refinement_rmse"]**2 for r in values))**0.5
        lines.append(f"| {name} | {n} | {dim} | {steps} | {cpus} | {backend} | {len(values)} | "
                     f"{statistics.median(medians):.3f} ({min(medians):.3f}–{max(medians):.3f}) | "
                     f"{first:.1f} | {peak} | {error:.3e} | {ref_gap:.3e} |")
    if problems:
        lines += ["", "## Failed or unsupported configurations", "", *problems]
    lines += ["", "These measurements compare the listed Euler implementations on this machine. "
              "They do not rank each library's best solver, RNG performance, GPU support, "
              "or end-to-end time to a user-specified statistical accuracy.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory / "summary.md"
    path.write_text(render(args.directory))
    print(path)


if __name__ == "__main__":
    main()
