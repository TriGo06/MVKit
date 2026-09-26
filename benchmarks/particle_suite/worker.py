"""One fresh process per backend/configuration, so compilation is observable."""

import argparse
import json
from pathlib import Path
import time
import sys

import numpy as np

from .backends import prepare
from .cases import Case


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, AttributeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend")
    parser.add_argument("directory", type=Path)
    parser.add_argument("repeats", type=int)
    args = parser.parse_args()
    directory = args.directory
    case = Case(**json.loads((directory / "case.json").read_text()))
    x0 = np.fromfile(directory / "x0.bin", dtype="<f8").reshape(case.n, case.width)
    z = np.fromfile(directory / "z.bin", dtype="<f8").reshape(case.steps, case.n, case.width)
    solve, metadata = prepare(args.backend, case, x0, z)
    start = time.perf_counter()
    result = solve()
    first = time.perf_counter() - start
    times = []
    for _ in range(args.repeats):
        del result
        start = time.perf_counter()
        result = solve()
        times.append(time.perf_counter() - start)
    np.asarray(result, dtype="<f8").tofile(directory / "result.bin")
    (directory / "timing.json").write_text(json.dumps({
        **metadata, "first_call_s": first, "warm_s": times,
        "self_peak_rss_bytes": peak_rss_bytes(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
