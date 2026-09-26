# Particle benchmark results

Generated: 2026-09-26T16:28:04Z

CPU: AMD EPYC 9V74 80-Core Processor; Linux x86_64.

Source revision: `2ea6c46c1d11b028b6d19b73bfd84cd47b80a270` (working tree dirty: True).
Source fingerprint: `ef867d98e42be781755add0c263c65b67f0d2ef26d3c0f5901fd7067c591f2f4`.

Fixed-grid Euler, float64, shared precomputed normals, initial and final states saved.
Warm timings exclude imports, compilation, input preparation and output serialization.
Reference errors compare coupled finite-particle paths with a finer NumPy Euler grid.
Peak RSS covers the whole worker process, including runtime, inputs and compilation; kernel high-water marks are preferred over sampled peaks. n/a means unavailable.

Each time is the median of per-seed warm medians. The range spans those medians.
CPU budget denotes allowed logical CPUs, not the number of threads actually used by every backend.

| Case | N | d | Steps | CPU budget | Backend | Seeds | Warm ms (range) | First call ms | Peak RSS MiB | Reference RMSE | Reference refinement RMSE |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| moments | 65536 | 1 | 128 | 1 | diffrax | 3 | 8.182 (8.144–8.460) | 477.4 | 617.9 | 5.487e-04 | 1.940e-04 |
| moments | 65536 | 1 | 128 | 1 | mvkit-core | 3 | 46.783 (45.128–46.787) | 46.9 | 129.9 | 5.487e-04 | 1.940e-04 |
| moments | 65536 | 1 | 128 | 1 | mvkit-python | 3 | 73.350 (71.411–75.681) | 75.7 | 617.9 | 5.487e-04 | 1.940e-04 |
| moments | 65536 | 1 | 128 | 1 | numba | 3 | 17.439 (17.011–17.591) | 3176.2 | 617.9 | 5.487e-04 | 1.940e-04 |
| moments | 65536 | 1 | 128 | 1 | numpy | 3 | 13.840 (13.616–14.088) | 40.2 | 617.9 | 5.487e-04 | 1.940e-04 |
| moments | 65536 | 1 | 128 | 1 | sciml | 3 | 55.449 (52.849–56.598) | 4762.8 | 785.7 | 5.487e-04 | 1.940e-04 |
| moments | 65536 | 1 | 128 | 4 | diffrax | 3 | 12.201 (11.913–12.216) | 384.5 | 617.9 | 5.487e-04 | 1.940e-04 |
| moments | 65536 | 1 | 128 | 4 | mvkit-core | 3 | 48.581 (42.208–50.844) | 49.2 | 129.9 | 5.487e-04 | 1.940e-04 |
| moments | 65536 | 1 | 128 | 4 | mvkit-python | 3 | 102.911 (96.922–106.046) | 103.4 | 617.9 | 5.487e-04 | 1.940e-04 |
| moments | 65536 | 1 | 128 | 4 | numba | 3 | 13.944 (12.649–22.663) | 3198.3 | 617.9 | 5.487e-04 | 1.940e-04 |
| moments | 65536 | 1 | 128 | 4 | numpy | 3 | 13.771 (13.724–13.996) | 40.5 | 617.9 | 5.487e-04 | 1.940e-04 |
| moments | 65536 | 1 | 128 | 4 | sciml | 3 | 64.349 (61.521–64.755) | 4726.9 | 816.2 | 5.487e-04 | 1.940e-04 |

These measurements compare the listed Euler implementations on this machine. They do not rank each library's best solver, RNG performance, GPU support, or end-to-end time to a user-specified statistical accuracy.
