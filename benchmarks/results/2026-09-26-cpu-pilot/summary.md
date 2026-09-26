# Particle benchmark results

Generated: 2026-09-26T16:19:47Z

CPU: AMD EPYC 9V74 80-Core Processor; Linux x86_64.

Source revision: `9fe50c8d9bf2ace7caa722ae454342d5b8bce9a5` (working tree dirty: False).
Source fingerprint: `d206e72a0cb30df519847f31edf55ae67b8918e5314a2360efa3114b05ff9882`.

Fixed-grid Euler, float64, shared precomputed normals, initial and final states saved.
Warm timings exclude imports, compilation, input preparation and output serialization.
Reference errors compare coupled finite-particle paths with a finer NumPy Euler grid.
Peak RSS covers the whole worker process, including runtime, inputs and compilation; kernel high-water marks are preferred over sampled peaks. n/a means unavailable.

Each time is the median of per-seed warm medians. The range spans those medians.
CPU budget denotes allowed logical CPUs, not the number of threads actually used by every backend.

| Case | N | d | Steps | CPU budget | Backend | Seeds | Warm ms (range) | First call ms | Peak RSS MiB | Reference RMSE | Reference refinement RMSE |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| moments | 4096 | 1 | 64 | 1 | diffrax | 3 | 0.190 (0.189–0.196) | 455.0 | 310.9 | 1.259e-03 | 1.929e-04 |
| moments | 4096 | 1 | 64 | 1 | mvkit-core | 3 | 2.225 (2.144–2.262) | 2.3 | 5.4 | 1.259e-03 | 1.929e-04 |
| moments | 4096 | 1 | 64 | 1 | mvkit-python | 3 | 2.291 (2.277–2.323) | 3.6 | 80.1 | 1.259e-03 | 1.929e-04 |
| moments | 4096 | 1 | 64 | 1 | numba | 3 | 0.604 (0.589–0.608) | 3156.7 | 234.2 | 1.259e-03 | 1.929e-04 |
| moments | 4096 | 1 | 64 | 1 | numpy | 3 | 0.907 (0.867–0.913) | 1.1 | 74.5 | 1.259e-03 | 1.929e-04 |
| moments | 4096 | 1 | 64 | 1 | sciml | 3 | 1.721 (1.693–1.828) | 4843.1 | 611.0 | 1.259e-03 | 1.929e-04 |
| moments | 4096 | 1 | 64 | 4 | diffrax | 3 | 0.186 (0.183–0.215) | 385.3 | 309.3 | 1.259e-03 | 1.929e-04 |
| moments | 4096 | 1 | 64 | 4 | mvkit-core | 3 | 7.082 (5.938–11.097) | 12.6 | 5.4 | 1.259e-03 | 1.929e-04 |
| moments | 4096 | 1 | 64 | 4 | mvkit-python | 3 | 8.528 (7.925–9.039) | 9.9 | 80.2 | 1.259e-03 | 1.929e-04 |
| moments | 4096 | 1 | 64 | 4 | numba | 3 | 0.775 (0.490–0.861) | 3180.0 | 231.8 | 1.259e-03 | 1.929e-04 |
| moments | 4096 | 1 | 64 | 4 | numpy | 3 | 0.899 (0.886–0.944) | 1.1 | 74.5 | 1.259e-03 | 1.929e-04 |
| moments | 4096 | 1 | 64 | 4 | sciml | 3 | 2.422 (2.419–2.554) | 4613.3 | 587.8 | 1.259e-03 | 1.929e-04 |
| moments | 4096 | 1 | 128 | 1 | diffrax | 3 | 0.373 (0.365–0.384) | 461.9 | 314.9 | 5.485e-04 | 1.929e-04 |
| moments | 4096 | 1 | 128 | 1 | mvkit-core | 3 | 4.486 (4.328–4.542) | 4.6 | 9.4 | 5.485e-04 | 1.929e-04 |
| moments | 4096 | 1 | 128 | 1 | mvkit-python | 3 | 4.466 (4.385–4.773) | 6.2 | 83.9 | 5.485e-04 | 1.929e-04 |
| moments | 4096 | 1 | 128 | 1 | numba | 3 | 1.143 (1.133–1.163) | 3194.1 | 236.6 | 5.485e-04 | 1.929e-04 |
| moments | 4096 | 1 | 128 | 1 | numpy | 3 | 1.816 (1.726–1.962) | 2.0 | 74.5 | 5.485e-04 | 1.929e-04 |
| moments | 4096 | 1 | 128 | 1 | sciml | 3 | 3.463 (3.330–3.633) | 4586.1 | 585.2 | 5.485e-04 | 1.929e-04 |
| moments | 4096 | 1 | 128 | 4 | diffrax | 3 | 0.366 (0.358–0.375) | 383.9 | 315.4 | 5.485e-04 | 1.929e-04 |
| moments | 4096 | 1 | 128 | 4 | mvkit-core | 3 | 11.198 (10.547–19.973) | 10.3 | 9.4 | 5.485e-04 | 1.929e-04 |
| moments | 4096 | 1 | 128 | 4 | mvkit-python | 3 | 20.419 (12.040–22.627) | 22.1 | 84.3 | 5.485e-04 | 1.929e-04 |
| moments | 4096 | 1 | 128 | 4 | numba | 3 | 1.505 (1.464–1.810) | 3217.1 | 236.9 | 5.485e-04 | 1.929e-04 |
| moments | 4096 | 1 | 128 | 4 | numpy | 3 | 1.747 (1.712–1.845) | 2.0 | 74.5 | 5.485e-04 | 1.929e-04 |
| moments | 4096 | 1 | 128 | 4 | sciml | 3 | 7.763 (7.423–13.549) | 4594.0 | 591.0 | 5.485e-04 | 1.929e-04 |
| multiplicative | 2048 | 4 | 64 | 1 | diffrax | 3 | 0.924 (0.864–0.953) | 457.8 | 310.4 | 3.980e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 64 | 1 | mvkit-core | 3 | 3.253 (3.217–3.469) | 3.6 | 9.5 | 3.980e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 64 | 1 | numba | 3 | 0.815 (0.694–0.914) | 3231.8 | 236.5 | 3.980e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 64 | 1 | numpy | 3 | 4.143 (3.934–5.245) | 4.5 | 130.8 | 3.980e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 64 | 1 | sciml | 3 | 2.717 (2.571–2.895) | 4714.2 | 583.4 | 3.980e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 64 | 4 | diffrax | 3 | 0.902 (0.868–0.965) | 376.6 | 314.6 | 3.980e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 64 | 4 | mvkit-core | 3 | 7.257 (6.830–11.717) | 8.5 | 9.5 | 3.980e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 64 | 4 | numba | 3 | 0.895 (0.734–1.606) | 3133.7 | 236.7 | 3.980e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 64 | 4 | numpy | 3 | 3.930 (3.901–4.016) | 4.2 | 130.8 | 3.980e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 64 | 4 | sciml | 3 | 6.659 (5.971–7.775) | 4776.0 | 587.4 | 3.980e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 128 | 1 | diffrax | 3 | 1.835 (1.734–1.857) | 445.7 | 318.0 | 2.575e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 128 | 1 | mvkit-core | 3 | 6.514 (6.404–6.517) | 6.9 | 17.5 | 2.575e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 128 | 1 | numba | 3 | 1.441 (1.394–1.483) | 3208.7 | 239.9 | 2.575e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 128 | 1 | numpy | 3 | 8.260 (8.118–8.601) | 8.8 | 130.8 | 2.575e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 128 | 1 | sciml | 3 | 4.977 (4.962–5.388) | 4684.6 | 617.8 | 2.575e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 128 | 4 | diffrax | 3 | 1.827 (1.717–1.953) | 368.0 | 323.8 | 2.575e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 128 | 4 | mvkit-core | 3 | 14.272 (13.656–14.857) | 19.4 | 17.5 | 2.575e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 128 | 4 | numba | 3 | 2.688 (1.606–3.316) | 3260.9 | 239.3 | 2.575e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 128 | 4 | numpy | 3 | 8.060 (7.900–8.324) | 8.4 | 130.8 | 2.575e-03 | 1.476e-03 |
| multiplicative | 2048 | 4 | 128 | 4 | sciml | 3 | 10.585 (7.506–13.068) | 4631.3 | 601.3 | 2.575e-03 | 1.476e-03 |
| pairwise | 128 | 2 | 64 | 1 | diffrax | 3 | 30.896 (30.789–32.475) | 521.4 | 309.5 | 2.127e-03 | 3.292e-04 |
| pairwise | 128 | 2 | 64 | 1 | mvkit-core | 3 | 55.302 (55.122–56.802) | 55.2 | 1.9 | 2.127e-03 | 3.292e-04 |
| pairwise | 128 | 2 | 64 | 1 | mvkit-python | 3 | 56.488 (56.285–57.812) | 56.7 | 76.7 | 2.127e-03 | 3.292e-04 |
| pairwise | 128 | 2 | 64 | 1 | numba | 3 | 15.933 (15.897–16.277) | 3325.8 | 232.7 | 2.127e-03 | 3.292e-04 |
| pairwise | 128 | 2 | 64 | 1 | numpy | 3 | 69.629 (67.815–69.989) | 69.2 | 74.5 | 2.127e-03 | 3.292e-04 |
| pairwise | 128 | 2 | 64 | 1 | sciml | 3 | 28.899 (28.685–29.142) | 4754.2 | 617.4 | 2.127e-03 | 3.292e-04 |
| pairwise | 128 | 2 | 64 | 4 | diffrax | 3 | 18.326 (17.575–19.784) | 436.5 | 311.2 | 2.127e-03 | 3.292e-04 |
| pairwise | 128 | 2 | 64 | 4 | mvkit-core | 3 | 22.270 (20.321–23.235) | 26.8 | 1.9 | 2.127e-03 | 3.292e-04 |
| pairwise | 128 | 2 | 64 | 4 | mvkit-python | 3 | 22.544 (20.348–22.773) | 29.4 | 76.6 | 2.127e-03 | 3.292e-04 |
| pairwise | 128 | 2 | 64 | 4 | numba | 3 | 6.726 (6.619–7.066) | 3352.9 | 230.1 | 2.127e-03 | 3.292e-04 |
| pairwise | 128 | 2 | 64 | 4 | numpy | 3 | 69.729 (68.315–76.261) | 72.5 | 74.5 | 2.127e-03 | 3.292e-04 |
| pairwise | 128 | 2 | 64 | 4 | sciml | 3 | 22.276 (20.840–23.422) | 4765.2 | 587.6 | 2.127e-03 | 3.292e-04 |
| pairwise | 128 | 2 | 128 | 1 | diffrax | 3 | 63.202 (61.730–64.703) | 555.6 | 308.8 | 9.349e-04 | 3.292e-04 |
| pairwise | 128 | 2 | 128 | 1 | mvkit-core | 3 | 112.249 (111.735–115.391) | 112.4 | 2.4 | 9.349e-04 | 3.292e-04 |
| pairwise | 128 | 2 | 128 | 1 | mvkit-python | 3 | 113.870 (112.829–115.524) | 113.7 | 77.2 | 9.349e-04 | 3.292e-04 |
| pairwise | 128 | 2 | 128 | 1 | numba | 3 | 30.977 (30.766–31.007) | 3293.3 | 233.1 | 9.349e-04 | 3.292e-04 |
| pairwise | 128 | 2 | 128 | 1 | numpy | 3 | 137.471 (137.070–141.973) | 137.8 | 74.5 | 9.349e-04 | 3.292e-04 |
| pairwise | 128 | 2 | 128 | 1 | sciml | 3 | 56.607 (55.689–59.768) | 4914.0 | 622.0 | 9.349e-04 | 3.292e-04 |
| pairwise | 128 | 2 | 128 | 4 | diffrax | 3 | 34.932 (32.967–41.413) | 434.3 | 309.3 | 9.349e-04 | 3.292e-04 |
| pairwise | 128 | 2 | 128 | 4 | mvkit-core | 3 | 42.358 (40.742–43.987) | 49.0 | 2.3 | 9.349e-04 | 3.292e-04 |
| pairwise | 128 | 2 | 128 | 4 | mvkit-python | 3 | 45.491 (44.798–47.978) | 56.3 | 76.9 | 9.349e-04 | 3.292e-04 |
| pairwise | 128 | 2 | 128 | 4 | numba | 3 | 13.563 (9.013–13.868) | 3244.8 | 233.3 | 9.349e-04 | 3.292e-04 |
| pairwise | 128 | 2 | 128 | 4 | numpy | 3 | 139.287 (136.209–142.846) | 135.0 | 74.5 | 9.349e-04 | 3.292e-04 |
| pairwise | 128 | 2 | 128 | 4 | sciml | 3 | 44.387 (38.747–54.010) | 4759.1 | 586.6 | 9.349e-04 | 3.292e-04 |

## Failed or unsupported configurations

- multiplicative, mvkit-python, steps=64, seed=7, CPU budget=1: unsupported
- multiplicative, mvkit-python, steps=64, seed=7, CPU budget=4: unsupported
- multiplicative, mvkit-python, steps=128, seed=7, CPU budget=1: unsupported
- multiplicative, mvkit-python, steps=128, seed=7, CPU budget=4: unsupported
- multiplicative, mvkit-python, steps=64, seed=19, CPU budget=1: unsupported
- multiplicative, mvkit-python, steps=64, seed=19, CPU budget=4: unsupported
- multiplicative, mvkit-python, steps=128, seed=19, CPU budget=1: unsupported
- multiplicative, mvkit-python, steps=128, seed=19, CPU budget=4: unsupported
- multiplicative, mvkit-python, steps=64, seed=41, CPU budget=1: unsupported
- multiplicative, mvkit-python, steps=64, seed=41, CPU budget=4: unsupported
- multiplicative, mvkit-python, steps=128, seed=41, CPU budget=1: unsupported
- multiplicative, mvkit-python, steps=128, seed=41, CPU budget=4: unsupported

These measurements compare the listed Euler implementations on this machine. They do not rank each library's best solver, RNG performance, GPU support, or end-to-end time to a user-specified statistical accuracy.
