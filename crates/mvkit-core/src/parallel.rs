//! Scheduling policy for inexpensive, independent coordinate updates.

use ndarray::{ArrayView2, ArrayViewMut2, Zip};
use rayon::prelude::*;

// Small memory-bound loops cost less than a Rayon dispatch. Keep each task
// large enough to amortize scheduling; expensive interactions choose their
// own granularity instead of using a particle-count threshold here.
pub(crate) const MIN_PARALLEL_LEN: usize = 65_536;

pub(crate) fn use_parallel(len: usize) -> bool {
    len >= 2 * MIN_PARALLEL_LEN && rayon::current_num_threads() > 1
}

pub(crate) fn map_coordinates(
    state: ArrayView2<f64>,
    out: ArrayViewMut2<f64>,
    f: impl Fn(f64) -> f64 + Sync + Send,
) {
    let parallel = use_parallel(state.len());
    let zip = Zip::from(out).and(state);
    if parallel {
        zip.into_par_iter()
            .with_min_len(MIN_PARALLEL_LEN)
            .for_each(|(out, &x)| *out = f(x));
    } else {
        zip.for_each(|out, &x| *out = f(x));
    }
}
