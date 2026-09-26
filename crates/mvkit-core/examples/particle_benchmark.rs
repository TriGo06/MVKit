//! Native benchmark worker. Inputs/outputs are little-endian Float64 in C order.
//! The custom multidimensional model lives here, outside the public Python API.

use mvkit_core::{
    models::{CuckerSmale, LinearQuadratic},
    schemes::euler_maruyama,
    MeanFieldSDE,
};
use ndarray::{Array2, Array3, ArrayView2, ArrayViewMut2, Axis};
use rayon::prelude::*;
use serde::Deserialize;
use std::{error::Error, fs, path::Path, time::Instant};

#[derive(Deserialize)]
struct Case {
    name: String,
    n: usize,
    dim: usize,
    steps: usize,
    t_final: f64,
    a: f64,
    b: f64,
    sigma: f64,
    beta: f64,
}

struct Multiplicative {
    dim: usize,
    a: f64,
    b: f64,
    sigma: f64,
}
impl MeanFieldSDE for Multiplicative {
    fn dim(&self) -> usize {
        self.dim
    }
    fn drift(&self, x: ArrayView2<f64>, mut out: ArrayViewMut2<f64>) {
        let mean = x.mean_axis(Axis(0)).expect("nonempty state");
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for k in 0..self.dim {
                    row[k] = self.a * x[[i, k]] + self.b * mean[k];
                }
            });
    }
    fn diffusion(&self, x: ArrayView2<f64>, mut out: ArrayViewMut2<f64>) {
        out.zip_mut_with(&x, |s, &v| *s = self.sigma * v);
    }
}

// Keep compatibility with toolchains predating slice::as_chunks.
#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
fn read_values(path: &Path) -> Result<Vec<f64>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 8 != 0 {
        return Err("invalid Float64 file".into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn peak_rss_bytes() -> Option<u64> {
    // Linux high-water mark; parent sampling is the fallback on other systems.
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()
        .map(|kib| kib * 1024)
}

fn benchmark<M: MeanFieldSDE>(
    model: &M,
    c: &Case,
    dir: &Path,
    repeats: usize,
) -> Result<(), Box<dyn Error>> {
    let x = Array2::from_shape_vec((c.n, model.dim()), read_values(&dir.join("x0.bin"))?)?;
    let z = Array3::from_shape_vec(
        (c.steps, c.n, model.dim()),
        read_values(&dir.join("z.bin"))?,
    )?;
    let solve = || euler_maruyama(model, &x, c.t_final, c.steps, 0, 0, Some(&z));
    let start = Instant::now();
    let mut result = solve();
    let first = start.elapsed().as_secs_f64();
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        // Drop the previous result before timing, as in the other workers.
        drop(result);
        let start = Instant::now();
        result = solve();
        times.push(start.elapsed().as_secs_f64());
    }
    let bytes: Vec<u8> = result.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(dir.join("result.bin"), bytes)?;
    fs::write(
        dir.join("timing.json"),
        serde_json::to_vec(&serde_json::json!({
            "first_call_s": first, "warm_s": times,
            "backend_version": env!("CARGO_PKG_VERSION"),
            "threads": rayon::current_num_threads(), "input_transfer_s": 0.0,
        "device": "cpu", "interface": "Rust core", "self_peak_rss_bytes": peak_rss_bytes()
        }))?,
    )?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: particle_benchmark INPUT_DIRECTORY REPEATS".into());
    }
    let dir = Path::new(&args[1]);
    let repeats = args[2].parse::<usize>()?;
    if repeats == 0 {
        return Err("repeats must be positive".into());
    }
    let c: Case = serde_json::from_slice(&fs::read(dir.join("case.json"))?)?;
    if c.n == 0 || c.dim == 0 || c.steps == 0 || !c.t_final.is_finite() || c.t_final <= 0.0 {
        return Err("invalid benchmark dimensions or time grid".into());
    }
    match c.name.as_str() {
        "moments" if c.dim == 1 => {
            benchmark(&LinearQuadratic::new(c.a, c.b, c.sigma), &c, dir, repeats)
        }
        "pairwise" => benchmark(&CuckerSmale::new(c.dim, c.beta, c.sigma), &c, dir, repeats),
        "multiplicative" => benchmark(
            &Multiplicative {
                dim: c.dim,
                a: c.a,
                b: c.b,
                sigma: c.sigma,
            },
            &c,
            dir,
            repeats,
        ),
        _ => Err("unsupported case".into()),
    }
}
