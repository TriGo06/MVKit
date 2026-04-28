//! Python bindings for mvkit-core.

use pyo3::prelude::*;

// Both pyfunctions live inside this submodule so that the
// clippy::useless_conversion allow can be scoped to the macro-expanded
// wrapper code that pyo3 0.22's #[pyfunction] generates as a sibling to
// the user function. Function-level #[allow(...)] does not reach those
// generated items, and we do not want a file-wide allow.
#[allow(clippy::useless_conversion)]
mod functions {
    use mvkit_core::models::{CuckerSmale, LinearQuadratic};
    use mvkit_core::schemes::euler_maruyama;
    use ndarray::Array2;
    use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2};
    use pyo3::prelude::*;

    /// Simulate the Cucker-Smale flocking model.
    ///
    /// See `python/mvkit/__init__.py` for the user-facing docstring.
    #[pyfunction]
    #[pyo3(signature = (
        x0,
        t_final,
        n_steps,
        spatial_dim,
        beta = 0.5,
        sigma = 0.1,
        record_every = 0,
        seed = 42,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn simulate_cucker_smale<'py>(
        py: Python<'py>,
        x0: PyReadonlyArray2<'py, f64>,
        t_final: f64,
        n_steps: usize,
        spatial_dim: usize,
        beta: f64,
        sigma: f64,
        record_every: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let x0_view = x0.as_array();
        let expected_cols = 2 * spatial_dim;
        if x0_view.ncols() != expected_cols {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "x0 has {} columns, expected 2 * spatial_dim = {}",
                x0_view.ncols(),
                expected_cols
            )));
        }
        if n_steps == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_steps must be > 0",
            ));
        }
        if !t_final.is_finite() || t_final <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "t_final must be a positive finite number",
            ));
        }

        let x0_owned: Array2<f64> = x0_view.to_owned();
        let model = CuckerSmale::new(spatial_dim, beta, sigma);

        // Release the GIL for the duration of the integration so the Rust
        // thread pool can run unimpeded and the Python interpreter stays
        // responsive (e.g. during long sims in a Jupyter kernel).
        let result = py.allow_threads(|| {
            euler_maruyama(&model, &x0_owned, t_final, n_steps, record_every, seed)
        });

        Ok(result.into_pyarray_bound(py))
    }

    /// Simulate the linear-quadratic McKean-Vlasov model.
    ///
    /// See `python/mvkit/__init__.py` for the user-facing docstring.
    #[pyfunction]
    #[pyo3(signature = (
        x0,
        t_final,
        n_steps,
        a,
        b,
        sigma,
        record_every = 0,
        seed = 42,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn simulate_linear_quadratic<'py>(
        py: Python<'py>,
        x0: PyReadonlyArray2<'py, f64>,
        t_final: f64,
        n_steps: usize,
        a: f64,
        b: f64,
        sigma: f64,
        record_every: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let x0_view = x0.as_array();
        if x0_view.ncols() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "x0 has {} columns, expected 1 (linear-quadratic state is scalar)",
                x0_view.ncols()
            )));
        }
        if n_steps == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_steps must be > 0",
            ));
        }
        if !t_final.is_finite() || t_final <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "t_final must be a positive finite number",
            ));
        }
        if !a.is_finite() || !b.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a and b must be finite",
            ));
        }
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sigma must be a non-negative finite number",
            ));
        }

        let x0_owned: Array2<f64> = x0_view.to_owned();
        let model = LinearQuadratic::new(a, b, sigma);

        let result = py.allow_threads(|| {
            euler_maruyama(&model, &x0_owned, t_final, n_steps, record_every, seed)
        });

        Ok(result.into_pyarray_bound(py))
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(functions::simulate_cucker_smale, m)?)?;
    m.add_function(wrap_pyfunction!(functions::simulate_linear_quadratic, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
