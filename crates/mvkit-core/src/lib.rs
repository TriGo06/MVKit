//! mvkit-core: McKean-Vlasov particle simulation in Rust.
//!
//! This crate provides the numerical core: trait abstraction for mean-field
//! SDEs, Euler-Maruyama and coordinatewise Milstein integrators, and models like
//! Cucker-Smale. The Python bindings live in the `mvkit-py` sibling crate.

pub mod models;
mod parallel;
pub mod schemes;
pub mod traits;

pub use traits::MeanFieldSDE;
