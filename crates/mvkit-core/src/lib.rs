//! mvkit-core: McKean-Vlasov particle simulation in Rust.
//!
//! This crate provides the numerical core: trait abstraction for mean-field
//! SDEs, integrators (Euler-Maruyama for now), and built-in models like
//! Cucker-Smale. The Python bindings live in the `mvkit-py` sibling crate.

pub mod models;
pub mod schemes;
pub mod traits;

pub use traits::MeanFieldSDE;
