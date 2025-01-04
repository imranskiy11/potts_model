pub mod lattice;
pub mod monte_carlo;
pub mod wang_landau;
pub mod energy;

pub use lattice::Lattice;
// pub use monte_carlo::MonteCarlo;
pub use wang_landau::WangLandau;  // Expose WangLandau
// pub use energy::Energy;  // Expose Energy
