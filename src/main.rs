mod axis; 
mod monitor; 
mod app;
mod viz; 
mod stats;
mod lattice;
mod ui_helpers;
mod wang_landau;

#[cfg(feature = "cuda")]
mod gpu;
#[cfg(not(feature = "cuda"))]
mod gpu;
mod utils;

#[cfg(feature = "cuda")]
use cust::CudaFlags;

fn main() -> Result<(), eframe::Error> {
    #[cfg(feature = "cuda")]
    cust::init(CudaFlags::empty()).expect("CUDA init");

    eframe::run_native(
        "Potts Model",
        Default::default(),
        Box::new(|cc| Box::new(app::App::default_with_repaint(cc))),
    )
}
