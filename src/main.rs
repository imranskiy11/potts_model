mod app;
mod lattice;
mod gpu;
mod ui;
mod utils;

use app::App;
use cust::CudaFlags;

fn main() -> Result<(), eframe::Error> {
    cust::init(CudaFlags::empty()).expect("Failed to init CUDA!");

    let opts= eframe::NativeOptions::default();

    eframe::run_native(
        "Potts Model (Big GPU, old UI)",
        opts,
        Box::new(|cc| {
            cc.egui_ctx.request_repaint_after(std::time::Duration::from_millis(30));
            Box::new(App::default())
        })
    )
}