use eframe::egui;
use cust::CudaFlags;

mod app;
mod lattice;
mod gpu;
mod ui;
mod utils;

use app::App;

fn main() -> Result<(), eframe::Error> {
    // Инициализируем CUDA.
    // Передаём пустой флаг, и при ошибке - panic!
    cust::init(CudaFlags::empty()).expect("Не удалось инициализировать CUDA!");

    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "Модель Поттса (GPU/CPU)",
        options,
        Box::new(|_| Box::new(App::default())),
    )
}