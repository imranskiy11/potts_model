// src/main.rs

mod app;
mod lattice;
mod gpu;
mod ui;
mod utils;

use app::App;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "Модель Поттса (GPU/CPU)",
        options,
        Box::new(|_| Box::new(App::default())),
    )
}
