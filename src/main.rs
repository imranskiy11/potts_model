mod app; 
mod lattice;
mod ui;
mod utils;
mod gpu;


use app::App;


fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "Модель Поттса с GPU/CPU",
        options,
        Box::new(|_| Box::new(App::default())),
    )
}