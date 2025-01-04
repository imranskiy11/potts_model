mod lattice;
mod ui;
mod utils;

use eframe::egui;
// use egui_plot::{Line, Plot, PlotPoints};
use std::time::{Duration, Instant};

use crate::lattice::Lattice;
use crate::ui::{show_settings_screen, show_visualization_screen};
use crate::utils::csv_export;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "Модель Поттса",
        options,
        Box::new(|_| Box::new(App::default())),
    )
}

struct App {
    nx: usize,
    ny: usize,
    nz: usize,
    q: u8,
    temperature: f64,
    steps_per_update: usize,
    lattice: Option<Lattice>,
    slice_z: usize,
    results: String,
    current_screen: Screen,
    is_running: bool,
    energy_history: Vec<(f64, f64)>,
    energy_squared_history: Vec<(f64, f64)>,
    characteristic: String,
    last_update: Instant,
    update_interval: f32,
    omega: Vec<f64>,
    histogram: Vec<u64>,
    f: f64,
    min_energy: i32,
    wang_landau_active: bool,
}

impl Default for App {
    fn default() -> Self {
        let min_energy = -200;
        let max_energy = 200;
        let energy_bins = (max_energy - min_energy + 1) as usize;

        Self {
            nx: 20,
            ny: 20,
            nz: 10,
            q: 4,
            temperature: 2.0,
            steps_per_update: 10,
            lattice: None,
            slice_z: 0,
            results: String::new(),
            current_screen: Screen::Settings,
            is_running: false,
            energy_history: Vec::new(),
            energy_squared_history: Vec::new(),
            characteristic: "Энергия".to_string(),
            last_update: Instant::now(),
            update_interval: 100.0,
            omega: vec![1.0; energy_bins],
            histogram: vec![0; energy_bins],
            f: 2.71828,
            min_energy,
            wang_landau_active: false,
        }
    }
}

enum Screen {
    Settings,
    Visualization,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_screen {
                Screen::Settings => show_settings_screen(ui, self),
                Screen::Visualization => show_visualization_screen(ui, self),
            }
        });
    }
}
