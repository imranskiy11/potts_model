mod lattice;
mod ui;
mod utils;
mod gpu;

use eframe::egui;
use std::time::Instant;

use crate::gpu::monte_carlo::run_monte_carlo_step_on_gpu;
use crate::lattice::Lattice;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "Модель Поттса с GPU/CPU",
        options,
        Box::new(|_| Box::new(App::default())),
    )
}

enum ComputeMode {
    CPU,
    GPU,
}

struct App {
    current_screen: Screen,
    nx: usize,
    ny: usize,
    nz: usize,
    q: u8,
    temperature: f64,
    lattice: Option<Lattice>,
    is_running: bool,
    last_update: Instant,
    compute_mode: ComputeMode,
    steps_per_update: usize,
    energy_history: Vec<(f64, f64)>,
    energy_squared_history: Vec<(f64, f64)>,
    omega: Vec<f64>,
    histogram: Vec<u64>,
    f: f64,
    min_energy: f64,
    slice_z: usize,
    update_interval: f64,
    wang_landau_active: bool,
    characteristic: String,
    results: String,
}

enum Screen {
    Settings,
    Visualization,
}


impl Default for App {
    fn default() -> Self {
        Self {
            nx: 20,
            ny: 20,
            nz: 10,
            q: 4,
            temperature: 2.0,
            lattice: None,
            is_running: false,
            last_update: Instant::now(),
            compute_mode: ComputeMode::CPU,
            steps_per_update: 1,
            energy_history: vec![],
            energy_squared_history: vec![],
            omega: vec![1.0; 100], // Пример значения
            histogram: vec![0; 100], // Пример значения
            f: 1.0,
            min_energy: -1.0,
            slice_z: 0,
            update_interval: 100.0,
            wang_landau_active: false,
            characteristic: String::new(),
            results: String::new(),
            current_screen: Screen::Settings,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Выбор режима выполнения:");
            ui.horizontal(|ui| {
                if ui.button("CPU").clicked() {
                    self.compute_mode = ComputeMode::CPU;
                }
                if ui.button("GPU").clicked() {
                    self.compute_mode = ComputeMode::GPU;
                }
            });

            if ui.button("Запустить Монте-Карло").clicked() {
                if let Some(ref mut lattice) = self.lattice {
                    match self.compute_mode {
                        ComputeMode::CPU => {
                            lattice.monte_carlo_step(self.temperature);
                        }
                        ComputeMode::GPU => {
                            run_monte_carlo_step_on_gpu(
                                lattice.nx,
                                lattice.ny,
                                lattice.nz,
                                lattice.q,
                                &mut lattice.states,
                                self.temperature,
                            );
                        }
                    }
                }
            }
        });
    }
}
