use std::time::Instant;
use eframe::egui;

use crate::lattice::Lattice;
// use crate::gpu::monte_carlo::run_monte_carlo_step_on_gpu;

/// Режим вычислений
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeMode {
    CPU,
    GPU,
}

/// Экран в UI (два экрана: настройки и визуализация)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Settings,
    Visualization,
}

/// Главное приложение
pub struct App {
    pub current_screen: Screen,

    // Параметры решётки
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub q: u8,
    pub temperature: f64,

    // Собственно решётка (None, пока не создана)
    pub lattice: Option<Lattice>,

    // Управление симуляцией
    pub is_running: bool,
    pub last_update: Instant,
    pub compute_mode: ComputeMode,
    pub steps_per_update: usize,

    // Хранение энергии
    pub energy_history: Vec<(f64, f64)>,
    pub energy_squared_history: Vec<(f64, f64)>,

    // Параметры для алгоритма Ванга-Ландау (по желанию)
    pub omega: Vec<f64>,
    pub histogram: Vec<u64>,
    pub f: f64,
    pub min_energy: f64,

    // Настройки визуализации
    pub slice_z: usize,
    pub update_interval: f64,
    pub wang_landau_active: bool,
    pub characteristic: String,

    // Строка для результатов/сообщений
    pub results: String,
}

impl Default for App {
    fn default() -> Self {
        Self {
            current_screen: Screen::Settings,

            nx: 50,
            ny: 50,
            nz: 50,
            q: 4,
            temperature: 2.0,

            lattice: None,

            is_running: false,
            last_update: Instant::now(),
            compute_mode: ComputeMode::CPU,
            steps_per_update: 1,

            energy_history: vec![],
            energy_squared_history: vec![],

            omega: vec![1.0; 200],
            histogram: vec![0; 200],
            f: 1.0,
            min_energy: -1.0,

            slice_z: 0,
            update_interval: 200.0,
            wang_landau_active: false,
            characteristic: String::new(),

            results: String::new(),
        }
    }
}

/// Имплементация eframe::App
impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        match self.current_screen {
            Screen::Settings => {
                crate::ui::settings::show_settings_screen(ctx, self);
            }
            Screen::Visualization => {
                crate::ui::visualization::show_visualization_screen(ctx, self);
            }
        }
    }
}