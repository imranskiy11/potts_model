// src/app.rs

use std::time::Instant;
use eframe::egui;

// Если используете вычисления на GPU (Монте-Карло)
use crate::gpu::monte_carlo::run_monte_carlo_step_on_gpu;
// Если нужна решётка для CPU-вычислений
use crate::lattice::Lattice;

/// Режим вычислений: CPU или GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeMode {
    CPU,
    GPU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Settings,
    Visualization,
}

/// Главная структура приложения.
///
/// Хранит всё состояние:  
/// - Параметры решётки (nx, ny, nz, q, temperature)  
/// - Ссылку на саму решётку (lattice: Option<Lattice>)  
/// - Настройки работы (compute_mode: CPU/GPU, steps_per_update и т.д.)  
/// - Данные для статистики (energy_history и т.п.)  
/// - Вспомогательные переменные для Ванга-Ландау (omega, histogram, f, min_energy)  
/// - Поля для интерфейса (slice_z, update_interval, is_running, current_screen и др.)  
/// - Строка results для вывода сообщений пользователю
pub struct App {
    // --- Какой экран сейчас показываем
    pub current_screen: Screen,

    // --- Параметры решётки
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub q: u8,
    pub temperature: f64,

    pub lattice: Option<Lattice>,

    pub is_running: bool,
    pub last_update: Instant,
    pub compute_mode: ComputeMode,
    pub steps_per_update: usize,

    pub energy_history: Vec<(f64, f64)>,
    pub energy_squared_history: Vec<(f64, f64)>,

    // --- Параметры алгоритма Ванга-Ландау
    pub omega: Vec<f64>,
    pub histogram: Vec<u64>,
    pub f: f64,
    pub min_energy: f64,

    // --- Параметры для визуализации
    pub slice_z: usize,
    pub update_interval: f64,
    pub wang_landau_active: bool,
    pub characteristic: String,


    pub results: String,
}

impl Default for App {
    fn default() -> Self {
        Self {

            current_screen: Screen::Settings,

            // Начальные размеры решётки, q-состояний и температура
            nx: 20,
            ny: 20,
            nz: 10,
            q: 4,
            temperature: 2.0,


            lattice: None,

            // Управление симуляцией
            is_running: false,
            last_update: Instant::now(),
            compute_mode: ComputeMode::CPU,
            steps_per_update: 1,

            // Очистим историю энергий
            energy_history: vec![],
            energy_squared_history: vec![],

            // Параметры Ванга-Ландау (просто пример)
            omega: vec![1.0; 100],
            histogram: vec![0; 100],
            f: 1.0,
            min_energy: -1.0,

            // Параметры визуализации
            slice_z: 0,
            update_interval: 100.0,
            wang_landau_active: false,
            characteristic: String::new(),

            // Сообщения пользователю
            results: String::new(),
        }
    }
}

/// Реализация eframe::App (главный метод update).
///
/// В зависимости от того, какой экран выбран (current_screen), вызываем код из `ui/settings.rs` или `ui/visualization.rs`.
impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        match self.current_screen {
            Screen::Settings => {
                // Перейдём в UI настроек
                crate::ui::settings::show_settings_screen(ctx, self);
            }
            Screen::Visualization => {
                // Перейдём в экран визуализации
                crate::ui::visualization::show_visualization_screen(ctx, self);
            }
        }
    }
}
