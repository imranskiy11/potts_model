use std::time::Instant;
use eframe::egui;
use crate::lattice::Lattice;
use std::thread;

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

pub struct App {
    pub current_screen: Screen,

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

    pub omega: Vec<f64>,
    pub histogram: Vec<u64>,
    pub f: f64,
    pub min_energy: f64,

    pub slice_z: usize,
    pub update_interval: f64,
    pub wang_landau_active: bool,
    pub characteristic: String,

    pub results: String,

    pub gpu_in_progress: bool,
    pub gpu_join_handle: Option<thread::JoinHandle<Vec<u8>>>,
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

            gpu_in_progress: false,
            gpu_join_handle: None,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(handle) = self.gpu_join_handle.take() {
            if handle.is_finished() {
                // join
                match handle.join() {
                    Ok(new_states) => {
                        if let Some(lat) = &mut self.lattice {
                            lat.states= new_states;
                            let e= compute_energy(lat);
                            self.energy_history.push((self.temperature, e));
                            self.energy_squared_history.push((self.temperature, e.powi(2)));
                        }
                        self.results="GPU расчет завершен".to_string();
                    }
                    Err(_)=> {
                        self.results="Ошибка join GPU".to_string();
                    }
                }
                self.gpu_in_progress= false;
            } else {
                self.gpu_join_handle= Some(handle);
            }
        }

        match self.current_screen {
            Screen::Settings => crate::ui::settings::show_settings_screen(ctx, self),
            Screen::Visualization => crate::ui::visualization::show_visualization_screen(ctx, self),
        }
        ctx.request_repaint();
    }
}

fn compute_energy(lat: &mut Lattice)-> f64 {
    let mut total=0;
    for i in 0.. lat.states.len() {
        let z= i/(lat.nx*lat.ny);
        let rest= i- z*(lat.nx*lat.ny);
        let y= rest/ lat.nx;
        let x= rest% lat.nx;
        total+= lat.calculate_energy(x,y,z);
    }
    total as f64/ (lat.nx*lat.ny*lat.nz) as f64
}