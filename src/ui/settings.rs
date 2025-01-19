use eframe::egui;
use crate::app::{App, Screen, ComputeMode};
use crate::lattice::Lattice;

pub fn show_settings_screen(ctx: &egui::Context, app: &mut App) {
    egui::CentralPanel::default().show(ctx, |ui| {
        ui.heading("Настройки модели Поттса (GPU/CPU)");
        ui.separator();

        ui.group(|ui|{
            ui.label("Режим вычислений:");
            ui.horizontal(|ui|{
                ui.radio_value(&mut app.compute_mode, ComputeMode::CPU, "CPU");
                ui.radio_value(&mut app.compute_mode, ComputeMode::GPU, "GPU");
            });
        });

        ui.separator();
        ui.group(|ui|{
            ui.label("Размеры решетки (nx, ny, nz):");
            ui.horizontal(|ui|{
                ui.label("nx:");
                ui.add(egui::DragValue::new(&mut app.nx).clamp_range(1..=9999));
                ui.label("ny:");
                ui.add(egui::DragValue::new(&mut app.ny).clamp_range(1..=9999));
                ui.label("nz:");
                ui.add(egui::DragValue::new(&mut app.nz).clamp_range(1..=9999));
            });
        });

        ui.group(|ui|{
            ui.label("Число состояний (q):");
            ui.add(egui::DragValue::new(&mut app.q).clamp_range(2..=255));
        });
        ui.group(|ui|{
            ui.label("steps_per_update (n_sweeps для GPU):");
            ui.add(egui::DragValue::new(&mut app.steps_per_update).speed(1));
        });

        if ui.button("Создать решетку").clicked() {
            if app.nx>0 && app.ny>0 && app.nz>0 && app.q>1 {
                app.lattice= Some(Lattice::new(app.nx, app.ny, app.nz, app.q));
                app.energy_history.clear();
                app.energy_squared_history.clear();
                app.results="Решетка создана!".to_string();
                app.current_screen= Screen::Visualization;
                app.is_running= true;
                app.last_update= std::time::Instant::now();
            } else {
                app.results="Некорректные параметры".to_string();
            }
        }

        ui.separator();
        ui.label(format!("Результат: {}", app.results));
    });
}