use eframe::egui;
use crate::app::{App, Screen, ComputeMode};
use crate::lattice::Lattice;

pub fn show_settings_screen(ctx: &egui::Context, app: &mut App) {
    egui::CentralPanel::default().show(ctx, |ui| {
        ui.heading("Настройки модели Поттса");
        ui.separator();

        // (CPU/GPU)
        ui.group(|ui| {
            ui.label("Режим вычислений:");
            ui.horizontal(|ui| {
                ui.radio_value(&mut app.compute_mode, ComputeMode::CPU, "CPU");
                ui.radio_value(&mut app.compute_mode, ComputeMode::GPU, "GPU");
            });
            ui.label(format!("Текущий режим: {:?}", app.compute_mode));
        });

        ui.separator();
        ui.group(|ui| {
            ui.label("Размеры решётки:");
            ui.horizontal(|ui| {
                ui.label("X:");
                ui.add(egui::DragValue::new(&mut app.nx).speed(1.0));
                ui.label("Y:");
                ui.add(egui::DragValue::new(&mut app.ny).speed(1.0));
                ui.label("Z:");
                ui.add(egui::DragValue::new(&mut app.nz).speed(1.0));
            });
        });

        ui.group(|ui| {
            ui.label("Параметры модели:");
            ui.horizontal(|ui| {
                ui.label("Число состояний (q):");
                ui.add(egui::DragValue::new(&mut app.q).speed(1.0));
            });

            ui.horizontal(|ui| {
                ui.label("Шаги на обновление (Метод Монте-Карло):");
                ui.add(egui::DragValue::new(&mut app.steps_per_update).speed(1.0));
            });
        });

        if ui.button("Создать решётку").clicked() {
            if app.nx > 0 && app.ny > 0 && app.nz > 0 && app.q > 1 {

                app.lattice = Some(Lattice::new(app.nx, app.ny, app.nz, app.q));
                app.results = "Решётка создана!".to_string();

                app.energy_history.clear();
                app.energy_squared_history.clear();
                app.omega.fill(1.0);
                app.histogram.fill(0);

                app.current_screen = Screen::Visualization;
                app.is_running = true;
                app.last_update = std::time::Instant::now();
            } else {
                app.results = "Ошибка: Проверьте параметры ввода!".to_string();
            }
        }

        ui.separator();
        ui.label(format!("Результат: {}", app.results));
    });
}
