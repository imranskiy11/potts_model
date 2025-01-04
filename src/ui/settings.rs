use eframe::egui;

pub fn show_settings_screen(ui: &mut egui::Ui, app: &mut crate::App) {
    ui.heading("Настройки модели Поттса");
    ui.separator();

    ui.vertical(|ui| {
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
                app.lattice = Some(crate::lattice::Lattice::new(app.nx, app.ny, app.nz, app.q));
                app.results = "Решётка создана!".to_string();
                app.energy_history.clear();
                app.energy_squared_history.clear();
                app.omega.fill(1.0);
                app.histogram.fill(0);
                app.current_screen = crate::Screen::Visualization;
                app.is_running = true;
                app.last_update = std::time::Instant::now();
            } else {
                app.results = "Ошибка: Проверьте параметры ввода!".to_string();
            }
        }
    });
}
