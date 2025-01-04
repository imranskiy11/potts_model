use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::time::{Duration, Instant};


pub fn show_visualization_screen(ui: &mut egui::Ui, app: &mut crate::App) {
    egui::SidePanel::right("graph_panel").min_width(200.0).show(ui.ctx(), |ui| {
        ui.heading("График характеристик");
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Энергия").clicked() {
                app.characteristic = "Энергия".to_string();
            }
            if ui.button("Теплоёмкость").clicked() {
                app.characteristic = "Теплоёмкость".to_string();
            }
        });

        Plot::new("Characteristics")
            .view_aspect(1.0)
            .show(ui, |plot_ui| {
                let points: PlotPoints = match app.characteristic.as_str() {
                    "Энергия" => app
                        .energy_history
                        .iter()
                        .map(|&(temp, energy)| [temp, energy])
                        .collect::<Vec<[f64; 2]>>()
                        .into(),
                    "Теплоёмкость" => {
                        let mut points = Vec::new();
                        for ((temp, energy), (_, energy_squared)) in app
                            .energy_history
                            .iter()
                            .zip(&app.energy_squared_history)
                        {
                            let cv = (energy_squared - energy.powi(2)) / temp.powi(2);
                            points.push([*temp, cv]);
                        }
                        points.into()
                    }
                    _ => Vec::new().into(),
                };
                plot_ui.line(Line::new(points).name(&app.characteristic));
            });

        ui.group(|ui| {
            ui.label("Экспорт данных:");
            if ui.button("Экспортировать").clicked() {
                if let Err(e) = crate::utils::csv_export::export_to_csv(&app.energy_history, &app.energy_squared_history, "results.csv") {
                    eprintln!("Ошибка при экспорте: {}", e);
                }
            }
        });

        ui.heading("Алгоритм Ванга-Ландау");
        ui.separator();

        if ui.button(if app.wang_landau_active { "Остановить" } else { "Запустить" }).clicked() {
            app.wang_landau_active = !app.wang_landau_active;
        }

        Plot::new("ln(Omega(E))")
            .view_aspect(2.0)
            .show(ui, |plot_ui| {
                let points: PlotPoints = app
                    .omega
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| [app.min_energy as f64 + i as f64, value.ln()])
                    .collect::<Vec<[f64; 2]>>()
                    .into();
                plot_ui.line(Line::new(points).name("ln(Ω(E))"));
            });
    });

    egui::CentralPanel::default().show(ui.ctx(), |ui| {
        ui.heading("Визуализация модели Поттса");
        ui.separator();

        ui.group(|ui| {
            ui.label("Управление температурой:");
            ui.add(egui::Slider::new(&mut app.temperature, 0.1..=5.0).text("Температура"));
        });

        ui.group(|ui| {
            ui.label("Скорость обновления (мс):");
            ui.add(egui::Slider::new(&mut app.update_interval, 10.0..=1000.0).text("Интервал"));
        });

        ui.group(|ui| {
            ui.label("Выбор сечения Z:");
            ui.add(
                egui::Slider::new(&mut app.slice_z, 0..=(app.nz - 1))
                    .text("Сечение Z")
                    .clamp_to_range(true),
            );
        });

        ui.group(|ui| {
            ui.label("Управление симуляцией:");
            ui.horizontal(|ui| {
                if ui.button(if app.is_running { "Пауза" } else { "Запуск" }).clicked() {
                    app.is_running = !app.is_running;
                }
                if ui.button("Перезапуск").clicked() {
                    if let Some(lattice) = &mut app.lattice {
                        *lattice = crate::lattice::Lattice::new(app.nx, app.ny, app.nz, app.q);
                    }
                    app.energy_history.clear();
                    app.energy_squared_history.clear();
                    app.is_running = true;
                }
            });
        });

        if let Some(lattice) = &mut app.lattice {
            if app.wang_landau_active {
                lattice.wang_landau_step(
                    &mut app.omega,
                    &mut app.histogram,
                    app.f,
                    app.min_energy as i32, // Преобразование в i32
                );
            } else if app.is_running && app.last_update.elapsed()
                >= Duration::from_millis(app.update_interval as u64)
            {
                for _ in 0..app.steps_per_update {
                    lattice.monte_carlo_step(app.temperature);
                }

                let total_energy: i32 = lattice
                    .states
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let z = i / (app.nx * app.ny);
                        let y = (i % (app.nx * app.ny)) / app.nx;
                        let x = i % app.nx;
                        lattice.calculate_energy(x, y, z)
                    })
                    .sum();

                let average_energy = total_energy as f64 / (app.nx * app.ny * app.nz) as f64;
                app.energy_history.push((app.temperature, average_energy));
                app.energy_squared_history.push((app.temperature, average_energy.powi(2)));

                app.last_update = Instant::now();
            }

            let states = lattice.get_slice(app.slice_z);
            for row in states {
                ui.horizontal(|ui| {
                    for &state in &row {
                        let color = match state {
                            0 => egui::Color32::RED,
                            1 => egui::Color32::GREEN,
                            2 => egui::Color32::BLUE,
                            _ => egui::Color32::GRAY,
                        };
                        ui.colored_label(color, "⬛");
                    }
                });
            }
        }

        if ui.button("Назад").clicked() {
            app.current_screen = crate::Screen::Settings;
        }
    });
    
    
}
