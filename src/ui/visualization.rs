use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::time::{Duration, Instant};

use crate::app::{App, Screen, ComputeMode};
use crate::utils::csv_export::export_to_csv;
use crate::gpu::monte_carlo::run_monte_carlo_step_on_gpu;

pub fn show_visualization_screen(ctx: &egui::Context, app: &mut App) {
    egui::SidePanel::right("graph_panel")
        .min_width(300.0)
        .show(ctx, |ui| {
            ui.heading("Графики");
            ui.separator();

            // 1) График энергии
            Plot::new("EnergyPlot")
                .view_aspect(1.5)
                .show(ui, |plot_ui| {
                    let points: PlotPoints = app
                        .energy_history
                        .iter()
                        .map(|&(t, e)| [t, e])
                        .collect::<Vec<[f64; 2]>>()
                        .into();
                    plot_ui.line(Line::new(points).name("Энергия"));
                });
            ui.separator();

            // 2) График теплоёмкости
            Plot::new("CvPlot")
                .view_aspect(1.5)
                .show(ui, |plot_ui| {
                    let points: PlotPoints = app
                        .energy_history
                        .iter()
                        .zip(&app.energy_squared_history)
                        .map(|(&(t, e), &(_, e2))| {
                            let cv = (e2 - e.powi(2)) / t.powi(2);
                            [t, cv]
                        })
                        .collect::<Vec<[f64; 2]>>()
                        .into();
                    plot_ui.line(Line::new(points).name("Теплоёмкость"));
                });

            ui.separator();
            ui.group(|ui| {
                ui.label("Экспорт CSV:");
                if ui.button("Экспортировать").clicked() {
                    if let Err(e) = export_to_csv(
                        &app.energy_history,
                        &app.energy_squared_history,
                        "results.csv",
                    ) {
                        eprintln!("Ошибка: {e}");
                        app.results = format!("Ошибка при экспорте: {e}");
                    } else {
                        app.results = "Данные экспортированы в results.csv".to_string();
                    }
                }
            });

            ui.separator();
            ui.label("Алгоритм Ванга-Ландау");
            if ui
                .button(if app.wang_landau_active {
                    "Остановить Ванга-Ландау"
                } else {
                    "Запустить Ванга-Ландау"
                })
                .clicked()
            {
                app.wang_landau_active = !app.wang_landau_active;
            }

            // График ln(Ω(E)) (при желании)
        });

    egui::CentralPanel::default().show(ctx, |ui| {
        ui.heading("Визуализация модели Поттса");
        ui.separator();

        // Слайдеры
        ui.group(|ui| {
            ui.label("Температура:");
            ui.add(egui::Slider::new(&mut app.temperature, 0.1..=5.0).text("T"));
        });

        ui.group(|ui| {
            ui.label("Скорость обновления (мс):");
            ui.add(egui::Slider::new(&mut app.update_interval, 10.0..=2000.0).text("Интервал"));
        });

        ui.group(|ui| {
            ui.label("Сечение Z:");
            ui.add(
                egui::Slider::new(&mut app.slice_z, 0..=(app.nz - 1))
                    .clamp_to_range(true)
                    .text("Z"),
            );
        });

        // Кнопки "Пауза/Запуск" и "Перезапуск"
        ui.group(|ui| {
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
                    app.results = "Решётка перезапущена".to_string();
                }
            });
        });

        // Либо Ванга–Ландау, либо обычное Монте-Карло
        if let Some(lattice) = &mut app.lattice {
            if app.wang_landau_active {
                // Запуск Wang-Landau шага (CPU-вариант)
                // ...
            } else {
                // Если "не на паузе" и пришло время обновления
                if app.is_running
                    && app.last_update.elapsed() >= Duration::from_millis(app.update_interval as u64)
                {
                    for _ in 0..app.steps_per_update {
                        match app.compute_mode {
                            ComputeMode::CPU => {
                                lattice.monte_carlo_step(app.temperature);
                            }
                            ComputeMode::GPU => {
                                // Запускаем GPU-ядер
                                let res = run_monte_carlo_step_on_gpu(
                                    lattice.nx,
                                    lattice.ny,
                                    lattice.nz,
                                    lattice.q,
                                    &mut lattice.states,
                                    app.temperature,
                                );
                                if let Err(e) = res {
                                    app.results = format!("GPU error: {:?}", e);
                                }
                            }
                        }
                    }
                    // Подсчитываем суммарную энергию
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
                    let avg_energy =
                        total_energy as f64 / (app.nx * app.ny * app.nz) as f64;
                    app.energy_history.push((app.temperature, avg_energy));
                    app.energy_squared_history.push((app.temperature, avg_energy.powi(2)));

                    app.last_update = Instant::now();
                }
            }

            // Рисуем срез Z
            let slice = lattice.get_slice(app.slice_z);
            for row in slice {
                ui.horizontal(|ui| {
                    for &st in &row {
                        let color = match st {
                            0 => egui::Color32::RED,
                            1 => egui::Color32::GREEN,
                            2 => egui::Color32::BLUE,
                            _ => egui::Color32::GRAY,
                        };
                        ui.colored_label(color, "⬛");
                    }
                });
            }
        } else {
            ui.label("Решётка не создана! Вернитесь в Настройки.");
        }

        // Кнопка "Назад"
        if ui.button("Назад").clicked() {
            app.current_screen = Screen::Settings;
        }

        ui.separator();
        ui.label(format!("Результат: {}", app.results));
    });
}