use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::time::{Duration, Instant};
use std::thread;

use crate::app::{App, Screen, ComputeMode};
use crate::gpu::monte_carlo::run_monte_carlo_step_on_gpu;
use crate::lattice::Lattice;
use crate::utils::csv_export::export_to_csv;

fn compute_total_energy(lat: &mut Lattice)-> f64 {
    let mut total=0;
    for i in 0.. lat.states.len(){
        let z= i/(lat.nx*lat.ny);
        let rest= i- z*(lat.nx*lat.ny);
        let y= rest/ lat.nx;
        let x= rest% lat.nx;
        total+= lat.calculate_energy(x,y,z);
    }
    total as f64/ (lat.nx* lat.ny* lat.nz) as f64
}

pub fn show_visualization_screen(ctx: &egui::Context, app: &mut App) {
    egui::SidePanel::right("graph_panel").min_width(280.0).show(ctx, |ui| {
        ui.heading("Графики");
        ui.separator();

        // График энергии
        Plot::new("EnergyPlot").view_aspect(1.5).show(ui, |plot_ui| {
            let points: PlotPoints= app.energy_history.iter()
                .map(|&(t,e)| [t,e])
                .collect::<Vec<[f64;2]>>()
                .into();
            plot_ui.line(Line::new(points).name("Energy"));
        });

        ui.separator();
        // График теплоёмкости
        Plot::new("CvPlot").view_aspect(1.5).show(ui, |plot_ui| {
            let points: PlotPoints= app.energy_history.iter()
                .zip(&app.energy_squared_history)
                .map(|(&(temp,e), &(_,e2))| {
                    let cv= (e2- e.powi(2))/ temp.powi(2);
                    [temp, cv]
                })
                .collect::<Vec<[f64;2]>>()
                .into();
            plot_ui.line(Line::new(points).name("Cv"));
        });

        ui.separator();
        ui.group(|ui|{
            ui.label("Экспорт CSV:");
            if ui.button("Экспортировать").clicked() {
                match export_to_csv(&app.energy_history,&app.energy_squared_history,"results.csv"){
                    Ok(_)=> app.results="Экспорт ок".to_string(),
                    Err(e)=> {
                        eprintln!("Ошибка экспорта: {e}");
                        app.results= format!("Ошибка экспорта: {e}");
                    }
                }
            }
        });

        ui.separator();
        ui.label(format!("Результат: {}", app.results));
    });

    egui::CentralPanel::default().show(ctx, |ui| {
        ui.heading("Визуализация модели Поттса");
        ui.separator();

        ui.group(|ui|{
            ui.label("Температура:");
            ui.add(egui::Slider::new(&mut app.temperature, 0.1..=5.0).text("T"));
        });
        ui.group(|ui|{
            ui.label("Скорость обновления (мс):");
            ui.add(egui::Slider::new(&mut app.update_interval, 10.0..=2000.0).text("Интервал"));
        });
        ui.group(|ui|{
            ui.label("Сечение Z:");
            if app.nz>0 {
                ui.add(
                    egui::Slider::new(&mut app.slice_z, 0..=(app.nz-1))
                        .clamp_to_range(true)
                        .text("Z slice")
                );
            }
        });

        ui.group(|ui|{
            ui.horizontal(|ui|{
                if ui.button(if app.is_running {"Пауза"} else {"Запуск"}).clicked() {
                    app.is_running= !app.is_running;
                }
                if ui.button("Перезапуск").clicked(){
                    if let Some(lat)= &mut app.lattice {
                        *lat= Lattice::new(app.nx,app.ny,app.nz,app.q);
                    }
                    app.energy_history.clear();
                    app.energy_squared_history.clear();
                    app.is_running= true;
                    app.results="Перезапущено".to_string();
                }
            });
        });


        if let Some(lat)= &mut app.lattice {
            if app.is_running &&
               app.last_update.elapsed() >= Duration::from_millis(app.update_interval as u64)
            {
                match app.compute_mode {
                    ComputeMode::CPU => {
                        // steps_per_update 
                        for _ in 0.. app.steps_per_update {
                            lat.monte_carlo_step(app.temperature);
                        }
                        let e= compute_total_energy(lat);
                        app.energy_history.push((app.temperature, e));
                        app.energy_squared_history.push((app.temperature,e.powi(2)));
                        app.results="CPU шаг завершен".to_string();
                        app.last_update= Instant::now();
                    }
                    ComputeMode::GPU => {
                        if !app.gpu_in_progress {
                            app.gpu_in_progress= true;
                            let mut states_clone= lat.states.clone();
                            let (nx,ny,nz,q)= (lat.nx, lat.ny, lat.nz, lat.q);
                            let temp= app.temperature;
                            let sweeps= app.steps_per_update;

                            app.results="Запускаем фоновый GPU-поток...".to_string();
                            app.last_update= Instant::now();

                            app.gpu_join_handle= Some(thread::spawn(move|| {
                                let _= run_monte_carlo_step_on_gpu(
                                    nx,ny,nz,q,
                                    &mut states_clone,
                                    temp,
                                    sweeps
                                );
                                states_clone
                            }));
                        }
                    }
                }
            }

            //  slice Z
            let z_s= app.slice_z.min(lat.nz.saturating_sub(1));
            let slice= lat.get_slice(z_s);
            for row in slice {
                ui.horizontal(|ui|{
                    for &st in &row {
                        let color= match st {
                            0=> egui::Color32::RED,
                            1=> egui::Color32::GREEN,
                            2=> egui::Color32::BLUE,
                            3=> egui::Color32::YELLOW,
                            _=> egui::Color32::GRAY,
                        };
                        ui.colored_label(color,"⬛");
                    }
                });
            }
        } else {
            ui.label("Решетка не создана! Перейдите в Settings.");
        }

        if ui.button("Назад").clicked() {
            app.current_screen= Screen::Settings;
        }
    });
}