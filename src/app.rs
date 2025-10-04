use std::time::{Duration, Instant};

use egui::*;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use glam::IVec3;
use rand::{thread_rng, Rng};

use crate::{
    axis, gpu::run_monte_carlo_step,
    lattice::{AnyLattice, Lattice},
    monitor::Monitor,
    stats::{Accum, StatRow},
    ui_helpers::drag_f64,
    utils::{random_spins, Couplings},
    viz,
    wang_landau::{WLParams, WLState},
};


#[derive(Clone, Copy, PartialEq)] enum Compute { CPU, GPU }
#[derive(Clone, Copy, PartialEq)] enum Screen  { Plot, Slice, Stats, WL }


pub struct App {

    dims:  IVec3,
    q:     u8,
    q_slider: f64,
    lat:   AnyLattice,
    spins: Vec<u8>,


    cpl:     Couplings,
    sweeps:  usize,
    compute: Compute,


    series:    Vec<(f64,f64)>,
    screen:    Screen,
    slice_z:   usize,
    slice_tex: Option<TextureHandle>,


    running:   bool,
    dt_ms:     u64,
    last_tick: Instant,


    acc:       Accum,
    rows:      Vec<StatRow>,
    beta_step: f64,
    samples:   usize,


    wl_p:       WLParams,
    wl_s:       Option<WLState>,
    wl_running: bool,

    monitor: Monitor,
}

impl Default for App {
    fn default() -> Self {
        let dims  = IVec3::splat(20);
        let lat   = AnyLattice::Bcc(crate::lattice::bcc::Bcc::new(dims, /*shells=*/2));
        let q     = 3;
        let spins = random_spins(q, lat.n_sites());

        Self {
            dims, q, q_slider: q as f64,
            lat, spins,

            cpl:     Couplings { js: vec![1.0, 1.0], beta: 1.0 },
            sweeps:  200,
            compute: Compute::CPU,

            series:  Vec::new(),
            screen:  Screen::Plot,
            slice_z: 0,
            slice_tex: None,

            running: false,
            dt_ms:   30,
            last_tick: Instant::now(),

            acc:       Accum::default(),
            rows:      Vec::new(),
            beta_step: 0.05,
            samples:   400,

            wl_p:       WLParams::default(),
            wl_s:       None,
            wl_running: false,

            monitor:    Monitor::new(),
        }
    }
}
impl App {
    pub fn default_with_repaint(cc:&eframe::CreationContext<'_>)->Self{
        cc.egui_ctx.request_repaint_after(Duration::from_millis(16));
        Self::default()
    }
}


impl eframe::App for App {
    fn update(&mut self, ctx:&egui::Context, _:&mut eframe::Frame) {
        /* авто-MCS */
        if self.running && self.last_tick.elapsed() >= Duration::from_millis(self.dt_ms) {
            step_scan(self);
            self.last_tick = Instant::now();
        }
        if self.wl_running { step_wl(self); }

        ctx.request_repaint();

        SidePanel::left("controls").show(ctx, |ui| self.left_panel(ui));

        CentralPanel::default().show(ctx, |ui| {
            ScrollArea::both().show(ui, |ui| match self.screen {
                Screen::Plot  => plot_panel (self, ui),
                Screen::Slice => slice_panel(self, ui),
                Screen::Stats => stats_panel(self, ui),
                Screen::WL    => wl_panel   (self, ui),
            });
        });
    }
}


fn plot_panel(a:&mut App, ui:&mut Ui){
    let pts: PlotPoints = a.series.iter().map(|(t,e)|[*t,*e]).collect();
    Plot::new("E/N")
        .legend(Legend::default())
        .view_aspect(2.0)
        .show(ui, |p| p.line(Line::new(pts).name("E/N")));
    axis::label(ui,"E / N",  true);
    axis::label(ui,"t (кадр)",false);
}


fn slice_panel(a:&mut App, ui:&mut Ui){
    viz::draw_slice(ui,&mut a.slice_tex,&a.spins,a.dims,
        a.slice_z.min((a.dims.z-1) as usize),a.q);
    if a.dims.z>1 {
        ui.add_space(4.);
        ui.horizontal(|ui|{
            ui.label("z");
            ui.add(Slider::new(&mut a.slice_z,0..=((a.dims.z-1) as usize)));
        });
    }
}


fn stats_panel(a:&mut App, ui:&mut Ui){
    ui.heading("T-scan stats");
    ScrollArea::vertical().max_height(200.).show(ui, |ui|{
        Grid::new("table").striped(true).show(ui, |ui|{
            ui.label("T"); ui.label("E/N"); ui.label("|M|");
            ui.label("Cᵥ"); ui.label("χ"); ui.end_row();
            for r in &a.rows {
                ui.label(format!("{:.3}",r.t ));
                ui.label(format!("{:.4}",r.e ));
                ui.label(format!("{:.4}",r.m ));
                ui.label(format!("{:.6}",r.cv));
                ui.label(format!("{:.6}",r.chi));
                ui.end_row();
            }
        });
    });

    if !a.rows.is_empty() {
        let cv_pts: PlotPoints = a.rows.iter().map(|r| [r.t, r.cv ]).collect();
        let ch_pts: PlotPoints = a.rows.iter().map(|r| [r.t, r.chi]).collect();
        Plot::new("thermo")
            .legend(Legend::default())
            .view_aspect(2.0)
            .show(ui, |p|{
                p.line(Line::new(cv_pts).name("Cᵥ"));
                p.line(Line::new(ch_pts).name("χ"));
            });
        axis::label(ui,"Cᵥ, χ",true);
        axis::label(ui,"T",false);
    }

    if ui.button("Export CSV").clicked() { _ = export_csv(&a.rows); }
}


fn wl_panel(a:&mut App, ui:&mut Ui){
    ui.collapsing("Range & bin", |ui|{
        ui.horizontal(|ui|{ ui.label("E_min"); ui.add(DragValue::new(&mut a.wl_p.e_min).speed(0.1)); });
        ui.horizontal(|ui|{ ui.label("E_max"); ui.add(DragValue::new(&mut a.wl_p.e_max).speed(0.1)); });
        ui.horizontal(|ui|{ ui.label("bin");   ui.add(DragValue::new(&mut a.wl_p.bin  ).speed(0.005));});
        ui.horizontal(|ui|{ ui.label("flat");  ui.add(DragValue::new(&mut a.wl_p.flat ).speed(0.02)); });
        if ui.button("⚠ reset WL state").clicked(){ a.wl_s = None; }
    });

    if a.wl_s.is_none() {
        if ui.button("Init WL").clicked(){ a.wl_s = Some(WLState::new(&a.wl_p)); }
        return;
    }
    let wl = a.wl_s.as_mut().unwrap();

    ui.horizontal(|ui|{
        ui.label(format!("ln f = {:.3e}", wl.ln_f));
        if ui.button(if a.wl_running{"⏸ Pause"}else{"▶ Run"}).clicked(){
            a.wl_running = !a.wl_running;
        }
        if ui.button("Refine (ln f / 2)").clicked(){ wl.refine(); }
    });

    if a.wl_running && wl.is_flat(&a.wl_p) {
        wl.refine();
        if wl.ln_f < 1e-3 { a.wl_running = false; }
    }

    ScrollArea::vertical().max_height(140.).show(ui, |ui|{
        for (i,lg) in wl.ln_g.iter().enumerate(){
            let e = a.wl_p.e_min + i as f64 * a.wl_p.bin;
            ui.label(format!("{:6.3}  {:>8.4}", e, lg));
        }
    });

    if !wl.ln_g.is_empty(){
        let pts: PlotPoints = wl.ln_g.iter().enumerate()
            .map(|(i,lg)|{
                let e = a.wl_p.e_min + i as f64 * a.wl_p.bin;
                [e,*lg]
            }).collect();
        Plot::new("lng")
            .legend(Legend::default())
            .view_aspect(2.0)
            .include_y(0.0)
            .show(ui, |p| p.line(Line::new(pts).name("ln g")));
        axis::label(ui,"ln g(E)", true);
        axis::label(ui,"E / N",   false);
    }
}


fn rebuild(a:&mut App){
    a.lat   = AnyLattice::Bcc(crate::lattice::bcc::Bcc::new(a.dims, a.cpl.js.len()));
    a.spins = random_spins(a.q, a.lat.n_sites());
    a.series.clear(); a.slice_tex=None;
    a.acc.clear();    a.rows.clear(); a.wl_s=None;
}


fn step_scan(a:&mut App){
    run_monte_carlo_step(a.compute==Compute::GPU,
        &a.lat,&mut a.spins,&a.cpl,a.sweeps);

    let n = a.spins.len() as f64;
    let e = energy(&a.lat,&a.spins,&a.cpl)/n;
    let m = magnetization(&a.spins,a.q);
    a.acc.add(e,m);

    if a.acc.n >= a.samples {
        a.rows.push(a.acc.finish(a.cpl.beta, n as usize));
        a.acc.clear();
        a.cpl.beta = (a.cpl.beta + a.beta_step).max(1e-4);
    }
    a.series.push((a.series.len() as f64, e));
}


fn step_wl(a:&mut App){
    let e_now = energy(&a.lat,&a.spins,&a.cpl) / a.spins.len() as f64;
    if e_now < a.wl_p.e_min { a.wl_p.e_min = (e_now-0.5).floor(); a.wl_s=None; }
    if e_now > a.wl_p.e_max { a.wl_p.e_max = (e_now+0.5).ceil();  a.wl_s=None; }
    if a.wl_s.is_none() { a.wl_s = Some(WLState::new(&a.wl_p)); }

    let wl = a.wl_s.as_mut().unwrap();

    let idx = thread_rng().gen_range(0..a.spins.len());
    let old = a.spins[idx];
    let new = thread_rng().gen_range(0..a.q);
    if new == old { return }

    let n = a.spins.len() as f64;
    let e_old = e_now;
    a.spins[idx] = new;
    let e_new = energy(&a.lat,&a.spins,&a.cpl)/n;

    if !wl.step(&a.wl_p,e_old,e_new) { a.spins[idx] = old; }
}


fn energy(lat:&AnyLattice, spins:&[u8], c:&Couplings)->f64{
    let mut e = 0.0;
    for i in 0..spins.len() {
        for (sh,&j) in c.js.iter().enumerate() {
            for &nb in lat.neighbours(i,sh){
                if spins[i] == spins[nb] { e -= j*0.5; }  // каждая пара ровно один раз
            }
        }
    } e
}
fn magnetization(spins:&[u8], q:u8)->f64{
    let mut cnt = vec![0usize; q as usize];
    for &s in spins { cnt[s as usize] += 1; }
    let max = *cnt.iter().max().unwrap() as f64;
    (q as f64*max/spins.len() as f64 -1.0)/(q as f64 -1.0)
}

fn export_csv(rows:&[StatRow])->csv::Result<()>{
    let mut w = csv::Writer::from_path("stats.csv")?;
    for r in rows { w.serialize(r)?; }
    w.flush()?;
    Ok(())
}


impl App {
    fn left_panel(&mut self, ui:&mut Ui) {
        ui.heading("Параметры");

        /* Temperature */
        ui.collapsing("Temperature", |ui|{
            ui.horizontal(|ui|{
                ui.label("β");
                ui.add(Slider::new(&mut self.cpl.beta,0.05..=5.0).show_value(false));
                ui.add(DragValue::new(&mut self.cpl.beta).speed(0.01));
            });
            ui.horizontal(|ui|{
                let mut t = 1.0/self.cpl.beta;
                ui.label("T");
                ui.add(Slider::new(&mut t,0.2..=20.0).show_value(false));
                ui.add(DragValue::new(&mut t).speed(0.05));
                self.cpl.beta = 1.0/t;
            });
        });


        ui.collapsing("Lattice", |ui|{
            ui.horizontal(|ui|{
                ui.label("Lx"); ui.add(DragValue::new(&mut self.dims.x).clamp_range(4..=64));
                ui.label("Ly"); ui.add(DragValue::new(&mut self.dims.y).clamp_range(4..=64));
                ui.label("Lz"); ui.add(DragValue::new(&mut self.dims.z).clamp_range(4..=64));
            });

            let old_q = self.q_slider;
            drag_f64(ui,"q",&mut self.q_slider,1.0);
            self.q_slider = self.q_slider.clamp(2.0, 8.0);


            if (self.q_slider - old_q).abs() > f64::EPSILON {
                self.q = self.q_slider as u8;
                rebuild(self);
            }

            if ui.button("Reset lattice").clicked(){ rebuild(self); }
        });


        ui.collapsing("Interaction", |ui|{
            for j in &mut self.cpl.js { drag_f64(ui,"J",j,0.05); }
            if ui.button("+ ещё оболочка").clicked(){ self.cpl.js.push(0.0); }
            if ui.button("Default J=[1,1]").clicked(){ self.cpl.js = vec![1.0,1.0]; }
        });


        ui.collapsing("Simulation", |ui|{
            ui.horizontal(|ui|{
                ui.radio_value(&mut self.compute,Compute::CPU,"CPU");
                ui.radio_value(&mut self.compute,Compute::GPU,"GPU");
            });
            ui.horizontal(|ui|{
                ui.label("MCS/step");
                ui.add(DragValue::new(&mut self.sweeps).clamp_range(1..=5000));
            });
            if ui.button(if self.running{"⏸ Pause"}else{"▶ Run"}).clicked(){
                self.running = !self.running;
            }
            ui.add_enabled(!self.running,
                Slider::new(&mut self.dt_ms,10..=200).text("dt ms"));
            if ui.button("Single step").clicked(){ step_scan(self); }
        });


        ui.collapsing("T-scan", |ui|{
            drag_f64(ui,"Δβ",&mut self.beta_step,0.005);
            ui.add(DragValue::new(&mut self.samples).clamp_range(10..=5000).suffix(" samp/T"));
            if ui.button("Reset scan").clicked(){ self.rows.clear(); self.acc.clear(); }
        });

        ui.separator(); self.monitor.draw(ui); ui.separator();
        ui.horizontal(|ui|{
            ui.selectable_value(&mut self.screen,Screen::Plot ,"Plot");
            ui.selectable_value(&mut self.screen,Screen::Slice,"Slice");
            ui.selectable_value(&mut self.screen,Screen::Stats,"Stats");
            ui.selectable_value(&mut self.screen,Screen::WL   ,"WL");
        });
    }
}
