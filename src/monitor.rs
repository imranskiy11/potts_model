use std::time::{Duration, Instant};
use sysinfo::{System, SystemExt, CpuExt};

#[cfg(feature = "cuda")]
use nvml_wrapper::Nvml;

pub struct Monitor {
    sys:       System,
    last:      Instant,
    cpu_perc:  f32,
    ram_mb:    u64,

    #[cfg(feature = "cuda")]
    gpu: Option<GpuInfo>,
}

#[cfg(feature = "cuda")]
struct GpuInfo {
    name:    String,
    util:    f32,
    vram_mb: u64,
}

impl Monitor {
    pub fn new() -> Self {
        let mut sys = System::new();
        sys.refresh_cpu();

        #[cfg(feature = "cuda")]
        let gpu = Nvml::init().ok().and_then(|nv| {
            let dev   = nv.device_by_index(0).ok()?;
            let name  = dev.name().unwrap_or_default();
            Some(GpuInfo { name, util: 0.0, vram_mb: 0 })
        });

        Self{
            sys,
            last: Instant::now(),
            cpu_perc: 0.0,
            ram_mb:   0,
            #[cfg(feature="cuda")]
            gpu,
        }
    }

    fn refresh(&mut self) {
        if self.last.elapsed() < Duration::from_millis(500) { return; }
        self.last = Instant::now();

        self.sys.refresh_cpu();
        self.cpu_perc = self.sys.global_cpu_info().cpu_usage();

        self.sys.refresh_memory();
        self.ram_mb = self.sys.used_memory() / (1024 * 1024); // KB → MB

        #[cfg(feature = "cuda")]
        if let Some(g) = &mut self.gpu {
            if let Ok(nv) = Nvml::init() {
                if let Ok(dev) = nv.device_by_index(0) {
                    if let Ok(u) = dev.utilization_rates() { g.util = u.gpu as f32; }
                    if let Ok(m) = dev.memory_info()       { g.vram_mb = (m.used >> 20) as u64; }
                }
            }
        }
    }

    pub fn draw(&mut self, ui:&mut egui::Ui) {
        self.refresh();

        ui.label(format!("CPU  {:>4.0} %", self.cpu_perc));
        ui.label(format!("RAM  {:>5} MB", self.ram_mb));

        #[cfg(feature = "cuda")]
        {
            let _ = if let Some(g) = &self.gpu {
                ui.label(format!("GPU  {:>4.0} % | {:>5} MB | {}", g.util, g.vram_mb, g.name))
            } else {
                ui.label("GPU  —")
            };
        }

        #[cfg(not(feature = "cuda"))]
        ui.label("GPU  (cuda off)");
    }
}
