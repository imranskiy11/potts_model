use egui::*;

pub fn drag_f64(ui: &mut Ui, label: &str, v: &mut f64, speed: f64) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(
            DragValue::new(v)
                .speed(speed)
                .clamp_range(-20.0..=20.0)
        );
    });
}
