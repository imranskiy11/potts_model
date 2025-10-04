use egui::*;

pub fn label(ui: &Ui, text: &str, vertical: bool) {
    let font  = TextStyle::Monospace.resolve(ui.style());
    let color = ui.style().visuals.text_color();

    if vertical {
        let stacked: String = text.chars().flat_map(|c| [c, '\n']).collect();
        let galley = ui.fonts(|f| f.layout(stacked, font, color, f32::INFINITY));
        let pos    = ui.max_rect().left_center() - vec2(14.0, galley.size().y * 0.5);
        ui.painter().galley(pos, galley);
    } else {
        let galley = ui.fonts(|f| f.layout_no_wrap(text.to_owned(), font, color));
        let pos    = ui.max_rect().center_bottom() + vec2(0.0, 6.0);
        ui.painter().galley(pos, galley);
    }
}
