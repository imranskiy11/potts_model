use egui::{ColorImage, TextureHandle, TextureOptions, Ui, Vec2};
use glam::IVec3;
use palette::{Hsl, Srgb, FromColor};


pub fn draw_slice(
    ui: &mut Ui,
    tex_slot: &mut Option<TextureHandle>,
    spins: &[u8],
    dims: IVec3,
    z: usize,
    q: u8,
) {
    let (nx,ny,nz)=(dims.x as usize,dims.y as usize,dims.z as usize);
    if z>=nz { return; }

    let mut rgb = Vec::<u8>::with_capacity(nx*ny*3);
    for y in 0..ny {
        for x in 0..nx {
            let idx = (z*ny+y)*nx+x;
            rgb.extend_from_slice(&spin_rgb(spins[idx]%q, q));
        }
    }
    let img = ColorImage::from_rgb([nx,ny], &rgb);

    let tex = tex_slot.get_or_insert_with(|| {
        ui.ctx().load_texture("slice", img.clone(), TextureOptions::NEAREST)
    });
    tex.set(img, TextureOptions::NEAREST);

    let side = ui.available_width().min(ui.available_height());
    ui.image((tex.id(), Vec2::splat(side)));
}

fn spin_rgb(s:u8, q:u8)->[u8;3]{
    let h = 360.0 * s as f32 / q.max(1) as f32;
    let rgb:Srgb = Srgb::from_color(Hsl::new(h,0.7,0.5));
    [(rgb.red*255.)as u8,(rgb.green*255.)as u8,(rgb.blue*255.)as u8]
}
