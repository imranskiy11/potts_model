use crate::gpu::gpu::run_on_gpu;
use cudarc::driver::{CudaDevice, CudaSlice};

pub fn run_wang_landau_step_on_gpu(
    nx: usize,
    ny: usize,
    nz: usize,
    q: u8,
    states: &mut [u8],
    f: f64,
    min_energy: i32,
) {
    unsafe fn wang_landau_kernel(
        nx: usize,
        ny: usize,
        nz: usize,
        q: u8,
        states: &mut [u8],
        params: (f64, i32),
    ) {
        let (f, min_energy) = params;
        let i = thread_idx_x();
        let j = thread_idx_y();
        let k = thread_idx_z();

        if i < nx && j < ny && k < nz {
            let idx = i + nx * (j + ny * k);
            let current_state = states[idx];
            let new_state = (rand::random::<u8>() % q) as u8;
            let delta_e = 0.0; 

            if rand::random::<f64>() < (f / f).exp() {
                states[idx] = new_state;
            }
        }
    }

    run_on_gpu(nx, ny, nz, q, states, (f, min_energy), wang_landau_kernel);
}
