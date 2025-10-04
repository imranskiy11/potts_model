use crate::utils::Couplings;
use crate::lattice::{AnyLattice,Lattice};

#[cfg(feature = "cuda")]
pub fn run_monte_carlo_step_on_gpu(
    lat: &AnyLattice,
    spins: &mut [u8],
    cpl: &Couplings,
    n_sweeps: usize,
) -> cust::error::CudaResult<()> {
    use cust::{prelude::*, memory::CopyDestination};

    static PTX: &str = include_str!("../../kernels/bcc_update.ptx");


    let ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;

    // constant memory для J
    let js_dev = module.get_global::<f64>("J")?;
    js_dev.copy_from(&cpl.js)?;

    // копируем решётку
    let mut spins_dev = DeviceBuffer::<u8>::from_slice(spins)?;
    let n = spins.len() as u32;

    // kernel
    let func = module.get_function("bcc_update")?;
    let grid = (n as u32 + 255) / 256;
    unsafe {
        launch!(func<<<grid,256,0,Stream::null()>>>(
            spins_dev.as_device_ptr(),
            n,
            cpl.js.len() as u32,
            cpl.beta,
            n_sweeps as u32
        ))?;
    }
    spins_dev.copy_to(spins)?;
    drop(ctx); // explicite
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub use super::monte_carlo::run_monte_carlo_step_on_gpu;
