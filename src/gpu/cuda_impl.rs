use std::ffi::CString;
use cust::error::CudaResult;
use cust::prelude::*;

use crate::{lattice::AnyLattice, utils::Couplings};

pub fn run_monte_carlo_step_gpu(
    lat:    &AnyLattice,
    spins:  &mut [u8],
    cpl:    &Couplings,
    sweeps: usize,
) -> CudaResult<()> {
    static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/bcc_update.ptx"));

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;

    if let Some(&j0) = cpl.js.first() {
        let cname = CString::new("J").unwrap();
        let mut symbol = module.get_global::<f64>(&cname)?;
        symbol.copy_from(&j0)?;
    }

    let mut d_spins = DeviceBuffer::from_slice(spins)?;
    let n_sites = spins.len() as u32;

    let func = module.get_function("bcc_update")?;
    let grid = (n_sites + 255) / 256;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    unsafe {
        launch!(func<<<grid, 256, 0, stream>>>(
            d_spins.as_device_ptr(),
            n_sites,
            cpl.js.len() as u32,
            cpl.beta,
            sweeps as u32
        ))?;
    }
    stream.synchronize()?;
    d_spins.copy_to(spins)?;
    Ok(())
}
