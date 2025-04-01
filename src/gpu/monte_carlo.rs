use cust::context::Context;
use cust::memory::{DeviceBuffer, CopyDestination};
use cust::stream::{Stream, StreamFlags};
use cust::module::Module;
use cust::launch;
use rand::Rng;
use std::error::Error;

/// Запуск random-site update на GPU: n_sweeps циклов, много потоков, inner_loops=8 в ядре
pub fn run_monte_carlo_step_on_gpu(
    nx: usize,
    ny: usize,
    nz: usize,
    q: u8,
    states: &mut [u8],
    temperature: f64,
    n_sweeps: usize,
) -> Result<(), Box<dyn Error>> {
    let device = cust::device::Device::get_device(0)?;
    let _context = Context::new(device)?; // контекст

    let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    let total_size = states.len();
    let d_states = DeviceBuffer::from_slice(states)?;

    // numThreads ~ min(1e6, total_size)
    let threads_per_block=1024u32;
    let desired = 2_000_000.min(total_size);
    let max_threads= (65535u64*1024u64) as usize;
    let mut num_threads= desired.min(max_threads).max(1);
    let num_threads_u32= num_threads as u32;
    let blocks= (num_threads_u32+ threads_per_block-1)/ threads_per_block;

    let rand_ids_len= n_sweeps* num_threads;
    let randoms_len= rand_ids_len*2;

    // rand_ids, randoms
    let mut rng= rand::thread_rng();
    let mut rand_ids_host= Vec::with_capacity(rand_ids_len);
    for _ in 0.. rand_ids_len {
        rand_ids_host.push(rng.gen_range(0..(total_size as u32)));
    }
    let mut randoms_host= Vec::with_capacity(randoms_len);
    for _ in 0.. randoms_len {
        randoms_host.push(rng.gen_range(0.0f32..1.0f32));
    }

    let d_rand_ids= DeviceBuffer::from_slice(&rand_ids_host)?;
    let d_randoms= DeviceBuffer::from_slice(&randoms_host)?;

    let ptx= include_str!("../../kernels/monte_carlo.ptx");
    let module= Module::from_ptx(ptx, &[])?;
    let function= module.get_function("metropolis_kernel")?;

    unsafe {
        launch!(
            function<<<(blocks,1,1),(threads_per_block,1,1),0,stream>>>(
                d_states.as_device_ptr(),
                d_rand_ids.as_device_ptr(),
                d_randoms.as_device_ptr(),
                nx as u32,
                ny as u32,
                nz as u32,
                q as u32,
                temperature as f32,
                total_size as u32,
                n_sweeps as u32
            )
        )?;
    }
    stream.synchronize()?;
    
    d_states.copy_to(states)?;

    Ok(())
}