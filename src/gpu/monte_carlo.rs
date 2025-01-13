use cust::context::Context;
use cust::memory::{DeviceBuffer, CopyDestination};
use cust::stream::{Stream, StreamFlags};
use cust::module::Module;
use cust::{launch, error::CudaResult};
use rand::Rng;
use std::error::Error;

/// Запускает метрополис на GPU
pub fn run_monte_carlo_step_on_gpu(
    nx: usize,
    ny: usize,
    nz: usize,
    q: u8,
    states: &mut [u8],
    temperature: f64,
) -> Result<(), Box<dyn Error>> {
    // Создаём контекст CUDA
    let device = cust::device::Device::get_device(0)?;
    let _context = Context::new(device)?;

    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    // Копируем states на GPU
    let mut d_states = DeviceBuffer::from_slice(states)?;

    // Генерируем random_numbers (каждый шаг => новое)
    // Тут можно побольше сделать (nx*ny*nz)
    let len = states.len();
    let random_numbers: Vec<f32> = (0..len)
        .map(|_| rand::thread_rng().gen_range(0.0..1.0))
        .collect();
    let d_random = DeviceBuffer::from_slice(&random_numbers)?;

    // Загрузим PTX (упрощённый пример ядра)
    let ptx = include_str!("../../kernels/monte_carlo.ptx");
    let module = Module::from_ptx(ptx, &[])?;
    let function = module.get_function("metropolis_kernel")?;

    // Запускаем с неким BlockSize, GridSize
    // Подбираем так, чтобы покрыть nx*ny*nz
    // Допустим block = 256, grid = (states.len() + 255)/256
    let threads_per_block = 256;
    let blocks = (len + threads_per_block - 1) / threads_per_block;

    unsafe {
        launch!(
            function<<<(blocks as u32, 1, 1), (threads_per_block as u32, 1, 1), 0, stream>>>(
                d_states.as_device_ptr(),
                d_random.as_device_ptr(),
                nx as u32,
                ny as u32,
                nz as u32,
                q as u32,
                temperature as f32,
                len as u32
            )
        )?;
    }

    stream.synchronize()?;

    // Копируем результаты обратно
    d_states.copy_to(states)?;

    Ok(())
}
