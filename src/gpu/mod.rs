use crate::{lattice::{AnyLattice,Lattice}, utils::Couplings};

#[cfg(feature = "cuda")]
pub mod cuda_impl;
#[cfg(feature = "cuda")]
use cuda_impl::run_monte_carlo_step_gpu;

//CPU

pub fn run_monte_carlo_step_cpu(
    lat:&AnyLattice,
    spins:&mut [u8],
    cpl:&Couplings,
    n_sweeps:usize)
{
    use rand::{Rng,SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let n = spins.len();
    if n==0 { return; }
    let q = 1+(*spins.iter().max().unwrap()) as u8;
    let mut rng = ChaCha8Rng::from_entropy();

    for _ in 0..n_sweeps {
        for parity in [0,1] {
            for idx in 0..n {
                let xyz = lat.xyz(idx);
                if ((xyz.x+xyz.y+xyz.z)&1)!=parity { continue; }

                let old = spins[idx];
                let mut new = old;
                while new==old { new=rng.gen_range(0..q); }

                let mut dE = 0.0;
                for (sh,&j) in cpl.js.iter().enumerate() {
                    for &nb in lat.neighbours(idx,sh) {
                        let s = spins[nb];
                        if s==old { dE+=j; }
                        if s==new { dE-=j; }
                    }
                }
                if dE<=0.0 || rng.gen::<f64>()<(-cpl.beta*dE).exp() {
                    spins[idx]=new;
                }
            }
        }
    }
}


/// true => GPU, false => CPU.
pub fn run_monte_carlo_step(
    use_gpu:bool,
    lat:&AnyLattice,
    spins:&mut [u8],
    cpl:&Couplings,
    sweeps:usize)
{
    if use_gpu {
        #[cfg(feature="cuda")]
        {
            if run_monte_carlo_step_gpu(lat,spins,cpl,sweeps).is_ok() { return; }
            eprintln!("GPU-kernel failed — fallback to CPU");
        }
    }
    run_monte_carlo_step_cpu(lat,spins,cpl,sweeps);
}
