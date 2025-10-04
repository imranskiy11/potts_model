use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub const DEFAULT_Q: u8 = 3;


#[derive(Clone)]
pub struct Couplings {
    pub js:   Vec<f64>,
    pub beta: f64,
}

impl Couplings {
    pub fn n_shells(&self) -> usize { self.js.len() }
}


pub fn random_spins(q: u8, n_sites: usize) -> Vec<u8> {
    let mut rng = ChaCha8Rng::from_entropy();
    (0..n_sites).map(|_| rng.gen_range(0..q)).collect()
}
