use rand::Rng;
use crate::lattice::Lattice;

pub struct MonteCarlo {

}

impl Lattice {
    pub fn metropolis_step(&mut self, x: usize, y: usize, z: usize, temperature: f64) {
        let mut rng = rand::thread_rng();
        let new_state = rng.gen_range(0..self.q);
        let delta_e = self.delta_energy(x, y, z, new_state);

        if delta_e <= 0 || rng.gen::<f64>() < (-delta_e as f64 / temperature).exp() {
            self.set_state(x, y, z, new_state);
        }
    }

    pub fn monte_carlo_step(&mut self, temperature: f64) {
        let mut rng = rand::thread_rng();
        for _ in 0..self.states.len() {
            let x = rng.gen_range(0..self.nx);
            let y = rng.gen_range(0..self.ny);
            let z = rng.gen_range(0..self.nz);
            self.metropolis_step(x, y, z, temperature);
        }
    }
}
