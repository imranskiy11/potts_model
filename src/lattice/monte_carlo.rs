use rand::Rng;
use crate::lattice::Lattice;

impl Lattice {
    /// Один шаг (обход всех спинов) методом Метрополиса на CPU
    pub fn monte_carlo_step(&mut self, temperature: f64) {
        let mut rng = rand::thread_rng();
        for _ in 0..self.states.len() {
            let x = rng.gen_range(0..self.nx);
            let y = rng.gen_range(0..self.ny);
            let z = rng.gen_range(0..self.nz);
            self.metropolis_step(x, y, z, temperature);
        }
    }

    /// Простейший вариант (обновление одного спина)
    pub fn metropolis_step(&mut self, x: usize, y: usize, z: usize, temperature: f64) {
        let mut rng = rand::thread_rng();
        let old_state = self.get_state(x, y, z);
        let new_state = rng.gen_range(0..self.q);
        if new_state == old_state {
            return;
        }
        let delta_e = self.delta_energy(x, y, z, new_state);
        if delta_e <= 0 {
            self.set_state(x, y, z, new_state);
        } else {
            let r: f64 = rng.gen();
            if r < (-delta_e as f64 / temperature).exp() {
                self.set_state(x, y, z, new_state);
            }
        }
    }
}