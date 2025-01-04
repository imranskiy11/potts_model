use rand::Rng;
use crate::lattice::Lattice;

pub struct WangLandau {
    // Fields for WangLandau, if any.
}

impl Lattice {
    pub fn wang_landau_step(
        &mut self,
        omega: &mut Vec<f64>,
        histogram: &mut Vec<u64>,
        f: f64,
        min_energy: i32,
    ) -> i32 {
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(0..self.nx);
        let y = rng.gen_range(0..self.ny);
        let z = rng.gen_range(0..self.nz);

        let current_state = self.get_state(x, y, z);
        let current_energy = self.calculate_energy(x, y, z);

        let new_state = rng.gen_range(0..self.q);
        if new_state == current_state {
            return current_energy;
        }

        let new_energy = self.delta_energy(x, y, z, new_state);
        let current_index = (current_energy - min_energy) as usize;
        let new_index = (new_energy - min_energy) as usize;

        if rng.gen::<f64>() < (omega[current_index] / omega[new_index]).exp() {
            self.set_state(x, y, z, new_state);
            histogram[new_index] += 1;
            omega[new_index] *= f;
            new_energy
        } else {
            histogram[current_index] += 1;
            omega[current_index] *= f;
            current_energy
        }
    }
}
