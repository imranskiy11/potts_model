use rand::Rng;
use crate::lattice::Lattice;

pub struct WangLandau {
    // ...
}

impl Lattice {
    pub fn wang_landau_step(
        &mut self,
        omega: &mut Vec<f64>,
        histogram: &mut Vec<u64>,
        f: f64,
        min_energy: i32,
    )-> i32 {
        let mut rng= rand::thread_rng();
        let x= rng.gen_range(0..self.nx);
        let y= rng.gen_range(0..self.ny);
        let z= rng.gen_range(0..self.nz);

        let old_st= self.get_state(x,y,z);
        let old_e= self.calculate_energy(x,y,z);

        let new_st= rng.gen_range(0.. self.q);
        if new_st== old_st {
            return old_e;
        }
        let new_e= self.calculate_energy(x,y,z);

        let i_old= (old_e- min_energy) as usize;
        let i_new= (new_e- min_energy) as usize;

        if rng.gen::<f64>() < (omega[i_old]/ omega[i_new]).exp() {
            self.set_state(x,y,z,new_st);
            histogram[i_new]+=1;
            omega[i_new]*= f;
            new_e
        } else {
            histogram[i_old]+=1;
            omega[i_old]*= f;
            old_e
        }
    }
}