use rand::Rng;
use crate::lattice::Lattice;

impl Lattice {
    pub fn monte_carlo_step(&mut self, temperature:f64) {
        let mut rng= rand::thread_rng();
        for _ in 0.. self.states.len() {
            let x= rng.gen_range(0..self.nx);
            let y= rng.gen_range(0..self.ny);
            let z= rng.gen_range(0..self.nz);
            self.metropolis_step(x,y,z, temperature);
        }
    }

    pub fn metropolis_step(&mut self, x:usize,y:usize,z:usize, temperature:f64) {
        let mut rng= rand::thread_rng();
        let old_st= self.get_state(x,y,z);
        let new_st= rng.gen_range(0.. self.q);
        if new_st== old_st {
            return;
        }
        let neigh= [
            (1,0,0),(-1,0,0),
            (0,1,0),(0,-1,0),
            (0,0,1),(0,0,-1),
        ];
        let mut count_old=0;
        let mut count_new=0;
        for &(dx,dy,dz) in &neigh {
            let xx= x as isize+ dx;
            let yy= y as isize+ dy;
            let zz= z as isize+ dz;
            if xx>=0 && xx<(self.nx as isize) &&
               yy>=0 && yy<(self.ny as isize) &&
               zz>=0 && zz<(self.nz as isize) {
                let idx_n= (zz as usize)*(self.nx*self.ny)
                          + (yy as usize)* self.nx
                          + (xx as usize);
                let stn= self.states[idx_n];
                if stn== old_st { count_old+=1; }
                if stn== new_st { count_new+=1; }
            }
        }
        let dE= count_old- count_new;
        if dE<=0 {
            self.set_state(x,y,z, new_st);
        } else {
            let r:f64= rng.gen();
            if r< (-dE as f64/ temperature).exp(){
                self.set_state(x,y,z, new_st);
            }
        }
    }
}