use crate::lattice::Lattice;

impl Lattice {
    pub fn calculate_energy(&self, x:usize, y:usize,z:usize)-> i32 {
        let st= self.get_state(x,y,z);
        let neigh= [
            (1,0,0),(-1,0,0),
            (0,1,0),(0,-1,0),
            (0,0,1),(0,0,-1),
        ];
        let mut e=0;
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
                if stn== st { e-=1; } else { e+=1;}
            }
        }
        e
    }
}