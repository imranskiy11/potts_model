use rand::Rng;

pub struct Lattice {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub q: u8,
    pub states: Vec<u8>,
}

impl Lattice {
    pub fn new(nx: usize, ny: usize, nz: usize, q: u8)-> Self {
        let mut rng= rand::thread_rng();
        let size= nx*ny*nz;
        let states= (0..size).map(|_| rng.gen_range(0..q)).collect();
        Self{ nx,ny,nz, q, states}
    }

    pub fn index(&self, x:usize,y:usize,z:usize)-> usize {
        x + self.nx*(y+ self.ny*z)
    }

    pub fn get_state(&self, x:usize, y:usize, z:usize)-> u8 {
        let idx= self.index(x,y,z);
        self.states[idx]
    }

    pub fn set_state(&mut self, x:usize, y:usize, z:usize, st:u8) {
        let idx= self.index(x,y,z);
        self.states[idx]= st;
    }

    pub fn get_slice(&self, z:usize)-> Vec<Vec<u8>> {
        let mut slice= vec![vec![0; self.nx]; self.ny];
        for yy in 0.. self.ny {
            for xx in 0.. self.nx {
                slice[yy][xx]= self.get_state(xx,yy,z);
            }
        }
        slice
    }
}