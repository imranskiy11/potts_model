use rand::Rng;

pub struct Lattice {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub q: u8,
    pub states: Vec<u8>,
}

impl Lattice {
    pub fn new(nx: usize, ny: usize, nz: usize, q: u8) -> Self {
        let mut rng = rand::thread_rng();
        let size = nx * ny * nz;
        let states = (0..size).map(|_| rng.gen_range(0..q)).collect();

        Self { nx, ny, nz, q, states }
    }

    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.nx * (y + self.ny * z)
    }

    pub fn get_state(&self, x: usize, y: usize, z: usize) -> u8 {
        let idx = self.index(x, y, z);
        self.states[idx]
    }

    pub fn set_state(&mut self, x: usize, y: usize, z: usize, state: u8) {
        let idx = self.index(x, y, z);
        self.states[idx] = state;
    }

    // Новый метод delta_energy
    pub fn delta_energy(&self, x: usize, y: usize, z: usize, new_state: u8) -> i32 {
        let current_state = self.get_state(x, y, z);
        if current_state == new_state {
            return 0;
        }

        let neighbors = [
            (-1, 0, 0), (1, 0, 0), // Соседи по X
            (0, -1, 0), (0, 1, 0), // Соседи по Y
            (0, 0, -1), (0, 0, 1), // Соседи по Z
        ];

        let mut delta_energy = 0;
        for (dx, dy, dz) in neighbors.iter() {
            let nx = (x as isize + dx) as usize;
            let ny = (y as isize + dy) as usize;
            let nz = (z as isize + dz) as usize;

            if nx < self.nx && ny < self.ny && nz < self.nz {
                let neighbor_state = self.get_state(nx, ny, nz);
                if current_state == neighbor_state {
                    delta_energy += 1;
                }
                if new_state == neighbor_state {
                    delta_energy -= 1;
                }
            }
        }
        delta_energy
    }

    pub fn get_slice(&self, z: usize) -> Vec<Vec<u8>> {
        let mut slice = vec![vec![0; self.nx]; self.ny];
        for y in 0..self.ny {
            for x in 0..self.nx {
                slice[y][x] = self.get_state(x, y, z);
            }
        }
        slice
    }
}
