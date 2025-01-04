use crate::lattice::Lattice;

impl Lattice {
    pub fn calculate_energy(&self, x: usize, y: usize, z: usize) -> i32 {
        let mut energy = 0;
        let current_state = self.get_state(x, y, z);

        let neighbors = [
            (-1, 0, 0), (1, 0, 0), // Соседи по X
            (0, -1, 0), (0, 1, 0), // Соседи по Y
            (0, 0, -1), (0, 0, 1), // Соседи по Z
        ];

        for (dx, dy, dz) in neighbors.iter() {
            let nx = (x as isize + dx) as usize;
            let ny = (y as isize + dy) as usize;
            let nz = (z as isize + dz) as usize;

            if nx < self.nx && ny < self.ny && nz < self.nz {
                let neighbor_state = self.get_state(nx, ny, nz);
                if current_state == neighbor_state {
                    energy -= 1; // Совпадение снижает энергию
                } else {
                    energy += 1; // Несовпадение повышает энергию
                }
            }
        }
        energy
    }
}
