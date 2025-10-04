use glam::{IVec3, ivec3};
use super::Lattice;

thread_local! {
    static BUF: std::cell::RefCell<Vec<usize>> = Default::default();
}

pub struct Bcc {
    l: IVec3,                 // размеры
    shells: Vec<Vec<IVec3>>,  // смещения каждой оболочки
}

impl Bcc {
    pub fn new(l: IVec3, want_shells: usize) -> Self {
        let mut this = Self { l, shells: Vec::new() };
        this.ensure_shells(want_shells);
        this
    }

    pub fn ensure_shells(&mut self, want: usize) {
        while self.shells.len() < want {
            let n = self.shells.len() + 1;
            self.shells.push(gen_shell(n));
        }
    }

    #[inline] fn pbc(&self, p: IVec3) -> IVec3 {
        ivec3(
            (p.x + self.l.x) % self.l.x,
            (p.y + self.l.y) % self.l.y,
            (p.z + self.l.z) % self.l.z,
        )
    }
}

impl Lattice for Bcc {
    #[inline] fn dims     (&self) -> IVec3  { self.l }
    #[inline] fn n_sites  (&self) -> usize  { (self.l.x*self.l.y*self.l.z) as usize }
    #[inline] fn n_shells (&self) -> usize  { self.shells.len() }

    #[inline] fn idx(&self, p: IVec3) -> usize {
        let p = self.pbc(p);
        (p.z * self.l.y * self.l.x + p.y * self.l.x + p.x) as usize
    }
    #[inline] fn xyz(&self, idx: usize) -> IVec3 {
        let (lx,ly) = (self.l.x, self.l.y);
        let z = idx as i32 / (lx*ly);
        let y = (idx as i32 - z*lx*ly) / lx;
        let x = idx as i32 - z*lx*ly - y*lx;
        ivec3(x,y,z)
    }

    fn neighbours(&self, idx: usize, shell: usize) -> &[usize] {
        if shell >= self.shells.len() { return &[]; }
        let p  = self.xyz(idx);

        BUF.with(|buf| {
            let mut buf = buf.borrow_mut();
            buf.clear();
            for &d in &self.shells[shell] {
                buf.push(self.idx(p + d));
            }
            unsafe { std::mem::transmute::<&[usize], &[usize]>(&buf[..]) }
        })
    }
}


fn gen_shell(n: usize) -> Vec<IVec3> {
    let mut v = Vec::new();
    match n {
        1 => {
            for &sx in &[-1,1] {
                for &sy in &[-1,1] {
                    for &sz in &[-1,1] {
                        v.push(ivec3(sx,sy,sz));
                    }
                }
            }
        }
        2 => {
            for &d in &[ivec3( 2,0,0), ivec3(-2,0,0),
                        ivec3( 0,2,0), ivec3( 0,-2,0),
                        ivec3( 0,0,2), ivec3( 0,0,-2)] {
                v.push(d);
            }
        }
        3 => {
            for &sx in &[-2,2] {
                for &sy in &[-2,2] {
                    v.push(ivec3(sx,sy,0));
                    v.push(ivec3(sx,0,sy));
                    v.push(ivec3(0,sx,sy));
                }
            }
        }
        _ => unimplemented!("shell {n} not implemented"),
    }
    v
}
