use glam::IVec3;


pub trait Lattice {
    fn dims(&self)                    -> IVec3;
    fn n_sites(&self)                 -> usize;
    fn idx(&self, p: IVec3)           -> usize;   // (x,y,z) → i ( c PBC )
    fn xyz(&self, idx: usize)         -> IVec3;   // i → (x,y,z)
    fn neighbours(&self, idx: usize, shell: usize) -> &[usize];
    fn n_shells(&self)                -> usize;
}

pub mod bcc;
pub use bcc::Bcc;


pub enum AnyLattice { Bcc(Bcc) }

macro_rules! fwd {
    ($self:ident.$field:ident() $($args:expr),*) => {
        match $self { AnyLattice::Bcc(b) => b.$field($($args),*) }
    };
}

impl Lattice for AnyLattice {
    fn dims         (&self)                -> IVec3      { fwd!(self.dims())                }
    fn n_sites      (&self)                -> usize      { fwd!(self.n_sites())             }
    fn idx          (&self, p:IVec3)       -> usize      { fwd!(self.idx()       p)         }
    fn xyz          (&self, i:usize)       -> IVec3      { fwd!(self.xyz()       i)         }
    fn neighbours   (&self,i:usize,s:usize)-> &[usize]   { fwd!(self.neighbours() i,s)      }
    fn n_shells     (&self)                -> usize      { fwd!(self.n_shells())            }
}
