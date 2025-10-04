use rand::{thread_rng, Rng};
use rand_distr::Uniform;

/*── параметры ──*/
#[derive(Clone)]
pub struct WLParams {
    pub e_min : f64,
    pub e_max : f64,
    pub bin   : f64,
    pub ln_f0 : f64,
    pub flat  : f64,
}
impl Default for WLParams {
    fn default() -> Self {
        Self { e_min:-4.0, e_max:0.0, bin:0.1, ln_f0:1.0, flat:0.8 }
    }
}


pub struct WLState {
    pub ln_g : Vec<f64>,
    pub hist : Vec<u32>,
    pub ln_f : f64,
}
impl WLState {
    pub fn new(p:&WLParams) -> Self {
        let bins = ((p.e_max - p.e_min)/p.bin).ceil() as usize + 1;
        Self { ln_g: vec![0.0; bins],
               hist: vec![0;   bins],
               ln_f: p.ln_f0 }
    }

    #[inline]
    fn idx(p:&WLParams, e:f64) -> Option<usize> {
        if e < p.e_min || e > p.e_max { None }
        else { Some( ((e - p.e_min)/p.bin).floor() as usize ) }
    }


    pub fn step(&mut self, p:&WLParams, e:f64, e_new:f64) -> bool {
        let (Some(i_old), Some(i_new)) =
            (Self::idx(p, e), Self::idx(p, e_new)) else { return false };

        let acc_prob = (self.ln_g[i_old] - self.ln_g[i_new]).exp();
        let u: f64 = thread_rng().sample(Uniform::new(0.0, 1.0));
        let accept = u < acc_prob;

        let k = if accept { i_new } else { i_old };
        self.ln_g[k] += self.ln_f;
        self.hist[k] += 1;
        accept
    }

    pub fn is_flat(&self, p:&WLParams) -> bool {
        let h_min = *self.hist.iter().min().unwrap() as f64;
        let h_avg = self.hist.iter().sum::<u32>() as f64 / self.hist.len() as f64;
        h_min > p.flat * h_avg
    }


    pub fn refine(&mut self) {
        self.ln_f *= 0.5;
        self.hist.fill(0);
    }
}
