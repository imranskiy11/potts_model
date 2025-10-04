use serde::Serialize;

#[derive(Default)]
pub struct Accum {
    pub n: usize,
    pub sum_e:  f64,
    pub sum_e2: f64,
    pub sum_m:  f64,
    pub sum_m2: f64,
}

impl Accum {
    pub fn add(&mut self, e: f64, m: f64) {
        self.n      += 1;
        self.sum_e  += e;
        self.sum_e2 += e * e;
        self.sum_m  += m;
        self.sum_m2 += m * m;
    }
    pub fn clear(&mut self) { *self = Self::default(); }

    pub fn finish(&self, beta: f64, n_sites: usize) -> StatRow {
        let n  = self.n as f64;
        let e  = self.sum_e  / n;
        let m  = self.sum_m  / n;
        let e2 = self.sum_e2 / n;
        let m2 = self.sum_m2 / n;
        let t  = 1.0 / beta;
        let cv  = (e2 - e*e) / (t*t * n_sites as f64);
        let chi = (m2 - m*m) / (t   * n_sites as f64);
        StatRow{beta,t,e,m,cv,chi}
    }
}

#[derive(Serialize, Clone)]
pub struct StatRow {
    pub beta: f64,
    pub t:    f64,
    pub e:    f64,
    pub m:    f64,
    pub cv:   f64,
    pub chi:  f64,
}
