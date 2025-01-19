use std::fs::File;
use std::io::{self, Write};

pub fn export_to_csv(
    energy_history: &Vec<(f64,f64)>,
    energy_squared_history: &Vec<(f64,f64)>,
    file_name: &str
) -> io::Result<()> {
    let mut file = File::create(file_name)?;
    writeln!(file, "Temperature,Energy,HeatCapacity")?;
    for ((t,e), &(_, e2)) in energy_history.iter().zip(energy_squared_history) {
        let cv = (e2 - e.powi(2)) / t.powi(2);
        writeln!(file, "{},{},{}", t,e,cv)?;
    }
    Ok(())
}