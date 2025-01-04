use std::fs::File;
use std::io::{self, Write};

pub fn export_to_csv(energy_history: &Vec<(f64, f64)>, energy_squared_history: &Vec<(f64, f64)>, file_name: &str) -> io::Result<()> {
    let mut file = File::create(file_name)?;
    writeln!(file, "Temperature,Energy,HeatCapacity")?;
    
    for ((temp, energy), (_, energy_squared)) in energy_history.iter().zip(energy_squared_history) {
        let heat_capacity = (energy_squared - energy.powi(2)) / temp.powi(2);
        writeln!(file, "{},{},{}", temp, energy, heat_capacity)?;
    }

    Ok(())
}
