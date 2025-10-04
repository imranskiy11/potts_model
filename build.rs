use std::{env, fs, path::Path, process::Command};

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let src = "kernels/bcc_update.cu";
    let out = Path::new("kernels").join("bcc_update.ptx");

    let ok = Command::new("nvcc")
        .args(["-ptx", src, "-o", out.to_str().unwrap(),
               "-allow-unsupported-compiler"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !ok {
        if !out.exists() {
            println!("cargo:warning=nvcc/cl.exe не найдены, создаю заглушку PTX (GPU будет недоступен)");
            fs::write(&out, "// dummy ptx\n").expect("write dummy ptx");
        } else {
            println!("cargo:warning=nvcc не найден, использую существующий PTX");
        }
    } else {
        println!("cargo:rerun-if-changed={src}");
    }

    let _ = fs::copy(&out, Path::new(&env::var("OUT_DIR").unwrap()).join("bcc_update.ptx"));
}
