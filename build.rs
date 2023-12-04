extern crate cbindgen;

use cbindgen::{Builder, Config};
use std::env;
use std::path::Path;

fn main() {
    let crate_env = env::var("CARGO_MANIFEST_DIR").unwrap();
    let crate_path = Path::new(&crate_env);
    let config = Config::from_root_or_default(crate_path);
    Builder::new()
        .with_crate(crate_path.to_str().unwrap())
        .with_config(config)
        .generate()
        .expect("Cannot generate header file!")
        .write_to_file("testprogram/headers/gpu.h");
}
