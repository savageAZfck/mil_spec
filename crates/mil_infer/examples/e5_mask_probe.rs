//! e5_mask_probe — compile one mil under each device mask to isolate
//! whether e5_compiler_compile failure is ANE-specific or universal.
//!
//!   cargo run -p mil_infer --example e5_mask_probe -- <model.mil>

use mil_infer::e5rt::{CompileCfg, DeviceMask, Program};
use std::path::PathBuf;

fn main() {
    let mil = PathBuf::from(std::env::args().nth(1).expect("usage: e5_mask_probe <model.mil>"));
    for (name, mask) in [
        ("cpu", DeviceMask::Cpu),
        ("gpu", DeviceMask::Gpu),
        ("ane", DeviceMask::Ane),
        ("auto", DeviceMask::Auto),
    ] {
        let cc = CompileCfg {
            cache_dir: std::env::temp_dir().join(format!("e5_mask_{name}")),
            ..CompileCfg::default()
        };
        match Program::compile_auto_cfg(&mil, mask, &cc) {
            Ok(_) => println!("{name}: OK"),
            Err(e) => println!("{name}: {e}"),
        }
    }
}
