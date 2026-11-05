use mil_infer::e5rt::{CompileCfg, DeviceMask, Program};
use std::path::PathBuf;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mil = PathBuf::from(&a[1]);
    let cache = PathBuf::from(&a[2]);
    let cc = CompileCfg { cache_dir: cache, force_recompile: true, profiling: Default::default() };
    match Program::compile_auto_cfg(&mil, DeviceMask::Ane, &cc) {
        Ok((prog, _)) => { drop(prog); println!("compiled ok"); }
        Err(e) => println!("compile failed: {e}"),
    }
}
