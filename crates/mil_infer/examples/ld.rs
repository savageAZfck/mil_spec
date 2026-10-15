use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn main() {
    for p in std::env::args().skip(1) {
        match EspressoProgram::load_precompiled(Path::new(&p), &[]) {
            Ok(_) => println!("OK {p}"),
            Err(e) => println!("FAIL {p}: {e:?}"),
        }
    }
}
