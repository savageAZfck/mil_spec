use mil_infer::e5rt::Program;
use std::path::Path;

fn main() {
    for p in std::env::args().skip(1) {
        match Program::from_e5_bundle(Path::new(&p)) {
            Ok((prog, _)) => { drop(prog); println!("OK   {p}"); }
            Err(e) => println!("FAIL {p}: {e}"),
        }
    }
}
