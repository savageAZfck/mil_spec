use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn main() {
    for p in std::env::args().skip(1) {
        match EspressoProgram::load_precompiled(Path::new(&p), &[]) {
            Ok(m) => println!(
                "OK   {p}: in={:?} out={:?}",
                m.input_descs.iter().map(|d| &d.name).collect::<Vec<_>>(),
                m.output_descs.iter().map(|d| &d.name).collect::<Vec<_>>()
            ),
            Err(e) => println!("FAIL {p}: {e}"),
        }
    }
}
