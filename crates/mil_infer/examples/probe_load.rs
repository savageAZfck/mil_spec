fn main() {
    let p = std::env::args().nth(1).expect("usage: probe_load <file.macho> [more...]");
    for f in std::iter::once(p.clone()).chain(std::env::args().skip(2)) {
        match mil_infer::espresso::EspressoProgram::load_precompiled(
            std::path::Path::new(&f),
            &[],
        ) {
            Ok(_) => println!("{f}: LOADED"),
            Err(e) => println!("{f}: FAIL {e}"),
        }
    }
}
