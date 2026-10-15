use mil_infer::anec::Anec;
use std::path::Path;
fn main() {
    let a = Anec::open().unwrap();
    let mil = std::env::args().nth(1).unwrap();
    let out = std::env::args().nth(2).unwrap();
    let arch = std::env::args().nth(3).unwrap_or("h16".into());
    let _ = std::fs::remove_dir_all(&out);
    match a.compile(Path::new(&mil), Path::new(&out), &arch) {
        Ok(v) => println!(
            "rc={} status={} dram={:?} artifact={:?} errs={:?}",
            v.rc, v.status, v.dram_usage, v.artifact, v.errors
        ),
        Err(e) => println!("compile err: {e}"),
    }
}
