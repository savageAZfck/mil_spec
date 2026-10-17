use std::fs;
fn main() {
    let mut files = Vec::new();
    for dir in [
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets/chain"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets/ioperm"),
    ] {
        for e in fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().map(|x| x == "mil").unwrap_or(false) {
                files.push(p);
            }
        }
    }
    files.sort();
    for f in &files {
        let out = "/tmp/route_sweep.macho";
        match mil_infer::hwx::emit_from_mil(f.to_str().unwrap(), out) {
            Ok(img) => println!(
                "OK   {} -> {} bytes",
                f.file_name().unwrap().to_str().unwrap(),
                img.len()
            ),
            Err(e) => println!(
                "ERR  {} -> {}",
                f.file_name().unwrap().to_str().unwrap(),
                &e[..e.len().min(90)]
            ),
        }
    }
}
