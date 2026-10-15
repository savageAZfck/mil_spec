//! emit_avw <r> <k> <mil_path> <out_path> — write the emitted artifact.
use mil_infer::hwx::emit_avw;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r: u32 = a[1].parse().unwrap();
    let k: u32 = a[2].parse().unwrap();
    let img = emit_avw(r, k, &a[3], &a[4]);
    std::fs::write(&a[4], &img).unwrap();
    println!("emitted {} bytes r={} k={}", img.len(), r, k);
}
