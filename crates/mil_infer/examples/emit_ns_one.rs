fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let cin: usize = a[0].parse().unwrap();
    let cout: usize = a[1].parse().unwrap();
    let w: Vec<u16> = (0..cin * cout).map(|i| (i % 65536) as u16).collect();
    let img = mil_infer::hwx::emit_conv1x1_ns(cin, cout, &w, "/tmp/x.mil", "/tmp/x.macho").unwrap();
    std::fs::write(&a[2], img).unwrap();
}
