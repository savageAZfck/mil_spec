//! emit_w_one <cin> <cout> <w> <output> [mil_path out_sig weights_mode]
//! emit a spatial conv1x1 (x `[1,Cin,1,W]` -> z `[1,Cout,1,W]`).
//! weights_mode: "ramp" (default) or "half" (all fp16 0.5).
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let cin: usize = a[0].parse().unwrap();
    let cout: usize = a[1].parse().unwrap();
    let w: usize = a[2].parse().unwrap();
    let mil = a.get(4).map(|s| s.as_str()).unwrap_or("/tmp/x.mil");
    let out = a.get(5).map(|s| s.as_str()).unwrap_or("/tmp/x.macho");
    let mode = a.get(6).map(|s| s.as_str()).unwrap_or("ramp");
    let wt: Vec<u16> = if mode == "half" {
        vec![0x3800u16; cin * cout]
    } else {
        (0..cin * cout).map(|i| (i % 65536) as u16).collect()
    };
    let img = mil_infer::hwx::emit_conv1x1_w(cin, cout, w, &wt, mil, out).unwrap();
    std::fs::write(&a[3], img).unwrap();
}
