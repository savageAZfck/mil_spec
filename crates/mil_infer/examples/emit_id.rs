fn main() {
    // identity conv 64->64 in fp16 bits
    let w: Vec<u16> = (0..64 * 64)
        .map(|i| if i / 64 == i % 64 { 0x3c00 } else { 0 })
        .collect();
    let img = mil_infer::hwx::emit_conv1x1_ns(64, 64, &w, "/tmp/id.mil", "/tmp/id.macho").unwrap();
    std::fs::write("/tmp/emit_id.hwx", &img).unwrap();
    println!("emitted {} bytes", img.len());
}
