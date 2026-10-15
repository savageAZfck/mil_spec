//! verify_ns — emit non-square `conv1x1` HWX images via
//! `emit_conv1x1_ns` for every `<cin>x<cout>` probe dir under the given
//! roots and byte-compare against the captured ANEC artifacts. Needs
//! HWX_SIG_ANEC=1 so the signature block matches.
//!
//!   HWX_SIG_ANEC=1 cargo run -p mil_infer --example verify_ns -- /tmp/cns4 /tmp/cnsg

use std::fs;

fn main() {
    let mut total = 0;
    let mut pass = 0;
    let mut fail = Vec::new();
    for root in std::env::args().skip(1) {
        let mut dirs: Vec<(usize, usize)> = fs::read_dir(&root)
            .unwrap()
            .filter_map(|e| {
                let n = e.unwrap().file_name().to_string_lossy().to_string();
                let (a, b) = n.split_once('x')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .collect();
        dirs.sort_unstable();
        for (cin, cout) in dirs {
            let mil = format!("{root}/{cin}x{cout}/net.mil");
            let out = format!("{root}/{cin}x{cout}/out/net.macho");
            let Ok(want) = fs::read(&out) else { continue };
            let wpath = format!("{root}/{cin}x{cout}/weights/weight.bin");
            let w: Vec<u16> = match fs::read(&wpath) {
                Ok(b) if b.len() >= 0x80 + cin * cout * 2 => b[0x80..0x80 + cin * cout * 2]
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect(),
                _ => (0..cin * cout).map(|i| i as u16).collect(),
            };
            total += 1;
            match mil_infer::hwx::emit_conv1x1_ns(cin, cout, &w, &mil, &out) {
                Ok(img) if img == want => {
                    pass += 1;
                    println!("{cin}x{cout}: OK ({} bytes)", img.len());
                }
                Ok(img) => {
                    let d = img
                        .iter()
                        .zip(&want)
                        .position(|(a, b)| a != b)
                        .map(|i| i as i64)
                        .unwrap_or(-1);
                    fail.push(format!(
                        "{cin}x{cout}: first diff {d:#x} len {:#x}/{:#x}",
                        img.len(),
                        want.len()
                    ));
                }
                Err(e) => fail.push(format!("{cin}x{cout}: {e}")),
            }
        }
    }
    println!("--- {pass}/{total} byte-identical ---");
    for f in &fail {
        println!("  {f}");
    }
    if !fail.is_empty() {
        std::process::exit(1);
    }
}
