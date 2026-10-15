//! verify_dyn — emit `conv1x1` HWX images via the parametric emitter
//! for every swept channel count under the given roots and byte-compare
//! against the captured ANEC artifacts. Requires HWX_SIG_ANEC=1 so the
//! signature block matches the reference compiler's format.
//!
//!   HWX_SIG_ANEC=1 cargo run -p mil_infer --example verify_dyn -- /tmp/cpar /tmp/cpar2 /tmp/cpar3

use std::fs;

fn main() {
    let mut total = 0;
    let mut pass = 0;
    let mut fail = Vec::new();
    for root in std::env::args().skip(1) {
        let mut cs: Vec<usize> = fs::read_dir(&root)
            .unwrap()
            .filter_map(|e| {
                let n = e.unwrap().file_name().to_string_lossy().to_string();
                n.strip_prefix('c').and_then(|s| s.parse().ok())
            })
            .collect();
        cs.sort_unstable();
        for c in cs {
            let mil = format!("{root}/c{c}/net.mil");
            let out = format!("{root}/c{c}/out/net.macho");
            let Ok(want) = fs::read(&out) else { continue };
            // prefer the probe's own weight.bin (BLOBFILE data at 0x80);
            // fall back to the integer ramp used by conv_probe
            let wpath = format!("{root}/c{c}/weights/weight.bin");
            let w: Vec<u16> = match fs::read(&wpath) {
                Ok(b) if b.len() >= 0x80 + c * c * 2 => b[0x80..0x80 + c * c * 2]
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect(),
                // conv_extract probes were compiled with zero weights
                _ if std::env::var_os("VERIFY_ZERO_W").is_some() => {
                    vec![0u16; c * c]
                }
                _ => (0..c * c).map(|i| i as u16).collect(),
            };
            total += 1;
            match mil_infer::hwx::emit_conv1x1_dyn(c, &w, &mil, &out) {
                Ok(img) if img == want => {
                    pass += 1;
                    println!("c{c}: OK ({} bytes)", img.len());
                }
                Ok(img) => {
                    let d = img
                        .iter()
                        .zip(&want)
                        .position(|(a, b)| a != b)
                        .map(|i| i as i64)
                        .unwrap_or(-1);
                    fail.push(format!(
                        "c{c}: first diff {d:#x} len {:#x}/{:#x}",
                        img.len(),
                        want.len()
                    ));
                }
                Err(e) => fail.push(format!("c{c}: {e}")),
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
