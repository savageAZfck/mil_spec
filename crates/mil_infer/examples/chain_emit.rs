//! chain_emit — verify emit_conv_chain64 byte-identity vs ANEC corpus.
//!
//!   chain_emit verify   — emit k-deep chains and diff vs fixtures

use std::fs;

fn weights_of(mil_dir_weights: &str, k: usize) -> Vec<u16> {
    // weight.bin: 64B record header at 0; then per task: record header
    // (64B) + 0x2000B fp16 payload. blob0 at off 64 (single hdr shared),
    // task i payload at 128 + i*0x2040.
    let raw = fs::read(mil_dir_weights).unwrap();
    let mut w = Vec::new();
    for i in 0..k {
        let base = 128 + i * 0x2040;
        for j in 0..0x1000 {
            w.push(u16::from_le_bytes([raw[base + 2 * j], raw[base + 2 * j + 1]]));
        }
    }
    w
}

fn main() {
    std::env::set_var("HWX_SIG_ANEC", "1");
    let mut cases: Vec<(usize, String, String, String)> = vec![
        (2, "crates/mil_infer/assets/chain/c64k2.hwx".into(), "/tmp/chain/c64k2/weights/weight.bin".into(), "y2".into()),
        (3, "crates/mil_infer/assets/chain/c64k3.hwx".into(), "/tmp/chain/c64k3/weights/weight.bin".into(), "y3".into()),
        (4, "crates/mil_infer/assets/chain/c64k4.hwx".into(), "/tmp/chain/c64k4/weights/weight.bin".into(), "y4".into()),
    ];
    for k in [5usize, 6, 8] {
        let h = format!("/tmp/chain568/c64k{k}/out/net.macho");
        let w = format!("/tmp/chain568/c64k{k}/weights/weight.bin");
        if std::path::Path::new(&h).exists() {
            cases.push((k, h, w, format!("y{k}")));
        }
    }
    let mut all_ok = true;
    for (k, refh, wpath, yname) in &cases {
        let want = match fs::read(refh.as_str()) {
            Ok(w) => w,
            Err(e) => {
                println!("k{k}: no fixture {refh}: {e}");
                continue;
            }
        };
        let wpath2 = if std::path::Path::new(wpath).exists() {
            wpath.clone()
        } else {
            // derive from committed corpus: weights live in the mil dir
            let alt = refh.replace(".hwx", "/weights/weight.bin");
            alt
        };
        let w = match std::panic::catch_unwind(|| weights_of(&wpath2, *k)) {
            Ok(w) => w,
            Err(_) => {
                println!("k{k}: no weights at {wpath2}");
                all_ok = false;
                continue;
            }
        };
        // paths embedded in signature must match the fixture's
        let (mil, outp) = if *k <= 4 {
            (
                format!("/tmp/chain/c64k{k}/net.mil"),
                format!("/tmp/chain/c64k{k}/out/net.macho"),
            )
        } else {
            (
                format!("/tmp/chain568/c64k{k}/net.mil"),
                format!("/tmp/chain568/c64k{k}/out/net.macho"),
            )
        };
        let img =
            mil_infer::hwx::emit_conv_chain64(*k, &w, yname, &mil, &outp).unwrap();
        if img == want {
            println!("k{k}: BYTE-IDENTICAL ({} bytes)", img.len());
            continue;
        }
        all_ok = false;
        let n = img.len().min(want.len());
        let mut shown = 0;
        for i in (0..n).step_by(4) {
            let a = u32::from_le_bytes(img[i..i + 4].try_into().unwrap());
            let b = u32::from_le_bytes(want[i..i + 4].try_into().unwrap());
            if a != b {
                if shown < 30 {
                    println!("k{k} +{i:#x}: got {a:08x} want {b:08x}");
                }
                shown += 1;
            }
        }
        println!(
            "k{k}: {} u32 diffs, len {} vs {}",
            shown,
            img.len(),
            want.len()
        );
    }
    std::process::exit(if all_ok { 0 } else { 1 });
}
