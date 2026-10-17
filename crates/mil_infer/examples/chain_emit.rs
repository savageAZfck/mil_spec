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
            w.push(u16::from_le_bytes([
                raw[base + 2 * j],
                raw[base + 2 * j + 1],
            ]));
        }
    }
    w
}

fn main() {
    std::env::set_var("HWX_SIG_ANEC", "1");
    let mut cases: Vec<(usize, String, String, String)> = vec![
        (
            2,
            "crates/mil_infer/assets/chain/c64k2.hwx".into(),
            "/tmp/chain/c64k2/weights/weight.bin".into(),
            "y2".into(),
        ),
        (
            3,
            "crates/mil_infer/assets/chain/c64k3.hwx".into(),
            "/tmp/chain/c64k3/weights/weight.bin".into(),
            "y3".into(),
        ),
        (
            4,
            "crates/mil_infer/assets/chain/c64k4.hwx".into(),
            "/tmp/chain/c64k4/weights/weight.bin".into(),
            "y4".into(),
        ),
    ];
    for k in [5usize, 6, 8] {
        let h = format!("/tmp/chain568/c64k{k}/out/net.macho");
        let w = format!("/tmp/chain568/c64k{k}/weights/weight.bin");
        if std::path::Path::new(&h).exists() {
            cases.push((k, h, w, format!("y{k}")));
        }
    }
    // shape-parametric chains: walk every m<a>x<b>x.. dir under
    // /tmp/chainmix plus the c32 homogeneous corpus
    {
        let mut dirs: Vec<std::path::PathBuf> = vec![];
        for root in ["/tmp/chainmix", "/tmp/chain32"] {
            if let Ok(rd) = fs::read_dir(root) {
                for e in rd.flatten() {
                    if e.path().join("weights/weight.bin").exists()
                        && e.path().join("out/net.macho").exists()
                    {
                        dirs.push(e.path());
                    }
                }
            }
        }
        dirs.sort();
        for d in dirs {
            let name = d.file_name().unwrap().to_str().unwrap().to_string();
            // `m<a>x<b>x..` = channel list; `c<N>k<K>` = homogeneous
            let cs: Vec<usize> = if let Some(rest) = name.strip_prefix('m') {
                rest.split('x').map(|s| s.parse().unwrap()).collect()
            } else {
                let body = name.strip_prefix('c').unwrap();
                let (c, kk) = body.split_once('k').unwrap();
                vec![c.parse().unwrap(); kk.parse::<usize>().unwrap() + 1]
            };
            let k = cs.len() - 1;
            let raw = fs::read(d.join("weights/weight.bin")).unwrap();
            let shapes: Vec<(usize, usize)> = cs.windows(2).map(|w| (w[0], w[1])).collect();
            // walk the per-task record headers for data offsets
            let mut ws = Vec::new();
            let mut offs = Vec::new();
            let mut pos = 64usize;
            for &(ci, co) in &shapes {
                let pay = u64::from_le_bytes(raw[pos + 8..pos + 16].try_into().unwrap()) as usize;
                let dat = u64::from_le_bytes(raw[pos + 16..pos + 24].try_into().unwrap()) as usize;
                debug_assert_eq!(pay, ci * co * 2);
                offs.push(dat as u64);
                for j in 0..ci * co {
                    ws.push(u16::from_le_bytes([raw[dat + 2 * j], raw[dat + 2 * j + 1]]));
                }
                pos = dat + pay;
            }
            let want = fs::read(d.join("out/net.macho")).unwrap();
            // the signature embeds the mil/out paths ANEC saw; recover
            // them from the artifact since corpus batches differ.
            let (milp, outp) = {
                let s = String::from_utf8_lossy(&want);
                let grab = |flag: &str| -> String {
                    s.split_whitespace()
                        .skip_while(|t| *t != flag)
                        .nth(1)
                        .unwrap_or("")
                        .to_string()
                };
                let mi = grab("-i");
                let ou = grab("-o");
                if mi.is_empty() || ou.is_empty() {
                    (
                        d.join("net.mil").to_str().unwrap().to_string(),
                        d.join("out/net.macho").to_str().unwrap().to_string(),
                    )
                } else {
                    (mi, ou)
                }
            };
            let img = mil_infer::hwx::emit_conv_chain(
                &shapes,
                &ws,
                &offs,
                &format!("y{k}"),
                &milp,
                &outp,
            )
            .unwrap();
            if img == want {
                println!("{name}: BYTE-IDENTICAL ({} bytes)", img.len());
            } else {
                let n = img.len().min(want.len());
                let mut shown = 0;
                for i in (0..n).step_by(4) {
                    let a = u32::from_le_bytes(img[i..i + 4].try_into().unwrap());
                    let b = u32::from_le_bytes(want[i..i + 4].try_into().unwrap());
                    if a != b {
                        if shown < 30 {
                            println!("{name} +{i:#x}: got {a:08x} want {b:08x}");
                        }
                        shown += 1;
                    }
                }
                println!(
                    "{name}: {shown} u32 diffs ({} vs {} bytes)",
                    img.len(),
                    want.len()
                );
            }
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

            refh.replace(".hwx", "/weights/weight.bin")
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
        let img = mil_infer::hwx::emit_conv_chain64(*k, &w, yname, &mil, &outp).unwrap();
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

// note: appended below
