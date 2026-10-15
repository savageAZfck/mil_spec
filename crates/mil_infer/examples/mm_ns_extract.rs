//! mm_ns_extract — non-square const-weight matmul parameter sweep.
//! Compiles `[1,1,1,K] x [1,N,1,K] (transpose_y) -> [1,1,1,N]` through
//! ANEC for each grid point, extracts the scheduler scalars and blob
//! fields, appends one CSV line to `<root>/mm_params.txt`, then
//! deletes the artifact.
//!
//!   mm_ns_extract /tmp/mmgrid 32 2048 32   # lo hi step
//!
//! Re-running skips pairs already present in `mm_params.txt`.

use mil_infer::Anec;
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::Path;

fn mm_mil(k: u32, n: u32) -> String {
    format!(
        "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, 1, 1, {k}]> x) {{\n            bool txx = const()[val = bool(false)];\n            bool tyy = const()[val = bool(true)];\n            tensor<fp16, [1, {n}, 1, {k}]> w = const()[val = tensor<fp16, [1, {n}, 1, {k}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<fp16, [1, 1, 1, {n}]> z = matmul(x = x, y = w, transpose_x = txx, transpose_y = tyy)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
    )
}

fn weight_blob(data_len: usize) -> Vec<u8> {
    let mut b = vec![0x0c, 0, 0, 0, 0x02, 0, 0, 0];
    b.resize(64, 0);
    b.extend_from_slice(&[0xef, 0xbe, 0xad, 0xde, 0x01, 0, 0, 0]);
    b.extend_from_slice(&(data_len as u64).to_le_bytes());
    b.extend_from_slice(&128u64.to_le_bytes());
    b.resize(128, 0);
    b
}

fn u32le(d: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(d[off..off + 4].try_into().unwrap())
}

/// Variable scalars of the non-square matmul `__text` record and the
/// command blobs — everything that isn't a plain dim or lane-table
/// echo. Returns `(f14, p88, p90, p1f4, nsyms, textlen, big)` where
/// `p88` is the packed lane-depth word and `p90`/`p1f4` carry the
/// row pad in bytes.
fn extract(d: &[u8]) -> (u32, u32, u32, u32, u32, u32, bool) {
    let ncmd = u32le(d, 0x10);
    let (mut off, mut tfo, mut tsz, mut nsyms, mut f14) = (0x20usize, 0usize, 0usize, 0u32, 0u32);
    for _ in 0..ncmd {
        let (cmd, sz) = (u32le(d, off), u32le(d, off + 4));
        if cmd == 0x19 {
            let nsec = u32le(d, off + 64);
            for j in 0..nsec as usize {
                let s = off + 72 + 80 * j;
                if &d[s..s + 7] == b"__text\0" {
                    tfo = u32le(d, s + 48) as usize;
                    tsz = u32le(d, s + 40) as usize;
                }
            }
        } else if cmd == 2 {
            nsyms = u32le(d, off + 12);
        } else if cmd == 4 && f14 == 0 {
            f14 = u32le(d, off + 0x890);
        }
        off += sz as usize;
    }
    let p88 = u32le(d, tfo + 0x88);
    let p90 = u32le(d, tfo + 0x90);
    let p1f4 = u32le(d, tfo + 0x1f4);
    let big = u32le(d, tfo + 0x1d4) == 0x244404;
    (f14, p88, p90, p1f4, nsyms, tsz as u32, big)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = Path::new(&args.next().unwrap_or("/tmp/mmgrid".into())).to_path_buf();
    let lo: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(32);
    let hi: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(2048);
    let step: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(32);
    fs::create_dir_all(&root).unwrap();
    let pfile = root.join("mm_params.txt");
    let mut seen: BTreeSet<(u32, u32)> = BTreeSet::new();
    if let Ok(txt) = fs::read_to_string(&pfile) {
        for l in txt.lines() {
            let mut it = l.split(',');
            let (a, b, ok) = (
                it.next().and_then(|s| s.parse().ok()),
                it.next().and_then(|s| s.parse().ok()),
                it.next().map(|s| s.parse::<u32>().is_ok()),
            );
            if let (Some(a), Some(b), Some(true)) = (a, b, ok) {
                seen.insert((a, b));
            }
        }
    }
    let anec = Anec::open().expect("ANEC unavailable");
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&pfile)
        .unwrap();
    let mut n = 0;
    for k in (lo..=hi).step_by(step as usize) {
        for nn in (lo..=hi).step_by(step as usize) {
            if seen.contains(&(k, nn)) {
                continue;
            }
            let dir = root.join(format!("k{k}n{nn}"));
            fs::create_dir_all(dir.join("out")).unwrap();
            fs::create_dir_all(dir.join("weights")).unwrap();
            // ANEC selects a sparsity plan when the payload is all
            // zeros — write nonzero data (pattern suffices) to get
            // the dense plan.
            let mut w = weight_blob(k as usize * nn as usize * 2);
            for _ in 0..(k as usize * nn as usize) {
                w.extend_from_slice(&[0, 0x3c]);
            }
            fs::write(dir.join("weights/weight.bin"), &w).unwrap();
            fs::write(dir.join("net.mil"), mm_mil(k, nn)).unwrap();
            match anec.compile(&dir.join("net.mil"), &dir.join("out"), "h16") {
                Ok(v) if v.status == 0 => {
                    let d = fs::read(dir.join("out/net.macho")).unwrap();
                    let (f14, p88, p90, p1f4, nsyms, tsz, big) = extract(&d);
                    writeln!(
                        f,
                        "{k},{nn},{f14},{p88},{p90},{p1f4},{nsyms},{tsz},{}",
                        big as u32
                    )
                    .unwrap();
                    f.flush().unwrap();
                    n += 1;
                    if n % 200 == 0 {
                        println!("{n} done ({k}x{nn})");
                    }
                }
                Ok(v) => {
                    writeln!(f, "{k},{nn},STATUS{}", v.status).unwrap();
                }
                Err(e) => {
                    writeln!(f, "{k},{nn},ERR {e}").unwrap();
                }
            }
            fs::remove_dir_all(&dir).ok();
        }
    }
    println!("done {n}");
}
