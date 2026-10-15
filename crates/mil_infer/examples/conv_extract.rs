//! conv_extract — extend `CONV_CHAN_TABLE` coverage by compiling a
//! `conv1x1` probe at each `C` through ANEC, extracting the three
//! scheduler scalars (`f14`, `pad16`, `big`) plus the symbol-plan
//! marker, then deleting the multi-megabyte artifact. Appends one CSV
//! line per `C` to `<root>/params.txt`:
//!
//!   cargo run -p mil_infer --release --example conv_extract -- /tmp/cdense2 2080 4096
//!
//! Re-running skips `C` values already present in `params.txt` and
//! reuses any surviving `out/net.macho` without recompiling.

use mil_infer::Anec;
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::Path;

fn weight_blob(data_len: usize) -> Vec<u8> {
    let mut b = vec![0x0c, 0, 0, 0, 0x02, 0, 0, 0];
    b.resize(64, 0);
    b.extend_from_slice(&[0xef, 0xbe, 0xad, 0xde, 0x01, 0, 0, 0]);
    b.extend_from_slice(&(data_len as u64).to_le_bytes());
    b.extend_from_slice(&128u64.to_le_bytes());
    b.resize(128, 0);
    b
}

fn conv1x1_mil(c: u32) -> String {
    format!(
        "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {c}, 1, 1]> x) {{\n            tensor<fp16, [{c}, {c}, 1, 1]> w = const()[val = tensor<fp16, [{c}, {c}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n            tensor<fp16, [1, {c}, 1, 1]> z = conv(x = x, weight = w, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
    )
}

fn u32le(d: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(d[off..off + 4].try_into().unwrap())
}

/// `(f14, pad16, big, nsyms)` — blob1 `0xbe8` copy is at the first
/// `cmd4` command + `0x890`; text fields are relative to the `__text`
/// section fileoff. `pad16 = (f158 - C*16)/16`.
fn extract(d: &[u8], c: u32) -> (u32, u32, bool, u32) {
    let ncmd = u32le(d, 0x10);
    let (mut off, mut tfo, mut nsyms, mut c4) = (0x20usize, 0usize, 0u32, Vec::new());
    for _ in 0..ncmd {
        let (cmd, sz) = (u32le(d, off), u32le(d, off + 4));
        if cmd == 0x19 {
            let nsec = u32le(d, off + 64);
            for j in 0..nsec as usize {
                let s = off + 72 + 80 * j;
                if &d[s..s + 7] == b"__text\0" {
                    tfo = u32le(d, s + 48) as usize;
                }
            }
        } else if cmd == 2 {
            nsyms = u32le(d, off + 12);
        } else if cmd == 4 {
            c4.push(off);
        }
        off += sz as usize;
    }
    let f14 = u32le(d, c4[0] + 0x890);
    let f158 = u32le(d, tfo + 0x158);
    let pad16 = (f158 - c * 16) / 16;
    let big = u32le(d, tfo + 0x124) == 0x244404;
    (f14, pad16, big, nsyms)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = Path::new(&args.next().unwrap_or("/tmp/cdense2".into())).to_path_buf();
    let lo: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(2080);
    let hi: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(4096);
    let params = root.join("params.txt");
    let mut have = BTreeSet::<u32>::new();
    if let Ok(txt) = fs::read_to_string(&params) {
        for line in txt.lines() {
            if let Some(c) = line.split(',').next().and_then(|s| s.parse().ok()) {
                have.insert(c);
            }
        }
    }
    let anec = Anec::open().expect("ANEC unavailable");
    let mut pf = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&params)
        .unwrap();
    for c in lo..=hi {
        if have.contains(&c) {
            continue;
        }
        let dir = root.join(format!("c{c}"));
        let macho = dir.join("out/net.macho");
        if !macho.exists() {
            fs::create_dir_all(dir.join("out")).unwrap();
            fs::create_dir_all(dir.join("weights")).unwrap();
            // Zero weights: the scheduler scalars don't read values.
            let mut w = weight_blob(c as usize * c as usize * 2);
            w.resize(128 + c as usize * c as usize * 2, 0);
            fs::write(dir.join("weights/weight.bin"), &w).unwrap();
            fs::write(dir.join("net.mil"), conv1x1_mil(c)).unwrap();
            match anec.compile(&dir.join("net.mil"), &dir.join("out"), "h16") {
                Ok(v) if v.status == 0 && macho.exists() => {}
                Ok(v) => {
                    writeln!(pf, "{c},FAIL,{},{:?}", v.status, v.errors).unwrap();
                    pf.flush().unwrap();
                    let _ = fs::remove_file(dir.join("weights/weight.bin"));
                    continue;
                }
                Err(e) => {
                    writeln!(pf, "{c},ERR,{e}").unwrap();
                    pf.flush().unwrap();
                    let _ = fs::remove_file(dir.join("weights/weight.bin"));
                    continue;
                }
            }
        }
        let d = fs::read(&macho).unwrap();
        let (f14, pad16, big, nsyms) = extract(&d, c);
        writeln!(pf, "{c},{f14},{pad16},{big},{nsyms}").unwrap();
        pf.flush().unwrap();
        println!("c{c}: f14={f14} pad16={pad16} big={big} nsyms={nsyms}");
        // Keep every 64th artifact for byte-verify spot checks; the
        // rest are regenerable and cost ~10-30 MB each.
        if c % 64 != 0 {
            let _ = fs::remove_file(&macho);
        }
        let _ = fs::remove_file(dir.join("weights/weight.bin"));
    }
}
