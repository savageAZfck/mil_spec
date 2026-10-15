//! conv_ns_extract — non-square conv1x1 parameter sweep. Compiles a
//! `[1,Cin,1,1] x [Cout,Cin,1,1]` probe through ANEC for each grid
//! point, extracts the scheduler scalars (`f14`, `pad16`, `big`,
//! `nsyms`, text size), appends one CSV line to `<root>/ns_params.txt`,
//! then deletes the multi-megabyte artifact.
//!
//!   conv_ns_extract /tmp/cnsgrid 32 2048 32   # range, step
//!
//! Re-running skips pairs already present in `ns_params.txt`.

use mil_infer::Anec;
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::Path;

fn conv1x1_mil(cin: u32, cout: u32) -> String {
    format!(
        "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {cin}, 1, 1]> x) {{\n            tensor<fp16, [{cout}, {cin}, 1, 1]> w = const()[val = tensor<fp16, [{cout}, {cin}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n            tensor<fp16, [1, {cout}, 1, 1]> z = conv(x = x, weight = w, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
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

/// `(f14, pad16, big, nsyms, textlen)`.
fn extract(d: &[u8], cin: u32) -> (u32, u32, bool, u32, u32) {
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
    // narrow regime drops the 0x124/0x128 words; the C*16 group slides
    let narrow = tsz == 0x1cc;
    let cx = tfo + if narrow { 0x154 } else { 0x158 };
    let pad16 = (u32le(d, cx) / 16).saturating_sub(cin);
    let big = !narrow && u32le(d, tfo + 0x124) == 0x244404;
    (f14, pad16, big, nsyms, tsz as u32)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = Path::new(&args.next().unwrap_or("/tmp/cnsgrid".into())).to_path_buf();
    let lo: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(32);
    let hi: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(2048);
    let step: u32 = args.next().map(|s| s.parse().unwrap()).unwrap_or(32);
    fs::create_dir_all(&root).unwrap();
    let pfile = root.join("ns_params.txt");
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
    for cin in (lo..=hi).step_by(step as usize) {
        for cout in (lo..=hi).step_by(step as usize) {
            if seen.contains(&(cin, cout)) {
                continue;
            }
            let dir = root.join(format!("{cin}x{cout}"));
            fs::create_dir_all(dir.join("out")).unwrap();
            fs::create_dir_all(dir.join("weights")).unwrap();
            // Zero weights: the scheduler scalars don't read values, but
            // the blob must carry the full payload ANEC expects.
            let mut w = weight_blob(cin as usize * cout as usize * 2);
            w.resize(128 + cin as usize * cout as usize * 2, 0);
            fs::write(dir.join("weights/weight.bin"), &w).unwrap();
            fs::write(dir.join("net.mil"), conv1x1_mil(cin, cout)).unwrap();
            match anec.compile(&dir.join("net.mil"), &dir.join("out"), "h16") {
                Ok(v) if v.status == 0 => {
                    let d = fs::read(dir.join("out/net.macho")).unwrap();
                    let (f14, pad16, big, nsyms, tsz) = extract(&d, cin);
                    writeln!(f, "{cin},{cout},{f14},{pad16},{},{nsyms},{tsz}", big as u32).unwrap();
                    f.flush().unwrap();
                    n += 1;
                    if n % 100 == 0 {
                        println!("{n} done ({cin}x{cout})");
                    }
                }
                Ok(v) => {
                    writeln!(f, "{cin},{cout},STATUS{}", v.status).unwrap();
                }
                Err(e) => {
                    writeln!(f, "{cin},{cout},ERR {e}").unwrap();
                }
            }
            fs::remove_dir_all(&dir).ok();
        }
    }
    println!("done {n}");
}
