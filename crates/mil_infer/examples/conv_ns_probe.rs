//! conv_ns_probe — compile non-square `conv1x1` probes through ANEC.
//! Weight shape is `[cout, cin, 1, 1]`, x `[1, cin, 1, 1]` -> z `[1, cout, 1, 1]`.
//! Writes `root/<cin>x<cout>/{net.mil,weights/weight.bin,out/net.macho}`.
//!
//!   cargo run -p mil_infer --example conv_ns_probe -- /tmp/cns4 128:32 32:128 64:96

use mil_infer::Anec;
use std::fs;
use std::path::Path;

fn weight_blob(data: &[u8]) -> Vec<u8> {
    let mut b = vec![0x0c, 0, 0, 0, 0x02, 0, 0, 0];
    b.resize(64, 0);
    b.extend_from_slice(&[0xef, 0xbe, 0xad, 0xde, 0x01, 0, 0, 0]);
    b.extend_from_slice(&(data.len() as u64).to_le_bytes());
    b.extend_from_slice(&128u64.to_le_bytes());
    b.resize(128, 0);
    b.extend_from_slice(data);
    b
}

fn conv1x1_mil(cin: u32, cout: u32) -> String {
    format!(
        "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {cin}, 1, 1]> x) {{\n            tensor<fp16, [{cout}, {cin}, 1, 1]> w = const()[val = tensor<fp16, [{cout}, {cin}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n            tensor<fp16, [1, {cout}, 1, 1]> z = conv(x = x, weight = w, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = Path::new(&args.next().unwrap_or("/tmp/cns4".into())).to_path_buf();
    let pairs: Vec<(u32, u32)> = args
        .map(|s| {
            let (a, b) = s.split_once('x').unwrap();
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .collect();
    let anec = Anec::open().expect("ANEC unavailable");
    let by_row = std::env::var_os("CONV_PROBE_WROWS").is_some();
    for (cin, cout) in pairs {
        let dir = root.join(format!("{cin}x{cout}"));
        fs::create_dir_all(dir.join("out")).unwrap();
        fs::create_dir_all(dir.join("weights")).unwrap();
        // Default weight w[oc][ic] = flat index so each emitted u16 names
        // its source element exactly. WROWS writes w[oc][ic]=oc instead.
        let mut d = Vec::with_capacity(cin as usize * cout as usize * 2);
        let by_ic = std::env::var_os("CONV_PROBE_WIC").is_some();
        for i in 0..(cin * cout) {
            let v = if by_row {
                i / cin
            } else if by_ic {
                i % cin
            } else {
                i
            };
            d.extend_from_slice(&(v as u16).to_le_bytes());
        }
        fs::write(dir.join("weights/weight.bin"), weight_blob(&d)).unwrap();
        fs::write(dir.join("net.mil"), conv1x1_mil(cin, cout)).unwrap();
        match anec.compile(&dir.join("net.mil"), &dir.join("out"), "h16") {
            Ok(v) => println!("{cin}x{cout}: status={} dram={:?}", v.status, v.dram_usage),
            Err(e) => println!("{cin}x{cout}: ERR {e}"),
        }
    }
}
