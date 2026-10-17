//! conv_probe — compile a `conv1x1` probe at arbitrary channel count
//! through ANEC for field-fitting. Writes `root/c<C>/{net.mil,
//! weights/weight.bin,out/net.macho}`.
//!
//!   cargo run -p mil_infer --example conv_probe -- /tmp/cpar2 144 176 352

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

fn conv1x1_mil(c: u32, w: u32) -> String {
    format!(
        "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {c}, {w}, 1]> x) {{\n            tensor<fp16, [{c}, {c}, 1, 1]> w = const()[val = tensor<fp16, [{c}, {c}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n            tensor<fp16, [1, {c}, {w}, 1]> z = conv(x = x, weight = w, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = Path::new(&args.next().unwrap_or("/tmp/cpar2".into())).to_path_buf();
    let cs: Vec<u32> = args.map(|s| s.parse().unwrap()).collect();
    let anec = Anec::open().expect("ANEC unavailable");
    for c in cs {
        let dir = root.join(format!("c{c}"));
        fs::create_dir_all(dir.join("out")).unwrap();
        fs::create_dir_all(dir.join("weights")).unwrap();
        // CONV_PROBE_WROWS=1 writes w[oc][ic]=oc so the emitted lane
        // stream exposes ANEC's row assignment directly.
        let by_row = std::env::var_os("CONV_PROBE_WROWS").is_some();
        let mut d = Vec::with_capacity(c as usize * c as usize * 2);
        for i in 0..(c * c) {
            let v = if by_row { i / c } else { i };
            d.extend_from_slice(&(v as u16).to_le_bytes());
        }
        fs::write(dir.join("weights/weight.bin"), weight_blob(&d)).unwrap();
        fs::write(dir.join("net.mil"), conv1x1_mil(c, 1)).unwrap();
        let v = anec.compile(&dir.join("net.mil"), &dir.join("out"), "h16");
        match v {
            Ok(v) => println!("c{c}: status={} dram={:?}", v.status, v.dram_usage),
            Err(e) => println!("c{c}: ERR {e}"),
        }
    }
}
