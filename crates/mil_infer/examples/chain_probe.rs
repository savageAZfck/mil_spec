//! chain_probe — compile a k-deep conv1x1 chain through ANEC to study
//! multi-task __text emission. Each op gets its own weight blob slice.
//!
//!   cargo run -p mil_infer --example chain_probe -- /tmp/chain 64 1 2 3 4

use mil_infer::Anec;
use std::fs;
use std::path::Path;

fn chain_mil(c: u32, k: usize) -> String {
    let mut s =
        format!("program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {c}, 1, 1]> x) {{\n");
    for i in 0..k {
        let rec = 64 + i * (64 + (c * c * 2) as usize);
        s += &format!("            tensor<fp16, [{c}, {c}, 1, 1]> w{i} = const()[val = tensor<fp16, [{c}, {c}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({rec})))];\n");
    }
    s += "            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n";
    for i in 0..k {
        let (inp, outp) = (
            if i == 0 { "x".into() } else { format!("y{i}") },
            format!("y{}", i + 1),
        );
        s += &format!("            tensor<fp16, [1, {c}, 1, 1]> {outp} = conv(x = {inp}, weight = w{i}, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"{outp}\")];\n");
    }
    s += &format!("    }} -> (y{k});\n}}\n");
    s
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = Path::new(&args.next().unwrap_or("/tmp/chain".into())).to_path_buf();
    let c: u32 = args.next().unwrap().parse().unwrap();
    let ks: Vec<usize> = args.map(|s| s.parse().unwrap()).collect();
    let anec = Anec::open().expect("ANEC unavailable");
    let kmax = *ks.iter().max().unwrap();
    for k in ks {
        let dir = root.join(format!("c{c}k{k}"));
        fs::create_dir_all(dir.join("out")).unwrap();
        fs::create_dir_all(dir.join("weights")).unwrap();
        // one deadbeef record per weight: 64B header + payload
        let payload = (c * c * 2) as usize;
        let mut b = vec![0x0c, 0, 0, 0, 0x02, 0, 0, 0];
        b.resize(64, 0);
        for i in 0..kmax {
            let data_start = b.len() + 64;
            b.extend_from_slice(&[0xef, 0xbe, 0xad, 0xde, 0x01, 0, 0, 0]);
            b.extend_from_slice(&(payload as u64).to_le_bytes());
            b.extend_from_slice(&(data_start as u64).to_le_bytes());
            b.resize(data_start, 0);
            for j in 0..(c * c) as u32 {
                b.extend_from_slice(&(((j + i as u32 * 7919) % 65536) as u16).to_le_bytes());
            }
        }
        fs::write(dir.join("weights/weight.bin"), &b).unwrap();
        fs::write(dir.join("net.mil"), chain_mil(c, k)).unwrap();
        let v = anec.compile(&dir.join("net.mil"), &dir.join("out"), "h16");
        match v {
            Ok(v) => println!(
                "c{c}k{k}: status={} dram={:?} errs={:?}",
                v.status, v.dram_usage, v.errors
            ),
            Err(e) => println!("c{c}k{k}: ERR {e}"),
        }
    }
}
