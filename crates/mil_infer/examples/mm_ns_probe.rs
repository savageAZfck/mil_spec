//! mm_ns_probe — compile a non-square const-weight `matmul` probe
//! `[1,M,1,K] x [1,N,1,K] (transpose_y) -> [1,M,1,N]` through ANEC.
//! Writes `root/m<M>k<K>n<N>/{net.mil,weights/weight.bin,out/net.macho}`.
//!
//!   cargo run -p mil_infer --example mm_ns_probe -- /tmp/mmns 1x64x32 1x32x64 32x64x48

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

fn mm_mil(m: u32, k: u32, n: u32) -> String {
    format!(
        "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {m}, 1, {k}]> x) {{\n            bool txx = const()[val = bool(false)];\n            bool tyy = const()[val = bool(true)];\n            tensor<fp16, [1, {n}, 1, {k}]> w = const()[val = tensor<fp16, [1, {n}, 1, {k}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<fp16, [1, {m}, 1, {n}]> z = matmul(x = x, y = w, transpose_x = txx, transpose_y = tyy)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = Path::new(&args.next().unwrap_or("/tmp/mmns".into())).to_path_buf();
    for spec in args {
        let d: Vec<u32> = spec
            .split('x')
            .map(|s| s.trim_start_matches(['m', 'k', 'n']))
            .map(|s| s.parse().unwrap())
            .collect();
        let (m, k, n) = (d[0], d[1], d[2]);
        let dir = root.join(format!("m{m}k{k}n{n}"));
        fs::create_dir_all(dir.join("out")).unwrap();
        fs::create_dir_all(dir.join("weights")).unwrap();
        let mut d2 = Vec::with_capacity(n as usize * k as usize * 2);
        for i in 0..(n * k) {
            d2.extend_from_slice(&(i as u16).to_le_bytes());
        }
        fs::write(dir.join("weights/weight.bin"), weight_blob(&d2)).unwrap();
        fs::write(dir.join("net.mil"), mm_mil(m, k, n)).unwrap();
        match anec(&dir) {
            Ok(s) => println!("m{m}k{k}n{n}: {s}"),
            Err(e) => println!("m{m}k{k}n{n}: ERR {e}"),
        }
    }
}

fn anec(dir: &Path) -> Result<String, String> {
    let anec = Anec::open().map_err(|e| e.to_string())?;
    match anec.compile(&dir.join("net.mil"), &dir.join("out"), "h16") {
        Ok(v) => Ok(format!("status={} dram={:?}", v.status, v.dram_usage)),
        Err(e) => Err(e.to_string()),
    }
}
