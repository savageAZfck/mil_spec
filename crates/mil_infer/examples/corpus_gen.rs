//! Regenerate the ANEC reference corpus used for HWX emitter
//! byte-diffing. Writes `corpus/<name>/{net.mil,out/net.macho}`
//! under the directory given (or `corpus/` in cwd).
//!
//!   cargo run -p mil_infer --example corpus_gen [dir]

use mil_infer::Anec;
use std::fs;
use std::path::{Path, PathBuf};

/// deadbeef-chunked weight blob; BLOBFILE offsets point at the
/// chunk header at byte 64. `data` is the fp16 payload.
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

/// All-ones fp16 payload.
fn ones_blob(n_elem: usize) -> Vec<u8> {
    weight_blob(&vec![0x00, 0x3c].repeat(n_elem))
}

/// Perm-pattern weights: element `i` is the raw u16 `i`, so the
/// transformed stream in `__KERN_0` directly reads as a source-index
/// map of the ANE weight interleave.
fn perm_blob(n_elem: usize) -> Vec<u8> {
    let mut d = Vec::with_capacity(n_elem * 2);
    for i in 0..n_elem {
        d.extend_from_slice(&(i as u16).to_le_bytes());
    }
    weight_blob(&d)
}

fn ew_mil(op: &str, n: u32) -> String {
    format!(
        "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {n}, 1, 1]> x, tensor<fp16, [1, {n}, 1, 1]> y) {{\n        tensor<fp16, [1, {n}, 1, 1]> z = {op}(x = x, y = y)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
    )
}

fn unary_mil(op: &str, n: u32) -> String {
    format!(
        "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {n}, 1, 1]> x) {{\n        tensor<fp16, [1, {n}, 1, 1]> z = {op}(x = x)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
    )
}

fn conv1x1_mil(c: u32, w: u32) -> String {
    format!(
        "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {c}, {w}, 1]> x) {{\n            tensor<fp16, [{c}, {c}, 1, 1]> w = const()[val = tensor<fp16, [{c}, {c}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n            tensor<fp16, [1, {c}, {w}, 1]> z = conv(x = x, weight = w, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
    )
}

fn matmul_mil(n: u32) -> String {
    format!(
        "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, [1, {n}, 1, {n}]> x) {{\n        bool txx = const()[val = bool(false)];\n        bool tyy = const()[val = bool(true)];\n        tensor<fp16, [1, {n}, 1, {n}]> w = const()[val = tensor<fp16, [1, {n}, 1, {n}]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n        tensor<fp16, [1, {n}, 1, {n}]> z = matmul(x = x, y = w, transpose_x = txx, transpose_y = tyy)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
    )
}

fn one(anec: &Anec, root: &Path, name: &str, mil: &str, weights: Option<Vec<u8>>) {
    let dir = root.join(name);
    let out = dir.join("out");
    fs::create_dir_all(&out).unwrap();
    if let Some(w) = weights {
        let wd = dir.join("weights");
        fs::create_dir_all(&wd).unwrap();
        fs::write(wd.join("weight.bin"), w).unwrap();
    }
    let p = dir.join("net.mil");
    fs::write(&p, mil).unwrap();
    let v = anec.compile(&p, &out, "h16").unwrap_or_else(|e| {
        eprintln!("{name}: api error {e}");
        std::process::exit(1);
    });
    if v.status == 0 {
        println!("{name}: OK (dram={:?})", v.dram_usage);
    } else {
        println!("{name}: FAIL {:?}", v.errors);
    }
}

fn main() {
    let root = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "corpus".into()));
    fs::create_dir_all(&root).unwrap();
    let anec = Anec::open().expect("ANECompiler.framework load");

    for n in [20u32, 32, 36, 48, 56, 64, 96, 128, 256] {
        one(&anec, &root, &format!("add{n}"), &ew_mil("add", n), None);
    }
    for (op, tag) in [
        ("mul", "mul64"),
        ("sub", "sub64"),
        ("maximum", "max64"),
        ("minimum", "min64"),
    ] {
        one(&anec, &root, tag, &ew_mil(op, 64), None);
    }
    for (op, tag, n) in [
        ("sigmoid", "sig32", 32u32),
        ("sigmoid", "sig128", 128),
        ("tanh", "tanh64", 64),
        ("relu", "relu64", 64),
    ] {
        one(&anec, &root, tag, &unary_mil(op, n), None);
    }
    for (tag, c, w) in [
        ("conv64s1", 64u32, 1u32),
        ("conv64s5", 64, 5),
        ("conv32", 32, 1),
        ("conv128", 128, 1),
    ] {
        one(
            &anec,
            &root,
            tag,
            &conv1x1_mil(c, w),
            Some(ones_blob((c * c) as usize)),
        );
    }
    // perm-weight probes: same-shape pairs isolate the transformed
    // weight region byte-for-byte (transform ground truth lives in
    // assets/conv_perm_vectors.json).
    for (tag, c, w) in [
        ("conv32p", 32u32, 1u32),
        ("conv64p", 64, 1),
        ("conv128p", 128, 1),
        ("conv64s5p", 64, 5),
        ("conv64s8p", 64, 8),
    ] {
        one(
            &anec,
            &root,
            tag,
            &conv1x1_mil(c, w),
            Some(perm_blob((c * c) as usize)),
        );
    }
    for n in [32u32, 64, 128, 256] {
        one(
            &anec,
            &root,
            &format!("mm{n}"),
            &matmul_mil(n),
            Some(ones_blob((n * n) as usize)),
        );
    }
    for (tag, n) in [("mm32p", 32u32), ("mm64p", 64)] {
        one(
            &anec,
            &root,
            tag,
            &matmul_mil(n),
            Some(perm_blob((n * n) as usize)),
        );
    }
}
