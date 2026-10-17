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
    weight_blob(&[0x00, 0x3c].repeat(n_elem))
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

/// `(conv1x1 -> act)` repeated `k` times. `n_blobs` distinct
/// weight files (`weight.bin`, `weight2.bin`, ...) are referenced
/// round-robin across layers — `n_blobs=1` yields the shared-weight
/// (deduplicated) chain, `n_blobs=k` the fully distinct one.
fn conv_sig_chain_mil(k: usize, n_blobs: usize) -> String {
    conv_act_chain_mil(k, n_blobs, "sigmoid")
}

/// Same chain with a parameterized activation op.
fn conv_act_chain_mil(k: usize, n_blobs: usize, act: &str) -> String {
    const NAMES: [&str; 26] = [
        "t", "u", "v", "p", "q", "r", "s", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l",
        "m", "n", "o", "w1", "w3", "w4", "w5", "w6",
    ];
    let mut s = String::from(
        "program(1.3)\n{\n    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {\n            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n",
    );
    if act == "gelu" {
        s += "            string gmode = const()[val = string(\"EXACT\")];\n";
    }
    let mut inp = "x".to_string();
    for i in 0..k {
        let blob = if i % n_blobs == 0 {
            "weight.bin".to_string()
        } else {
            format!("weight{}.bin", i % n_blobs + 1)
        };
        let (wn, cn) = (format!("w{i}"), NAMES[i].to_string());
        s += &format!("            tensor<fp16, [64, 64, 1, 1]> {wn} = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/{blob}\"), offset = uint64(64)))];\n");
        s += &format!("            tensor<fp16, [1, 64, 1, 1]> {cn} = conv(x = {inp}, weight = {wn}, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"{cn}\")];\n");
        let sn = if i + 1 == k {
            "z".into()
        } else {
            format!("s{i}")
        };
        let call = if act == "gelu" {
            format!("gelu(x = {cn}, mode = gmode)")
        } else {
            format!("{act}(x = {cn})")
        };
        s += &format!(
            "            tensor<fp16, [1, 64, 1, 1]> {sn} = {call}[name = string(\"{sn}\")];\n"
        );
        inp = sn;
    }
    s += "    } -> (z);\n}\n";
    s
}

/// k-layer conv+silu chain — `mul(t, sigmoid(t))` lowered to the
/// fused LUT kernel by ANEC (record-identical to the unary acts).
fn conv_silu_chain_mil(k: usize, n_blobs: usize) -> String {
    const NAMES: [&str; 26] = [
        "t", "u", "v", "p", "q", "r", "s", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l",
        "m", "n", "o", "w1", "w3", "w4", "w5", "w6",
    ];
    let mut s = String::from(
        "program(1.3)\n{\n    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {\n            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n",
    );
    let mut inp = "x".to_string();
    for i in 0..k {
        let blob = if i % n_blobs == 0 {
            "weight.bin".to_string()
        } else {
            format!("weight{}.bin", i % n_blobs + 1)
        };
        let (wn, cn) = (format!("w{i}"), NAMES[i].to_string());
        s += &format!("            tensor<fp16, [64, 64, 1, 1]> {wn} = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/{blob}\"), offset = uint64(64)))];\n");
        s += &format!("            tensor<fp16, [1, 64, 1, 1]> {cn} = conv(x = {inp}, weight = {wn}, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"{cn}\")];\n");
        let sn = if i + 1 == k {
            "z".into()
        } else {
            format!("s{i}")
        };
        s += &format!("            tensor<fp16, [1, 64, 1, 1]> g{i} = sigmoid(x = {cn})[name = string(\"g{i}\")];\n");
        s += &format!("            tensor<fp16, [1, 64, 1, 1]> {sn} = mul(x = {cn}, y = g{i})[name = string(\"{sn}\")];\n");
        inp = sn;
    }
    s += "    } -> (z);\n}\n";
    s
}

/// k-layer conv+softmax chain — the multi-task record family: each
/// layer decomposes into ~4 task descriptors rather than one fused
/// kernel.
fn conv_sm_chain_mil(k: usize, n_blobs: usize) -> String {
    const NAMES: [&str; 26] = [
        "t", "u", "v", "p", "q", "r", "s", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l",
        "m", "n", "o", "w1", "w3", "w4", "w5", "w6",
    ];
    let mut s = String::from(
        "program(1.3)\n{\n    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {\n            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n            int32 ax = const()[val = int32(1)];\n",
    );
    let mut inp = "x".to_string();
    for i in 0..k {
        let blob = if i % n_blobs == 0 {
            "weight.bin".to_string()
        } else {
            format!("weight{}.bin", i % n_blobs + 1)
        };
        let (wn, cn) = (format!("w{i}"), NAMES[i].to_string());
        s += &format!("            tensor<fp16, [64, 64, 1, 1]> {wn} = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/{blob}\"), offset = uint64(64)))];\n");
        s += &format!("            tensor<fp16, [1, 64, 1, 1]> {cn} = conv(x = {inp}, weight = {wn}, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"{cn}\")];\n");
        let sn = if i + 1 == k {
            "z".into()
        } else {
            format!("s{i}")
        };
        s += &format!("            tensor<fp16, [1, 64, 1, 1]> {sn} = softmax(x = {cn}, axis = ax)[name = string(\"{sn}\")];\n");
        inp = sn;
    }
    s += "    } -> (z);\n}\n";
    s
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

    // multi-op chains: probe how op composition maps into the artifact
    // (kernel count, intermediate TDs, instruction-stream layout).
    one(
        &anec,
        &root,
        "chain_add_mul",
        "program(1.3)\n{\n    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x, tensor<fp16, [1, 64, 1, 1]> y) {\n        tensor<fp16, [1, 64, 1, 1]> t = add(x = x, y = y)[name = string(\"t\")];\n        tensor<fp16, [1, 64, 1, 1]> z = mul(x = t, y = x)[name = string(\"z\")];\n    } -> (z);\n}\n",
        None,
    );
    one(
        &anec,
        &root,
        "chain_add_add",
        "program(1.3)\n{\n    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x, tensor<fp16, [1, 64, 1, 1]> y) {\n        tensor<fp16, [1, 64, 1, 1]> t = add(x = x, y = y)[name = string(\"t\")];\n        tensor<fp16, [1, 64, 1, 1]> z = add(x = t, y = y)[name = string(\"z\")];\n    } -> (z);\n}\n",
        None,
    );
    // conv feeding a unary activation — the real-model pattern.
    one(
        &anec,
        &root,
        "chain_conv_sig",
        "program(1.3)\n{\n    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {\n            tensor<fp16, [64, 64, 1, 1]> w = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n            tensor<fp16, [1, 64, 1, 1]> t = conv(x = x, weight = w, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"t\")];\n            tensor<fp16, [1, 64, 1, 1]> z = sigmoid(x = t)[name = string(\"z\")];\n    } -> (z);\n}\n",
        Some(ones_blob(64 * 64)),
    );
    // perm-weight twin: isolates the weight bytes inside the chained
    // LUT|weight band layout for byte-identity testing.
    one(
        &anec,
        &root,
        "chain_conv_sigp",
        "program(1.3)\n{\n    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {\n            tensor<fp16, [64, 64, 1, 1]> w = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n            tensor<fp16, [1, 64, 1, 1]> t = conv(x = x, weight = w, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"t\")];\n            tensor<fp16, [1, 64, 1, 1]> z = sigmoid(x = t)[name = string(\"z\")];\n    } -> (z);\n}\n",
        Some(perm_blob(64 * 64)),
    );
    // conv->sig->conv->sig: does composition repeat additively?
    one(
        &anec,
        &root,
        "chain_conv2_sig2",
        "program(1.3)\n{\n    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {\n            tensor<fp16, [64, 64, 1, 1]> w = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n            tensor<fp16, [1, 64, 1, 1]> t = conv(x = x, weight = w, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"t\")];\n            tensor<fp16, [1, 64, 1, 1]> u = sigmoid(x = t)[name = string(\"u\")];\n            tensor<fp16, [64, 64, 1, 1]> w2 = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<fp16, [1, 64, 1, 1]> v = conv(x = u, weight = w2, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"v\")];\n            tensor<fp16, [1, 64, 1, 1]> z = sigmoid(x = v)[name = string(\"z\")];\n    } -> (z);\n}\n",
        Some(ones_blob(64 * 64)),
    );
    // third repetition: fields that scale with op position become
    // linear functions of layer index — the wiring rule equations.
    one(
        &anec,
        &root,
        "chain_conv3_sig3",
        "program(1.3)\n{\n    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {\n            tensor<fp16, [64, 64, 1, 1]> w = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n            string pt = const()[val = string(\"valid\")];\n            tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n            tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n            int32 gr = const()[val = int32(1)];\n            tensor<fp16, [1, 64, 1, 1]> t = conv(x = x, weight = w, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"t\")];\n            tensor<fp16, [1, 64, 1, 1]> u = sigmoid(x = t)[name = string(\"u\")];\n            tensor<fp16, [64, 64, 1, 1]> w2 = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<fp16, [1, 64, 1, 1]> v = conv(x = u, weight = w2, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"v\")];\n            tensor<fp16, [1, 64, 1, 1]> p = sigmoid(x = v)[name = string(\"p\")];\n            tensor<fp16, [64, 64, 1, 1]> w3 = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n            tensor<fp16, [1, 64, 1, 1]> q = conv(x = p, weight = w3, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gr)[name = string(\"q\")];\n            tensor<fp16, [1, 64, 1, 1]> z = sigmoid(x = q)[name = string(\"z\")];\n    } -> (z);\n}\n",
        Some(ones_blob(64 * 64)),
    );

    // deeper chains pin the per-position wiring formulas; the
    // distinct-weight variants isolate the __KERN_0 weight-stream
    // concatenation and per-record source ordinals.
    for (tag, k, n_blobs) in [
        ("chain4", 4usize, 1usize),
        ("chain5", 5, 1),
        ("chain2_dw", 2, 2),
        ("chain3_dw", 3, 3),
        ("chain3_pdw", 3, 2),
        ("chain4_dw", 4, 4),
        ("chain5_dw", 5, 5),
        ("chain6_dw", 6, 6),
        ("chain8_dw", 8, 8),
    ] {
        let dir = root.join(tag);
        fs::create_dir_all(dir.join("out")).unwrap();
        fs::create_dir_all(dir.join("weights")).unwrap();
        for j in 0..n_blobs {
            let name = if j == 0 {
                "weight.bin".to_string()
            } else {
                format!("weight{}.bin", j + 1)
            };
            // shared chains use the ones blob (same bytes as the
            // captured skeletons); distinct blobs carry perm-offset
            // patterns so each __KERN_0 stream self-identifies
            let blob = if n_blobs == 1 {
                ones_blob(64 * 64)
            } else {
                let mut d = Vec::with_capacity(64 * 64 * 2);
                for i in 0..64 * 64u32 {
                    d.extend_from_slice(&((i + j as u32 * 0x5555) as u16).to_le_bytes());
                }
                weight_blob(&d)
            };
            fs::write(dir.join("weights").join(name), blob).unwrap();
        }
        fs::write(dir.join("net.mil"), conv_sig_chain_mil(k, n_blobs)).unwrap();
        let v = anec
            .compile(&dir.join("net.mil"), &dir.join("out"), "h16")
            .unwrap();
        if v.status == 0 {
            println!("{tag}: OK (dram={:?})", v.dram_usage);
        } else {
            println!("{tag}: FAIL {:?}", v.errors);
        }
    }

    // fused conv->activation variants: each activation family gets
    // its own LUT table, record wiring markers, and kernel-name set;
    // shared + dw probes cover the name table per stream ordinal
    for (tag, act, k, n_blobs) in [
        ("chain_conv_relu", "relu", 1usize, 1usize),
        ("chain_conv_tanh", "tanh", 1, 1),
        ("chain_conv2_relu2", "relu", 2, 1),
        ("chain_conv2_tanh2", "tanh", 2, 1),
        ("chain5_relu", "relu", 5, 1),
        ("chain5_tanh", "tanh", 5, 1),
        ("chain8_relu_dw", "relu", 8, 8),
        ("chain8_tanh_dw", "tanh", 8, 8),
        ("chain_conv_gelu", "gelu", 1, 1),
        ("chain5_gelu", "gelu", 5, 1),
        ("chain8_gelu_dw", "gelu", 8, 8),
        ("chain_conv_silu", "silu", 1, 1),
        ("chain_conv2_silu2", "silu", 2, 1),
        ("chain5_silu", "silu", 5, 1),
        ("chain8_silu_dw", "silu", 8, 8),
        ("chain_conv_sm", "softmax", 1, 1),
        ("chain_conv2_sm2", "softmax", 2, 1),
        ("smp", "softmax", 1, 1),
    ] {
        let dir = root.join(tag);
        fs::create_dir_all(dir.join("out")).unwrap();
        fs::create_dir_all(dir.join("weights")).unwrap();
        for j in 0..n_blobs {
            let name = if j == 0 {
                "weight.bin".to_string()
            } else {
                format!("weight{}.bin", j + 1)
            };
            let mut d = Vec::with_capacity(64 * 64 * 2);
            for i in 0..64 * 64u32 {
                let v = if tag == "smp" {
                    i + 1
                } else {
                    i + j as u32 * 0x5555
                };
                d.extend_from_slice(&(v as u16).to_le_bytes());
            }
            let blob = if n_blobs == 1 && tag != "smp" {
                ones_blob(64 * 64)
            } else {
                weight_blob(&d)
            };
            fs::write(dir.join("weights").join(name), blob).unwrap();
        }
        let mil = match act {
            "silu" => conv_silu_chain_mil(k, n_blobs),
            "softmax" => conv_sm_chain_mil(k, n_blobs),
            _ => conv_act_chain_mil(k, n_blobs, act),
        };
        fs::write(dir.join("net.mil"), mil).unwrap();
        let v = anec
            .compile(&dir.join("net.mil"), &dir.join("out"), "h16")
            .unwrap();
        if v.status == 0 {
            println!("{tag}: OK (dram={:?})", v.dram_usage);
        } else {
            println!("{tag}: FAIL {:?}", v.errors);
        }
    }
}
