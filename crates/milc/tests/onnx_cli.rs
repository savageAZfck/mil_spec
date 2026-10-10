//! `milc onnx` / `milc attest` end-to-end: ONNX (hand-encoded, no
//! onnx package) → package → coremlc → mil_infer, plus provenance.
//! Prediction legs return early when `coremlc` is absent.

use mil_infer::{ComputeUnits, Input, Model};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

// ---------- minimal ONNX encoder (copied subset of mil_onnx tests) ----------

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}
fn f_i(field: u32, v: i64, out: &mut Vec<u8>) {
    varint((field as u64) << 3, out);
    varint(v as u64, out);
}
fn f_len(field: u32, b: &[u8], out: &mut Vec<u8>) {
    varint(((field as u64) << 3) | 2, out);
    varint(b.len() as u64, out);
    out.extend_from_slice(b);
}
fn f_str(field: u32, s: &str, out: &mut Vec<u8>) {
    f_len(field, s.as_bytes(), out);
}
fn t_f32(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
    let mut m = Vec::new();
    for &d in dims {
        f_i(1, d, &mut m);
    }
    f_i(2, 1, &mut m);
    f_str(8, name, &mut m);
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    f_len(9, &raw, &mut m);
    m
}
/// `ValueInfoProto` float tensor; a dim that parses as an integer is a
/// `dim_value`, anything else a `dim_param`.
fn value_info(name: &str, dims: &[&str]) -> Vec<u8> {
    let mut shape = Vec::new();
    for p in dims {
        let mut dim = Vec::new();
        match p.parse::<i64>() {
            Ok(d) => f_i(1, d, &mut dim),
            Err(_) => f_str(2, p, &mut dim),
        }
        f_len(1, &dim, &mut shape);
    }
    let mut tt = Vec::new();
    f_i(1, 1, &mut tt);
    f_len(2, &shape, &mut tt);
    let mut ty = Vec::new();
    f_len(1, &tt, &mut ty);
    let mut vi = Vec::new();
    f_str(1, name, &mut vi);
    f_len(2, &ty, &mut vi);
    vi
}
fn node(op: &str, inputs: &[&str], outputs: &[&str], name: &str) -> Vec<u8> {
    let mut m = Vec::new();
    for i in inputs {
        f_str(1, i, &mut m);
    }
    for o in outputs {
        f_str(2, o, &mut m);
    }
    f_str(3, name, &mut m);
    f_str(4, op, &mut m);
    m
}
fn model(nodes: &[Vec<u8>], inits: &[Vec<u8>], ins: &[Vec<u8>], outs: &[Vec<u8>]) -> Vec<u8> {
    let mut g = Vec::new();
    for n in nodes {
        f_len(1, n, &mut g);
    }
    f_str(2, "g", &mut g);
    for i in inits {
        f_len(5, i, &mut g);
    }
    for v in ins {
        f_len(11, v, &mut g);
    }
    for v in outs {
        f_len(12, v, &mut g);
    }
    let mut m = Vec::new();
    f_i(1, 8, &mut m);
    f_str(2, "milc onnx_cli test encoder", &mut m);
    f_len(7, &g, &mut m);
    let mut os = Vec::new();
    f_str(1, "", &mut os);
    f_i(2, 13, &mut os);
    f_len(8, &os, &mut m);
    m
}

// ---------- harness ----------

struct Tmp(PathBuf);
impl Tmp {
    fn new(tag: &str) -> Tmp {
        let p = std::env::temp_dir().join(format!("milc_onnx_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Tmp(p)
    }
}
impl Drop for Tmp {
    fn drop(&mut self) {
        if std::env::var("MIL_KEEP_ARTIFACTS").as_deref() != Ok("1") {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn milc(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_milc"))
        .args(args)
        .output()
        .expect("spawn milc")
}
fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}
fn p(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn randv(n: usize, lo: f32, hi: f32, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            (lo as f64 + (hi - lo) as f64 * u) as f32
        })
        .collect()
}
fn f16_bytes(vs: &[f32]) -> Vec<u8> {
    vs.iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect()
}
fn qh(x: &[f32]) -> Vec<f64> {
    x.iter()
        .map(|v| half::f16::from_f32(*v).to_f32() as f64)
        .collect()
}
fn softmax_row(v: &[f64]) -> Vec<f64> {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}
fn check(got: &[f32], want: &[f64], rtol: f64, atol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let maxd = got
        .iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - *w).abs())
        .fold(0.0, f64::max);
    println!(
        "{what}: predicted {} values, max|Δ| = {maxd:.3e}",
        got.len()
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (*g as f64 - *w).abs();
        assert!(
            g.is_finite() && d <= atol + rtol * w.abs(),
            "{what}[{i}]: got {g}, want {w}"
        );
    }
}

/// Compile + predict `pkg` with one fp16 input; `None` without coremlc.
fn predict(pkg: &Path, dir: &Path, input: &str, shape: &[i64], x: &[f32]) -> Option<Vec<f32>> {
    mil_compile::coremlc_path()?;
    let cm = mil_compile::compile(pkg, &dir.join("compiled")).expect("compile");
    let m = Model::load(&cm.path, ComputeUnits::All).expect("load");
    let data = f16_bytes(x);
    let pr = m
        .predict(&[Input {
            name: input,
            shape,
            data: &data,
            dtype: mil_spec::DType::Fp16,
        }])
        .expect("predict");
    Some(pr.outputs[0].values())
}

/// `a(m,k)·b(k,n) + c(n)` → relu → softmax(rows), in f64.
fn gemm_chain_ref(a: &[f32], b: &[f32], c: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let (aq, bq, cq) = (qh(a), qh(b), qh(c));
    let mut want = Vec::new();
    for i in 0..m {
        let row: Vec<f64> = (0..n)
            .map(|j| (cq[j] + (0..k).map(|l| aq[i * k + l] * bq[l * n + j]).sum::<f64>()).max(0.0))
            .collect();
        want.extend(softmax_row(&row));
    }
    want
}

/// Gemm→Relu→Softmax with input dims `a_dims` (strings so a dim can be
/// symbolic) and static output dims.
fn gemm_chain_model(
    a_dims: &[&str],
    y_dims: &[&str],
    k: i64,
    n: i64,
    seed: u64,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let b = randv((k * n) as usize, -1.0, 1.0, seed ^ 0xB2);
    let c = randv(n as usize, -0.5, 0.5, seed ^ 0xC3);
    let bytes = model(
        &[
            node("Gemm", &["a", "b", "c"], &["g_out"], "gemm1"),
            node("Relu", &["g_out"], &["r_out"], "relu1"),
            node("Softmax", &["r_out"], &["y"], "sm1"),
        ],
        &[t_f32("b", &[k, n], &b), t_f32("c", &[n], &c)],
        &[value_info("a", a_dims)],
        &[value_info("y", y_dims)],
    );
    (bytes, b, c)
}

#[test]
fn onnx_gemm_chain_predicts() {
    let t = Tmp::new("gemm");
    let (m, k, n) = (2usize, 4usize, 3usize);
    let (bytes, b, c) = gemm_chain_model(&["2", "4"], &["2", "3"], k as i64, n as i64, 1);
    let onnx = t.0.join("m.onnx");
    std::fs::write(&onnx, &bytes).unwrap();
    let pkg = t.0.join("m.mlpackage");
    let o = milc(&["onnx", p(&onnx), "-o", p(&pkg)]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let out = text(&o.stdout);
    for want in ["input", "a [2, 4]", "output", "y [2, 3]", "ops", "Gemm"] {
        // op names are lowered (Gemm → linear/matmul…); only check summary shape
        if want == "Gemm" {
            continue;
        }
        assert!(out.contains(want), "summary lacks {want:?}:\n{out}");
    }
    assert!(out.contains("weight.bin"), "{out}");
    let a = randv(m * k, -1.0, 1.0, 0xA1);
    let Some(got) = predict(&pkg, &t.0, "a", &[m as i64, k as i64], &a) else {
        return;
    };
    check(
        &got,
        &gemm_chain_ref(&a, &b, &c, m, k, n),
        1e-2,
        5e-3,
        "gemm→relu→softmax",
    );
}

#[test]
fn onnx_dim_param_binding() {
    let t = Tmp::new("dim");
    let (k, n) = (4usize, 3usize);
    let (bytes, b, c) = gemm_chain_model(&["batch", "4"], &["batch", "3"], k as i64, n as i64, 2);
    let onnx = t.0.join("m.onnx");
    std::fs::write(&onnx, &bytes).unwrap();
    let pkg = t.0.join("m.mlpackage");

    // unbound → actionable error naming the param
    let o = milc(&["onnx", p(&onnx), "-o", p(&pkg)]);
    assert!(!o.status.success());
    let e = text(&o.stderr);
    assert!(e.contains("'batch'") && e.contains("--dim batch=N"), "{e}");

    // typo → error listing the real params
    let o = milc(&["onnx", p(&onnx), "-o", p(&pkg), "--dim", "btach=2"]);
    assert!(!o.status.success());
    let e = text(&o.stderr);
    assert!(e.contains("btach") && e.contains("batch"), "{e}");

    // non-positive binding rejected
    let o = milc(&["onnx", p(&onnx), "-o", p(&pkg), "--dim", "batch=0"]);
    assert!(!o.status.success());
    assert!(text(&o.stderr).contains("positive"), "{}", text(&o.stderr));

    // bound → works, predicts at batch 2
    let o = milc(&["onnx", p(&onnx), "-o", p(&pkg), "--dim", "batch=2"]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let out = text(&o.stdout);
    assert!(
        out.contains("a [2, 4]") && out.contains("dims bound: batch=2"),
        "{out}"
    );
    let a = randv(2 * k, -1.0, 1.0, 0xA2);
    let Some(got) = predict(&pkg, &t.0, "a", &[2, k as i64], &a) else {
        return;
    };
    check(
        &got,
        &gemm_chain_ref(&a, &b, &c, 2, k, n),
        1e-2,
        5e-3,
        "batch=2",
    );
}

#[test]
fn classical_json_predicts() {
    let t = Tmp::new("classical");
    let json = t.0.join("glm.json");
    std::fs::write(
        &json,
        br#"{"kind":"glm_regressor","n_features":3,"input_name":"features",
             "weights":[0.5,-1.0,2.0],"intercept":[0.25]}"#,
    )
    .unwrap();
    let pkg = t.0.join("m.mlpackage");
    let o = milc(&["onnx", p(&json), "-o", p(&pkg)]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let x = randv(3, -2.0, 2.0, 0xA1);
    let Some(got) = predict(&pkg, &t.0, "features", &[3], &x) else {
        return;
    };
    let xq = qh(&x);
    check(
        &got,
        &[0.5 * xq[0] - xq[1] + 2.0 * xq[2] + 0.25],
        1e-2,
        5e-3,
        "glm",
    );
}

#[test]
fn attest_onnx_package_with_weights() {
    let t = Tmp::new("attest_w");
    // 32×16 weights = 2 KiB > inline threshold → weight.bin exists
    let (bytes, _, _) = gemm_chain_model(&["1", "32"], &["1", "16"], 32, 16, 3);
    let onnx = t.0.join("big.onnx");
    std::fs::write(&onnx, &bytes).unwrap();
    let pkg = t.0.join("m.mlpackage");
    let o = milc(&["onnx", p(&onnx), "-o", p(&pkg)]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let wbin = pkg.join("Data/com.apple.CoreML/weights/weight.bin");
    assert!(wbin.exists(), "expected a weight.bin");

    let o = milc(&["attest", p(&pkg), "--source", p(&onnx)]);
    let out = text(&o.stdout);
    assert!(o.status.success(), "{out}{}", text(&o.stderr));
    assert!(out.contains("weight.bin sha256 OK"), "{out}");
    assert!(out.contains("src big.onnx sha256 OK"), "{out}");
    assert!(out.contains("1/1 source file(s) verified"), "{out}");
    assert!(!out.contains("mil.prov.config"), "{out}");
    println!("{out}");

    // modified source → mismatch
    let mut tampered = bytes.clone();
    *tampered.last_mut().unwrap() ^= 0x01;
    let other = t.0.join("other");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(other.join("big.onnx"), &tampered).unwrap();
    let o = milc(&["attest", p(&pkg), "--source", p(&other.join("big.onnx"))]);
    assert!(!o.status.success(), "source tamper must fail");

    // flip one weight.bin byte → TAMPERED, non-zero exit
    let mut w = std::fs::read(&wbin).unwrap();
    let mid = w.len() / 2;
    w[mid] ^= 0x01;
    std::fs::write(&wbin, &w).unwrap();
    let o = milc(&["attest", p(&pkg), "--source", p(&onnx)]);
    assert!(!o.status.success(), "weight tamper must fail");
    assert!(text(&o.stdout).contains("TAMPERED"), "{}", text(&o.stdout));
}

#[test]
fn attest_onnx_package_without_weight_blob() {
    let t = Tmp::new("attest_nw");
    let (bytes, _, _) = gemm_chain_model(&["2", "4"], &["2", "3"], 4, 3, 4);
    let onnx = t.0.join("small.onnx");
    std::fs::write(&onnx, &bytes).unwrap();
    let pkg = t.0.join("m.mlpackage");
    let o = milc(&["onnx", p(&onnx), "-o", p(&pkg)]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(
        text(&o.stdout).contains("weight.bin none"),
        "{}",
        text(&o.stdout)
    );
    let wdir = pkg.join("Data/com.apple.CoreML/weights");
    assert!(!wdir.join("weight.bin").exists());

    let o = milc(&["attest", p(&pkg), "--source", p(&onnx)]);
    let out = text(&o.stdout);
    assert!(o.status.success(), "{out}{}", text(&o.stderr));
    assert!(out.contains("mil.prov.weights = none"), "{out}");
    assert!(out.contains("no weight.bin, as recorded"), "{out}");

    // a blob appearing in a package that recorded none is tampering
    std::fs::create_dir_all(&wdir).unwrap();
    std::fs::write(wdir.join("weight.bin"), b"planted").unwrap();
    let o = milc(&["attest", p(&pkg)]);
    assert!(!o.status.success());
    assert!(
        text(&o.stdout).contains("UNEXPECTED weight.bin"),
        "{}",
        text(&o.stdout)
    );
}

/// Replace the first `from` with the same-length `to` in the package spec.
fn patch_spec(pkg: &Path, from: &[u8], to: &[u8]) {
    assert_eq!(from.len(), to.len());
    let p = pkg.join("Data/com.apple.CoreML/model.mlmodel");
    let mut b = std::fs::read(&p).unwrap();
    let at = b
        .windows(from.len())
        .position(|w| w == from)
        .unwrap_or_else(|| panic!("pattern not found in spec"));
    b[at..at + to.len()].copy_from_slice(to);
    std::fs::write(&p, &b).unwrap();
}

#[test]
fn attest_detects_inline_weight_edit_in_spec() {
    let t = Tmp::new("attest_prog");
    let (bytes, b, _) = gemm_chain_model(&["2", "4"], &["2", "3"], 4, 3, 5);
    let onnx = t.0.join("small.onnx");
    std::fs::write(&onnx, &bytes).unwrap();
    let pkg = t.0.join("m.mlpackage");
    let o = milc(&["onnx", p(&onnx), "-o", p(&pkg)]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(!pkg
        .join("Data/com.apple.CoreML/weights/weight.bin")
        .exists());

    let o = milc(&["attest", p(&pkg), "--source", p(&onnx)]);
    let out = text(&o.stdout);
    assert!(o.status.success(), "{out}");
    assert!(out.contains("mil.prov.program = "), "{out}");
    assert!(out.contains("program (model.mlmodel) sha256 OK"), "{out}");
    println!("{out}");

    // flip one bit of an inline constant (the Gemm B matrix, fp16 inline)
    let needle = f16_bytes(&b[..4]);
    let mut edited = needle.clone();
    edited[0] ^= 0x01;
    patch_spec(&pkg, &needle, &edited);
    let o = milc(&["attest", p(&pkg), "--source", p(&onnx)]);
    assert!(!o.status.success(), "inline-weight edit must fail attest");
    assert!(
        text(&o.stdout).contains("TAMPERED program"),
        "{}",
        text(&o.stdout)
    );
    // the source and (absent) weight checks alone would have passed
    assert!(text(&o.stdout).contains("src small.onnx sha256 OK"));
}

#[test]
fn attest_detects_description_edit_in_onnx_package() {
    let t = Tmp::new("attest_desc");
    let (bytes, _, _) = gemm_chain_model(&["2", "4"], &["2", "3"], 4, 3, 6);
    let onnx = t.0.join("small.onnx");
    std::fs::write(&onnx, &bytes).unwrap();
    let pkg = t.0.join("m.mlpackage");
    assert!(milc(&["onnx", p(&onnx), "-o", p(&pkg)]).status.success());
    patch_spec(&pkg, b"converted by mil_onnx", b"converted bY mil_onnx");
    let o = milc(&["attest", p(&pkg)]);
    assert!(!o.status.success());
    assert!(text(&o.stdout).contains("TAMPERED program"));
}
