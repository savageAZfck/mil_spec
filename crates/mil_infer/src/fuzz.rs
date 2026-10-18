//! `fuzz` — random MIL program generator for differential testing of
//! `ANECompiler.framework`.
//!
//! Generates programs inside the empirically mapped ANE-eligible
//! dialect (see `ANE_FRONTEND_SPEC.md`), tagged with a validity hint:
//! `Valid` cases probe acceptance; `Reject` cases carry a construct
//! the spec or direct probing documents as illegal and verify the
//! frontend actually refuses it — an *accepted* reject case is a
//! validation hole in ANECompiler, a *crashed* process is a bug either
//! way.
//!
//! The driver (`milc ane fuzz`) fork-isolates every case through
//! `milc ane fuzz-worker` so a compiler crash lands as a corpus
//! artifact instead of killing the run.
//!
//! ## Dialect boundaries discovered by this fuzzer (M4, h16g)
//!
//! - `const` with `BLOBFILE` weights is rejected in every spelling
//!   (`@model_path/...`, absolute, relative, `tensor<string,[]>`-wrapped
//!   path) — including `assets/chain/c64k1.mil`, a corpus file that
//!   produced byte-identical HWX. Weight-bearing programs need the
//!   `FileConstants`/`ExternConstants` dict lane (unprobed); inline
//!   fp16 list consts are accepted.
//! - `gelu` is rejected in all spellings (`mode` first/last, bare or
//!   `tensor<string,[]>` const) despite the spec matrix listing it OK.
//! - `clip` requires fp16 alpha/beta consts; int32 bounds reject.
//! - Arg order is free; dead consts are tolerated; tiny `[1,4,1,1]`
//!   shapes pass; bare `int32`/`bool`/`string`/`fp16` and
//!   `tensor<T, []>` scalar consts are both legal.

use std::fmt::Write as _;

/// Whether the generated program is *expected* to pass the frontend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expected {
    /// Spec-conforming program; rejection is a finding.
    Valid,
    /// Carries a construct probing shows illegal *on this build*;
    /// acceptance is a validation hole (or a grammar change upstream).
    Reject,
}

/// One generated case: MIL text plus the optional weight blob to drop
/// at `weights/weight.bin` (64-byte corpus-style header + fp16 data).
#[derive(Debug)]
pub struct GenCase {
    /// Case family name (e.g. `ew_chain`, `conv1x1`, `reject_gather`).
    pub family: &'static str,
    /// The `.mil` source.
    pub mil: String,
    /// `weights/weight.bin` contents when the program uses BLOBFILE.
    pub weights: Option<Vec<u8>>,
    /// Expected frontend verdict.
    pub expected: Expected,
    /// Short descriptor of what this case probes.
    pub note: String,
}

/// Outcome classification for one executed case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Valid case accepted — baseline conformance.
    AcceptOk,
    /// Reject case rejected — expected behavior.
    RejectOk,
    /// Valid case rejected — spec gap or grammar bug (triage).
    UnexpectedReject,
    /// Reject case accepted — validation hole in ANECompiler.
    UnexpectedAccept,
    /// Worker died on a signal or nonzero exit — compiler crash.
    Crash,
    /// Worker exceeded its time budget — possible hang.
    Timeout,
}

impl Class {
    /// Case is a finding worth keeping.
    pub fn is_finding(self) -> bool {
        !matches!(self, Class::AcceptOk | Class::RejectOk)
    }
}

/// Deterministic xorshift64* — no external deps, `--seed`-reproducible.
pub struct Rng(u64);

impl Rng {
    /// Seed; 0 is folded to a fixed nonzero constant.
    pub fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    /// Next u64.
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in `0..n` (n > 0).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// Pick from a slice.
    pub fn pick<'a, T>(&mut self, v: &'a [T]) -> &'a T {
        &v[self.below(v.len() as u64) as usize]
    }

    /// Bool with probability `pct`%.
    pub fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
}

/// fp16 bytes for a deterministic pseudo-random weight tensor of `n`
/// elements (values spread over ±0.5 — safe dynamic range), with the
/// 64-byte corpus-style header BLOBFILE offsets expect.
fn weight_blob(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut blob = vec![0u8; 64];
    for _ in 0..n {
        let f = (rng.below(1024) as f32 / 1024.0) - 0.5;
        blob.extend_from_slice(&half::f16::from_f32(f).to_le_bytes());
    }
    blob
}

/// fp16 values as a MIL literal list nested to `shape`.
fn fp16_list(rng: &mut Rng, shape: &[usize]) -> String {
    fn rec(rng: &mut Rng, dims: &[usize]) -> String {
        if dims.len() == 1 {
            let vals: Vec<String> = (0..dims[0])
                .map(|_| {
                    let f = (rng.below(1024) as f32 / 1024.0) - 0.5;
                    format!("{f:.4}")
                })
                .collect();
            format!("[{}]", vals.join(", "))
        } else {
            let parts: Vec<String> = (0..dims[0]).map(|_| rec(rng, &dims[1..])).collect();
            format!("[{}]", parts.join(", "))
        }
    }
    rec(rng, shape)
}

fn shape4(rng: &mut Rng) -> (u64, u64, u64) {
    let cands = [
        (1, 16, 1),
        (1, 32, 1),
        (1, 64, 1),
        (1, 64, 4),
        (1, 128, 1),
        (2, 32, 2),
        (1, 96, 1),
        (1, 48, 2),
        (1, 4, 1),
        (4, 64, 1),
    ];
    *rng.pick(&cands)
}

fn blob_const(name: &str, shape: &str, offset: u64) -> String {
    format!(
        "tensor<fp16, {shape}> {name} = const()[val = tensor<fp16, {shape}>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({offset})))];\n"
    )
}

/// Wrap a body + output name into a full 1.3 program with `params` as
/// fp16 inputs of `in_shape`.
fn wrap(in_shape: &str, params: &[&str], body: &str, out: &str) -> String {
    let sig = params
        .iter()
        .map(|p| format!("tensor<fp16, {in_shape}> {p}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("program(1.3)\n{{\n    func main<ios18>({sig}) {{\n{body}    }} -> ({out});\n}}\n")
}

fn gen_ew_chain(rng: &mut Rng) -> GenCase {
    let (n, c, w) = shape4(rng);
    let shape = format!("[{n}, {c}, {w}, 1]");
    let ops: &[&str] = &["add", "mul", "sub", "real_div", "minimum", "maximum", "pow"];
    let n_ops = 2 + rng.below(5) as usize;
    let mut mil = String::new();
    let mut cur = "x0".to_string();
    let mut count = 0usize;
    for i in 0..n_ops {
        let op = rng.pick(ops);
        let out = format!("t{i}");
        let _ = writeln!(
            mil,
            "        tensor<fp16, {shape}> {out} = {op}(x = {cur}, y = x1)[name = string(\"{out}\")];"
        );
        cur = out;
        count += 1;
        if rng.chance(35) && count < n_ops + 4 {
            let u: &[&str] = &["sigmoid", "tanh", "silu", "exp", "sqrt", "abs"];
            let uop = rng.pick(u);
            let uout = format!("t{i}u");
            let _ = writeln!(
                mil,
                "        tensor<fp16, {shape}> {uout} = {uop}(x = {cur})[name = string(\"{uout}\")];"
            );
            cur = uout;
            count += 1;
        }
    }
    let note = format!("{count}-op elementwise chain on {shape}");
    GenCase {
        family: "ew_chain",
        mil: wrap(&shape, &["x0", "x1"], &mil, &cur),
        weights: None,
        expected: Expected::Valid,
        note,
    }
}

/// matmul with an inline fp16 weight const — small `n` keeps the MIL
/// literal readable.
fn gen_matmul(rng: &mut Rng) -> GenCase {
    let n = *rng.pick(&[4usize, 8, 16]);
    let shape = format!("[1, {n}, 1, {n}]");
    // Discovered: ANECompiler requires exactly one transpose flag —
    // tx == ty (both set or both clear) → InvalidMILProgram.
    let tx = rng.chance(50);
    let ty = !tx;
    let mut body = String::new();
    let _ = writeln!(
        body,
        "        bool txx = const()[val = bool({})];\n        bool tyy = const()[val = bool({})];",
        tx, ty
    );
    let _ = writeln!(
        body,
        "        tensor<fp16, {shape}> w = const()[val = tensor<fp16, {shape}>({})];",
        fp16_list(rng, &[1, n, 1, n])
    );
    let _ = writeln!(
        body,
        "        tensor<fp16, {shape}> z = matmul(transpose_x = txx, transpose_y = tyy, x = x0, y = w)[name = string(\"z\")];"
    );
    GenCase {
        family: "matmul",
        mil: wrap(&shape, &["x0"], &body, "z"),
        weights: None,
        expected: Expected::Valid,
        note: format!("matmul tx={tx} ty={ty} n={n} inline weights"),
    }
}

fn conv_prelude(body: &mut String) {
    body.push_str(
        "        tensor<string, []> pt = const()[val = string(\"valid\")];\n\
         \x20       tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n\
         \x20       tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n\
         \x20       tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n\
         \x20       tensor<int32, []> gr = const()[val = int32(1)];\n",
    );
}

/// conv1x1 with inline weight const.
fn gen_conv(rng: &mut Rng) -> GenCase {
    let square = rng.chance(60);
    let (cin, cout) = if square {
        let c = *rng.pick(&[4usize, 8, 16]);
        (c, c)
    } else {
        let cin = *rng.pick(&[4usize, 8, 16]);
        let cout = *rng.pick(&[4usize, 8, 16]);
        (cin, cout)
    };
    let w_dim = *rng.pick(&[1u64, 2, 4]);
    let xs = format!("[1, {cin}, {w_dim}, 1]");
    let ws = format!("[{cout}, {cin}, 1, 1]");
    let ys = format!("[1, {cout}, {w_dim}, 1]");
    let mut body = String::new();
    conv_prelude(&mut body);
    let _ = writeln!(
        body,
        "        tensor<fp16, {ws}> w = const()[val = tensor<fp16, {ws}>({})];",
        fp16_list(rng, &[cout, cin, 1, 1])
    );
    let _ = writeln!(
        body,
        "        tensor<fp16, {ys}> z = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = w, x = x0)[name = string(\"z\")];"
    );
    let family: &'static str = if square { "conv1x1" } else { "conv_ns" };
    GenCase {
        family,
        mil: wrap(&xs, &["x0"], &body, "z"),
        weights: None,
        expected: Expected::Valid,
        note: format!("conv1x1 {cin}->{cout} w={w_dim} inline"),
    }
}

/// Two-conv chain, inline weights.
fn gen_conv_chain(rng: &mut Rng) -> GenCase {
    let k = 2usize;
    let c0 = *rng.pick(&[4usize, 8]);
    let w_dim = *rng.pick(&[1u64, 2]);
    let mut body = String::new();
    conv_prelude(&mut body);
    let mut cur_c = c0;
    for i in 0..k {
        let next_c = *rng.pick(&[4usize, 8]);
        let ws = format!("[{next_c}, {cur_c}, 1, 1]");
        let _ = writeln!(
            body,
            "        tensor<fp16, {ws}> w{i} = const()[val = tensor<fp16, {ws}>({})];",
            fp16_list(rng, &[next_c, cur_c, 1, 1])
        );
        let inp = if i == 0 {
            "x0".to_string()
        } else {
            format!("z{}", i - 1)
        };
        let _ = writeln!(
            body,
            "        tensor<fp16, [1, {next_c}, {w_dim}, 1]> z{i} = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = w{i}, x = {inp})[name = string(\"z{i}\")];"
        );
        cur_c = next_c;
    }
    GenCase {
        family: "conv_chain",
        mil: wrap(
            &format!("[1, {c0}, {w_dim}, 1]"),
            &["x0"],
            &body,
            &format!("z{}", k - 1),
        ),
        weights: None,
        expected: Expected::Valid,
        note: format!("{k}-conv chain c0={c0} w={w_dim} inline"),
    }
}

fn gen_layout(rng: &mut Rng) -> GenCase {
    let (n, c, w) = shape4(rng);
    let shape = format!("[{n}, {c}, {w}, 1]");
    let mut body = String::new();
    let mut cur = "x0".to_string();
    let mut cur_shape = shape.clone();
    let kind = rng.below(3);
    match kind {
        0 => {
            let _ = writeln!(
                body,
                "        tensor<int32, [3]> sh = const()[val = tensor<int32, [3]>([{n}, {c}, {w}])];"
            );
            let _ = writeln!(
                body,
                "        tensor<fp16, {shape}> r = reshape(shape = sh, x = {cur})[name = string(\"r\")];"
            );
            cur = "r".into();
            cur_shape = format!("[{n}, {c}, {w}]");
        }
        1 => {
            let _ = writeln!(
                body,
                "        tensor<int32, [4]> p1 = const()[val = tensor<int32, [4]>([0, 1, 3, 2])];\n\
                 \x20       tensor<int32, [4]> p2 = const()[val = tensor<int32, [4]>([0, 1, 3, 2])];"
            );
            let _ = writeln!(
                body,
                "        tensor<fp16, {shape}> t = transpose(perm = p1, x = {cur})[name = string(\"t\")];"
            );
            let _ = writeln!(
                body,
                "        tensor<fp16, {shape}> t2 = transpose(perm = p2, x = t)[name = string(\"t2\")];"
            );
            cur = "t2".into();
        }
        _ => {
            let _ = writeln!(
                body,
                "        int32 ax = const()[val = int32(1)];\n        bool il = const()[val = bool(false)];"
            );
            let _ = writeln!(
                body,
                "        tensor<fp16, [{n}, {}, {w}, 1]> cc = concat(axis = ax, interleave = il, values = ({cur}, x1))[name = string(\"cc\")];",
                c * 2
            );
            cur = "cc".into();
            cur_shape = format!("[{n}, {}, {w}, 1]", c * 2);
        }
    }
    let sig_in = if kind == 2 {
        &["x0", "x1"][..]
    } else {
        &["x0"][..]
    };
    let sig = sig_in
        .iter()
        .map(|p| format!("tensor<fp16, {shape}> {p}"))
        .collect::<Vec<_>>()
        .join(", ");
    GenCase {
        family: "layout",
        mil: format!(
            "program(1.3)\n{{\n    func main<ios18>({sig}) {{\n{body}    }} -> ({cur});\n}}\n"
        ),
        weights: None,
        expected: Expected::Valid,
        note: format!("layout op kind={kind} on {shape}"),
    }
}

fn gen_softmax_reduce(rng: &mut Rng) -> GenCase {
    let (n, c, w) = shape4(rng);
    let shape = format!("[{n}, {c}, {w}, 1]");
    let mut body = String::new();
    let kind = rng.below(2);
    let out;
    if kind == 0 {
        let _ = writeln!(body, "        int32 ax = const()[val = int32(-1)];");
        let _ = writeln!(
            body,
            "        tensor<fp16, {shape}> z = softmax(axis = ax, x = x0)[name = string(\"z\")];"
        );
        out = "z";
    } else {
        let _ = writeln!(
            body,
            "        tensor<int32, [1]> ax = const()[val = tensor<int32, [1]>([1])];\n        bool kd = const()[val = bool(true)];"
        );
        let _ = writeln!(
            body,
            "        tensor<fp16, {shape}> z = reduce_mean(axes = ax, keep_dims = kd, x = x0)[name = string(\"z\")];"
        );
        out = "z";
    }
    GenCase {
        family: if kind == 0 { "softmax" } else { "reduce" },
        mil: wrap(&shape, &["x0"], &body, out),
        weights: None,
        expected: Expected::Valid,
        note: format!(
            "{} on {shape}",
            if kind == 0 {
                "softmax(-1)"
            } else {
                "reduce_mean(ax=1)"
            }
        ),
    }
}

/// clip with fp16 scalar bounds + rsqrt with fp16 epsilon + cast —
/// the fp16-const attribute ops this build accepts.
fn gen_fp_attr(rng: &mut Rng) -> GenCase {
    let (n, c, w) = shape4(rng);
    let shape = format!("[{n}, {c}, {w}, 1]");
    let kind = rng.below(3);
    let mut body = String::new();
    let (family, note) = match kind {
        0 => {
            let lo = -1.5 + rng.below(100) as f32 / 100.0;
            let hi = 0.5 + rng.below(200) as f32 / 100.0;
            let _ = writeln!(
                body,
                "        fp16 lo = const()[val = fp16({lo:.2})];\n        fp16 hi = const()[val = fp16({hi:.2})];"
            );
            let _ = writeln!(
                body,
                "        tensor<fp16, {shape}> z = clip(alpha = lo, beta = hi, x = x0)[name = string(\"z\")];"
            );
            ("clip", format!("clip({lo:.2},{hi:.2}) on {shape}"))
        }
        1 => {
            let _ = writeln!(body, "        fp16 eps = const()[val = fp16(0.001)];");
            let _ = writeln!(
                body,
                "        tensor<fp16, {shape}> z = rsqrt(epsilon = eps, x = x0)[name = string(\"z\")];"
            );
            ("rsqrt", format!("rsqrt on {shape}"))
        }
        _ => {
            let _ = writeln!(body, "        string dt = const()[val = string(\"fp32\")];");
            let _ = writeln!(
                body,
                "        tensor<fp32, {shape}> z = cast(dtype = dt, x = x0)[name = string(\"z\")];"
            );
            ("cast", format!("cast fp16->fp32 on {shape}"))
        }
    };
    GenCase {
        family,
        mil: wrap(&shape, &["x0"], &body, "z"),
        expected: Expected::Valid,
        weights: None,
        note,
    }
}

fn gen_reject(rng: &mut Rng, id: usize) -> GenCase {
    let kind = id % 6;
    let (n, c, w) = shape4(rng);
    let shape = format!("[{n}, {c}, {w}, 1]");
    match kind {
        0 => {
            // int32 tensor compute — spec: always InvalidMILProgram
            let body = "        tensor<int32, [1]> one = const()[val = tensor<int32, [1]>([1])];\n"
                .to_string()
                + "        tensor<int32, [1, 4]> z = add(x = xi, y = one)[name = string(\"z\")];\n";
            GenCase {
                family: "reject_int_compute",
                mil: format!(
                    "program(1.3)\n{{\n    func main<ios18>(tensor<int32, [1, 4]> xi) {{\n{body}    }} -> (z);\n}}\n"
                ),
                weights: None,
                expected: Expected::Reject,
                note: "int32 add — non-float compute".into(),
            }
        }
        1 => {
            // gather — spec: rejected in all spellings
            let body = "        tensor<int32, [2]> idx = const()[val = tensor<int32, [2]>([0, 1])];\n".to_string()
                + &format!(
                    "        tensor<fp16, {shape}> z = gather(x = x0, indices = idx)[name = string(\"z\")];\n"
                );
            GenCase {
                family: "reject_gather",
                mil: wrap(&shape, &["x0"], &body, "z"),
                weights: None,
                expected: Expected::Reject,
                note: "gather — rejected per spec".into(),
            }
        }
        2 => {
            // dynamic slice bound — begin comes from a runtime input,
            // not a compile-time const (spec: indices must be const)
            let mil = format!(
                "program(1.3)\n{{\n    func main<ios18>(tensor<fp16, {shape}> x0, tensor<int32, [4]> bgn, tensor<int32, [4]> end, tensor<int32, [4]> st) {{\n        tensor<int32, [4]> bm = const()[val = tensor<int32, [4]>([1, 1, 1, 1])];\n        tensor<int32, [4]> em = const()[val = tensor<int32, [4]>([1, 1, 1, 1])];\n        tensor<int32, [4]> sq = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n        tensor<fp16, [1, 4, 1, 1]> z = slice_by_index(begin = bgn, end = end, stride = st, begin_mask = bm, end_mask = em, squeeze_mask = sq, x = x0)[name = string(\"z\")];\n    }} -> (z);\n}}\n"
            );
            GenCase {
                family: "reject_dyn_slice",
                mil,
                weights: None,
                expected: Expected::Reject,
                note: "dynamic slice begin — runtime index".into(),
            }
        }
        3 => {
            // zero-extent slice — begin == end
            let body = "        tensor<int32, [4]> b = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n".to_string()
                + "        tensor<int32, [4]> e = const()[val = tensor<int32, [4]>([1, 4, 0, 0])];\n"
                + "        tensor<int32, [4]> s = const()[val = tensor<int32, [4]>([1, 1, 1, 1])];\n"
                + "        tensor<int32, [4]> bm = const()[val = tensor<int32, [4]>([1, 1, 1, 1])];\n"
                + "        tensor<int32, [4]> em = const()[val = tensor<int32, [4]>([1, 1, 1, 1])];\n"
                + "        tensor<int32, [4]> sq = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
                + "        tensor<fp16, [1, 4, 1, 1]> z = slice_by_index(begin = b, end = e, stride = s, begin_mask = bm, end_mask = em, squeeze_mask = sq, x = x0)[name = string(\"z\")];\n";
            GenCase {
                family: "reject_zero_extent",
                mil: wrap(&shape, &["x0"], &body, "z"),
                weights: None,
                expected: Expected::Reject,
                note: "zero-extent slice".into(),
            }
        }
        4 => {
            // BLOBFILE const — rejected in every spelling on this
            // build (weights need the FileConstants lane, unprobed).
            // Kept as a Reject probe: acceptance would mean the
            // frontend opened embedded weight blobs.
            let mut body = String::new();
            conv_prelude(&mut body);
            body.push_str(&blob_const("w", "[4, 4, 1, 1]", 64));
            let _ = writeln!(
                body,
                "        tensor<fp16, [1, 4, 1, 1]> z = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = w, x = x0)[name = string(\"z\")];"
            );
            GenCase {
                family: "blobfile_probe",
                mil: wrap("[1, 4, 1, 1]", &["x0"], &body, "z"),
                weights: Some(weight_blob(rng, 16)),
                expected: Expected::Reject,
                note: "BLOBFILE const — rejected on this build".into(),
            }
        }
        _ => {
            // gelu — rejected in every spelling on this build.
            let mut body = String::new();
            let _ = writeln!(
                body,
                "        tensor<string, []> md = const()[val = string(\"tanh\")];"
            );
            let _ = writeln!(
                body,
                "        tensor<fp16, {shape}> z = gelu(mode = md, x = x0)[name = string(\"z\")];"
            );
            GenCase {
                family: "reject_gelu",
                mil: wrap(&shape, &["x0"], &body, "z"),
                weights: None,
                expected: Expected::Reject,
                note: "gelu — rejected on this build".into(),
            }
        }
    }
}

/// Boundary-stress programs — legal grammar pushed to extremes that
/// tend to crash plan builders: huge dims, long chains, aggressive
/// stride/pad/dilation/groups, unusual ranks.
fn gen_stress(rng: &mut Rng) -> GenCase {
    match rng.below(6) {
        0 => {
            // large channel count elementwise
            let c = *rng.pick(&[256u64, 512, 1024, 2048]);
            let shape = format!("[1, {c}, 1, 1]");
            let mut body = String::new();
            let mut cur = "x0".to_string();
            for i in 0..4 {
                let out = format!("t{i}");
                let _ = writeln!(
                    body,
                    "        tensor<fp16, {shape}> {out} = {}(x = {cur}, y = x1)[name = string(\"{out}\")];",
                    rng.pick(&["add", "mul", "sub"])
                );
                cur = out;
            }
            GenCase {
                family: "stress_bigchan",
                mil: wrap(&shape, &["x0", "x1"], &body, &cur),
                weights: None,
                expected: Expected::Valid,
                note: format!("c={c} elementwise chain"),
            }
        }
        1 => {
            // long op chain — plan-builder stress
            let shape = "[1, 64, 1, 1]";
            let n_ops = 12 + rng.below(20) as usize;
            let mut body = String::new();
            let mut cur = "x0".to_string();
            for i in 0..n_ops {
                let out = format!("t{i}");
                let _ = writeln!(
                    body,
                    "        tensor<fp16, {shape}> {out} = {}(x = {cur}, y = x1)[name = string(\"{out}\")];",
                    rng.pick(&["add", "mul", "sub", "minimum", "maximum"])
                );
                cur = out;
            }
            GenCase {
                family: "stress_longchain",
                mil: wrap(shape, &["x0", "x1"], &body, &cur),
                weights: None,
                expected: Expected::Valid,
                note: format!("{n_ops}-op elementwise chain"),
            }
        }
        2 => {
            // conv with aggressive stride/dilation/pad/groups
            let c = *rng.pick(&[8usize, 16]);
            let w_dim = *rng.pick(&[4u64, 8]);
            let stride = *rng.pick(&[1u64, 2]);
            let dil = *rng.pick(&[1u64, 2]);
            let pad = *rng.pick(&[0u64, 1, 2]);
            let ws = format!("[{c}, {c}, 1, 1]");
            let mut body = String::new();
            let _ = writeln!(
                body,
                "        tensor<string, []> pt = const()[val = string(\"valid\")];\n\
                 \x20       tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([{stride}, {stride}])];\n\
                 \x20       tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([{pad}, {pad}, {pad}, {pad}])];\n\
                 \x20       tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([{dil}, {dil}])];\n\
                 \x20       tensor<int32, []> gr = const()[val = int32(1)];"
            );
            let _ = writeln!(
                body,
                "        tensor<fp16, {ws}> w = const()[val = tensor<fp16, {ws}>({})];",
                fp16_list(rng, &[c, c, 1, 1])
            );
            // output spatial shrinks under valid pad+stride
            let ow = (w_dim + 2 * pad - dil * (1 - 1) - 1) / stride + 1;
            let _ = writeln!(
                body,
                "        tensor<fp16, [1, {c}, {ow}, 1]> z = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = w, x = x0)[name = string(\"z\")];"
            );
            GenCase {
                family: "stress_conv_params",
                mil: wrap(&format!("[1, {c}, {w_dim}, 1]"), &["x0"], &body, "z"),
                weights: None,
                expected: Expected::Valid,
                note: format!("conv stride={stride} dil={dil} pad={pad} w={w_dim}->{ow}"),
            }
        }
        3 => {
            // grouped conv — groups > 1
            let groups = *rng.pick(&[2u64, 4]);
            let cin = 4 * groups as usize;
            let cout = 4 * groups as usize;
            let ws = format!("[{cout}, {}, 1, 1]", cin / groups as usize);
            let mut body = String::new();
            let _ = writeln!(
                body,
                "        tensor<string, []> pt = const()[val = string(\"valid\")];\n\
                 \x20       tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];\n\
                 \x20       tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];\n\
                 \x20       tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1, 1])];\n\
                 \x20       tensor<int32, []> gr = const()[val = int32({groups})];"
            );
            let _ = writeln!(
                body,
                "        tensor<fp16, {ws}> w = const()[val = tensor<fp16, {ws}>({})];",
                fp16_list(rng, &[cout, cin / groups as usize, 1, 1])
            );
            let _ = writeln!(
                body,
                "        tensor<fp16, [1, {cout}, 1, 1]> z = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = w, x = x0)[name = string(\"z\")];"
            );
            GenCase {
                family: "stress_grouped",
                mil: wrap(&format!("[1, {cin}, 1, 1]"), &["x0"], &body, "z"),
                weights: None,
                expected: Expected::Valid,
                note: format!("grouped conv groups={groups} {cin}->{cout}"),
            }
        }
        4 => {
            // conv weight/input channel mismatch — semantic check:
            // must reject cleanly, not crash
            let cin = *rng.pick(&[4usize, 8]);
            let wcin = cin + 4;
            let ws = format!("[4, {wcin}, 1, 1]");
            let mut body = String::new();
            conv_prelude(&mut body);
            let _ = writeln!(
                body,
                "        tensor<fp16, {ws}> w = const()[val = tensor<fp16, {ws}>({})];",
                fp16_list(rng, &[4, wcin, 1, 1])
            );
            let _ = writeln!(
                body,
                "        tensor<fp16, [1, 4, 1, 1]> z = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = w, x = x0)[name = string(\"z\")];"
            );
            GenCase {
                family: "stress_mismatch",
                mil: wrap(&format!("[1, {cin}, 1, 1]"), &["x0"], &body, "z"),
                weights: None,
                expected: Expected::Reject,
                note: format!("conv cin={cin} vs weight cin={wcin}"),
            }
        }
        _ => {
            // wide spatial + batch simultaneously
            let (n, c, w) = (
                *rng.pick(&[4u64, 8]),
                *rng.pick(&[64u64, 128]),
                *rng.pick(&[16u64, 64]),
            );
            let shape = format!("[{n}, {c}, {w}, 1]");
            let mut body = String::new();
            let _ = writeln!(
                body,
                "        tensor<fp16, {shape}> z = {}(x = x0, y = x1)[name = string(\"z\")];",
                rng.pick(&["add", "mul", "sub"])
            );
            GenCase {
                family: "stress_wide",
                mil: wrap(&shape, &["x0", "x1"], &body, "z"),
                weights: None,
                expected: Expected::Valid,
                note: format!("wide shape {shape}"),
            }
        }
    }
}

/// Take a valid generated program and corrupt its text — truncations,
/// spliced spans, token drops. The parser's only correct answer is a
/// clean rejection; acceptance is a hole, a crash is a bug.
fn gen_mutation(rng: &mut Rng, id: usize) -> GenCase {
    let mut base = match id % 4 {
        0 => gen_ew_chain(rng),
        1 => gen_conv(rng),
        2 => gen_layout(rng),
        _ => gen_softmax_reduce(rng),
    };
    let bytes = base.mil.as_bytes();
    let kind = rng.below(4);
    let (mutated, desc) = match kind {
        0 => {
            // truncate at a random byte
            let cut = 8 + rng.below(bytes.len().saturating_sub(16) as u64) as usize;
            (base.mil[..cut].to_string(), format!("truncate@{cut}"))
        }
        1 => {
            // delete a random span
            let lo = 16 + rng.below(bytes.len().saturating_sub(48) as u64) as usize;
            let hi = (lo + 8 + rng.below(32) as usize).min(bytes.len());
            (
                format!("{}{}", &base.mil[..lo], &base.mil[hi..]),
                format!("delete[{lo}..{hi}]"),
            )
        }
        2 => {
            // delete the Nth occurrence of a delimiter — guaranteed hit
            let dels = [b')', b';', b'{', b'}', b'[', b']', b'"', b'='];
            let d = *rng.pick(&dels);
            let mut hits: Vec<usize> = bytes
                .iter()
                .enumerate()
                .filter(|(_, &b)| b == d)
                .map(|(i, _)| i)
                .collect();
            let (v, desc) = if hits.is_empty() {
                (bytes.to_vec(), format!("no {d} present"))
            } else {
                let pos = hits[rng.below(hits.len() as u64) as usize];
                hits.clear();
                let mut v = bytes.to_vec();
                v.remove(pos);
                (v, format!("delete {d:?}@{pos}"))
            };
            (String::from_utf8_lossy(&v).to_string(), desc)
        }
        _ => {
            // corrupt a guaranteed-present keyword
            let (from, to) = *rng.pick(&[
                ("func main", "func frob"),
                ("program(1.3)", "program(9.9)"),
                ("tensor<", "tensor<<"),
                ("main<ios18>", "main<ios99>"),
                ("const()", "const("),
            ]);
            let s = base.mil.replacen(from, to, 1);
            (s, format!("splice {from:?} -> {to:?}"))
        }
    };
    base.family = "mutant";
    base.mil = mutated;
    base.expected = Expected::Reject;
    base.note = format!("mutated {}: {desc}", base.note);
    base
}

/// Generate case `id` — cycles families, ~20% reject probes, ~20%
/// boundary stress, ~15% parser mutations.
pub fn gen_program(rng: &mut Rng, id: usize) -> GenCase {
    if rng.chance(15) {
        return gen_mutation(rng, id);
    }
    if rng.chance(20) {
        return gen_reject(rng, id);
    }
    if rng.chance(25) {
        return gen_stress(rng);
    }
    match id % 6 {
        0 => gen_ew_chain(rng),
        1 => gen_matmul(rng),
        2 => gen_conv(rng),
        3 => gen_conv_chain(rng),
        4 => gen_layout(rng),
        _ => match rng.below(2) {
            0 => gen_softmax_reduce(rng),
            _ => gen_fp_attr(rng),
        },
    }
}

/// Classify a completed worker outcome.
pub fn classify(expected: Expected, accepted: bool, crashed: bool, timed_out: bool) -> Class {
    if crashed {
        return Class::Crash;
    }
    if timed_out {
        return Class::Timeout;
    }
    match (expected, accepted) {
        (Expected::Valid, true) => Class::AcceptOk,
        (Expected::Valid, false) => Class::UnexpectedReject,
        (Expected::Reject, true) => Class::UnexpectedAccept,
        (Expected::Reject, false) => Class::RejectOk,
    }
}
