//! apple_mil — translate a coyote-dialect net.mil into ONE valid Apple
//! MIL program that aned compiles for the local Neural Engine.
//!
//! Why: on M1-class chips (H13, also H14/H15) aned refuses third-party
//! precompiled HWX (ANEProgramCreate() 0x17), so the per-node HWX
//! patching in crate::graph cannot load. aned *does* compile a plain-text
//! MIL program from a directory (kANEFModelType = kANEFModelMIL), so the
//! whole coyote graph becomes one Apple-MIL function instead.
//!
//! Coyote's dialect uses shape tags as layout hints ([1,N,1,1] strided vs
//! [1,1,1,N] contiguous vectors, [1,A,1,B] / [R,C] matrices) and leans on
//! operand kinds for matmul meaning. Here every value gets a canonical
//! rank-4 shape: vectors [1,1,1,N], matrices [1,1,R,C].
//!   matmul(vec, vec, transpose_x)      -> outer product [A,B]
//!   matmul(vec, mat, transpose_y)      -> x · Wᵀ
//!   matmul(vec, mat[, transpose_x])    -> x · W (transpose_x is coyote's
//!                                         tag for "δ·W", ignored)
//!   rms_norm(x, w, eps)                -> mul/reduce_mean/rsqrt/mul/mul
//! Function inputs keep their declared shapes (host staging unchanged)
//! and reshape to canonical; outputs reshape back to their declared shape
//! (matrices to [1,R,1,C]). BLOBFILE consts are baked into
//! weights/weight.bin (Apple blob format v2) — no per-eval staging.

use crate::hwx;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Canonical 2-D view: r == 1 is a vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sh {
    r: i64,
    c: i64,
}

impl Sh {
    fn vec(n: i64) -> Self {
        Sh { r: 1, c: n }
    }
    fn is_vec(self) -> bool {
        self.r == 1
    }
    fn numel(self) -> i64 {
        self.r * self.c
    }
    fn mil(self) -> String {
        format!("[1, 1, {}, {}]", self.r, self.c)
    }
}

/// Coyote shape tag -> canonical view. Unit dims drop out; at most two
/// non-unit dims are allowed (vector or matrix).
fn canon(dims: &[i64]) -> Result<Sh, String> {
    let nu: Vec<i64> = dims.iter().copied().filter(|d| *d != 1).collect();
    match nu.len() {
        0 => Ok(Sh::vec(1)),
        1 => Ok(Sh::vec(nu[0])),
        2 => Ok(Sh { r: nu[0], c: nu[1] }),
        _ => Err(format!("shape {dims:?}: more than two non-unit dims")),
    }
}

fn dims_mil(d: &[i64]) -> String {
    let s: Vec<String> = d.iter().map(|v| v.to_string()).collect();
    format!("[{}]", s.join(", "))
}

/// Shape a function output is written back as: the declared tag when it
/// holds the same element count and kind, else [1,N,1,1] (vectors) /
/// [1,R,1,C] (matrices) — the layouts coyote's host readers expect
/// (row on c, col on w).
fn out_dims(declared: Option<&Vec<i64>>, s: Sh) -> Vec<i64> {
    if let Some(d) = declared {
        let n: i64 = d.iter().product();
        let rank4_flat = d.len() == 4 && d[0] == 1 && d[2] == 1;
        if n == s.numel() && rank4_flat && (s.is_vec() || (d[1] == s.r && d[3] == s.c)) {
            return d.clone();
        }
    }
    if s.is_vec() {
        vec![1, s.c, 1, 1]
    } else {
        vec![1, s.r, 1, s.c]
    }
}

/// A const() declaration from the coyote text.
struct ConstDecl {
    dtype: String,
    dims: Vec<i64>,
    /// Some((path, offset)) for BLOBFILE consts.
    blob: Option<(String, u64)>,
    /// The raw "val = ..." expression for inline consts.
    val: String,
}

fn parse_consts(text: &str) -> HashMap<String, ConstDecl> {
    let mut out = HashMap::new();
    for raw in text.lines() {
        let l = raw.trim();
        let Some(eq) = l.find(" = const()") else {
            continue;
        };
        let lhs = &l[..eq];
        let Some(name) = lhs.rsplit(' ').next() else {
            continue;
        };
        let dtype = lhs
            .strip_prefix("tensor<")
            .and_then(|s| s.split([',', '>']).next())
            .unwrap_or("")
            .trim()
            .to_string();
        let dims: Vec<i64> = lhs
            .find('[')
            .zip(lhs.find(']'))
            .map(|(a, b)| {
                lhs[a + 1..b]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect()
            })
            .unwrap_or_default();
        const BLOB_PFX: &str = "BLOBFILE(path = string(\"";
        const OFF_PFX: &str = "offset = uint64(";
        let blob = l.find(BLOB_PFX).map(|p| {
            let rest = &l[p + BLOB_PFX.len()..];
            let path = rest[..rest.find('"').unwrap_or(0)].to_string();
            let off = rest
                .find(OFF_PFX)
                .and_then(|o| {
                    let r = &rest[o + OFF_PFX.len()..];
                    r[..r.find(')')?].trim().parse().ok()
                })
                .unwrap_or(0);
            (path, off)
        });
        let val = l
            .find("val = ")
            .map(|p| {
                let v = &l[p + 6..];
                v.strip_suffix("];").unwrap_or(v).to_string()
            })
            .unwrap_or_default();
        out.insert(
            name.to_string(),
            ConstDecl {
                dtype,
                dims,
                blob,
                val,
            },
        );
    }
    out
}

/// Names in the "-> (a, b, c)" return list.
fn func_outputs(text: &str) -> Vec<String> {
    let Some(p) = text.rfind("-> (") else {
        return Vec::new();
    };
    let rest = &text[p + 4..];
    rest[..rest.find(')').unwrap_or(0)]
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Bool const value (__bt style) or inline bool(true).
fn bool_of(consts: &HashMap<String, ConstDecl>, v: Option<&str>) -> bool {
    let Some(v) = v else { return false };
    let lit = consts.get(v).map(|c| c.val.as_str()).unwrap_or(v);
    lit.contains("true")
}

/// Float of an inline literal (fp16(0.00001)) or an inline const name.
fn f32_of(consts: &HashMap<String, ConstDecl>, v: &str) -> Option<f32> {
    let lit = consts.get(v).map(|c| c.val.as_str()).unwrap_or(v);
    let inner = lit
        .find('(')
        .map(|a| lit[a + 1..].trim_end_matches(')'))
        .unwrap_or(lit);
    inner.trim().parse().ok()
}

/// Apple MIL blob storage v2: 64 B file header, then per blob a 64 B
/// metadata record (sentinel 0xDEADBEEF, dtype, size, data offset) and
/// the 64-aligned payload. Returns the metadata offset to cite.
fn push_blob(w: &mut Vec<u8>, data: &[u8]) -> u64 {
    const ALIGN: usize = 64;
    const FP16: u32 = 1;
    const SENTINEL: u32 = 0xDEAD_BEEF;
    let meta = w.len();
    let data_off = meta + ALIGN;
    let mut rec = vec![0u8; ALIGN];
    rec[0..4].copy_from_slice(&SENTINEL.to_le_bytes());
    rec[4..8].copy_from_slice(&FP16.to_le_bytes());
    rec[8..16].copy_from_slice(&(data.len() as u64).to_le_bytes());
    rec[16..24].copy_from_slice(&(data_off as u64).to_le_bytes());
    w.extend_from_slice(&rec);
    w.extend_from_slice(data);
    let pad = (ALIGN - w.len() % ALIGN) % ALIGN;
    w.resize(w.len() + pad, 0);
    let count = u32::from_le_bytes(w[0..4].try_into().unwrap()) + 1;
    w[0..4].copy_from_slice(&count.to_le_bytes());
    meta as u64
}

/// The translated program: MIL text, weight blob, and its IO names.
pub struct AppleMil {
    pub text: String,
    pub weights: Vec<u8>,
    /// Function inputs in signature order.
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// `(output, input)` pairs written back in place after each eval:
    /// packed 4-D `add` accumulators on a function input (KV caches,
    /// SGD weights) — parity with the HWX path's out0->in0 alias.
    pub inplace: Vec<(String, String)>,
}

/// Emitter state: MIL lines plus each value's canonical shape.
struct Em {
    body: String,
    shapes: HashMap<String, Sh>,
    /// Internal (canonical) MIL name of each coyote value.
    alias: HashMap<String, String>,
    n_tmp: usize,
    consts_emitted: HashMap<String, String>,
}

impl Em {
    fn line(&mut self, s: String) {
        self.body.push_str("        ");
        self.body.push_str(&s);
        self.body.push('\n');
    }
    fn tmp(&mut self, stem: &str) -> String {
        self.n_tmp += 1;
        format!("_t{}_{stem}", self.n_tmp)
    }
    /// Named scalar/shape const, emitted once per distinct literal.
    fn konst(&mut self, ty: &str, val: &str) -> String {
        let key = format!("{ty}|{val}");
        if let Some(n) = self.consts_emitted.get(&key) {
            return n.clone();
        }
        let n = format!("_k{}", self.consts_emitted.len());
        self.line(format!(
            "{ty} {n} = const()[name = string(\"{n}\"), val = {val}];"
        ));
        self.consts_emitted.insert(key, n.clone());
        n
    }
    fn bool_k(&mut self, b: bool) -> String {
        self.konst("tensor<bool, []>", &format!("bool({b})"))
    }
    fn shape_k(&mut self, d: &[i64]) -> String {
        let ty = format!("tensor<int32, [{}]>", d.len());
        let val = format!("tensor<int32, [{}]>({})", d.len(), dims_mil(d));
        self.konst(&ty, &val)
    }
    fn val(&self, v: &str) -> Result<(String, Sh), String> {
        let n = self.alias.get(v).ok_or_else(|| {
            format!("operand {v}: not defined (unsupported const or forward ref)")
        })?;
        Ok((n.clone(), self.shapes[n]))
    }
    fn def(&mut self, coyote: &str, mil: String, s: Sh) {
        self.shapes.insert(mil.clone(), s);
        self.alias.insert(coyote.to_string(), mil);
    }
}

fn arg<'a>(o: &'a hwx::MilOp, k: &str) -> Option<&'a str> {
    o.args.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}

fn bcast(a: Sh, b: Sh, op: &str) -> Result<Sh, String> {
    let d = |x: i64, y: i64| -> Result<i64, String> {
        if x == y || y == 1 {
            Ok(x)
        } else if x == 1 {
            Ok(y)
        } else {
            Err(format!("{op}: cannot broadcast {a:?} with {b:?}"))
        }
    };
    Ok(Sh {
        r: d(a.r, b.r)?,
        c: d(a.c, b.c)?,
    })
}

const BINARY: &[&str] = &["add", "sub", "mul", "real_div", "maximum", "minimum", "pow"];
const UNARY: &[&str] = &[
    "sigmoid", "tanh", "relu", "silu", "exp", "sqrt", "abs", "sin", "cos",
];
/// Blob file name aned resolves against the program dir.
const WEIGHT_REF: &str = "@model_path/weights/weight.bin";
const DEFAULT_RMS_EPS: f32 = 1e-5;

/// Translate coyote MIL text (blob consts resolved against mil_dir).
pub fn translate(text: &str, mil_dir: &Path) -> Result<AppleMil, String> {
    let (mut ops, _) = hwx::mil_ops(text);
    let sigs = hwx::func_input_shapes(text);
    // In-place parity with the HWX runtime: there a packed 4-D `add` on a
    // function input X writes X's surface in place, and programs run in
    // MIL order, so every later reader of X sees the new value (e.g.
    // decode attends over the cache after `kv.k[pos] = ...`). One Apple
    // program has no such side effect, so later reads of X are rewritten
    // to the add's result here; the last such result is written back.
    let mut cur: HashMap<String, String> = HashMap::new();
    let mut inplace_src: Vec<(String, String)> = Vec::new(); // (result, input)
    for o in ops.iter_mut() {
        let x0 = arg(o, "x").map(str::to_string);
        for (_, v) in o.args.iter_mut() {
            if let Some(n) = cur.get(v.as_str()) {
                *v = n.clone();
            }
        }
        let packed = o.op == "add" && o.shape.len() == 4 && o.shape[1] > 1 && o.shape[3] > 1;
        if let Some(x) = x0.filter(|x| packed && sigs.iter().any(|(n, _)| n == x)) {
            cur.insert(x.clone(), o.result.clone());
            inplace_src.push((o.result.clone(), x));
        }
    }
    let consts = parse_consts(text);
    let outputs = func_outputs(text);
    if outputs.is_empty() {
        return Err("no return list in func main".into());
    }
    let declared: HashMap<String, Vec<i64>> = sigs
        .iter()
        .cloned()
        .chain(ops.iter().map(|o| (o.result.clone(), o.shape.clone())))
        .collect();
    let mut em = Em {
        body: String::new(),
        shapes: HashMap::new(),
        alias: HashMap::new(),
        n_tmp: 0,
        consts_emitted: HashMap::new(),
    };
    // Liveness from the outputs: aned's compiler fails the whole program
    // (CompilationFailure) on any dead op, e.g. an unused gradient.
    let mut live: HashSet<&str> = outputs.iter().map(|s| s.as_str()).collect();
    for o in ops.iter().rev() {
        if live.contains(o.result.as_str()) {
            live.extend(o.args.iter().map(|(_, v)| v.as_str()));
        }
    }
    // weight blob: 64 B file header (count patched per blob, version 2)
    let mut weights = vec![0u8; 64];
    weights[4..8].copy_from_slice(&2u32.to_le_bytes());

    // function inputs -> canonical (live ones only; unused inputs are fine)
    for (name, dims) in sigs.iter().filter(|(n, _)| live.contains(n.as_str())) {
        let s = canon(dims)?;
        let t = format!("_c_{name}");
        let sk = em.shape_k(&[1, 1, s.r, s.c]);
        em.line(format!(
            "tensor<fp16, {}> {t} = reshape(x = {name}, shape = {sk})[name = string(\"{t}\")];",
            s.mil()
        ));
        em.def(name, t, s);
    }
    // blob consts -> baked canonical consts (live ones only)
    let mut blob_names: Vec<&String> = consts
        .iter()
        .filter(|(n, c)| c.blob.is_some() && live.contains(n.as_str()))
        .map(|(n, _)| n)
        .collect();
    blob_names.sort();
    for name in blob_names {
        let c = &consts[name];
        if c.dtype != "fp16" {
            return Err(format!("const {name}: dtype {} (only fp16 blobs)", c.dtype));
        }
        let (path, off) = c.blob.clone().unwrap();
        let s = canon(&c.dims)?;
        let nbytes = s.numel() as usize * 2;
        let raw = hwx::load_blob_bytes_at(&hwx::resolve_blob(mil_dir, &path), off, nbytes)?;
        if raw.len() < nbytes {
            return Err(format!(
                "const {name}: blob {path} holds {} B, need {nbytes}",
                raw.len()
            ));
        }
        let meta = push_blob(&mut weights, &raw[..nbytes]);
        let t = format!("_w_{name}");
        em.line(format!(
            "tensor<fp16, {sh}> {t} = const()[name = string(\"{t}\"), val = tensor<fp16, {sh}>(BLOBFILE(path = string(\"{WEIGHT_REF}\"), offset = uint64({meta})))];",
            sh = s.mil()
        ));
        em.def(name, t, s);
    }

    for o in ops.iter().filter(|o| live.contains(o.result.as_str())) {
        let res = format!("_v_{}", o.result);
        let x = arg(o, "x").ok_or_else(|| format!("{} {}: no x operand", o.op, o.result))?;
        let (xn, xs) = em.val(x)?;
        let op = o.op.as_str();
        if BINARY.contains(&op) {
            let y = arg(o, "y").ok_or_else(|| format!("{op} {}: no y operand", o.result))?;
            let (yn, ys) = em.val(y)?;
            let s = bcast(xs, ys, op)?;
            em.line(format!(
                "tensor<fp16, {}> {res} = {op}(x = {xn}, y = {yn})[name = string(\"{res}\")];",
                s.mil()
            ));
            em.def(&o.result, res, s);
        } else if UNARY.contains(&op) {
            em.line(format!(
                "tensor<fp16, {}> {res} = {op}(x = {xn})[name = string(\"{res}\")];",
                xs.mil()
            ));
            em.def(&o.result, res, xs);
        } else if op == "softmax" {
            let ax = em.konst("tensor<int32, []>", "int32(-1)");
            em.line(format!(
                "tensor<fp16, {}> {res} = softmax(x = {xn}, axis = {ax})[name = string(\"{res}\")];",
                xs.mil()
            ));
            em.def(&o.result, res, xs);
        } else if op == "matmul" {
            let y = arg(o, "y").ok_or_else(|| format!("matmul {}: no y operand", o.result))?;
            let (yn, ys) = em.val(y)?;
            let tx = bool_of(&consts, arg(o, "transpose_x"));
            let ty = bool_of(&consts, arg(o, "transpose_y"));
            let (tx, ty) = match (xs.is_vec(), ys.is_vec()) {
                // outer product: xᵀ·y
                (true, true) if tx => (true, false),
                // δ·W — coyote tags it transpose_x
                (true, false) if !ty => (false, false),
                _ => (tx, ty),
            };
            let (m, kx) = if tx { (xs.c, xs.r) } else { (xs.r, xs.c) };
            let (ky, n) = if ty { (ys.c, ys.r) } else { (ys.r, ys.c) };
            if kx != ky {
                return Err(format!(
                    "matmul {}: inner dims differ (x {xs:?} tx={tx}, y {ys:?} ty={ty})",
                    o.result
                ));
            }
            let s = Sh { r: m, c: n };
            let (bx, by) = (em.bool_k(tx), em.bool_k(ty));
            em.line(format!(
                "tensor<fp16, {}> {res} = matmul(x = {xn}, y = {yn}, transpose_x = {bx}, transpose_y = {by})[name = string(\"{res}\")];",
                s.mil()
            ));
            em.def(&o.result, res, s);
        } else if op == "rms_norm" {
            if !xs.is_vec() {
                return Err(format!("rms_norm {}: matrix input {xs:?}", o.result));
            }
            let eps = arg(o, "eps")
                .and_then(|v| f32_of(&consts, v))
                .unwrap_or(DEFAULT_RMS_EPS);
            let ax = em.konst("tensor<int32, [1]>", "tensor<int32, [1]>([-1])");
            let kt = em.bool_k(true);
            let ek = em.konst("tensor<fp16, []>", &format!("fp16({eps:e})"));
            let (sq, ms, rs, xn2) = (em.tmp("sq"), em.tmp("ms"), em.tmp("rs"), em.tmp("xn"));
            let one = Sh::vec(1);
            let (xm, om) = (xs.mil(), one.mil());
            em.line(format!(
                "tensor<fp16, {xm}> {sq} = mul(x = {xn}, y = {xn})[name = string(\"{sq}\")];"
            ));
            em.line(format!(
                "tensor<fp16, {om}> {ms} = reduce_mean(x = {sq}, axes = {ax}, keep_dims = {kt})[name = string(\"{ms}\")];"
            ));
            em.line(format!(
                "tensor<fp16, {om}> {rs} = rsqrt(x = {ms}, epsilon = {ek})[name = string(\"{rs}\")];"
            ));
            match arg(o, "w") {
                Some(g) => {
                    let (gn, gs) = em.val(g)?;
                    bcast(xs, gs, "rms_norm gamma")?;
                    em.line(format!(
                        "tensor<fp16, {xm}> {xn2} = mul(x = {xn}, y = {rs})[name = string(\"{xn2}\")];"
                    ));
                    em.line(format!(
                        "tensor<fp16, {xm}> {res} = mul(x = {xn2}, y = {gn})[name = string(\"{res}\")];"
                    ));
                }
                None => em.line(format!(
                    "tensor<fp16, {xm}> {res} = mul(x = {xn}, y = {rs})[name = string(\"{res}\")];"
                )),
            }
            em.def(&o.result, res, xs);
        } else {
            return Err(format!(
                "op {op} ({}) has no Apple-MIL translation yet (H13 backend)",
                o.result
            ));
        }
    }

    // outputs: reshape back to their host layout under the coyote name
    for out in &outputs {
        if sigs.iter().any(|(n, _)| n == out) {
            return Err(format!(
                "output {out} is a function input (passthrough unsupported)"
            ));
        }
        let (n, s) = em.val(out)?;
        let d = out_dims(declared.get(out), s);
        let sk = em.shape_k(&d);
        em.line(format!(
            "tensor<fp16, {}> {out} = reshape(x = {n}, shape = {sk})[name = string(\"{out}\")];",
            dims_mil(&d)
        ));
    }

    let params: Vec<String> = sigs
        .iter()
        .map(|(n, d)| format!("tensor<fp16, {}> {n}", dims_mil(d)))
        .collect();
    let mut text_out = String::from("program(1.3)\n{\n");
    let _ = writeln!(text_out, "    func main<ios18>({}) {{", params.join(", "));
    text_out.push_str(&em.body);
    let _ = writeln!(text_out, "    }} -> ({});\n}}", outputs.join(", "));
    let inplace: Vec<(String, String)> = inplace_src
        .iter()
        .filter(|(r, x)| outputs.contains(r) && cur.get(x) == Some(r))
        .cloned()
        .collect();
    Ok(AppleMil {
        text: text_out,
        weights,
        inplace,
        inputs: sigs.into_iter().map(|(n, _)| n).collect(),
        outputs,
    })
}

/// FNV-1a over program text + weights — names the program dir so a
/// changed program never hits aned's compile cache for the old one.
fn fingerprint(p: &AppleMil) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in p.text.as_bytes().iter().chain(p.weights.iter()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Prefix of program dirs under a build out_dir.
pub const DIR_PREFIX: &str = "mil13_";

/// Write {out_dir}/mil13_<fp>/{model.mil, weights/weight.bin}; stale
/// sibling program dirs are removed. Returns the program dir.
pub fn write_program(p: &AppleMil, out_dir: &Path) -> Result<PathBuf, String> {
    let dir = out_dir.join(format!("{DIR_PREFIX}{:016x}", fingerprint(p)));
    if let Ok(rd) = std::fs::read_dir(out_dir) {
        for e in rd.flatten() {
            let path = e.path();
            let stale = path != dir
                && path.is_dir()
                && e.file_name().to_string_lossy().starts_with(DIR_PREFIX);
            if stale {
                let _ = std::fs::remove_dir_all(&path);
            }
        }
    }
    let wdir = dir.join("weights");
    std::fs::create_dir_all(&wdir).map_err(|e| format!("{}: {e}", wdir.display()))?;
    let mil = dir.join(crate::espresso::MIL_FILE_NAME);
    std::fs::write(&mil, &p.text).map_err(|e| format!("{}: {e}", mil.display()))?;
    let w = wdir.join("weight.bin");
    std::fs::write(&w, &p.weights).map_err(|e| format!("{}: {e}", w.display()))?;
    Ok(dir)
}

/// ANE architecture string (h13g, h16g, ...) from the IORegistry.
pub fn ane_arch() -> Option<String> {
    static ARCH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    ARCH.get_or_init(|| {
        const KEY: &str = "ANEDevicePropertyTypeANEArchitectureTypeStr";
        let out = std::process::Command::new("/usr/sbin/ioreg")
            .args(["-r", "-c", "H11ANEIn", "-d", "1"])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&out.stdout);
        // one-line dict: ..."<KEY>"="h13g"}
        let after = &s[s.find(KEY)? + KEY.len()..];
        let after = &after[after.find('=')? + 1..];
        let v: String = after
            .trim_start()
            .trim_start_matches('"')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect();
        (!v.is_empty()).then_some(v)
    })
    .clone()
}

/// Use this backend? COYOTE_ANE=mil|hwx overrides; otherwise MIL on
/// H13–H15 (aned there rejects third-party HWX), HWX on H16+.
pub fn wanted() -> bool {
    match std::env::var("COYOTE_ANE").ok().as_deref() {
        Some("mil") => return true,
        Some("hwx") => return false,
        _ => {}
    }
    const MIL_ARCHS: &[&str] = &["h13", "h14", "h15"];
    ane_arch().is_some_and(|a| MIL_ARCHS.iter().any(|p| a.starts_with(p)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_shapes_and_matmul_forms() {
        let dir = std::env::temp_dir().join(format!("apple_mil_t_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("w.bin"), vec![0u8; 4 * 3 * 2]).unwrap();
        let src = r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 3, 1, 1]> x, tensor<fp16, [1, 4, 1, 1]> t) {
        tensor<bool, []> __bt = const()[val = bool(true)];
        tensor<fp16, [4, 3]> w = const()[val = tensor<fp16, [4, 3]>(BLOBFILE(path = string("w.bin"), offset = uint64(0)))];
        tensor<fp16, [1, 1, 1, 4]> v1 = matmul(x = x, y = w, transpose_y = __bt)[name = string("v1")];
        tensor<fp16, [1, 4, 1, 1]> v2 = sub(x = v1, y = t)[name = string("v2")];
        tensor<fp16, [1, 1, 1, 3]> v3 = matmul(x = v2, y = w, transpose_x = __bt)[name = string("v3")];
        tensor<fp16, [1, 4, 1, 3]> v4 = matmul(x = v2, y = x, transpose_x = __bt)[name = string("v4")];
        tensor<fp16, [1, 4, 1, 1]> v5 = rms_norm(x = v2, eps = fp16(0.00001))[name = string("v5")];
        tensor<fp16, [1, 4, 1, 1]> dead = relu(x = v2)[name = string("dead")];
    } -> (v3, v4, v5);
}
"#;
        let p = translate(src, &dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        // (case, expected substring) — every failing cell is reported
        let cases = [
            ("x canon", "tensor<fp16, [1, 1, 1, 3]> _c_x = reshape(x = x"),
            (
                "x·Wᵀ",
                "tensor<fp16, [1, 1, 1, 4]> _v_v1 = matmul(x = _c_x, y = _w_w",
            ),
            ("δ·W", "tensor<fp16, [1, 1, 1, 3]> _v_v3 = matmul"),
            ("outer", "tensor<fp16, [1, 1, 4, 3]> _v_v4 = matmul"),
            ("weight const", "tensor<fp16, [1, 1, 4, 3]> _w_w = const()"),
            ("out vec", "tensor<fp16, [1, 1, 1, 3]> v3 = reshape"),
            ("out mat", "tensor<fp16, [1, 4, 1, 3]> v4 = reshape"),
            ("rms", "reduce_mean(x = _t1_sq"),
            ("ret", "} -> (v3, v4, v5);"),
        ];
        let bad: Vec<&str> = cases
            .iter()
            .filter(|(_, s)| !p.text.contains(s))
            .map(|(c, _)| *c)
            .collect();
        assert!(bad.is_empty(), "missing {bad:?} in:\n{}", p.text);
        assert!(!p.text.contains("relu"), "dead op kept:\n{}", p.text);

        // in-place accumulator: reads of kv after `kv + outer` see the sum
        let src2 = r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 4, 1, 1]> pos, tensor<fp16, [1, 1, 1, 3]> v, tensor<fp16, [1, 4, 1, 3]> kv) {
        tensor<bool, []> __bt = const()[val = bool(true)];
        tensor<fp16, [1, 4, 1, 3]> o1 = matmul(x = pos, y = v, transpose_x = __bt)[name = string("o1")];
        tensor<fp16, [1, 4, 1, 3]> k2 = add(x = kv, y = o1)[name = string("k2")];
        tensor<fp16, [1, 1, 1, 3]> r = matmul(x = pos, y = kv, transpose_x = __bt)[name = string("r")];
    } -> (r, k2);
}
"#;
        let p2 = translate(src2, &dir).unwrap();
        assert!(
            p2.text.contains("_v_r = matmul(x = _c_pos, y = _v_k2"),
            "stale read:\n{}",
            p2.text
        );
        assert_eq!(p2.inplace, vec![("k2".to_string(), "kv".to_string())]);
        // header count=1, version=2; one 64 B meta + 24 B payload padded to 64
        assert_eq!(&p.weights[0..8], &[1, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(p.weights.len(), 64 + 64 + 64);
        assert_eq!(p.inputs, vec!["x", "t"]);
    }
}
