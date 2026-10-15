//! `graph` — multi-op graph scheduler: MIL DAG -> per-op emitted HWX
//! programs wired by IOSurface adoption.
//!
//! `emit_from_mil` covers whole-graph families that fit one `__text`
//! (conv chains, fused matmul+softmax partitions). This module handles
//! the general case: each op becomes its own emitted program, layout
//! adapters (strided <-> contiguous repacks) are inserted automatically
//! when producer/consumer currencies differ, and a [`Plan`] records the
//! surface bindings plus a topological eval order so [`GraphRuntime`]
//! executes the whole DAG as one unit.
//!
//! Op routes (all hw-verified emitters):
//!   add/sub/mul/min/max   elementwise, strided `[1,N,1,1]`
//!   add [1,T,1,D]         packed 4-D in-place (KV-cache accumulate)
//!   sigmoid/tanh          strided LUT; relu fixed C=64
//!   softmax               axis=1 -> channel (strided), else w-axis
//!   transpose             `[1,N,1,1]` -> `[1,1,1,N]` repack
//!   matmul const ty M=1   `emit_matmul_ns`; M>1 -> `emit_conv1x1_w`
//!   matmul x=[1,1,1,K] y=[1,R,1,K] ty   broadcast matvec (`emit_avw`)
//!   matmul x=[1,1,T,1] y=[1,1,T,D] tx   attn·V (`emit_attn_v`)
//!   matmul x=[1,T,1,1] y=[1,1,1,D] tx   outer product (`emit_outer`)
//!   rms_norm              expanded: mul -> repack -> ones-matvec ->
//!                         host rsqrt -> mul(x,rvec) -> mul(gamma)?

use crate::espresso::EspressoProgram;
use crate::hwx::{
    self, emit_add4d64, emit_attn_v, emit_avw, emit_conv1x1_w, emit_elementwise2, emit_matmul_ns,
    emit_outer, emit_relu, emit_repack128, emit_repack64, emit_sigmoid128, emit_softmax,
    emit_softmax_ch, emit_unary_lut, EwOp, LutOp, MilOp,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Physical tensor currency — byte-level layouts that must match for
/// zero-copy surface adoption.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cur {
    /// One fp16 per 64B channel plane (`[1,N,1,1]`, plane_stride 64).
    Strided(u32),
    /// Packed fp16 on the w axis (`[1,1,1,N]`).
    Contig(u32),
    /// Packed `[1,T,1,D]` rows (cache frames, outer-product outputs).
    Packed(u32, u32),
}

/// Where a program input port draws its bytes.
#[derive(Clone, Debug)]
pub enum BindSrc {
    /// Function input or const-filled surface — the program keeps its
    /// own surface; the host writes through [`GraphRuntime::write_ext`].
    Ext(String),
    /// Adopt `nodes[i]`'s output surface (port 0).
    Out(usize),
}

/// Host-side step between evals.
#[derive(Clone, Debug)]
pub enum HostOp {
    /// Read `src` node's output channel 0 (a partial-sum vector),
    /// compute `rsqrt(sum/n + eps)`, broadcast it strided into every
    /// surface bound to external input `ext`.
    RsqrtBroadcast {
        src: usize,
        ext: String,
        n: u32,
        eps: f32,
    },
}

#[derive(Debug)]
pub enum Node {
    /// An emitted HWX program on disk.
    Prog {
        name: String,
        path: PathBuf,
        binds: Vec<BindSrc>,
        /// `(out_port, in_port)` in-place aliases (persistent state).
        alias: Vec<(usize, usize)>,
        /// Input ports zeroed once before the first `run()`.
        zero: Vec<usize>,
    },
    Host(HostOp),
}

pub struct Plan {
    pub nodes: Vec<Node>,
    /// Eval order over `nodes` — adapters appear before their consumer.
    pub order: Vec<usize>,
    /// External inputs by name -> every (node, port) bound to it.
    pub externals: Vec<(String, Vec<(usize, usize)>)>,
    /// Externals pre-filled from blob consts (`name`, fp16 values laid
    /// out strided over `n` channels).
    pub const_fills: Vec<(String, Vec<f32>)>,
    /// Function outputs -> producing node index.
    pub outputs: Vec<(String, usize)>,
}

fn arg_of<'a>(o: &'a MilOp, k: &str) -> Option<&'a str> {
    o.args.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}

/// `axis = int32(1)` inline or a scalar const name.
fn mil_int(text: &str, v: &str) -> Option<i64> {
    let v = v.trim();
    if let Some(a) = v.find('(') {
        return v[a + 1..].trim_end_matches(')').trim().parse().ok();
    }
    if let Ok(n) = v.parse() {
        return Some(n);
    }
    for line in text.lines() {
        let l = line.trim();
        if (l.starts_with(&format!("{v} =")) || l.contains(&format!(" {v} =")))
            && l.contains("val =")
        {
            let p = l.find("val =").unwrap();
            let rest = &l[p + 5..];
            if let Some(a) = rest.find('(') {
                if let Ok(n) = rest[a + 1..].trim_end_matches(')').trim().parse() {
                    return Some(n);
                }
            }
        }
    }
    None
}

/// Expand `rms_norm(x, w, d, eps)` into the verified primitive
/// sub-plan. Returns the extra ops to splice in place; the host
/// rsqrt node is recorded via `hosts`.
fn expand_rms(o: &MilOp, eps: f32) -> Vec<MilOp> {
    let x = arg_of(o, "x").unwrap_or("x").to_string();
    let n: i64 = o.shape.iter().map(|d| (*d).max(1)).product();
    let mut v = vec![
        MilOp {
            op: "mul".into(),
            result: format!("{}_sq", o.result),
            shape: o.shape.clone(),
            args: vec![("x".into(), x.clone()), ("y".into(), x)],
        },
        MilOp {
            op: "transpose".into(),
            result: format!("{}_sqc", o.result),
            shape: vec![1, 1, 1, n],
            args: vec![("x".into(), format!("{}_sq", o.result))],
        },
        MilOp {
            op: "matmul".into(),
            result: format!("{}_ss", o.result),
            shape: vec![1, 32, 1, 1],
            args: vec![
                ("x".into(), format!("{}_sqc", o.result)),
                ("y".into(), format!("{}_ones", o.result)),
                ("transpose_y".into(), "__true".into()),
            ],
        },
        // marker consumed by the emitter loop: host rsqrt broadcast
        MilOp {
            op: "__rsqrt_bcast".into(),
            result: format!("{}_rvec", o.result),
            shape: o.shape.clone(),
            args: vec![
                ("x".into(), format!("{}_ss", o.result)),
                ("eps".into(), format!("__eps_{}", eps)),
            ],
        },
        MilOp {
            op: "mul".into(),
            result: format!("{}_xn", o.result),
            shape: o.shape.clone(),
            args: vec![
                ("x".into(), arg_of(o, "x").unwrap_or("x").to_string()),
                ("y".into(), format!("{}_rvec", o.result)),
            ],
        },
    ];
    if let Some(g) = arg_of(o, "w") {
        v.push(MilOp {
            op: "mul".into(),
            result: o.result.clone(),
            shape: o.shape.clone(),
            args: vec![
                ("x".into(), format!("{}_xn", o.result)),
                ("y".into(), g.to_string()),
            ],
        });
    } else {
        // result aliases the xn node
        v.last_mut().unwrap().result = o.result.clone();
    }
    v
}

/// Compile a MIL program into a [`Plan`]. `out_dir` receives one
/// `.macho` per node — distinct paths are mandatory because aned's
/// patch cache keys on the embedded out-path.
pub fn compile_graph(mil_path: &str, out_dir: &str) -> Result<Plan, String> {
    let text = std::fs::read_to_string(mil_path).map_err(|e| format!("{mil_path}: {e}"))?;
    let mil_dir = Path::new(mil_path)
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let (ops, blobs) = hwx::mil_ops(&text);
    if ops.is_empty() {
        return Err(format!("{mil_path}: no compute ops found"));
    }
    let blob_of = |name: &str| -> Option<String> {
        blobs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p.clone())
    };
    let blob_f32 = |name: &str| -> Option<Vec<f32>> {
        let path = blob_of(name)?;
        Some(
            hwx::load_blob_u16(&hwx::resolve_blob(&mil_dir, &path))
                .ok()?
                .iter()
                .map(|b| half::f16::from_bits(*b).to_f32())
                .collect(),
        )
    };
    let sigs = hwx::func_input_shapes(&text);
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{out_dir}: {e}"))?;

    // flatten rms_norm ops in place (before shape resolution so
    // internal results are visible to consumers)
    let mut flat: Vec<MilOp> = Vec::new();
    for o in ops {
        if o.op == "rms_norm" {
            let eps = arg_of(&o, "eps")
                .and_then(|v| mil_int(&text, v).map(|i| i as f32))
                .or_else(|| {
                    arg_of(&o, "eps").and_then(|v| {
                        hwx::mil_scalar_f16(&text, v).map(|b| half::f16::from_bits(b).to_f32())
                    })
                })
                .unwrap_or(1e-5);
            flat.extend(expand_rms(&o, eps));
        } else {
            flat.push(o);
        }
    }
    // shapes resolve against func inputs AND internal results
    let mut result_shapes: HashMap<String, Vec<i64>> = HashMap::new();
    for o in &flat {
        result_shapes.insert(o.result.clone(), o.shape.clone());
    }
    let shape_of = |n: &str| -> Vec<i64> {
        result_shapes.get(n).cloned().unwrap_or_else(|| {
            sigs.iter()
                .find(|(nm, _)| nm == n)
                .map(|(_, s)| s.clone())
                .unwrap_or_default()
        })
    };
    let weights_of = |o: &MilOp| -> Result<Vec<u16>, String> {
        let wname = arg_of(o, "weight")
            .or_else(|| arg_of(o, "y"))
            .ok_or("op without weight operand")?;
        let path = blob_of(wname).ok_or_else(|| format!("`{wname}` is not a blob const"))?;
        hwx::load_blob_u16(&hwx::resolve_blob(&mil_dir, &path))
    };

    let mut nodes: Vec<Node> = Vec::new();
    let mut created: Vec<usize> = Vec::new(); // node creation order
                                              // producer table: result name -> (node idx, currency)
    let mut prod: HashMap<String, (usize, Cur)> = HashMap::new();
    // pending bindings: node idx -> (arg names, want currencies)
    let mut wants: Vec<(usize, Vec<(String, Cur)>)> = Vec::new();
    let mut ext_uses: Vec<(String, usize, usize)> = Vec::new();
    let mut const_fills: Vec<(String, Vec<f32>)> = Vec::new();

    let mut push_prog = |nodes: &mut Vec<Node>,
                         created: &mut Vec<usize>,
                         name: String,
                         img: Vec<u8>,
                         out_dir: &str|
     -> Result<usize, String> {
        let idx = nodes.len();
        let path = format!("{out_dir}/n{idx}_{name}.macho");
        std::fs::write(&path, &img).map_err(|e| format!("{path}: {e}"))?;
        nodes.push(Node::Prog {
            name,
            path: PathBuf::from(path),
            binds: Vec::new(),
            alias: Vec::new(),
            zero: Vec::new(),
        });
        created.push(idx);
        Ok(idx)
    };

    for o in &flat {
        let elems: u32 = o.shape.iter().map(|d| (*d).max(1) as u32).product();
        match o.op.as_str() {
            "__rsqrt_bcast" => {
                // host node: rvec surfaces are externals on consumers
                let src = arg_of(o, "x").unwrap();
                let src_idx = prod
                    .get(src)
                    .ok_or_else(|| format!("rsqrt src `{src}` has no producer"))?
                    .0;
                let eps = arg_of(o, "eps")
                    .and_then(|v| v.strip_prefix("__eps_"))
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(1e-5);
                nodes.push(Node::Host(HostOp::RsqrtBroadcast {
                    src: src_idx,
                    ext: o.result.clone(),
                    n: elems,
                    eps,
                }));
                created.push(nodes.len() - 1);
                // rvec is virtual — consumers see it as an external
                continue;
            }
            "add" | "sub" | "mul" | "minimum" | "maximum" => {
                let op = match o.op.as_str() {
                    "add" => EwOp::Add,
                    "sub" => EwOp::Sub,
                    "mul" => EwOp::Mul,
                    "minimum" => EwOp::Min,
                    _ => EwOp::Max,
                };
                let (x, y) = (
                    arg_of(o, "x").unwrap_or("x").to_string(),
                    arg_of(o, "y").unwrap_or("y").to_string(),
                );
                // packed 4-D add `[1,T,1,D]` — the KV-cache accumulator
                if o.op == "add"
                    && o.shape.len() == 4
                    && o.shape[1] > 1
                    && o.shape[3] > 1
                    && o.shape[1] == 64
                    && o.shape[3] == 64
                {
                    let img = emit_add4d64(mil_path, &format!("{out_dir}/.tmp4d_{}", o.result));
                    let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                    if let Node::Prog { alias, zero, .. } = &mut nodes[idx] {
                        *alias = vec![(0, 0)];
                        *zero = vec![0];
                    }
                    prod.insert(o.result.clone(), (idx, Cur::Packed(64, 64)));
                    wants.push((
                        idx,
                        vec![(x, Cur::Packed(64, 64)), (y, Cur::Packed(64, 64))],
                    ));
                    continue;
                }
                let img =
                    emit_elementwise2(op, elems, mil_path, &format!("{out_dir}/.tmp_{}", o.result));
                let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                prod.insert(o.result.clone(), (idx, Cur::Strided(elems)));
                wants.push((
                    idx,
                    vec![(x, Cur::Strided(elems)), (y, Cur::Strided(elems))],
                ));
            }
            "sigmoid" | "tanh" | "relu" => {
                let x = arg_of(o, "x").unwrap_or("x").to_string();
                let img = match o.op.as_str() {
                    "sigmoid" if elems == 128 => {
                        emit_sigmoid128(mil_path, &format!("{out_dir}/.tmps_{}", o.result))
                    }
                    "sigmoid" => emit_unary_lut(
                        LutOp::Sigmoid,
                        elems,
                        mil_path,
                        &format!("{out_dir}/.tmps_{}", o.result),
                    ),
                    "tanh" => emit_unary_lut(
                        LutOp::Tanh,
                        elems,
                        mil_path,
                        &format!("{out_dir}/.tmpt_{}", o.result),
                    ),
                    _ => {
                        if elems != 64 {
                            return Err(format!("relu n={elems} off skeleton (64 only)"));
                        }
                        emit_relu(mil_path, &format!("{out_dir}/.tmpr_{}", o.result))
                    }
                };
                let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                prod.insert(o.result.clone(), (idx, Cur::Strided(elems)));
                wants.push((idx, vec![(x, Cur::Strided(elems))]));
            }
            "softmax" => {
                let x = arg_of(o, "x").unwrap_or("x").to_string();
                let axis = arg_of(o, "axis").and_then(|v| mil_int(&text, v));
                let t = o.shape.get(1).copied().unwrap_or(1).max(1) as u32;
                let (img, cur) = if axis == Some(1) && t > 1 {
                    (
                        emit_softmax_ch(t, mil_path, &format!("{out_dir}/.tmpsm_{}", o.result))?,
                        Cur::Strided(t),
                    )
                } else {
                    (
                        emit_softmax(elems, mil_path, &format!("{out_dir}/.tmpsm_{}", o.result))?,
                        Cur::Contig(elems),
                    )
                };
                let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                prod.insert(o.result.clone(), (idx, cur));
                wants.push((idx, vec![(x, cur)]));
            }
            "transpose" => {
                let x = arg_of(o, "x").unwrap_or("x").to_string();
                let img = match elems {
                    64 => emit_repack64(mil_path, &format!("{out_dir}/.tmpt_{}", o.result)),
                    128 => emit_repack128(mil_path, &format!("{out_dir}/.tmpt_{}", o.result)),
                    _ => return Err(format!("transpose n={elems} off repack table")),
                };
                let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                prod.insert(o.result.clone(), (idx, Cur::Contig(elems)));
                wants.push((idx, vec![(x, Cur::Strided(elems))]));
            }
            "matmul" => {
                let (x, y) = (
                    arg_of(o, "x").unwrap_or("x").to_string(),
                    arg_of(o, "y").unwrap_or_default().to_string(),
                );
                let ty = match arg_of(o, "transpose_y") {
                    Some(v) => {
                        hwx::mil_scalar_bool(&text, v).unwrap_or(v == "true" || v == "__true")
                    }
                    None => false,
                };
                let tx = match arg_of(o, "transpose_x") {
                    Some(v) => {
                        hwx::mil_scalar_bool(&text, v).unwrap_or(v == "true" || v == "__true")
                    }
                    None => false,
                };
                let xs = shape_of(&x);
                let ys = shape_of(&y);
                // synthetic ones-matvec from the rms expansion:
                // `y` names a missing blob -> build it ourselves
                if y.ends_with("_ones") {
                    let n = o.shape.get(3).copied().unwrap_or(0) as usize;
                    let k = xs.get(3).copied().unwrap_or(0) as usize;
                    let mut w = vec![0u16; 32 * k];
                    for i in 0..k {
                        w[i] = half::f16::from_f32(1.0).to_bits();
                    }
                    let img = emit_matmul_ns(
                        k,
                        n.max(32),
                        &w,
                        mil_path,
                        &format!("{out_dir}/.tmpm_{}", o.result),
                    )?;
                    let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                    prod.insert(o.result.clone(), (idx, Cur::Strided(n.max(32) as u32)));
                    wants.push((idx, vec![(x, Cur::Contig(k as u32))]));
                    continue;
                }
                if blob_of(&y).is_some() {
                    let w = weights_of(o)?;
                    let m = o.shape.get(1).copied().unwrap_or(0) as usize;
                    let n = o.shape.get(3).copied().unwrap_or(0) as usize;
                    if ty && n >= 32 && w.len() % n == 0 {
                        let k = w.len() / n;
                        let seq = if xs.len() == 2 {
                            xs[0].max(1) as usize
                        } else {
                            xs.get(2).copied().unwrap_or(1).max(1) as usize
                        };
                        let img = if seq == 1 || m == 1 {
                            emit_matmul_ns(
                                k,
                                n,
                                &w,
                                mil_path,
                                &format!("{out_dir}/.tmpm_{}", o.result),
                            )?
                        } else {
                            emit_conv1x1_w(
                                k,
                                n,
                                seq,
                                &w,
                                mil_path,
                                &format!("{out_dir}/.tmpm_{}", o.result),
                            )?
                        };
                        let idx =
                            push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                        prod.insert(o.result.clone(), (idx, Cur::Strided(n as u32)));
                        wants.push((idx, vec![(x, Cur::Contig(k as u32))]));
                        continue;
                    }
                    return Err(format!("matmul const x{xs:?} -> {:?} unsupported", o.shape));
                }
                // transpose_x matmuls: outer product (x strided row-mask,
                // y a vector) vs attn·V (x strided scores, y packed
                // cache rows). Distinguish by y's shape: a lone trailing
                // dim means the vector form.
                let y_spatial: Vec<i64> = ys.iter().skip(1).copied().collect();
                let y_widem = y_spatial.iter().filter(|d| **d > 1).count();
                let xt = xs
                    .get(1)
                    .copied()
                    .unwrap_or(1)
                    .max(1)
                    .max(xs.get(2).copied().unwrap_or(1).max(1));
                if tx && xs.len() == 4 && ys.len() == 4 && y_widem == 1 && ys[3] > 1 {
                    // outer: x=[1,T,1,1] strided one-hot, y=[1,1,1,D]
                    let (t, d) = (xt as u32, ys[3] as u32);
                    if t != 64 || d != 64 {
                        return Err(format!("outer ({t},{d}) off skeleton table"));
                    }
                    let img = emit_outer(mil_path, &format!("{out_dir}/.tmpo_{}", o.result));
                    let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                    prod.insert(o.result.clone(), (idx, Cur::Packed(t, d)));
                    wants.push((idx, vec![(x, Cur::Strided(t)), (y, Cur::Contig(d))]));
                } else if tx && xs.len() == 4 && ys.len() == 4 && y_widem == 2 {
                    // attn·V: x strided scores (either axis spelling),
                    // y packed [1,T,1,D] or [1,1,T,D] cache frame
                    let (t, d) = (xt as u32, ys[3] as u32);
                    let img =
                        emit_attn_v(t, d, mil_path, &format!("{out_dir}/.tmpv_{}", o.result))?;
                    let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                    prod.insert(o.result.clone(), (idx, Cur::Contig(d)));
                    wants.push((idx, vec![(x, Cur::Strided(t)), (y, Cur::Packed(t, d))]));
                } else if ty
                    && xs.len() == 4
                    && ys.len() == 4
                    && xs[..3] == [1, 1, 1]
                    && ys[0] == 1
                    && ys[2] == 1
                    && xs[3] == ys[3]
                {
                    let (r, k) = (ys[1].max(1) as u32, xs[3].max(1) as u32);
                    let img = emit_avw(r, k, mil_path, &format!("{out_dir}/.tmpa_{}", o.result));
                    let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                    prod.insert(o.result.clone(), (idx, Cur::Strided(r)));
                    wants.push((idx, vec![(x, Cur::Contig(k)), (y, Cur::Packed(r, k))]));
                } else {
                    return Err(format!(
                        "matmul x{xs:?} y{ys:?} tx={tx} ty={ty} not on a route"
                    ));
                }
            }
            other => return Err(format!("op `{other}` not on a graph route")),
        }
    }

    // ---- resolve binds, inserting strided->contig adapters ----------
    // adapters created during binding eval immediately before their
    // consumer: eval order = for each created node, its adapters first.
    let mut pre: HashMap<usize, Vec<usize>> = HashMap::new();
    for (nidx, ws) in &wants {
        let mut bs = Vec::new();
        for (port, (name, want)) in ws.iter().enumerate() {
            if let Some((pn, have)) = prod.get(name.as_str()).copied() {
                let src = if have == *want {
                    pn
                } else {
                    match (have, *want) {
                        (Cur::Strided(n), Cur::Contig(m)) if n == m => {
                            let img = match n {
                                64 => emit_repack64(mil_path, &format!("{out_dir}/.tmpa64_{name}")),
                                128 => {
                                    emit_repack128(mil_path, &format!("{out_dir}/.tmpa128_{name}"))
                                }
                                _ => return Err(format!("no repack skeleton for N={n}")),
                            };
                            let aidx = nodes.len();
                            let path = format!("{out_dir}/n{aidx}_{name}_rp.macho");
                            std::fs::write(&path, &img).map_err(|e| format!("{path}: {e}"))?;
                            nodes.push(Node::Prog {
                                name: format!("{name}_rp"),
                                path: PathBuf::from(path),
                                binds: vec![BindSrc::Out(pn)],
                                alias: Vec::new(),
                                zero: Vec::new(),
                            });
                            pre.entry(*nidx).or_default().push(aidx);
                            aidx
                        }
                        _ => {
                            return Err(format!(
                                "currency mismatch on `{name}`: have {have:?} want {want:?}"
                            ))
                        }
                    }
                };
                bs.push(BindSrc::Out(src));
            } else {
                // external or const-fill or virtual rvec
                if let Some(vals) = blob_f32(name) {
                    const_fills.push((name.clone(), vals));
                }
                ext_uses.push((name.clone(), *nidx, port));
                bs.push(BindSrc::Ext(name.clone()));
            }
        }
        if let Node::Prog { binds: b, .. } = &mut nodes[*nidx] {
            *b = bs;
        }
    }
    let mut order = Vec::with_capacity(nodes.len());
    for idx in created {
        if let Some(ps) = pre.get(&idx) {
            order.extend_from_slice(ps);
        }
        order.push(idx);
    }

    // ---- externals + outputs ------------------------------------------
    let mut ext_map: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for (name, nidx, port) in ext_uses {
        ext_map.entry(name).or_default().push((nidx, port));
    }
    let externals: Vec<(String, Vec<(usize, usize)>)> =
        ext_map.into_iter().map(|(k, v)| (k, v)).collect();
    let mut outputs = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if let Some(p) = l.find("-> (") {
            for nm in l[p + 4..]
                .trim_end_matches(|c| c == ')' || c == ';')
                .split(',')
            {
                let nm = nm.trim();
                if let Some((idx, _)) = prod.get(nm) {
                    outputs.push((nm.to_string(), *idx));
                }
            }
        }
    }
    Ok(Plan {
        nodes,
        order,
        externals,
        const_fills,
        outputs,
    })
}

// ---------------------------------------------------------------------------
// runtime
// ---------------------------------------------------------------------------

pub struct GraphRuntime {
    pub progs: Vec<Option<EspressoProgram>>,
    pub plan: Plan,
}

impl GraphRuntime {
    /// Load every node program, apply aliases/zero-init, then wire the
    /// adopted surfaces and const fills.
    pub fn load(plan: Plan) -> Result<Self, String> {
        let mut progs: Vec<Option<EspressoProgram>> = Vec::new();
        for n in &plan.nodes {
            match n {
                Node::Prog { path, .. } => progs.push(Some(
                    EspressoProgram::load_precompiled(path, &[])
                        .map_err(|e| format!("{}: {e:?}", path.display()))?,
                )),
                Node::Host(_) => progs.push(None),
            }
        }
        for (i, n) in plan.nodes.iter().enumerate() {
            if let Node::Prog { alias, zero, .. } = n {
                let p = progs[i].as_mut().unwrap();
                for (o, ii) in alias {
                    p.alias_out_to_in(*o, *ii);
                }
                for ii in zero {
                    for b in p.input_mut(*ii).iter_mut() {
                        *b = 0;
                    }
                }
            }
        }
        for (i, n) in plan.nodes.iter().enumerate() {
            if let Node::Prog { binds, .. } = n {
                for (port, b) in binds.iter().enumerate() {
                    if let BindSrc::Out(src) = b {
                        let s = progs[*src].as_ref().unwrap().output_surf(0);
                        progs[i]
                            .as_mut()
                            .unwrap()
                            .adopt_input(port, s)
                            .map_err(|e| format!("bind node{i} port{port}: {e:?}"))?;
                    }
                }
            }
        }
        let mut rt = Self { progs, plan };
        let fills = rt.plan.const_fills.clone();
        for (name, vals) in &fills {
            rt.write_ext(name, |b, d| {
                for (i, v) in vals.iter().enumerate() {
                    let o = d.offset(0, 0, i as u64, 0, 0);
                    if o + 2 <= b.len() {
                        b[o..o + 2].copy_from_slice(&half::f16::from_f32(*v).to_le_bytes());
                    }
                }
            });
        }
        Ok(rt)
    }

    /// Write external input `name` into every bound port via `f(bytes,
    /// desc)`.
    pub fn write_ext(&mut self, name: &str, f: impl Fn(&mut [u8], &crate::espresso::TensorDesc)) {
        let uses: Vec<(usize, usize)> = self
            .plan
            .externals
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, u)| u.clone())
            .unwrap_or_default();
        for (ni, port) in uses {
            let p = self.progs[ni].as_mut().unwrap();
            let d = p.input_descs[port].clone();
            f(p.input_mut(port), &d);
        }
    }

    pub fn output(&self, i: usize) -> &[u8] {
        self.progs[i].as_ref().unwrap().output(0)
    }
    pub fn out_desc(&self, i: usize) -> crate::espresso::TensorDesc {
        self.progs[i].as_ref().unwrap().output_descs[0].clone()
    }
    pub fn out_node(&self, name: &str) -> Option<usize> {
        self.plan
            .outputs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, i)| *i)
    }

    /// Execute the whole graph once in scheduled order (host ops run
    /// their scalar bridge inline).
    pub fn run(&mut self) -> Result<(), String> {
        for i in self.plan.order.clone() {
            match &self.plan.nodes[i] {
                Node::Prog { .. } => self.progs[i]
                    .as_ref()
                    .unwrap()
                    .eval()
                    .map_err(|e| format!("node {i}: {e:?}"))?,
                Node::Host(HostOp::RsqrtBroadcast { src, ext, n, eps }) => {
                    let (src, ext, n, eps) = (*src, ext.clone(), *n, *eps);
                    let d = self.out_desc(src);
                    let sum = half::f16::from_bits(u16::from_le_bytes([
                        self.output(src)[d.offset(0, 0, 0, 0, 0)],
                        self.output(src)[d.offset(0, 0, 0, 0, 0) + 1],
                    ]))
                    .to_f32();
                    let r = (sum / n as f32 + eps).sqrt().recip();
                    self.write_ext(&ext, |b, d| {
                        for i in 0..d.k.max(d.w) {
                            let o = if d.k > 1 {
                                d.offset(0, 0, i, 0, 0)
                            } else {
                                d.offset(0, 0, 0, 0, i)
                            };
                            if o + 2 <= b.len() {
                                b[o..o + 2].copy_from_slice(&half::f16::from_f32(r).to_le_bytes());
                            }
                        }
                    });
                }
            }
        }
        Ok(())
    }
}
