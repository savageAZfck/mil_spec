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

use crate::espresso::{EspressoProgram, SharedSurface, TensorDesc};
use crate::hwx::{
    self, emit_add4d, emit_attn_v, emit_avw, emit_conv1x1_w, emit_conv_act, emit_conv_chain,
    emit_elementwise2, emit_matmul_ns, emit_outer_t, emit_relu, emit_repack, emit_sigmoid_strided,
    emit_softmax, emit_softmax_ch, emit_unary_lut, EwOp, FusedAct, LutOp, MilOp,
};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::cell::RefCell;

thread_local! {
    /// Process-wide surface pool. IOSurfaces are a bounded global
    /// resource — create/destroy churn outruns the driver's reclaim
    /// path, so released surfaces go back to a bounded reuse pool
    /// instead of dying. Live set stays at one program's working set
    /// regardless of runtime count. (GraphRuntime is single-threaded;
    /// Rc everywhere already.)
    static SURF_POOL: RefCell<VecDeque<(usize, Rc<SharedSurface>)>> =
        RefCell::new(VecDeque::new());
}
const SURF_POOL_CAP: usize = 4096;

/// Take a surface of at least `bytes` from the pool, else allocate.
fn surface_pooled(bytes: usize) -> Result<Rc<SharedSurface>, String> {
    let s = SURF_POOL.with(|p| {
        let mut p = p.borrow_mut();
        // smallest fit — keeps big surfaces free for big requests
        let best = p
            .iter()
            .enumerate()
            .filter(|(_, (b, _))| *b >= bytes)
            .min_by_key(|(_, (b, _))| *b)
            .map(|(i, _)| i);
        best.map(|i| p.remove(i).unwrap().1)
    });
    match s {
        Some(surf) => {
            for b in surf.bytes_mut().iter_mut() {
                *b = 0;
            }
            Ok(surf)
        }
        None => SharedSurface::new(bytes).map(Rc::new).map_err(|e| format!("{e:?}")),
    }
}

fn surface_release(surf: Rc<SharedSurface>) {
    SURF_POOL.with(|p| {
        let mut p = p.borrow_mut();
        p.push_back((surf.bytes().len(), surf));
        while p.len() > SURF_POOL_CAP {
            p.pop_front();
        }
    });
}

/// Physical tensor currency — byte-level layouts that must match for
/// zero-copy surface adoption.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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
    /// Read `src` surface (`n` elements), compute `rsqrt(mean(x^2) + eps)`
    /// (scaled by `gamma` when present), and broadcast it strided into
    /// external input `ext` for the downstream ANE `mul(x, rvec)`.
    RsqrtBroadcast {
        src: BindSrc,
        ext: String,
        n: u32,
        eps: f32,
        gamma: Option<Vec<f32>>,
    },
    /// Elementwise reciprocal `1.0 / src[i]` written into external input
    /// `ext` for the downstream ANE `mul(a, b_recip)` (`real_div`).
    Recip {
        src: BindSrc,
        ext: String,
        n: u32,
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
                let inner = rest[a + 1..].trim_end_matches(|c: char| !c.is_numeric() && c != '-');
                let inner = inner.trim_end_matches(')').trim();
                if let Ok(n) = inner.parse() {
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
fn expand_rms(o: &MilOp, eps: f32, fold_gamma: bool) -> Vec<MilOp> {
    let x = arg_of(o, "x").unwrap_or("x").to_string();
    let mut bcast_args = vec![
        ("x".into(), x.clone()),
        ("eps".into(), format!("__eps_{}", eps)),
    ];
    if fold_gamma {
        if let Some(g) = arg_of(o, "w") {
            bcast_args.push(("w".into(), g.to_string()));
        }
    }
    let mut v = vec![
        // marker consumed by the emitter loop: host rsqrt broadcast
        MilOp {
            op: "__rsqrt_bcast".into(),
            result: format!("{}_rvec", o.result),
            shape: o.shape.clone(),
            args: bcast_args,
        },
        MilOp {
            op: "mul".into(),
            result: format!("{}_xn", o.result),
            shape: o.shape.clone(),
            args: vec![("x".into(), x), ("y".into(), format!("{}_rvec", o.result))],
        },
    ];
    if let Some(g) = arg_of(o, "w").filter(|_| !fold_gamma) {
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
        blob_of(name)?;
        Some(
            hwx::load_mil_weight_u16(&text, &mil_dir, name)
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
            let fold_gamma = arg_of(&o, "w").and_then(&blob_f32).is_some();
            flat.extend(expand_rms(&o, eps, fold_gamma));
        } else if o.op == "silu" {
            let x = arg_of(&o, "x").unwrap_or("x").to_string();
            let sig = format!("{}_sig", o.result);
            flat.push(MilOp {
                op: "sigmoid".into(),
                result: sig.clone(),
                shape: o.shape.clone(),
                args: vec![("x".into(), x.clone())],
            });
            flat.push(MilOp {
                op: "mul".into(),
                result: o.result.clone(),
                shape: o.shape.clone(),
                args: vec![("x".into(), x), ("y".into(), sig)],
            });
        } else if o.op == "real_div" {
            let x = arg_of(&o, "x").unwrap_or("x").to_string();
            let y = arg_of(&o, "y").unwrap_or("y").to_string();
            let r = format!("{}_rcp", o.result);
            flat.push(MilOp {
                op: "__recip".into(),
                result: r.clone(),
                shape: o.shape.clone(),
                args: vec![("x".into(), y)],
            });
            flat.push(MilOp {
                op: "mul".into(),
                result: o.result.clone(),
                shape: o.shape.clone(),
                args: vec![("x".into(), x), ("y".into(), r)],
            });
        } else if o.op == "conv" && arg_of(&o, "bias").is_some() {
            let bname = arg_of(&o, "bias").unwrap().to_string();
            let mut c_op = o.clone();
            c_op.result = format!("{}_nb", o.result);
            c_op.args.retain(|(k, _)| k != "bias");
            let c_res = c_op.result.clone();
            flat.push(c_op);
            flat.push(MilOp {
                op: "add".into(),
                result: o.result,
                shape: o.shape,
                args: vec![("x".into(), c_res), ("y".into(), bname)],
            });
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
        hwx::load_mil_weight_u16(&text, &mil_dir, wname)
    };

    // ---- fusion pass: conv+act (C=64) and k-deep conv1x1 chains -----
    let mut func_outs = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if let Some(p) = l.find("-> (") {
            for nm in l[p + 4..].trim_end_matches([')', ';']).split(',') {
                func_outs.push(nm.trim().to_string());
            }
        }
    }
    let mut use_cnt: HashMap<String, usize> = HashMap::new();
    for o in &flat {
        for (_, v) in &o.args {
            *use_cnt.entry(v.clone()).or_insert(0) += 1;
        }
    }
    let is_1x1 = |s: &[i64]| s.len() == 4 && s[0] == 1 && s[2] == 1 && s[3] == 1 && s[1] >= 32;
    let mut fused = Vec::with_capacity(flat.len());
    let mut fi = 0usize;
    while fi < flat.len() {
        let o = &flat[fi];
        // 1) 64->64 conv + silu (`g = conv; gs = sigmoid(g); y = mul(g, gs)`)
        if o.op == "conv"
            && o.shape == [1, 64, 1, 1]
            && !func_outs.contains(&o.result)
            && weights_of(o).map(|w| w.len() == 64 * 64).unwrap_or(false)
        {
            if fi + 2 < flat.len()
                && flat[fi + 1].op == "sigmoid"
                && arg_of(&flat[fi + 1], "x") == Some(o.result.as_str())
                && !func_outs.contains(&flat[fi + 1].result)
                && use_cnt.get(&o.result) == Some(&2)
                && use_cnt.get(&flat[fi + 1].result) == Some(&1)
                && flat[fi + 2].op == "mul"
                && ((arg_of(&flat[fi + 2], "x") == Some(o.result.as_str())
                    && arg_of(&flat[fi + 2], "y") == Some(flat[fi + 1].result.as_str()))
                    || (arg_of(&flat[fi + 2], "y") == Some(o.result.as_str())
                        && arg_of(&flat[fi + 2], "x") == Some(flat[fi + 1].result.as_str())))
            {
                fused.push(MilOp {
                    op: "__conv_act".into(),
                    result: flat[fi + 2].result.clone(),
                    shape: vec![1, 64, 1, 1],
                    args: vec![
                        ("x".into(), arg_of(o, "x").unwrap_or("x").to_string()),
                        (
                            "weight".into(),
                            arg_of(o, "weight").unwrap_or_default().to_string(),
                        ),
                        ("act".into(), "silu".into()),
                    ],
                });
                fi += 3;
                continue;
            }
            if fi + 1 < flat.len()
                && matches!(
                    flat[fi + 1].op.as_str(),
                    "sigmoid" | "tanh" | "relu" | "gelu"
                )
                && arg_of(&flat[fi + 1], "x") == Some(o.result.as_str())
                && use_cnt.get(&o.result) == Some(&1)
            {
                fused.push(MilOp {
                    op: "__conv_act".into(),
                    result: flat[fi + 1].result.clone(),
                    shape: vec![1, 64, 1, 1],
                    args: vec![
                        ("x".into(), arg_of(o, "x").unwrap_or("x").to_string()),
                        (
                            "weight".into(),
                            arg_of(o, "weight").unwrap_or_default().to_string(),
                        ),
                        ("act".into(), flat[fi + 1].op.clone()),
                    ],
                });
                fi += 2;
                continue;
            }
        }
        // 2) k >= 2 consecutive 1x1 convs -> single multi-task emit_conv_chain
        if o.op == "conv"
            && is_1x1(&o.shape)
            && !func_outs.contains(&o.result)
            && use_cnt.get(&o.result) == Some(&1)
        {
            let mut end = fi + 1;
            while end < flat.len()
                && flat[end].op == "conv"
                && is_1x1(&flat[end].shape)
                && arg_of(&flat[end], "x") == Some(flat[end - 1].result.as_str())
            {
                let res = &flat[end].result;
                end += 1;
                if func_outs.contains(res) || use_cnt.get(res) != Some(&1) {
                    break;
                }
            }
            if end - fi >= 2 {
                let mut args = vec![("x".into(), arg_of(o, "x").unwrap_or("x").to_string())];
                for (k_idx, co_op) in flat[fi..end].iter().enumerate() {
                    args.push((
                        format!("w{k_idx}"),
                        arg_of(co_op, "weight").unwrap_or_default().to_string(),
                    ));
                    args.push((format!("c{k_idx}"), co_op.shape[1].to_string()));
                }
                fused.push(MilOp {
                    op: "__conv_chain".into(),
                    result: flat[end - 1].result.clone(),
                    shape: flat[end - 1].shape.clone(),
                    args,
                });
                fi = end;
                continue;
            }
        }
        fused.push(o.clone());
        fi += 1;
    }
    let flat = fused;

    let mut nodes: Vec<Node> = Vec::new();
    let mut created: Vec<usize> = Vec::new(); // node creation order
                                              // producer table: result name -> (node idx, currency)
    let mut prod: HashMap<String, (usize, Cur)> = HashMap::new();
    // pending bindings: node idx -> (arg names, want currencies)
    let mut wants: Vec<(usize, Vec<(String, Cur)>)> = Vec::new();
    let mut ext_uses: Vec<(String, usize, usize)> = Vec::new();
    let mut const_fills: Vec<(String, Vec<f32>)> = Vec::new();

    let push_prog = |nodes: &mut Vec<Node>,
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
                let src_bind = match prod.get(src) {
                    Some((idx, _)) => BindSrc::Out(*idx),
                    None => BindSrc::Ext(src.to_string()),
                };
                let eps = arg_of(o, "eps")
                    .and_then(|v| v.strip_prefix("__eps_"))
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(1e-5);
                let gamma = arg_of(o, "w").and_then(&blob_f32);
                nodes.push(Node::Host(HostOp::RsqrtBroadcast {
                    src: src_bind,
                    ext: o.result.clone(),
                    n: elems,
                    eps,
                    gamma,
                }));
                created.push(nodes.len() - 1);
                // rvec is virtual — consumers see it as an external
                continue;
            }
            "__recip" => {
                let src = arg_of(o, "x").unwrap();
                let src_bind = match prod.get(src) {
                    Some((idx, _)) => BindSrc::Out(*idx),
                    None => BindSrc::Ext(src.to_string()),
                };
                nodes.push(Node::Host(HostOp::Recip {
                    src: src_bind,
                    ext: o.result.clone(),
                    n: elems,
                }));
                created.push(nodes.len() - 1);
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
                if o.op == "add" && o.shape.len() == 4 && o.shape[1] > 1 && o.shape[3] > 1 {
                    let (t, dd) = (o.shape[1] as u32, o.shape[3] as u32);
                    let img =
                        emit_add4d(t, dd, mil_path, &format!("{out_dir}/.tmp4d_{}", o.result))?;
                    let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                    if let Node::Prog { alias, zero, .. } = &mut nodes[idx] {
                        *alias = vec![(0, 0)];
                        *zero = vec![0];
                    }
                    prod.insert(o.result.clone(), (idx, Cur::Packed(t, dd)));
                    wants.push((idx, vec![(x, Cur::Packed(t, dd)), (y, Cur::Packed(t, dd))]));
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
                    "sigmoid" if matches!(elems, 128 | 256 | 512 | 1024) => emit_sigmoid_strided(
                        elems,
                        mil_path,
                        &format!("{out_dir}/.tmps_{}", o.result),
                    )?,
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
                let img = emit_repack(elems, mil_path, &format!("{out_dir}/.tmpt_{}", o.result))?;
                let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                prod.insert(o.result.clone(), (idx, Cur::Contig(elems)));
                wants.push((idx, vec![(x, Cur::Strided(elems))]));
            }
            "reshape" | "cast" => {
                let x = arg_of(o, "x").unwrap_or("x").to_string();
                if let Some(&pc) = prod.get(x.as_str()) {
                    prod.insert(o.result.clone(), pc);
                } else {
                    let img =
                        emit_repack(elems, mil_path, &format!("{out_dir}/.tmpr_{}", o.result))?;
                    let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                    prod.insert(o.result.clone(), (idx, Cur::Contig(elems)));
                    wants.push((idx, vec![(x, Cur::Strided(elems))]));
                }
            }
            "__conv_act" => {
                let x = arg_of(o, "x").unwrap_or("x").to_string();
                let w = weights_of(o)?;
                let act = match arg_of(o, "act").unwrap_or("sigmoid") {
                    "tanh" => FusedAct::Tanh,
                    "relu" => FusedAct::Relu,
                    "gelu" => FusedAct::Gelu,
                    "silu" => FusedAct::Silu,
                    _ => FusedAct::Sigmoid,
                };
                let img =
                    emit_conv_act(act, &w, mil_path, &format!("{out_dir}/.tmpca_{}", o.result))?;
                let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                prod.insert(o.result.clone(), (idx, Cur::Strided(64)));
                wants.push((idx, vec![(x, Cur::Strided(64))]));
            }
            "__conv_chain" => {
                let x = arg_of(o, "x").unwrap_or("x").to_string();
                let k = (o.args.len() - 1) / 2;
                let mut shapes = Vec::with_capacity(k);
                let mut flat_w = Vec::new();
                let mut offs = Vec::with_capacity(k);
                for ti in 0..k {
                    let wname = arg_of(o, &format!("w{ti}")).unwrap();
                    let cout: usize = arg_of(o, &format!("c{ti}")).unwrap().parse().unwrap();
                    let w = hwx::load_mil_weight_u16(&text, &mil_dir, wname)?;
                    let cin = w.len() / cout;
                    shapes.push((cin, cout));
                    flat_w.extend_from_slice(&w);
                    offs.push((128 + ti * (cin * cout * 2 + 64)) as u64);
                }
                let cin0 = shapes[0].0 as u32;
                let cout_last = shapes[k - 1].1 as u32;
                let img = emit_conv_chain(
                    &shapes,
                    &flat_w,
                    &offs,
                    &o.result,
                    mil_path,
                    &format!("{out_dir}/.tmpcc_{}", o.result),
                )?;
                let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                prod.insert(o.result.clone(), (idx, Cur::Strided(cout_last)));
                wants.push((idx, vec![(x, Cur::Strided(cin0))]));
            }
            "conv" => {
                let x = arg_of(o, "x").unwrap_or("x").to_string();
                let w = weights_of(o)?;
                let cout = o.shape.get(1).copied().unwrap_or(0).max(1) as usize;
                let spatial = (o.shape.get(2).copied().unwrap_or(1).max(1)
                    * o.shape.get(3).copied().unwrap_or(1).max(1))
                    as usize;
                if cout < 32 || w.len() % cout != 0 || w.len() / cout < 32 {
                    return Err(format!(
                        "conv {:?} weight len {} unsupported",
                        o.shape,
                        w.len()
                    ));
                }
                let cin = w.len() / cout;
                let img = emit_conv1x1_w(
                    cin,
                    cout,
                    spatial,
                    &w,
                    mil_path,
                    &format!("{out_dir}/.tmpc_{}", o.result),
                )?;
                let idx = push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                prod.insert(
                    o.result.clone(),
                    (idx, Cur::Strided((cout * spatial) as u32)),
                );
                wants.push((idx, vec![(x, Cur::Strided((cin * spatial) as u32))]));
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
                        let x_strided = match prod.get(x.as_str()) {
                            Some((_, Cur::Strided(_))) => true,
                            Some(_) => false,
                            None => xs.get(1).copied().unwrap_or(1) > 1,
                        };
                        let (img, want_cur) = if x_strided || (seq > 1 && m != 1) {
                            (
                                emit_conv1x1_w(
                                    k,
                                    n,
                                    seq,
                                    &w,
                                    mil_path,
                                    &format!("{out_dir}/.tmpm_{}", o.result),
                                )?,
                                Cur::Strided((k * seq) as u32),
                            )
                        } else {
                            (
                                emit_matmul_ns(
                                    k,
                                    n,
                                    &w,
                                    mil_path,
                                    &format!("{out_dir}/.tmpm_{}", o.result),
                                )?,
                                Cur::Contig(k as u32),
                            )
                        };
                        let idx =
                            push_prog(&mut nodes, &mut created, o.result.clone(), img, out_dir)?;
                        prod.insert(o.result.clone(), (idx, Cur::Strided((n * seq) as u32)));
                        wants.push((idx, vec![(x, want_cur)]));
                        continue;
                    }
                    return Err(format!("matmul const x{xs:?} -> {:?} unsupported", o.shape));
                }
                // transpose_x matmuls: outer product (x strided row-mask,
                // y a vector) vs attn·V (x strided scores, y packed
                // cache rows). Distinguish by y's shape: a lone trailing
                // dim means the vector form.
                let y_widem = ys.iter().skip(1).filter(|d| **d > 1).count();
                let xt = xs
                    .get(1)
                    .copied()
                    .unwrap_or(1)
                    .max(1)
                    .max(xs.get(2).copied().unwrap_or(1).max(1));
                if tx && xs.len() == 4 && ys.len() == 4 && y_widem == 1 {
                    // outer: x=[1,T,1,1] strided one-hot, y a D-vector
                    // (declared `[1,1,1,D]` or strided `[1,D,1,1]`)
                    let (t, d) = (xt as u32, ys[1].max(ys[3]).max(1) as u32);
                    let img =
                        emit_outer_t(t, d, mil_path, &format!("{out_dir}/.tmpo_{}", o.result))?;
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
                    && ys[0] == 1
                    && ys[1] > 1
                    && ys[2] == 1
                    && ys[3] > 1
                    && xs.iter().skip(1).filter(|d| **d > 1).count() == 1
                    && xs[1].max(xs[3]) == ys[3]
                {
                    // broadcast matvec: a vector (contig `[1,1,1,K]` or
                    // strided `[1,K,1,1]`) x matrix rows `[1,R,1,K]`
                    let (r, k) = (ys[1].max(1) as u32, ys[3].max(1) as u32);
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
    let mut adapters: HashMap<(usize, Cur), usize> = HashMap::new();
    for (nidx, ws) in &wants {
        let mut bs = Vec::new();
        for (port, (name, want)) in ws.iter().enumerate() {
            if let Some((pn, have)) = prod.get(name.as_str()).copied() {
                let src = if have == *want {
                    pn
                } else if let Some(&aidx) = adapters.get(&(pn, *want)) {
                    aidx
                } else {
                    match (have, *want) {
                        (Cur::Strided(n), Cur::Contig(m)) if n == m => {
                            let img =
                                emit_repack(n, mil_path, &format!("{out_dir}/.tmpa{n}_{name}"))?;
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
                            adapters.insert((pn, *want), aidx);
                            aidx
                        }
                        (Cur::Contig(n), Cur::Strided(m)) if n == m => {
                            let k = n as usize;
                            let mut eye = vec![0u16; k * k];
                            for i in 0..k {
                                eye[i * k + i] = 0x3c00;
                            }
                            let img = emit_matmul_ns(
                                k,
                                k,
                                &eye,
                                mil_path,
                                &format!("{out_dir}/.tmpu{n}_{name}"),
                            )?;
                            let aidx = nodes.len();
                            let path = format!("{out_dir}/n{aidx}_{name}_unrp.macho");
                            std::fs::write(&path, &img).map_err(|e| format!("{path}: {e}"))?;
                            nodes.push(Node::Prog {
                                name: format!("{name}_unrp"),
                                path: PathBuf::from(path),
                                binds: vec![BindSrc::Out(pn)],
                                alias: Vec::new(),
                                zero: Vec::new(),
                            });
                            pre.entry(*nidx).or_default().push(aidx);
                            adapters.insert((pn, *want), aidx);
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
    let externals: Vec<(String, Vec<(usize, usize)>)> = ext_map.into_iter().collect();
    let mut outputs = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if let Some(p) = l.find("-> (") {
            for nm in l[p + 4..].trim_end_matches([')', ';']).split(',') {
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
    /// One shared surface per node output (Rc: aliased nodes share the
    /// bound input surface — e.g. in-place KV-cache adds).
    bufs: Vec<Option<Rc<SharedSurface>>>,
    /// External input surfaces by name.
    ext_bufs: HashMap<String, Rc<SharedSurface>>,
    /// Producer output descs cached from each node's first load.
    out_descs: Vec<Option<TensorDesc>>,
    ext_descs: HashMap<String, TensorDesc>,
    /// Ext writes recorded before the surface materialized (first
    /// consumer bind creates it).
    pending: HashMap<String, Vec<ExtWrite>>,
    /// Resident node indices, oldest first.
    lru: VecDeque<usize>,
}

type ExtWrite = Rc<dyn Fn(&mut [u8], &TensorDesc)>;

impl GraphRuntime {
    /// Wire the plan without loading every program: surfaces are
    /// runtime-owned and programs load lazily on first eval, with
    /// LRU-style eviction whenever `loadModel` reports resource
    /// exhaustion (the ANE residency pool is finite).
    pub fn load(plan: Plan) -> Result<Self, String> {
        let n = plan.nodes.len();
        let mut rt = Self {
            progs: (0..n).map(|_| None).collect(),
            bufs: vec![None; n],
            out_descs: vec![None; n],
            ext_bufs: HashMap::new(),
            ext_descs: HashMap::new(),
            pending: HashMap::new(),
            lru: VecDeque::new(),
            plan,
        };
        let fills = rt.plan.const_fills.clone();
        for (name, vals) in fills {
            rt.write_ext(&name, move |b, d| {
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

    fn touch(&mut self, i: usize) {
        if let Some(p) = self.lru.iter().position(|v| *v == i) {
            self.lru.remove(p);
        }
        self.lru.push_back(i);
    }

    /// The shared surface backing node `i`'s output, honoring in-place
    /// aliases (`add`'s out aliases its cache input).
    fn node_surf(&self, i: usize) -> Option<Rc<SharedSurface>> {
        if let Node::Prog { alias, binds, .. } = &self.plan.nodes[i] {
            for (o, ii) in alias {
                if *o == 0 {
                    return self.port_surf(i, *ii);
                }
            }
            let _ = binds;
        }
        self.bufs[i].clone()
    }

    /// The surface bound to node `i` input port `p`.
    fn port_surf(&self, i: usize, p: usize) -> Option<Rc<SharedSurface>> {
        if let Node::Prog { binds, .. } = &self.plan.nodes[i] {
            match binds.get(p)? {
                BindSrc::Ext(name) => self.ext_bufs.get(name).cloned(),
                BindSrc::Out(src) => self.node_surf(*src),
            }
        } else {
            None
        }
    }

    /// Materialize the ext surface for `name` using `desc`, apply
    /// pending writes.
    fn ext_surface(&mut self, name: &str, desc: &TensorDesc) -> Result<Rc<SharedSurface>, String> {
        if let Some(s) = self.ext_bufs.get(name) {
            return Ok(s.clone());
        }
        let rc = surface_pooled(desc.bytes())
            .map_err(|e| format!("ext {name}: {e}"))?;
        self.ext_descs.insert(name.to_string(), desc.clone());
        if let Some(ws) = self.pending.remove(name) {
            for w in ws {
                w(rc.bytes_mut(), desc);
            }
        }
        self.ext_bufs.insert(name.to_string(), rc.clone());
        Ok(rc)
    }

    /// Load program `i` and bind all ports to runtime surfaces.
    fn load_node(&mut self, i: usize) -> Result<(), String> {
        let Node::Prog {
            path,
            binds,
            alias,
            zero,
            ..
        } = &self.plan.nodes[i]
        else {
            return Ok(());
        };
        let path = path.clone();
        let binds = binds.clone();
        let alias = alias.clone();
        let _ = zero.clone(); // fresh SharedSurfaces are zeroed at creation
        let mut p = EspressoProgram::load_precompiled(&path, &[])
            .map_err(|e| format!("{}: {e:?}", path.display()))?;
        // producer output desc + surface
        if self.out_descs[i].is_none() {
            self.out_descs[i] = p.output_descs.first().cloned();
        }
        let aliased = alias.iter().any(|(o, _)| *o == 0);
        if !aliased {
            if self.bufs[i].is_none() {
                let bytes = self.out_descs[i]
                    .as_ref()
                    .map(|d| d.bytes())
                    .unwrap_or(16384);
                self.bufs[i] = Some(
                    surface_pooled(bytes).map_err(|e| format!("n{i} out: {e}"))?,
                );
            }
            let s = self.bufs[i].clone().unwrap();
            p.adopt_output(0, s.as_ptr())
                .map_err(|e| format!("n{i} adopt out: {e:?}"))?;
        }
        // input ports
        for (port, b) in binds.iter().enumerate() {
            let surf = match b {
                BindSrc::Out(src) => self.node_surf(*src),
                BindSrc::Ext(name) => {
                    let d = p.input_descs[port].clone();
                    Some(self.ext_surface(name, &d)?)
                }
            };
            if let Some(s) = surf {
                p.adopt_input(port, s.as_ptr())
                    .map_err(|e| format!("bind node{i} port{port}: {e:?}"))?;
            }
        }
        // in-place aliases: out port rides the input's surface
        for (o, ii) in &alias {
            if let Some(s) = self.port_surf(i, *ii) {
                p.adopt_output(*o, s.as_ptr())
                    .map_err(|e| format!("n{i} alias: {e:?}"))?;
                if *o == 0 {
                    self.bufs[i] = Some(s);
                }
            }
        }
        self.progs[i] = Some(p);
        self.touch(i);
        Ok(())
    }

    /// Ensure node `i`'s program is resident; on resource exhaustion
    /// evict the least-recently-used resident program and retry.
    fn ensure(&mut self, i: usize) -> Result<(), String> {
        if self.progs[i].is_some() {
            self.touch(i);
            return Ok(());
        }
        loop {
            match self.load_node(i) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if !e.contains("no ANE resources") {
                        return Err(e);
                    }
                    // Bélády-optimal for cyclic DAG execution: the node
                    // that ran most recently has its next use furthest
                    // in the future (`N - 1` steps away).
                    let victim = self
                        .lru
                        .iter()
                        .rposition(|v| *v != i)
                        .map(|p| self.lru.remove(p).unwrap());
                    match victim {
                        Some(v) => self.progs[v] = None,
                        None => return Err(e),
                    }
                }
            }
        }
    }

    /// Write external input `name` into its shared surface via
    /// `f(bytes, desc)`; defers if no consumer has materialized it yet.
    pub fn write_ext(
        &mut self,
        name: &str,
        f: impl Fn(&mut [u8], &crate::espresso::TensorDesc) + 'static,
    ) {
        if let (Some(s), Some(d)) = (self.ext_bufs.get(name), self.ext_descs.get(name)) {
            f(s.bytes_mut(), d);
            return;
        }
        self.pending
            .entry(name.to_string())
            .or_default()
            .push(Rc::new(f));
    }

    pub fn output(&self, i: usize) -> &[u8] {
        self.bufs[i].as_ref().map(|s| s.bytes()).unwrap_or(&[])
    }
    pub fn out_desc(&self, i: usize) -> crate::espresso::TensorDesc {
        self.out_descs[i].clone().unwrap_or_default()
    }
    /// Node index by node/result name (internal intermediates too).
    pub fn node_named(&self, name: &str) -> Option<usize> {
        self.plan.nodes.iter().position(|n| match n {
            Node::Prog { name: n2, .. } => n2 == name,
            Node::Host(_) => false,
        })
    }

    pub fn out_node(&self, name: &str) -> Option<usize> {
        self.plan
            .outputs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, i)| *i)
    }

    /// Evict every resident program — frees the runtime's share of the
    /// ANE residency pool. Surfaces persist; programs lazily reload on
    /// the next `run`. For multi-runtime hosts (one GraphRuntime per
    /// compiled program) the pool is global, so a finished call should
    /// give its residents back.
    pub fn release_residents(&mut self) {
        for p in self.progs.iter_mut() {
            *p = None;
        }
        self.lru.clear();
    }

    /// Drop everything — resident programs AND every IOSurface. For
    /// hosts that stage all inputs per call and mirror state outside
    /// the runtime (coyote's VM), nothing here needs to survive a
    /// call boundary; both bounded pools (programs, IOSurfaces) get
    /// fully released. Pending ext writes are kept — they re-apply
    /// when the surface re-materializes.
    pub fn release_all(&mut self) {
        self.release_residents();
        // surfaces go back to the pool for reuse — create/destroy
        // churn outruns the driver's reclaim path.
        for b in self.bufs.iter_mut() {
            if let Some(s) = b.take() {
                surface_release(s);
            }
        }
        for d in self.out_descs.iter_mut() {
            *d = None;
        }
        for (_, s) in self.ext_bufs.drain() {
            surface_release(s);
        }
        self.ext_descs.clear();
    }

    /// Zero all persistent state surfaces (KV-cache accumulators marked
    /// with `zero` ports) for a fresh decode horizon without unloading
    /// resident ANE programs.
    pub fn reset_state(&mut self) {
        let mut ports = Vec::new();
        for (i, n) in self.plan.nodes.iter().enumerate() {
            if let Node::Prog { zero, .. } = n {
                for &port in zero {
                    ports.push((i, port));
                }
            }
        }
        for (i, port) in ports {
            if let Some(s) = self.port_surf(i, port) {
                for b in s.bytes_mut().iter_mut() {
                    *b = 0;
                }
            }
        }
    }

    /// Execute the whole graph once in scheduled order; programs load
    /// lazily and are evicted LRU-first under ANE residency pressure.
    pub fn run(&mut self) -> Result<(), String> {
        for i in self.plan.order.clone() {
            match &self.plan.nodes[i] {
                Node::Prog { .. } => {
                    self.ensure(i)?;
                    self.progs[i]
                        .as_ref()
                        .unwrap()
                        .eval()
                        .map_err(|e| format!("node {i}: {e:?}"))?
                }
                Node::Host(HostOp::RsqrtBroadcast {
                    src,
                    ext,
                    n,
                    eps,
                    gamma,
                }) => {
                    let (src, ext, n, eps, gamma) =
                        (src.clone(), ext.clone(), *n, *eps, gamma.clone());
                    let (surf, d) = match &src {
                        BindSrc::Out(idx) => {
                            (self.bufs[*idx].clone().unwrap(), self.out_desc(*idx))
                        }
                        BindSrc::Ext(name) => {
                            if !self.ext_bufs.contains_key(name) {
                                if let Some((_, uses)) =
                                    self.plan.externals.iter().find(|(nm, _)| nm == name)
                                {
                                    if let Some(&(ni, _)) = uses.first() {
                                        self.ensure(ni)?;
                                    }
                                }
                            }
                            (
                                self.ext_bufs.get(name).unwrap().clone(),
                                self.ext_descs.get(name).unwrap().clone(),
                            )
                        }
                    };
                    let b = surf.bytes();
                    // element axis: small-n layouts use channels (k), the
                    // n>256 family moves elements to the batch dim (n),
                    // and W handles the strided-vector fallthrough.
                    let off = |d: &crate::espresso::TensorDesc, i: u64| {
                        if d.k > 1 {
                            d.offset(0, 0, i, 0, 0)
                        } else if d.n > 1 {
                            d.offset(i, 0, 0, 0, 0)
                        } else {
                            d.offset(0, 0, 0, 0, i)
                        }
                    };
                    let sum: f32 = (0..n as u64)
                        .map(|i| {
                            let o = off(&d, i);
                            let v =
                                half::f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32();
                            v * v
                        })
                        .sum();
                    let r = (sum / n as f32 + eps).sqrt().recip();
                    self.write_ext(&ext, move |b, d| {
                        for i in 0..d.n.max(d.k).max(d.w) {
                            let o = off(d, i);
                            let g = gamma
                                .as_ref()
                                .and_then(|gv| gv.get(i as usize).copied())
                                .unwrap_or(1.0);
                            if o + 2 <= b.len() {
                                b[o..o + 2]
                                    .copy_from_slice(&half::f16::from_f32(r * g).to_le_bytes());
                            }
                        }
                    });
                }
                Node::Host(HostOp::Recip { src, ext, n }) => {
                    let (src, ext, n) = (src.clone(), ext.clone(), *n);
                    let (surf, d) = match &src {
                        BindSrc::Out(idx) => {
                            (self.bufs[*idx].clone().unwrap(), self.out_desc(*idx))
                        }
                        BindSrc::Ext(name) => {
                            if !self.ext_bufs.contains_key(name) {
                                if let Some((_, uses)) =
                                    self.plan.externals.iter().find(|(nm, _)| nm == name)
                                {
                                    if let Some(&(ni, _)) = uses.first() {
                                        self.ensure(ni)?;
                                    }
                                }
                            }
                            (
                                self.ext_bufs.get(name).unwrap().clone(),
                                self.ext_descs.get(name).unwrap().clone(),
                            )
                        }
                    };
                    let b = surf.bytes();
                    let off = |d: &crate::espresso::TensorDesc, i: u64| {
                        if d.k > 1 {
                            d.offset(0, 0, i, 0, 0)
                        } else if d.n > 1 {
                            d.offset(i, 0, 0, 0, 0)
                        } else {
                            d.offset(0, 0, 0, 0, i)
                        }
                    };
                    let rvals: Vec<f32> = (0..n as u64)
                        .map(|i| {
                            let o = off(&d, i);
                            let v =
                                half::f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32();
                            1.0 / v
                        })
                        .collect();
                    self.write_ext(&ext, move |b, d| {
                        for (i, &rv) in rvals.iter().enumerate() {
                            let o = off(d, i as u64);
                            if o + 2 <= b.len() {
                                b[o..o + 2].copy_from_slice(&half::f16::from_f32(rv).to_le_bytes());
                            }
                        }
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn graph_scheduler_rms_q8_swiglu_exec() {
        let dir = std::env::temp_dir().join(format!("graph_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let (d, ff) = (64usize, 128usize);
        let gam: Vec<u8> = (0..d)
            .flat_map(|i| f16::from_f32(0.8 + (i % 7) as f32 * 0.05).to_le_bytes())
            .collect();
        std::fs::write(dir.join("gam.bin"), &gam).unwrap();

        let wu_q8: Vec<i8> = (0..ff * d)
            .map(|i| (((i * 23 + 7) % 199) as i32 - 99) as i8)
            .collect();
        let wu_sc: Vec<f32> = (0..ff * 2)
            .map(|i| f16::from_f32(0.002 * ((i * 11 % 31) as f32 + 1.0)).to_f32())
            .collect();
        std::fs::write(
            dir.join("wu_q8.bin"),
            wu_q8.iter().map(|&c| c as u8).collect::<Vec<u8>>(),
        )
        .unwrap();
        std::fs::write(
            dir.join("wu_sc.bin"),
            wu_sc
                .iter()
                .flat_map(|&s| f16::from_f32(s).to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();

        let wg: Vec<f32> = (0..ff * d)
            .map(|i| f16::from_f32(((i * 17 + 3) % 29) as f32 / 64.0 - 0.22).to_f32())
            .collect();
        let wd: Vec<f32> = (0..d * ff)
            .map(|i| f16::from_f32(((i * 13 + 5) % 23) as f32 / 64.0 - 0.17).to_f32())
            .collect();
        std::fs::write(
            dir.join("wg.bin"),
            wg.iter()
                .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        std::fs::write(
            dir.join("wd.bin"),
            wd.iter()
                .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();

        let mil = format!(
            r#"program(1.3)
{{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {{
        tensor<fp16, [64]> gam = const()[val = tensor<fp16, [64]>(BLOBFILE(path = string("{d}/gam.bin"), offset = uint64(0)))];
        tensor<int8, [128, 64]> Wu_q = const()[val = tensor<int8, [128, 64]>(BLOBFILE(path = string("{d}/wu_q8.bin"), offset = uint64(0)))];
        tensor<fp16, [128, 2]> Wu_s = const()[val = tensor<fp16, [128, 2]>(BLOBFILE(path = string("{d}/wu_sc.bin"), offset = uint64(0)))];
        tensor<fp16, [128, 64]> Wu = constexpr_blockwise_shift_scale(data = Wu_q, scale = Wu_s)[name = string("Wu")];
        tensor<fp16, [128, 64]> Wg = const()[val = tensor<fp16, [128, 64]>(BLOBFILE(path = string("{d}/wg.bin"), offset = uint64(0)))];
        tensor<fp16, [64, 128]> Wd = const()[val = tensor<fp16, [64, 128]>(BLOBFILE(path = string("{d}/wd.bin"), offset = uint64(0)))];
        tensor<bool, []> bt = const()[val = bool(true)];
        tensor<fp16, [1, 64, 1, 1]> xn = rms_norm(x = x, w = gam, eps = fp16(0.00001))[name = string("xn")];
        tensor<fp16, [1, 1, 1, 128]> u = matmul(x = xn, y = Wu, transpose_y = bt)[name = string("u")];
        tensor<fp16, [1, 1, 1, 128]> g = matmul(x = xn, y = Wg, transpose_y = bt)[name = string("g")];
        tensor<fp16, [1, 128, 1, 1]> gs = sigmoid(x = g)[name = string("gs")];
        tensor<fp16, [1, 128, 1, 1]> gsl = mul(x = g, y = gs)[name = string("gsl")];
        tensor<fp16, [1, 128, 1, 1]> fu = mul(x = u, y = gsl)[name = string("fu")];
        tensor<fp16, [1, 1, 1, 64]> dn = matmul(x = fu, y = Wd, transpose_y = bt)[name = string("dn")];
        tensor<fp16, [1, 64, 1, 1]> y = add(x = x, y = dn)[name = string("y")];
    }} -> (y);
}}"#,
            d = dir.display()
        );
        let mil_path = dir.join("m.mil");
        std::fs::write(&mil_path, &mil).unwrap();
        let plan = compile_graph(
            mil_path.to_str().unwrap(),
            dir.join("out").to_str().unwrap(),
        )
        .unwrap();
        let mut rt = GraphRuntime::load(plan).unwrap();

        let xv: Vec<f32> = (0..d)
            .map(|i| f16::from_f32(((i * 9 + 4) % 17) as f32 / 16.0 - 0.5).to_f32())
            .collect();
        let xc = xv.clone();
        rt.write_ext("x", move |b, desc| {
            for i in 0..d {
                let o = desc.offset(0, 0, i as u64, 0, 0);
                b[o..o + 2].copy_from_slice(&f16::from_f32(xc[i]).to_le_bytes());
            }
        });
        rt.run().unwrap();

        let r = (xv.iter().map(|a| a * a).sum::<f32>() / d as f32 + 1e-5)
            .sqrt()
            .recip();
        let xnr: Vec<f32> = (0..d)
            .map(|i| {
                let g = f16::from_f32(0.8 + (i % 7) as f32 * 0.05).to_f32();
                xv[i] * f16::from_f32(r * g).to_f32()
            })
            .collect();
        let ur: Vec<f32> = (0..ff)
            .map(|j| {
                (0..d)
                    .map(|i| {
                        let w =
                            f16::from_f32(wu_q8[j * d + i] as f32 * wu_sc[j * 2 + i / 32]).to_f32();
                        xnr[i] * w
                    })
                    .sum()
            })
            .collect();
        let gr: Vec<f32> = (0..ff)
            .map(|j| (0..d).map(|i| xnr[i] * wg[j * d + i]).sum())
            .collect();
        let fur: Vec<f32> = ur
            .iter()
            .zip(&gr)
            .map(|(u, g)| u * g / (1.0 + (-g).exp()))
            .collect();
        let dnr: Vec<f32> = (0..d)
            .map(|j| (0..ff).map(|i| fur[i] * wd[j * ff + i]).sum())
            .collect();
        let yr: Vec<f32> = xv.iter().zip(&dnr).map(|(a, b)| a + b).collect();

        let yn = rt.out_node("y").unwrap();
        let dy = rt.out_desc(yn);
        let zb = rt.output(yn);
        for j in 0..d {
            let o = dy.offset(0, 0, j as u64, 0, 0);
            let hw = f16::from_bits(u16::from_le_bytes([zb[o], zb[o + 1]])).to_f32();
            assert!(
                (hw - yr[j]).abs() < 0.03 + yr[j].abs() * 0.05,
                "y[{j}]: hw={hw} ref={}",
                yr[j]
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn graph_fuses_conv_chain_and_conv_silu() {
        let dir = std::env::temp_dir().join(format!("graph_fuse_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mk = |tag: u8, rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| {
                    f16::from_f32(((i * 11 + tag as usize * 7) % 23) as f32 / 64.0 - 0.17).to_f32()
                })
                .collect()
        };
        let w1 = mk(1, 96, 64);
        let w2 = mk(2, 64, 96);
        let w3 = mk(3, 64, 64);
        for (name, v) in [("w1.bin", &w1), ("w2.bin", &w2), ("w3.bin", &w3)] {
            let b: Vec<u8> = v
                .iter()
                .flat_map(|&x| f16::from_f32(x).to_le_bytes())
                .collect();
            std::fs::write(dir.join(name), b).unwrap();
        }
        // 2-op conv chain (64->96->64) + fused conv+silu (64->64 + sigmoid + mul) + residual add
        let mil = format!(
            r#"program(1.3)
{{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {{
        tensor<fp16, [96, 64, 1, 1]> W1 = const()[val = tensor<fp16, [96, 64, 1, 1]>(BLOBFILE(path = string("{d}/w1.bin"), offset = uint64(0)))];
        tensor<fp16, [64, 96, 1, 1]> W2 = const()[val = tensor<fp16, [64, 96, 1, 1]>(BLOBFILE(path = string("{d}/w2.bin"), offset = uint64(0)))];
        tensor<fp16, [64, 64, 1, 1]> W3 = const()[val = tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path = string("{d}/w3.bin"), offset = uint64(0)))];
        tensor<fp16, [1, 96, 1, 1]> c1 = conv(weight = W1, x = x)[name = string("c1")];
        tensor<fp16, [1, 64, 1, 1]> c2 = conv(weight = W2, x = c1)[name = string("c2")];
        tensor<fp16, [1, 64, 1, 1]> g = conv(weight = W3, x = c2)[name = string("g")];
        tensor<fp16, [1, 64, 1, 1]> gs = sigmoid(x = g)[name = string("gs")];
        tensor<fp16, [1, 64, 1, 1]> gsl = mul(x = g, y = gs)[name = string("gsl")];
        tensor<fp16, [1, 64, 1, 1]> y = add(x = c2, y = gsl)[name = string("y")];
    }} -> (y);
}}"#,
            d = dir.display()
        );
        let mil_path = dir.join("m.mil");
        std::fs::write(&mil_path, &mil).unwrap();
        let plan = compile_graph(
            mil_path.to_str().unwrap(),
            dir.join("out").to_str().unwrap(),
        )
        .unwrap();
        // 6 logical ops (c1, c2, g, gs, gsl, y) fused into 3 ANE programs:
        // node 0 = __conv_chain(c1->c2), node 1 = __conv_act(g->gs->gsl), node 2 = add(c2, gsl)
        assert_eq!(
            plan.nodes.len(),
            3,
            "expected 3 fused nodes, got {}",
            plan.nodes.len()
        );

        let mut rt = GraphRuntime::load(plan).unwrap();
        let xv: Vec<f32> = (0..64)
            .map(|i| f16::from_f32(((i * 5 + 3) % 13) as f32 / 16.0 - 0.35).to_f32())
            .collect();
        let xc = xv.clone();
        rt.write_ext("x", move |b, desc| {
            for i in 0..64 {
                let o = desc.offset(0, 0, i as u64, 0, 0);
                b[o..o + 2].copy_from_slice(&f16::from_f32(xc[i]).to_le_bytes());
            }
        });
        rt.run().unwrap();

        let mm = |w: &[f32], v: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            (0..rows)
                .map(|r| (0..cols).map(|c| v[c] * w[r * cols + c]).sum())
                .collect()
        };
        let c1r = mm(&w1, &xv, 96, 64);
        let c2r = mm(&w2, &c1r, 64, 96);
        let gr = mm(&w3, &c2r, 64, 64);
        let yr: Vec<f32> = c2r
            .iter()
            .zip(&gr)
            .map(|(a, g)| a + g / (1.0 + (-g).exp()))
            .collect();

        let yn = rt.out_node("y").unwrap();
        let dy = rt.out_desc(yn);
        let zb = rt.output(yn);
        for j in 0..64 {
            let o = dy.offset(0, 0, j as u64, 0, 0);
            let hw = f16::from_bits(u16::from_le_bytes([zb[o], zb[o + 1]])).to_f32();
            assert!(
                (hw - yr[j]).abs() < 0.03 + yr[j].abs() * 0.05,
                "y[{j}]: hw={hw} ref={}",
                yr[j]
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// World-first proof: 2-layer neural network forward + backward
    /// propagation + in-place SGD weight updates running 100% on the
    /// Apple Neural Engine via self-emitted HWX programs and
    /// `SharedSurface` aliasing — zero CPU/GPU gradient math.
    #[test]
    fn ane_onchip_backprop_training_exec() {
        let dir = std::env::temp_dir().join(format!("ane_train_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let d = 64usize;
        let lr = 0.15f32;
        // 2-layer MLP training graph on ANE:
        //   Forward:
        //     z1 = W1 * x                     (emit_avw)
        //     h  = sigmoid(z1)                (emit_unary_lut)
        //     y  = W2 * h                     (emit_avw)
        //   Backward:
        //     dy      = y - target            (ew sub)
        //     ndy     = dy * neg_lr           (ew mul)
        //     dW2     = outer(ndy, h)         (emit_outer_t)
        //     W2_new  = W2 + dW2              (emit_add4d, aliased in-place to W2!)
        //     dh      = W2^T * dy             (emit_attn_v, auto-unrepacked to Strided)
        //     omh     = one - h               (ew sub)
        //     ds      = h * omh               (ew mul: sigmoid'(z1))
        //     dz1     = dh * ds               (ew mul)
        //     ndz1    = dz1 * neg_lr          (ew mul)
        //     dW1     = outer(ndz1, x)        (emit_outer_t)
        //     W1_new  = W1 + dW1              (emit_add4d, aliased in-place to W1!)
        let mil = r#"program(1.3)
{
    func main<ios18>(tensor<fp16, [1, 1, 1, 64]> x,
                     tensor<fp16, [1, 64, 1, 1]> target,
                     tensor<fp16, [1, 64, 1, 1]> neg_lr,
                     tensor<fp16, [1, 64, 1, 1]> one,
                     tensor<fp16, [1, 64, 1, 64]> W1,
                     tensor<fp16, [1, 64, 1, 64]> W2) {
        tensor<bool, []> bt = const()[val = bool(true)];
        tensor<fp16, [1, 64, 1, 1]> z1 = matmul(x = x, y = W1, transpose_y = bt)[name = string("z1")];
        tensor<fp16, [1, 64, 1, 1]> h = sigmoid(x = z1)[name = string("h")];
        tensor<fp16, [1, 64, 1, 1]> y = matmul(x = h, y = W2, transpose_y = bt)[name = string("y")];
        tensor<fp16, [1, 64, 1, 1]> dy = sub(x = y, y = target)[name = string("dy")];
        tensor<fp16, [1, 64, 1, 1]> ndy = mul(x = dy, y = neg_lr)[name = string("ndy")];
        tensor<fp16, [1, 1, 1, 64]> dh = matmul(x = dy, y = W2, transpose_x = bt)[name = string("dh")];
        tensor<fp16, [1, 64, 1, 1]> omh = sub(x = one, y = h)[name = string("omh")];
        tensor<fp16, [1, 64, 1, 1]> ds = mul(x = h, y = omh)[name = string("ds")];
        tensor<fp16, [1, 64, 1, 1]> dz1 = mul(x = dh, y = ds)[name = string("dz1")];
        tensor<fp16, [1, 64, 1, 1]> ndz1 = mul(x = dz1, y = neg_lr)[name = string("ndz1")];
        tensor<fp16, [1, 64, 1, 64]> dW2 = matmul(x = ndy, y = h, transpose_x = bt)[name = string("dW2")];
        tensor<fp16, [1, 64, 1, 64]> W2_new = add(x = W2, y = dW2)[name = string("W2_new")];
        tensor<fp16, [1, 64, 1, 64]> dW1 = matmul(x = ndz1, y = x, transpose_x = bt)[name = string("dW1")];
        tensor<fp16, [1, 64, 1, 64]> W1_new = add(x = W1, y = dW1)[name = string("W1_new")];
    } -> (y, W1_new, W2_new);
}"#;
        let mil_path = dir.join("train.mil");
        std::fs::write(&mil_path, mil).unwrap();
        let plan = compile_graph(
            mil_path.to_str().unwrap(),
            dir.join("out").to_str().unwrap(),
        )
        .unwrap();
        let mut rt = GraphRuntime::load(plan).unwrap();

        let xv: Vec<f32> = (0..d)
            .map(|i| f16::from_f32(((i * 7 + 3) % 19) as f32 / 19.0 - 0.45).to_f32())
            .collect();
        let tv: Vec<f32> = (0..d)
            .map(|i| f16::from_f32(((i * 11 + 5) % 23) as f32 / 23.0 - 0.5).to_f32())
            .collect();
        let mut w1_ref: Vec<f32> = (0..d * d)
            .map(|i| f16::from_f32((((i * 13 + 7) % 31) as f32 - 15.0) * 0.015).to_f32())
            .collect();
        let mut w2_ref: Vec<f32> = (0..d * d)
            .map(|i| f16::from_f32((((i * 17 + 11) % 29) as f32 - 14.0) * 0.015).to_f32())
            .collect();

        let xc = xv.clone();
        rt.write_ext("x", move |b, desc| {
            for i in 0..d {
                let o = desc.offset(0, 0, 0, 0, i as u64);
                b[o..o + 2].copy_from_slice(&f16::from_f32(xc[i]).to_le_bytes());
            }
        });
        let tc = tv.clone();
        rt.write_ext("target", move |b, desc| {
            for i in 0..d {
                let o = desc.offset(0, 0, i as u64, 0, 0);
                b[o..o + 2].copy_from_slice(&f16::from_f32(tc[i]).to_le_bytes());
            }
        });
        rt.write_ext("neg_lr", move |b, desc| {
            for i in 0..d {
                let o = desc.offset(0, 0, i as u64, 0, 0);
                b[o..o + 2].copy_from_slice(&f16::from_f32(-lr).to_le_bytes());
            }
        });
        rt.write_ext("one", move |b, desc| {
            for i in 0..d {
                let o = desc.offset(0, 0, i as u64, 0, 0);
                b[o..o + 2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
            }
        });
        let w1_init = w1_ref.clone();
        rt.write_ext("W1", move |b, desc| {
            for r in 0..d {
                for c in 0..d {
                    let o = desc.offset(0, 0, r as u64, 0, c as u64);
                    b[o..o + 2].copy_from_slice(&f16::from_f32(w1_init[r * d + c]).to_le_bytes());
                }
            }
        });
        let w2_init = w2_ref.clone();
        rt.write_ext("W2", move |b, desc| {
            for r in 0..d {
                for c in 0..d {
                    let o = desc.offset(0, 0, r as u64, 0, c as u64);
                    b[o..o + 2].copy_from_slice(&f16::from_f32(w2_init[r * d + c]).to_le_bytes());
                }
            }
        });

        let mut loss_first = 0.0f32;
        let mut loss_last = 0.0f32;
        for step in 0..16usize {
            rt.run().unwrap();

            // CPU reference step (matching fp16 intermediate ops)
            let z1: Vec<f32> = (0..d)
                .map(|r| (0..d).map(|c| xv[c] * w1_ref[r * d + c]).sum())
                .collect();
            let h: Vec<f32> = z1.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
            let y_ref: Vec<f32> = (0..d)
                .map(|r| (0..d).map(|c| h[c] * w2_ref[r * d + c]).sum())
                .collect();
            let dy: Vec<f32> = y_ref.iter().zip(&tv).map(|(y, t)| y - t).collect();
            let dh: Vec<f32> = (0..d)
                .map(|c| (0..d).map(|r| dy[r] * w2_ref[r * d + c]).sum())
                .collect();
            let dz1: Vec<f32> = (0..d).map(|i| dh[i] * h[i] * (1.0 - h[i])).collect();
            for r in 0..d {
                for c in 0..d {
                    w2_ref[r * d + c] += (-lr * dy[r]) * h[c];
                    w1_ref[r * d + c] += (-lr * dz1[r]) * xv[c];
                }
            }

            let yn = rt.out_node("y").unwrap();
            let dy_desc = rt.out_desc(yn);
            let yb = rt.output(yn);
            let mut mse_hw = 0.0f32;
            for j in 0..d {
                let o = dy_desc.offset(0, 0, j as u64, 0, 0);
                let hw = f16::from_bits(u16::from_le_bytes([yb[o], yb[o + 1]])).to_f32();
                mse_hw += (hw - tv[j]).powi(2);
                assert!(
                    (hw - y_ref[j]).abs() < 0.04 + y_ref[j].abs() * 0.05,
                    "step {step} y[{j}]: hw={hw} ref={}",
                    y_ref[j]
                );
            }
            mse_hw /= d as f32;
            if step == 0 {
                loss_first = mse_hw;
            }
            loss_last = mse_hw;
        }
        println!("ANE 2-layer backprop MSE: step0={loss_first:.6} -> step15={loss_last:.6}");
        assert!(
            loss_last < loss_first * 0.05,
            "expected >95% loss reduction on ANE, got {loss_first} -> {loss_last}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
