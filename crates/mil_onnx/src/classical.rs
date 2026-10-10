//! Classical ML → MIL: sklearn-style tree ensembles and GLMs.
//!
//! # Why MIL programs, not spec `treeEnsembleRegressor`/`glmRegressor`
//!
//! coremltools emits classical models into the *non-neural* spec
//! sections (`TreeEnsembleClassifier`, `GLMRegressor`, `Pipeline`). Those
//! messages are a second, larger protobuf schema we'd have to hand-encode
//! blind — and nothing in this workspace can verify them end-to-end
//! (mil_verify/mil_infer exercise the `mlProgram` path). So this module
//! lowers classical models to ordinary MIL tensor programs instead:
//! same `.mlpackage`, verified through `coremlc` + `mil_infer`.
//!
//! - **GLM** → `linear` + post-eval nonlinearity + `argmax` label.
//! - **Tree ensemble** → *violation-matrix* decomposition. Every leaf of
//!   every tree is a conjunction of branch decisions, so with
//!   `le_j = x[f_j] <= t_j` and `lt_j = x[f_j] < t_j` over all `J`
//!   internal nodes, leaf `l` is active iff
//!   `off_l + le·Mle[l,:] + lt·Mlt[l,:] == 0` — two matmuls, an `equal`,
//!   and a `matmul` over leaf values produce the ensemble score. Exact
//!   for axis-aligned trees; memory is `O(leaves × internal nodes)`.
//!
//! # JSON schema (documented; `pickle` is out of scope by design)
//!
//! ```json
//! {
//!   "kind": "tree_ensemble_classifier",          // or tree_ensemble_regressor,
//!                                                //    glm_classifier, glm_regressor
//!   "n_features": 4,
//!   "input_name": "features",                    // optional, default "features"
//!   "post_transform": "none|logistic|softmax|probit",
//!
//!   // classifiers:
//!   "class_labels": [0, 1, 2],                   // ints; enables a "label" output
//!
//!   // tree ensembles:
//!   "n_outputs": 3,                              // regressor: targets; classifier: classes
//!   "base_values": [0.0],
//!   "aggregate": "sum|average",                  // regressor only
//!   "trees": [{
//!     "weight": 1.0,
//!     "nodes": [
//!       {"id": 0, "feature": 2, "threshold": 1.5, "mode": "leq",
//!        "left": 1, "right": 2},                 // internal node
//!       {"id": 1, "value": [0.7, -0.2, -0.5]}    // leaf (value per output)
//!     ]
//!   }],
//!
//!   // GLMs:
//!   "weights":   [[w0, w1, ...], ...],           // (n_out × n_features), or flat [n_features]
//!   "intercept": [b0, b1, ...]
//! }
//! ```
//!
//! `mode` may be `leq` (sklearn default), `lt`, `gte`, `gt`. Tree `id`s
//! are per-tree; `left` is the branch taken when the predicate holds.
//!
//! # `ai.onnx.ml` inside ONNX graphs
//!
//! [`map_ml_node`] handles `TreeEnsembleClassifier/Regressor`,
//! `LinearClassifier/Regressor`, `Normalizer`, and `Scaler` nodes —
//! skl2onnx exports land here with the same lowering.

use crate::map::Ctx;
use crate::model::NodeProto;
use crate::{BuiltModel, OnnxError};
use mil_spec::{bind, DType, Feature, TensorType, Value, ValueType, NVT};
use std::collections::{HashMap, HashSet};

type Res<T> = Result<T, OnnxError>;

// ================= shared IR =================

/// Branch predicate mode (`x ⋄ threshold` → true child).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Leq,
    Lt,
    Gte,
    Gt,
}

impl Mode {
    fn parse(s: &str) -> Res<Mode> {
        match s {
            "leq" | "LEQ" | "BRANCH_LEQ" => Ok(Mode::Leq),
            "lt" | "LT" | "BRANCH_LT" => Ok(Mode::Lt),
            "gte" | "GTE" | "BRANCH_GTE" => Ok(Mode::Gte),
            "gt" | "GT" | "BRANCH_GT" => Ok(Mode::Gt),
            m => Err(OnnxError::Unsupported(format!(
                "tree node mode '{m}' — supported: leq/lt/gte/gt"
            ))),
        }
    }
    /// Complement predicate (false branch).
    fn complement(self) -> Mode {
        match self {
            Mode::Leq => Mode::Gt,
            Mode::Lt => Mode::Gte,
            Mode::Gte => Mode::Lt,
            Mode::Gt => Mode::Leq,
        }
    }
    /// Violation contribution of requiring predicate `self`:
    /// `(off, cle, clt)` such that violation adds `off + cle·le + clt·lt`,
    /// where `le = (d<=0)`, `lt = (d<0)`.
    fn coeffs(self) -> (f64, f64, f64) {
        match self {
            Mode::Leq => (1.0, -1.0, 0.0), // viol = 1 - le
            Mode::Lt => (1.0, 0.0, -1.0),  // viol = 1 - lt
            Mode::Gte => (0.0, 0.0, 1.0),  // viol = lt
            Mode::Gt => (0.0, 1.0, 0.0),   // viol = le
        }
    }
}

/// Post-evaluation transform on raw scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostTransform {
    /// Identity.
    None,
    /// `1/(1+e^-x)` per class.
    Logistic,
    /// `softmax` over classes.
    Softmax,
    /// `Φ(x)` per class (`0.5(1+erf(x/√2))`).
    Probit,
}

impl PostTransform {
    fn parse(s: &str) -> Res<PostTransform> {
        match s {
            "none" | "NONE" => Ok(PostTransform::None),
            "logistic" | "LOGISTIC" => Ok(PostTransform::Logistic),
            "softmax" | "SOFTMAX" => Ok(PostTransform::Softmax),
            "softmax_zero" | "SOFTMAX_ZERO" => Ok(PostTransform::Softmax),
            "probit" | "PROBIT" => Ok(PostTransform::Probit),
            p => Err(OnnxError::Unsupported(format!("post_transform '{p}'"))),
        }
    }
}

/// One decision-tree node.
#[derive(Clone, Debug)]
enum TNode {
    /// Internal nodes: split predicate + children (true/false).
    Internal {
        feature: usize,
        threshold: f64,
        mode: Mode,
        on_true: i64,
        on_false: i64,
    },
    /// Leaf: per-output values (already × tree weight).
    Leaf(Vec<f64>),
}

/// Assembled violation-matrix lowering for an ensemble.
struct EnsMats {
    /// Internal-node count.
    j: usize,
    /// Leaf count.
    l: usize,
    /// Per-internal-node feature index.
    feat: Vec<i32>,
    /// Per-internal-node threshold.
    thr: Vec<f32>,
    /// `(J, L)` violation matrix for the `le` mask — already transposed
    /// for `matmul(le_f (B,J), mle_t (J,L))`.
    mle_t: Vec<f32>,
    /// `(J, L)` for the `lt` mask.
    mlt_t: Vec<f32>,
    /// `(L,)` constant violation offsets.
    off: Vec<f32>,
    /// `(L, K)` leaf values (tree weight folded in).
    v: Vec<f32>,
    /// Outputs per score (classes/targets).
    k: usize,
}

/// DFS a tree, accumulating path predicates into the violation matrices.
/// `path` holds `(pivot_col, off, cle, clt)` violation contributions per
/// ancestor step.
fn build_tree_mats(nodes: &HashMap<i64, TNode>, root: i64, ctx: &mut EnsBuild) -> Res<()> {
    fn walk(
        nodes: &HashMap<i64, TNode>,
        id: i64,
        ctx: &mut EnsBuild,
        path: &mut Vec<(usize, f64, f64, f64)>,
    ) -> Res<()> {
        let n = nodes.get(&id).ok_or_else(|| {
            OnnxError::Schema(format!("tree: node id {id} referenced but absent"))
        })?;
        match n {
            TNode::Leaf(v) => {
                if v.len() != ctx.k {
                    return Err(OnnxError::Schema(format!(
                        "leaf value has {} elems, expected {}",
                        v.len(),
                        ctx.k
                    )));
                }
                let leaf_row = ctx.leaf_count;
                ctx.leaf_count += 1;
                for &(j, o, cle, clt) in path.iter() {
                    ctx.off[leaf_row] += o as f32;
                    ctx.mle_t[j * ctx.l_cap + leaf_row] += cle as f32;
                    ctx.mlt_t[j * ctx.l_cap + leaf_row] += clt as f32;
                }
                for (ki, &val) in v.iter().enumerate() {
                    ctx.v[leaf_row * ctx.k + ki] += val as f32;
                }
                Ok(())
            }
            TNode::Internal {
                feature,
                threshold,
                mode,
                on_true,
                on_false,
            } => {
                let j = ctx.node_col(*feature, *threshold, *mode);
                let (ot, alet, alt_t) = mode.coeffs();
                let (of, alef, alt_f) = mode.complement().coeffs();
                path.push((j, ot, alet, alt_t));
                walk(nodes, *on_true, ctx, path)?;
                path.pop();
                path.push((j, of, alef, alt_f));
                walk(nodes, *on_false, ctx, path)?;
                path.pop();
                Ok(())
            }
        }
    }
    walk(nodes, root, ctx, &mut Vec::new())
}

/// Builder state while accumulating violation matrices.
struct EnsBuild<'a> {
    k: usize,
    l_cap: usize,
    leaf_count: usize,
    /// pivot column → (feature, threshold, mode)
    pivots: Vec<(usize, f64, Mode)>,
    /// pivot dedup: same (feature, threshold, mode) shares a column.
    pivot_map: HashMap<(usize, u64, u8), usize>,
    mle_t: Vec<f32>,
    mlt_t: Vec<f32>,
    off: Vec<f32>,
    v: &'a mut Vec<f32>,
}

impl<'a> EnsBuild<'a> {
    fn node_col(&mut self, feature: usize, thr: f64, mode: Mode) -> usize {
        let key = (feature, thr.to_bits(), mode as u8);
        if let Some(&j) = self.pivot_map.get(&key) {
            return j;
        }
        let j = self.pivots.len();
        // grow the (J,L) matrices by one row
        self.mle_t.resize((j + 1) * self.l_cap, 0.0);
        self.mlt_t.resize((j + 1) * self.l_cap, 0.0);
        self.pivots.push((feature, thr, mode));
        self.pivot_map.insert(key, j);
        j
    }
}

/// Total leaf count across trees (capacity for the matrices).
fn count_leaves(trees: &[TreeJSON]) -> usize {
    trees
        .iter()
        .map(|t| t.nodes.iter().filter(|n| n.leaf.is_some()).count())
        .sum()
}

/// Lower an ensemble to MIL ops on `x` (rank-2 `(B, F)` producer name).
/// Returns the `(B, K)` **votes** tensor — the raw leaf-value
/// combination *without* `base_values`, so callers can apply the
/// aggregate (`SUM`/`AVERAGE`) before adding the base (ONNX semantics:
/// `post(aggregate(votes) + base)`).
fn lower_ensemble(
    cx: &mut Ctx,
    trees: &[TreeJSON],
    n_out: usize,
    x: &str,
    x_shape: &[i64],
    pfx: &str,
) -> Res<String> {
    let b = x_shape[0];
    let k = n_out;
    let l_cap = count_leaves(trees);
    if l_cap == 0 {
        return Err(OnnxError::Schema("tree ensemble has no leaves".into()));
    }
    let mut v = vec![0.0f32; l_cap * k];
    let mut build = EnsBuild {
        k,
        l_cap,
        leaf_count: 0,
        pivots: Vec::new(),
        pivot_map: HashMap::new(),
        mle_t: Vec::new(),
        mlt_t: Vec::new(),
        off: vec![0.0; l_cap],
        v: &mut v,
    };
    for t in trees {
        let map: HashMap<i64, TNode> = t
            .nodes
            .iter()
            .map(|n| n.to_tnode(t.weight).map(|tn| (n.id, tn)))
            .collect::<Res<_>>()?;
        let root = find_root(&map, t)?;
        build_tree_mats(&map, root, &mut build)?;
    }
    let l = build.leaf_count;
    let jj = build.pivots.len();
    // The (J, l_cap) accumulator rows were laid out with stride l_cap;
    // compact to stride l now that the true leaf count is known.
    let compact = |m: &[f32]| -> Vec<f32> {
        (0..jj)
            .flat_map(|j| m[j * l_cap..j * l_cap + l].iter().copied())
            .collect()
    };
    let mats = EnsMats {
        j: jj,
        l,
        feat: build.pivots.iter().map(|p| p.0 as i32).collect(),
        thr: build.pivots.iter().map(|p| p.1 as f32).collect(),
        mle_t: compact(&build.mle_t),
        mlt_t: compact(&build.mlt_t),
        off: build.off[..l].to_vec(),
        v: v[..l * k].to_vec(),
        k,
    };
    emit_ensemble(cx, &mats, x, b, pfx)
}

fn find_root(map: &HashMap<i64, TNode>, t: &TreeJSON) -> Res<i64> {
    // root = node never referenced as a child
    let mut child = HashSet::new();
    for n in &t.nodes {
        if let Some(l) = n.left {
            child.insert(l);
        }
        if let Some(r) = n.right {
            child.insert(r);
        }
    }
    t.nodes
        .iter()
        .map(|n| n.id)
        .find(|id| !child.contains(id) && map.contains_key(id))
        .ok_or_else(|| OnnxError::Schema("tree has no root".into()))
}

/// Emit the violation-matrix ops. `x` is `(B, F)` fp16.
/// Returns `(B, K)` votes name (leaf values only — no base).
fn emit_ensemble(cx: &mut Ctx, m: &EnsMats, x: &str, b: i64, pfx: &str) -> Res<String> {
    let (j, l, k) = (m.j as i64, m.l as i64, m.k as i64);
    if m.j == 0 {
        // Stump-only ensemble: every leaf is unconditional, so the votes
        // are `Σ_l V[l]` — an input-independent const.
        let mut row = vec![0.0f32; m.k];
        for li in 0..m.l {
            for (ki, r) in row.iter_mut().enumerate() {
                *r += m.v[li * m.k + ki];
            }
        }
        let data: Vec<f32> = (0..b).flat_map(|_| row.iter().copied()).collect();
        return Ok(cx.k_f16t_named_ret(&format!("{pfx}_votes"), &[b, k], &data));
    }
    // F = x[:, feat] — gather over feature axis
    let fidx = cx.k_i32t(&[j], &m.feat);
    let ax = cx.k_i32s(1);
    let bd = cx.k_i32s(0);
    let vi = cx.k_bool(false);
    let f = cx.e1(
        "gather",
        vec![
            ("x".into(), bind(x).1),
            ("indices".into(), bind(&fidx).1),
            ("axis".into(), bind(&ax).1),
            ("batch_dims".into(), bind(&bd).1),
            ("validate_indices".into(), bind(&vi).1),
        ],
        &format!("{pfx}_f"),
        DType::Fp16,
        &[b, j],
    );
    let thr = cx.k_f16t(&[j], &m.thr);
    let d = cx.binary("sub", &f, &thr, &format!("{pfx}_d"))?;
    let zero = cx.k_f16(0.0);
    let le_b = cx.compare("less_equal", &d, &zero, &format!("{pfx}_leb"))?;
    let lt_b = cx.compare("less", &d, &zero, &format!("{pfx}_ltb"))?;
    let dtf = cx.k_str("fp16");
    let le_f = cx.e1(
        "cast",
        vec![("x".into(), bind(&le_b).1), ("dtype".into(), bind(&dtf).1)],
        &format!("{pfx}_lef"),
        DType::Fp16,
        &[b, j],
    );
    let dtf2 = cx.k_str("fp16");
    let lt_f = cx.e1(
        "cast",
        vec![("x".into(), bind(&lt_b).1), ("dtype".into(), bind(&dtf2).1)],
        &format!("{pfx}_ltf"),
        DType::Fp16,
        &[b, j],
    );
    let tx = cx.k_bool(false);
    let ty = cx.k_bool(false);
    let mle = cx.k_f16t_w(&[j, l], &m.mle_t);
    let v1 = cx.e1(
        "matmul",
        vec![
            ("x".into(), bind(&le_f).1),
            ("y".into(), bind(&mle).1),
            ("transpose_x".into(), bind(&tx).1),
            ("transpose_y".into(), bind(&ty).1),
        ],
        &format!("{pfx}_v1"),
        DType::Fp16,
        &[b, l],
    );
    let tx2 = cx.k_bool(false);
    let ty2 = cx.k_bool(false);
    let mlt = cx.k_f16t_w(&[j, l], &m.mlt_t);
    let v2 = cx.e1(
        "matmul",
        vec![
            ("x".into(), bind(&lt_f).1),
            ("y".into(), bind(&mlt).1),
            ("transpose_x".into(), bind(&tx2).1),
            ("transpose_y".into(), bind(&ty2).1),
        ],
        &format!("{pfx}_v2"),
        DType::Fp16,
        &[b, l],
    );
    let viol = cx.binary("add", &v1, &v2, &format!("{pfx}_viol0"))?;
    let off = cx.k_f16t(&[1, l], &m.off);
    let viol = cx.binary("add", &viol, &off, &format!("{pfx}_viol"))?;
    let zl = cx.k_f16t(&[1, l], &vec![0.0; m.l]);
    let ind_b = cx.compare("equal", &viol, &zl, &format!("{pfx}_indb"))?;
    let dtf3 = cx.k_str("fp16");
    let ind = cx.e1(
        "cast",
        vec![
            ("x".into(), bind(&ind_b).1),
            ("dtype".into(), bind(&dtf3).1),
        ],
        &format!("{pfx}_ind"),
        DType::Fp16,
        &[b, l],
    );
    let tx3 = cx.k_bool(false);
    let ty3 = cx.k_bool(false);
    let vv = cx.k_f16t_w(&[l, k], &m.v);
    Ok(cx.e1(
        "matmul",
        vec![
            ("x".into(), bind(&ind).1),
            ("y".into(), bind(&vv).1),
            ("transpose_x".into(), bind(&tx3).1),
            ("transpose_y".into(), bind(&ty3).1),
        ],
        &format!("{pfx}_votes"),
        DType::Fp16,
        &[b, k],
    ))
}

/// `scores = aggregate(votes) + base_values`, per ONNX
/// `TreeEnsemble*` semantics — base is added *after* aggregation so
/// `AVERAGE` doesn't scale it.
fn apply_agg_base(
    cx: &mut Ctx,
    votes: &str,
    k: i64,
    aggregate: &str,
    base_values: &[f64],
    n_trees: usize,
    pfx: &str,
) -> Res<String> {
    let mut s = votes.to_string();
    match aggregate {
        "sum" | "SUM" => {}
        "average" | "AVERAGE" => {
            let inv = cx.k_f16(1.0 / n_trees.max(1) as f32);
            s = cx.binary("mul", &s, &inv, &format!("{pfx}_avg"))?;
        }
        a => {
            return Err(OnnxError::Unsupported(format!(
                "tree ensemble aggregate '{a}' — supported: sum|average"
            )))
        }
    }
    if base_values.is_empty() {
        return Ok(s);
    }
    if base_values.len() != k as usize {
        return Err(OnnxError::Schema(format!(
            "base_values: {} elems, expected {k}",
            base_values.len()
        )));
    }
    let bv: Vec<f32> = base_values.iter().map(|&v| v as f32).collect();
    let base = cx.k_f16t(&[1, k], &bv);
    cx.binary("add", &s, &base, &format!("{pfx}_scores"))
}

impl<'a> Ctx<'a> {
    /// Named f16 tensor const that registers env (for `emit` symmetry).
    fn k_f16t_named_ret(&mut self, name: &str, shape: &[i64], vs: &[f32]) -> String {
        let vt = self.tt(DType::Fp16, shape);
        self.b.op(
            "const",
            vec![],
            vec![(name, vt)],
            vec![("val".into(), Value::f16s(shape, vs))],
        );
        self.env.insert(
            name.to_string(),
            crate::map::TInfo {
                shape: shape.to_vec(),
                dtype: DType::Fp16,
            },
        );
        *self.hist.entry("const".to_string()).or_insert(0) += 1;
        name.to_string()
    }
}

// ================= post-evaluation =================

/// Apply a post-transform to `(B,K)` scores; returns same-shape name.
fn apply_post(
    cx: &mut Ctx,
    scores: &str,
    shape: &[i64],
    post: PostTransform,
    pfx: &str,
) -> Res<String> {
    match post {
        PostTransform::None => Ok(scores.to_string()),
        PostTransform::Logistic => Ok(cx.unary("sigmoid", scores, &format!("{pfx}_sig"))?),
        PostTransform::Softmax => {
            let a = cx.k_i32s(-1);
            Ok(cx.e1(
                "softmax",
                vec![("x".into(), bind(scores).1), ("axis".into(), bind(&a).1)],
                &format!("{pfx}_sm"),
                DType::Fp16,
                shape,
            ))
        }
        PostTransform::Probit => {
            // Φ(x) = 0.5 (1 + erf(x/√2))
            let inv = cx.k_f16(std::f32::consts::FRAC_1_SQRT_2);
            let t = cx.binary("mul", scores, &inv, &format!("{pfx}_pb1"))?;
            let e = cx.unary("erf", &t, &format!("{pfx}_erf"))?;
            let one = cx.k_f16(1.0);
            let p1 = cx.binary("add", &e, &one, &format!("{pfx}_pb2"))?;
            let half = cx.k_f16(0.5);
            cx.binary("mul", &p1, &half, &format!("{pfx}_prob"))
        }
    }
}

/// Emit `(B,K)` scores → `(K,)` feature + optional `argmax→label` `(1,)`
/// int32 output. Returns `(scores_name, label_name)` for the block's
/// declared outputs.
fn emit_outputs(
    cx: &mut Ctx,
    scores: &str,
    shape: &[i64],
    post: PostTransform,
    labels: Option<&[i64]>,
    scores_out: &str,
    label_out: &str,
) -> Res<(String, Option<String>)> {
    let postd = apply_post(cx, scores, shape, post, "post")?;
    // (B,K) → (K,) when B==1, else keep
    let k = *shape.last().unwrap_or(&1);
    let sq = if shape.len() == 2 && shape[0] == 1 {
        reshape(cx, &postd, &[k], scores_out)?
    } else {
        cx.e1(
            "identity",
            vec![("x".into(), bind(&postd).1)],
            scores_out,
            DType::Fp16,
            shape,
        )
    };
    let label = labels.map(|ls| {
        // argmax runs on the rank-2 post-transformed scores so the
        // output is (B,), not a scalar.
        let am_shape = &[shape[0]];
        let a = cx.k_i32s(-1);
        let kd = cx.k_bool(false);
        let am = cx.e1(
            "reduce_argmax",
            vec![
                ("x".into(), bind(&postd).1),
                ("axis".into(), bind(&a).1),
                ("keep_dims".into(), bind(&kd).1),
            ],
            "classical_am",
            DType::Int32,
            am_shape,
        );
        let li: Vec<i32> = ls.iter().map(|&v| v as i32).collect();
        let lt = cx.k_i32t(&[ls.len() as i64], &li);
        let axc = cx.k_i32s(0);
        let bd = cx.k_i32s(0);
        let vi = cx.k_bool(false);
        cx.e1(
            "gather",
            vec![
                ("x".into(), bind(&lt).1),
                ("indices".into(), bind(&am).1),
                ("axis".into(), bind(&axc).1),
                ("batch_dims".into(), bind(&bd).1),
                ("validate_indices".into(), bind(&vi).1),
            ],
            label_out,
            DType::Int32,
            am_shape,
        )
    });
    Ok((sq, label))
}

fn reshape(cx: &mut Ctx, x: &str, shape: &[i64], out: &str) -> Res<String> {
    let dt = cx.info(x)?.dtype;
    let s = cx.k_i32v(&shape.iter().map(|&d| d as i32).collect::<Vec<_>>());
    Ok(cx.e1(
        "reshape",
        vec![("x".into(), bind(x).1), ("shape".into(), bind(&s).1)],
        out,
        dt,
        shape,
    ))
}

/// Expand a rank-1 `(F,)` input to `(1, F)`.
fn ensure_rank2(cx: &mut Ctx, x: &str, shape: &[i64], pfx: &str) -> Res<(String, Vec<i64>)> {
    if shape.len() == 2 {
        return Ok((x.to_string(), shape.to_vec()));
    }
    if shape.len() == 1 {
        let s2 = vec![1, shape[0]];
        return Ok((reshape(cx, x, &s2, &format!("{pfx}_r2"))?, s2));
    }
    Err(OnnxError::Schema(format!(
        "classical model input must be rank 1 or 2, got {shape:?}"
    )))
}

// ================= JSON schema =================

/// One tree node in the JSON schema.
#[derive(Clone, Debug)]
struct NodeJSON {
    id: i64,
    feature: Option<usize>,
    threshold: Option<f64>,
    mode: Option<String>,
    left: Option<i64>,
    right: Option<i64>,
    leaf: Option<Vec<f64>>,
}

impl NodeJSON {
    fn to_tnode(&self, weight: f64) -> Res<TNode> {
        if let Some(v) = &self.leaf {
            return Ok(TNode::Leaf(v.iter().map(|x| x * weight).collect()));
        }
        Ok(TNode::Internal {
            feature: self.feature.ok_or_else(|| {
                OnnxError::Schema(format!("node {}: internal node needs 'feature'", self.id))
            })?,
            threshold: self
                .threshold
                .ok_or_else(|| OnnxError::Schema(format!("node {}: needs 'threshold'", self.id)))?,
            mode: Mode::parse(self.mode.as_deref().unwrap_or("leq"))?,
            on_true: self
                .left
                .ok_or_else(|| OnnxError::Schema(format!("node {}: needs 'left'", self.id)))?,
            on_false: self
                .right
                .ok_or_else(|| OnnxError::Schema(format!("node {}: needs 'right'", self.id)))?,
        })
    }
}

/// One tree in the JSON schema.
#[derive(Clone, Debug)]
struct TreeJSON {
    weight: f64,
    nodes: Vec<NodeJSON>,
}

fn jf<'a>(v: &'a serde_json::Value, key: &str) -> Res<&'a serde_json::Value> {
    v.get(key)
        .ok_or_else(|| OnnxError::Schema(format!("missing field '{key}'")))
}
fn jarr(v: &serde_json::Value, key: &str) -> Res<Vec<serde_json::Value>> {
    Ok(jf(v, key)?
        .as_array()
        .ok_or_else(|| OnnxError::Schema(format!("'{key}' is not an array")))?
        .clone())
}
fn jnum(v: &serde_json::Value) -> Res<f64> {
    v.as_f64()
        .ok_or_else(|| OnnxError::Schema(format!("expected number, got {v}")))
}
fn jf64s(v: &serde_json::Value, key: &str) -> Res<Vec<f64>> {
    jarr(v, key)?.iter().map(jnum).collect()
}
fn ji64s(v: &serde_json::Value, key: &str) -> Res<Vec<i64>> {
    jarr(v, key)?
        .iter()
        .map(|e| {
            e.as_i64()
                .ok_or_else(|| OnnxError::Schema(format!("expected int, got {e}")))
        })
        .collect()
}
fn jstr<'a>(v: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(|s| s.as_str())
}

fn parse_trees(v: &serde_json::Value) -> Res<Vec<TreeJSON>> {
    let mut trees = Vec::new();
    for (ti, t) in jarr(v, "trees")?.iter().enumerate() {
        let weight = t.get("weight").and_then(|w| w.as_f64()).unwrap_or(1.0);
        let mut nodes = Vec::new();
        for nv in jarr(t, "nodes")? {
            let id = nv
                .get("id")
                .and_then(|i| i.as_i64())
                .ok_or_else(|| OnnxError::Schema(format!("tree {ti}: node missing 'id'")))?;
            let leaf = match nv.get("value") {
                Some(vv) => Some(
                    vv.as_array()
                        .ok_or_else(|| OnnxError::Schema("leaf 'value' not array".into()))?
                        .iter()
                        .map(|e| e.as_f64().unwrap_or(f64::NAN))
                        .collect(),
                ),
                None => None,
            };
            nodes.push(NodeJSON {
                id,
                feature: nv
                    .get("feature")
                    .and_then(|f| f.as_i64())
                    .map(|f| f as usize),
                threshold: nv.get("threshold").and_then(|f| f.as_f64()),
                mode: nv.get("mode").and_then(|m| m.as_str()).map(String::from),
                left: nv.get("left").and_then(|l| l.as_i64()),
                right: nv.get("right").and_then(|r| r.as_i64()),
                leaf,
            });
        }
        trees.push(TreeJSON { weight, nodes });
    }
    Ok(trees)
}

/// Convert a JSON classical model into a [`BuiltModel`].
pub fn convert_json(bytes: &[u8]) -> Res<BuiltModel> {
    let v: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| OnnxError::Schema(format!("json: {e}")))?;
    let kind = jstr(&v, "kind").ok_or_else(|| OnnxError::Schema("missing 'kind'".into()))?;
    let n_feat = jf(&v, "n_features")?
        .as_i64()
        .ok_or_else(|| OnnxError::Schema("'n_features' not an int".into()))?;
    if n_feat <= 0 {
        return Err(OnnxError::Schema("'n_features' must be > 0".into()));
    }
    let input_name = jstr(&v, "input_name").unwrap_or("features");
    let post = PostTransform::parse(jstr(&v, "post_transform").unwrap_or("none"))?;
    let labels: Option<Vec<i64>> = match v.get("class_labels") {
        Some(_) => Some(ji64s(&v, "class_labels")?),
        None => None,
    };

    let mut cx = Ctx::empty(13);
    let sn = cx.san(input_name);
    let x = sn.clone();
    cx.env.insert(
        sn.clone(),
        crate::map::TInfo {
            shape: vec![n_feat],
            dtype: DType::Fp16,
        },
    );
    let (x2, x2_shape) = ensure_rank2(&mut cx, &x, &[n_feat], "in")?;

    match kind {
        "glm_regressor" | "glm_classifier" => {
            let wv = jf(&v, "weights")?;
            // accept (n_out × n_feat) or flat (n_feat) for single-output
            let (k, wflat): (usize, Vec<f64>) = if wv.as_array().map(|a| {
                a.first().map(|e| e.is_array()).unwrap_or(false)
            }) == Some(true)
            {
                let rows = jarr(&v, "weights")?;
                let k = rows.len();
                let mut flat = Vec::new();
                for r in &rows {
                    flat.extend(
                        r.as_array()
                            .ok_or_else(|| OnnxError::Schema("weights row not array".into()))?
                            .iter()
                            .map(|e| e.as_f64().unwrap_or(0.0)),
                    );
                }
                (k, flat)
            } else {
                (1, jf64s(&v, "weights")?)
            };
            if wflat.len() != k * n_feat as usize {
                return Err(OnnxError::Schema(format!(
                    "weights: {} elems, expected {}",
                    wflat.len(),
                    k * n_feat as usize
                )));
            }
            let intercept = match v.get("intercept") {
                Some(_) => jf64s(&v, "intercept")?,
                None => vec![0.0; k],
            };
            if intercept.len() != k {
                return Err(OnnxError::Schema(format!(
                    "intercept: {} elems, expected {k}",
                    intercept.len()
                )));
            }
            lower_glm(
                &mut cx,
                &x2,
                &x2_shape,
                &wflat.iter().map(|&w| w as f32).collect::<Vec<_>>(),
                &intercept.iter().map(|&w| w as f32).collect::<Vec<_>>(),
                k,
                post,
                labels.as_deref(),
                kind == "glm_classifier",
            )?;
        }
        "tree_ensemble_regressor" | "tree_ensemble_classifier" => {
            let n_out = v
                .get("n_outputs")
                .and_then(|n| n.as_i64())
                .unwrap_or_else(|| labels.as_ref().map(|l| l.len() as i64).unwrap_or(1))
                as usize;
            let base = match v.get("base_values") {
                Some(_) => jf64s(&v, "base_values")?,
                None => vec![0.0; n_out],
            };
            if base.len() != n_out {
                return Err(OnnxError::Schema(format!(
                    "base_values: {} elems, expected {n_out}",
                    base.len()
                )));
            }
            let trees = parse_trees(&v)?;
            let votes = lower_ensemble(&mut cx, &trees, n_out, &x2, &x2_shape, "ens")?;
            let aggregate = jstr(&v, "aggregate").unwrap_or("sum");
            let scores = apply_agg_base(
                &mut cx,
                &votes,
                n_out as i64,
                aggregate,
                &base,
                trees.len(),
                "ens",
            )?;
            let _ = emit_outputs(
                &mut cx,
                &scores,
                &[1, n_out as i64],
                post,
                labels.as_deref(),
                "scores",
                "label",
            )?;
        }
        k => {
            return Err(OnnxError::Schema(format!(
                "unknown kind '{k}' — expected glm_regressor|glm_classifier|tree_ensemble_regressor|tree_ensemble_classifier"
            )))
        }
    }

    finish_classical(cx, &sn, n_feat)
}

/// Emit `linear` GLM path; returns nothing — outputs are wired inside.
#[allow(clippy::too_many_arguments)] // mirrors the JSON/ONNX parameter lists
fn lower_glm(
    cx: &mut Ctx,
    x: &str,
    x_shape: &[i64],
    wflat: &[f32], // (K, F) row-major
    intercept: &[f32],
    k: usize,
    post: PostTransform,
    labels: Option<&[i64]>,
    is_classifier: bool,
) -> Res<()> {
    let b = x_shape[0];
    let k64 = k as i64;
    let w = cx.k_f16t(&[k64, x_shape[1]], wflat);
    let bias = cx.k_f16t(&[k64], intercept);
    let scores = cx.e1(
        "linear",
        vec![
            ("x".into(), bind(x).1),
            ("weight".into(), bind(&w).1),
            ("bias".into(), bind(&bias).1),
        ],
        "glm_lin",
        DType::Fp16,
        &[b, k64],
    );
    emit_outputs(
        cx,
        &scores,
        &[b, k64],
        post,
        labels.filter(|_| is_classifier),
        "scores",
        "label",
    )?;
    Ok(())
}

fn finish_classical(mut cx: Ctx, input_sn: &str, n_feat: i64) -> Res<BuiltModel> {
    // declare outputs: "scores" always; "label" when it was produced
    let mut outputs = Vec::new();
    let mut block_outs = Vec::new();
    for name in ["scores", "label"] {
        if let Some(t) = cx.env.get(name).cloned() {
            outputs.push(Feature {
                name: name.to_string(),
                shape: t.shape.clone(),
                dtype: t.dtype,
                is_state: false,
            });
            block_outs.push(name.to_string());
        }
    }
    cx.b.outputs = block_outs;
    let wb = if cx.has_blob {
        Some(cx.wb.finish())
    } else {
        None
    };
    Ok(BuiltModel {
        block: cx.b,
        inputs: vec![Feature {
            name: input_sn.to_string(),
            shape: vec![n_feat],
            dtype: DType::Fp16,
            is_state: false,
        }],
        outputs,
        fn_inputs: vec![NVT {
            name: input_sn.to_string(),
            ty: ValueType::Tensor(TensorType::f16(&[n_feat])),
        }],
        weight_bin: wb,
        op_histogram: cx.hist,
    })
}

// ================= ai.onnx.ml nodes =================

fn attr_ints(n: &NodeProto, k: &str) -> Vec<i64> {
    n.attr_ints(k)
}
fn attr_floats(n: &NodeProto, k: &str) -> Vec<f32> {
    n.attr_floats(k)
}
fn attr_strs(n: &NodeProto, k: &str) -> Vec<String> {
    n.attrs
        .get(k)
        .map(|a| {
            a.strings
                .iter()
                .map(|s| String::from_utf8_lossy(s).into_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// Lower an `ai.onnx.ml` node inside an ONNX graph.
pub(crate) fn map_ml_node(cx: &mut Ctx, node: &NodeProto, label: &str) -> Res<()> {
    let out = node.outputs.first().cloned().unwrap_or_default();
    let out_sn = cx.san(&out);
    let _ = out_sn;
    match node.op_type.as_str() {
        "TreeEnsembleClassifier" | "TreeEnsembleRegressor" => map_tree_ensemble_ml(cx, node, label),
        "LinearClassifier" | "LinearRegressor" => map_linear_ml(cx, node, label),
        "Normalizer" => {
            let xs0 = cx.resolve(&node.inputs[0], label)?;
            let t = cx.info(&xs0)?.clone();
            let norm = node.attr_str("norm").unwrap_or_else(|| "MAX".into());
            let axes: Vec<i64> = (1..t.shape.len() as i64).collect();
            let reduced = match norm.as_str() {
                "L1" => cx.reduce(
                    "reduce_l1_norm",
                    &xs0,
                    &axes,
                    true,
                    &format!("{label}_n"),
                    t.dtype,
                )?,
                "L2" => cx.reduce(
                    "reduce_l2_norm",
                    &xs0,
                    &axes,
                    true,
                    &format!("{label}_n"),
                    t.dtype,
                )?,
                "MAX" => {
                    let a = cx.unary("abs", &xs0, &format!("{label}_abs"))?;
                    cx.reduce(
                        "reduce_max",
                        &a,
                        &axes,
                        true,
                        &format!("{label}_n"),
                        t.dtype,
                    )?
                }
                n => {
                    return Err(OnnxError::Unsupported(format!(
                        "Normalizer '{label}': norm '{n}'"
                    )))
                }
            };
            let out = cx.binary("real_div", &xs0, &reduced, &out_sn)?;
            let _ = out;
            Ok(())
        }
        "Scaler" => {
            // y = (x - offset) * scale   (ai.onnx.ml Scaler: offset, scale
            // are per-feature float lists)
            let xs0 = cx.resolve(&node.inputs[0], label)?;
            let t = cx.info(&xs0)?.clone();
            let f = *t.shape.last().unwrap_or(&1);
            let off = node.attr_floats("offset");
            let scale = node.attr_floats("scale");
            let off = if off.is_empty() {
                vec![0.0; f as usize]
            } else {
                off
            };
            let scale = if scale.is_empty() {
                vec![1.0; f as usize]
            } else {
                scale
            };
            if off.len() != f as usize || scale.len() != f as usize {
                return Err(OnnxError::BadShape(format!(
                    "Scaler '{label}': offset/scale must have {f} entries"
                )));
            }
            let oc = cx.k_f16t(&[f], &off);
            let sc = cx.k_f16t(&[f], &scale);
            let sub = cx.binary("sub", &xs0, &oc, &format!("{label}_sub"))?;
            cx.binary("mul", &sub, &sc, &out_sn)?;
            Ok(())
        }
        other => Err(OnnxError::UnknownOp {
            op: other.to_string(),
            node: label.to_string(),
            domain: "ai.onnx.ml".into(),
        }),
    }
}

/// Assemble ai.onnx.ml TreeEnsemble* parallel-array attributes into
/// [`TreeJSON`]-shaped structures, then lower via [`lower_ensemble`].
fn map_tree_ensemble_ml(cx: &mut Ctx, node: &NodeProto, label: &str) -> Res<()> {
    let is_clf = node.op_type == "TreeEnsembleClassifier";
    let xs0 = cx.resolve(&node.inputs[0], label)?;
    let t = cx.info(&xs0)?.clone();
    let (x2, x2_shape) = ensure_rank2(cx, &xs0, &t.shape, label)?;

    let treeids = attr_ints(node, "nodes_treeids");
    let nodeids = attr_ints(node, "nodes_nodeids");
    let featids = attr_ints(node, "nodes_featureids");
    let vals = attr_floats(node, "nodes_values");
    let modes = attr_strs(node, "nodes_modes");
    let trueids = attr_ints(node, "nodes_truenodeids");
    let falseids = attr_ints(node, "nodes_falsenodeids");
    let n = treeids.len();
    if nodeids.len() != n || featids.len() != n || vals.len() != n || modes.len() != n {
        return Err(OnnxError::Malformed {
            what: node.op_type.clone(),
            detail: format!("'{label}': nodes_* attribute length mismatch"),
        });
    }

    let (valids, targetids, weights, k, wids) = if is_clf {
        let labels_i = attr_ints(node, "classlabels_ints");
        let k = labels_i.len().max(1);
        (
            attr_ints(node, "class_nodeids"),
            attr_ints(node, "class_treeids"),
            attr_floats(node, "class_weights"),
            k,
            attr_ints(node, "class_ids"),
        )
    } else {
        let n_targets = node.attr_i("n_targets", 1).max(1) as usize;
        (
            attr_ints(node, "target_nodeids"),
            attr_ints(node, "target_treeids"),
            attr_floats(node, "target_weights"),
            n_targets,
            attr_ints(node, "target_ids"),
        )
    };
    if valids.len() != targetids.len()
        || valids.len() != weights.len()
        || valids.len() != wids.len()
    {
        return Err(OnnxError::Malformed {
            what: node.op_type.clone(),
            detail: format!("'{label}': leaf weight arrays length mismatch"),
        });
    }

    // Build per-tree node tables.
    let n_trees = treeids.iter().max().copied().unwrap_or(-1) + 1;
    let mut trees: Vec<Vec<NodeJSON>> = vec![Vec::new(); n_trees as usize];
    for i in 0..n {
        let ti = treeids[i] as usize;
        let mode_str = &modes[i];
        if mode_str == "LEAF" {
            // collect per-class/target weights for this leaf
            let mut vv = vec![0.0f64; k];
            for w in 0..valids.len() {
                if targetids[w] == treeids[i] && valids[w] == nodeids[i] {
                    let cls = wids[w] as usize;
                    if cls < k {
                        vv[cls] += weights[w] as f64;
                    }
                }
            }
            trees[ti].push(NodeJSON {
                id: nodeids[i],
                feature: None,
                threshold: None,
                mode: None,
                left: None,
                right: None,
                leaf: Some(vv),
            });
        } else {
            let m = Mode::parse(mode_str)?;
            trees[ti].push(NodeJSON {
                id: nodeids[i],
                feature: Some(featids[i] as usize),
                threshold: Some(vals[i] as f64),
                mode: Some(match m {
                    Mode::Leq => "leq".into(),
                    Mode::Lt => "lt".into(),
                    Mode::Gte => "gte".into(),
                    Mode::Gt => "gt".into(),
                }),
                left: Some(trueids[i]),
                right: Some(falseids[i]),
                leaf: None,
            });
        }
    }
    let trees: Vec<TreeJSON> = trees
        .into_iter()
        .map(|nodes| TreeJSON { weight: 1.0, nodes })
        .collect();
    let base: Vec<f64> = attr_floats(node, "base_values")
        .iter()
        .map(|&v| v as f64)
        .collect();
    let post = PostTransform::parse(
        &node
            .attr_str("post_transform")
            .unwrap_or_else(|| "NONE".into()),
    )?;
    let votes = lower_ensemble(cx, &trees, k, &x2, &x2_shape, label)?;

    // aggregate_function is a regressor attribute; classifiers always
    // sum per-class votes. Either way base_values is added after.
    let agg = if is_clf {
        "SUM".to_string()
    } else {
        node.attr_str("aggregate_function")
            .unwrap_or_else(|| "SUM".into())
    };
    let scores = apply_agg_base(cx, &votes, k as i64, &agg, &base, trees.len(), label)?;
    let labels_i = if is_clf {
        Some(attr_ints(node, "classlabels_ints"))
    } else {
        None
    };
    finish_ml_out(
        cx,
        node,
        &scores,
        &x2_shape,
        post,
        labels_i.as_deref(),
        label,
    )
}

/// ai.onnx.ml outputs: classifier gives Y (label) + Z (scores);
/// regressor gives a single scores output.
fn finish_ml_out(
    cx: &mut Ctx,
    node: &NodeProto,
    scores: &str,
    x2_shape: &[i64],
    post: PostTransform,
    labels: Option<&[i64]>,
    label: &str,
) -> Res<()> {
    let k = cx.info(scores)?.shape[1];
    if let Some(ls) = labels {
        // outputs: [Y (label), Z (scores)]
        let ysn = cx.san(&node.outputs[0]);
        let zsn = cx.san(&node.outputs.get(1).cloned().unwrap_or_default());
        let postd = apply_post(cx, scores, &[x2_shape[0], k], post, label)?;
        // label = gather(classlabels, argmax(postd))
        let a = cx.k_i32s(-1);
        let kd = cx.k_bool(false);
        let am = cx.e1(
            "reduce_argmax",
            vec![
                ("x".into(), bind(&postd).1),
                ("axis".into(), bind(&a).1),
                ("keep_dims".into(), bind(&kd).1),
            ],
            &format!("{label}_am"),
            DType::Int32,
            &[x2_shape[0]],
        );
        let li: Vec<i32> = ls.iter().map(|&v| v as i32).collect();
        let lt = cx.k_i32t(&[ls.len() as i64], &li);
        let axc = cx.k_i32s(0);
        let bd = cx.k_i32s(0);
        let vi = cx.k_bool(false);
        cx.e1(
            "gather",
            vec![
                ("x".into(), bind(&lt).1),
                ("indices".into(), bind(&am).1),
                ("axis".into(), bind(&axc).1),
                ("batch_dims".into(), bind(&bd).1),
                ("validate_indices".into(), bind(&vi).1),
            ],
            &ysn,
            DType::Int32,
            &[x2_shape[0]],
        );
        if !zsn.is_empty() {
            let sc_shape = cx.info(&postd)?.shape.clone();
            cx.e1(
                "identity",
                vec![("x".into(), bind(&postd).1)],
                &zsn,
                DType::Fp16,
                &sc_shape,
            );
        }
    } else {
        let postd = apply_post(cx, scores, &[x2_shape[0], k], post, label)?;
        let osn = cx.san(&node.outputs[0]);
        let sc_shape = cx.info(&postd)?.shape.clone();
        cx.e1(
            "identity",
            vec![("x".into(), bind(&postd).1)],
            &osn,
            DType::Fp16,
            &sc_shape,
        );
    }
    Ok(())
}

/// ai.onnx.ml LinearClassifier/LinearRegressor.
fn map_linear_ml(cx: &mut Ctx, node: &NodeProto, label: &str) -> Res<()> {
    let is_clf = node.op_type == "LinearClassifier";
    let xs0 = cx.resolve(&node.inputs[0], label)?;
    let t = cx.info(&xs0)?.clone();
    let (x2, x2_shape) = ensure_rank2(cx, &xs0, &t.shape, label)?;
    let f = *x2_shape.last().unwrap_or(&0);
    let coef = attr_floats(node, "coefficients");
    let intercepts = attr_floats(node, "intercepts");
    let k = if coef.len() % f.max(1) as usize == 0 {
        (coef.len() / f.max(1) as usize).max(1)
    } else {
        return Err(OnnxError::BadShape(format!(
            "LinearClassifier '{label}': {} coefficients vs {f} features",
            coef.len()
        )));
    };
    let n_labels = attr_ints(node, "classlabels_ints").len();
    // binary sklearn LR exports K=1 row with 2 labels → scores [-s, s]
    let (kk, wflat, bvec) = if is_clf && k == 1 && n_labels == 2 {
        let mut w2 = Vec::with_capacity(coef.len() * 2);
        w2.extend(coef.iter().map(|v| -v));
        w2.extend(&coef);
        let b0 = intercepts.first().copied().unwrap_or(0.0);
        (2usize, w2, vec![-b0, b0])
    } else {
        let b = if intercepts.len() == k {
            intercepts
        } else {
            vec![0.0; k]
        };
        (k, coef, b)
    };
    let w = cx.k_f16t_w(&[kk as i64, f], &wflat);
    let bias = cx.k_f16t(&[kk as i64], &bvec);
    let scores = cx.e1(
        "linear",
        vec![
            ("x".into(), bind(&x2).1),
            ("weight".into(), bind(&w).1),
            ("bias".into(), bind(&bias).1),
        ],
        &format!("{label}_lin"),
        DType::Fp16,
        &[x2_shape[0], kk as i64],
    );
    let post = PostTransform::parse(
        &node
            .attr_str("post_transform")
            .unwrap_or_else(|| "NONE".into()),
    )?;
    let labels_i = if is_clf {
        Some(attr_ints(node, "classlabels_ints"))
    } else {
        None
    };
    finish_ml_out(
        cx,
        node,
        &scores,
        &x2_shape,
        post,
        labels_i.as_deref(),
        label,
    )
}
