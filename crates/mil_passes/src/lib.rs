//! `mil_passes` — optimization passes over [`mil_spec::Block`] graphs.
//!
//! The typed `Block` helpers emit correct but unoptimized MIL: every
//! `conv1x1` writes five fresh const ops for strides/padding/dilations,
//! every reshape writes a new shape vector, every scalar is its own op.
//! Real graphs carry thousands of duplicate consts. These passes shrink
//! the spec — fewer ops means fewer dispatches, and dispatch count is the
//! metric that decides whether a graph lives or dies on the ANE.
//!
//! All passes are deterministic, run in program order, and report what
//! they changed so callers can diff behavior. A pass never alters op
//! semantics — only which values are produced where.
//!
//! # Passes
//!
//! - [`dedup_consts`] — merge identical const ops (the big win)
//! - [`noop_elim`] — remove reshapes/transposes/squeezes that change nothing
//! - [`const_fold`] — evaluate ops whose inputs are all inline consts
//! - [`fuse_silu`] — `mul(a, sigmoid(a))` → `silu(a)` (SwiGLU pattern)
//! - [`dead_code`] — drop ops whose outputs feed nothing
//! - [`optimize`] — all of the above to a fixpoint

#![forbid(unsafe_code)]

use mil_spec::{Binding, Block, DType, Immediate, Op, TensorType, Value, ValueType};
use std::collections::{HashMap, HashSet};

/// What one pass did. `removed`/`added` are op counts; `renamed` counts
/// bindings rewritten to a different producer name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassReport {
    /// Ops removed from the block.
    pub removed: usize,
    /// Ops added (folded consts).
    pub added: usize,
    /// Bindings rewritten to a different name.
    pub renamed: usize,
}

// ---------- internals: const introspection ----------

/// Fingerprint of everything that determines a const's value: all attrs'
/// wire bytes in order. Two const ops with the same fingerprint produce
/// the same tensor.
fn const_fingerprint(op: &Op) -> Option<Vec<u8>> {
    if op.ty != "const" {
        return None;
    }
    let mut fp = Vec::new();
    let mut any = false;
    for (k, v) in &op.attrs {
        fp.extend_from_slice(k.as_bytes());
        fp.extend_from_slice(&v.wire_bytes());
        any = true;
    }
    any.then_some(fp)
}

/// Decode an int32 vector from a const op's `val` attr.
fn const_i32s(op: &Op) -> Option<Vec<i32>> {
    if op.ty != "const" {
        return None;
    }
    for (_, v) in &op.attrs {
        if let Value::Imm(_, Immediate::Ints(vs)) = v {
            return Some(vs.clone());
        }
    }
    None
}

/// Decode fp32 numbers from a const op's `val` attr. Handles fp16/fp32
/// immediate floats and raw fp16 Bytes storage.
fn const_f32s(op: &Op) -> Option<Vec<f32>> {
    if op.ty != "const" {
        return None;
    }
    for (_, v) in &op.attrs {
        match v {
            Value::Bytes(ValueType::Tensor(t), raw) if t.dtype == DType::Fp16 => {
                if raw.len() % 2 == 0 {
                    return Some(
                        raw.chunks_exact(2)
                            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                            .collect(),
                    );
                }
            }
            Value::Imm(ValueType::Tensor(t), Immediate::Floats(vs))
                if t.dtype == DType::Fp16 || t.dtype == DType::Fp32 =>
            {
                return Some(vs.clone());
            }
            _ => {}
        }
    }
    None
}

/// Shape declared for a const op's output.
fn const_out_shape(op: &Op) -> Option<Vec<i64>> {
    if op.ty != "const" {
        return None;
    }
    let nvt = op.outputs.first()?;
    if let ValueType::Tensor(t) = &nvt.ty {
        Some(t.shape.clone())
    } else {
        None
    }
}

/// Whether a const op carries its value inline (foldable) rather than
/// referencing `weight.bin` (not foldable — we don't read the blob).
fn const_is_inline(op: &Op) -> bool {
    op.ty == "const"
        && op
            .attrs
            .iter()
            .any(|(_, v)| matches!(v, Value::Imm(..) | Value::Bytes(..)))
}

/// Build a `const` op holding `vals` as fp16 with the given shape.
fn make_f16_const(name: &str, shape: &[i64], vals: &[f32]) -> Op {
    Op {
        ty: "const".into(),
        inputs: vec![],
        outputs: vec![mil_spec::NVT {
            name: name.into(),
            ty: ValueType::Tensor(TensorType::f16(shape)),
        }],
        attrs: vec![("val".into(), Value::f16s(shape, vals))],
    }
}

/// Value-type → shape lookup for every produced name in the block.
fn shape_env(b: &Block) -> HashMap<String, Vec<i64>> {
    let mut env = HashMap::new();
    for op in &b.ops {
        for out in &op.outputs {
            if let ValueType::Tensor(t) = &out.ty {
                env.insert(out.name.clone(), t.shape.clone());
            }
        }
    }
    env
}

/// Find a previously-emitted op by output name (SSA: producers precede
/// consumers, so only earlier ops can be referenced).
fn producer<'a>(ops: &'a [Op], name: &str) -> Option<&'a Op> {
    ops.iter()
        .find(|op| op.outputs.iter().any(|o| o.name == name))
}

/// First named binding for a single-name input key.
fn arg_name(op: &Op, key: &str) -> Option<String> {
    op.inputs.iter().find(|(k, _)| k == key).and_then(|(_, a)| {
        a.0.iter().find_map(|bd| {
            if let Binding::Name(n) = bd {
                Some(n.clone())
            } else {
                None
            }
        })
    })
}

/// All names in a variadic input (e.g. `concat.values`).
fn arg_list(op: &Op, key: &str) -> Option<Vec<String>> {
    op.inputs.iter().find(|(k, _)| k == key).map(|(_, a)| {
        a.0.iter()
            .filter_map(|bd| {
                if let Binding::Name(n) = bd {
                    Some(n.clone())
                } else {
                    None
                }
            })
            .collect()
    })
}

/// Rewrite every `Binding::Name` in the block through `alias`
/// (chases chains a→b→c). Returns rewrite count.
fn apply_alias(b: &mut Block, alias: &HashMap<String, String>) -> usize {
    if alias.is_empty() {
        return 0;
    }
    let resolve = |name: &str| -> String {
        let mut cur = name.to_string();
        for _ in 0..alias.len() + 1 {
            match alias.get(&cur) {
                Some(next) => cur = next.clone(),
                None => break,
            }
        }
        cur
    };
    let mut n = 0;
    for op in &mut b.ops {
        for (_, arg) in &mut op.inputs {
            for binding in &mut arg.0 {
                if let Binding::Name(name) = binding {
                    let new = resolve(name);
                    if new != *name {
                        *name = new;
                        n += 1;
                    }
                }
            }
        }
    }
    for out in &mut b.outputs {
        let new = resolve(out);
        if new != *out {
            *out = new;
            n += 1;
        }
    }
    n
}

/// All value names an op consumes.
fn consumed_names(op: &Op) -> Vec<String> {
    let mut names = Vec::new();
    for (_, arg) in &op.inputs {
        for binding in &arg.0 {
            if let Binding::Name(n) = binding {
                names.push(n.clone());
            }
        }
    }
    names
}

// ---------- pass 1: const dedup ----------

/// Merge const ops that encode the same value. The typed helpers emit
/// five consts per conv1x1 (`strides`, `pad_type`, `pads`, `dilations`,
/// `groups`) plus a fresh shape vector per reshape — on a 28-layer graph
/// that is hundreds of identical ops. Every duplicate after the first
/// is deleted and its consumers rebound to the surviving const.
pub fn dedup_consts(b: &mut Block) -> PassReport {
    let mut seen: HashMap<Vec<u8>, String> = HashMap::new();
    let mut alias: HashMap<String, String> = HashMap::new();
    let mut keep = Vec::with_capacity(b.ops.len());
    let mut removed = 0;

    for op in std::mem::take(&mut b.ops) {
        if let Some(fp) = const_fingerprint(&op) {
            if let Some(canon) = seen.get(&fp) {
                for out in &op.outputs {
                    alias.insert(out.name.clone(), canon.clone());
                }
                removed += 1;
                continue;
            }
            if let Some(out) = op.outputs.first() {
                seen.insert(fp, out.name.clone());
            }
        }
        keep.push(op);
    }
    b.ops = keep;
    let renamed = apply_alias(b, &alias);
    PassReport {
        removed,
        added: 0,
        renamed,
    }
}

// ---------- pass 2: no-op elimination ----------

/// Remove ops that provably do nothing: `reshape` to the same shape,
/// `transpose` with the identity permutation, `expand_dims`/`squeeze`
/// with empty axes. Consumers are rebound to the op's input.
pub fn noop_elim(b: &mut Block) -> PassReport {
    let env = shape_env(b);
    // name -> i32 const payload (perm/axes/shape producers)
    let mut i32_env: HashMap<String, Vec<i32>> = HashMap::new();
    for op in &b.ops {
        if let Some(vs) = const_i32s(op) {
            if let Some(out) = op.outputs.first() {
                i32_env.insert(out.name.clone(), vs);
            }
        }
    }

    let mut alias: HashMap<String, String> = HashMap::new();
    let mut keep = Vec::with_capacity(b.ops.len());
    let mut removed = 0;

    for op in std::mem::take(&mut b.ops) {
        let noop = match op.ty.as_str() {
            "reshape" => {
                let xs = arg_name(&op, "x").and_then(|n| env.get(&n).cloned());
                let out = op.outputs.first().and_then(|o| {
                    if let ValueType::Tensor(t) = &o.ty {
                        Some(t.shape.clone())
                    } else {
                        None
                    }
                });
                matches!((xs, out), (Some(a), Some(c)) if a == c)
            }
            "transpose" => {
                let perm = arg_name(&op, "perm").and_then(|n| i32_env.get(&n).cloned());
                matches!(perm, Some(p) if p.iter().enumerate().all(|(i, &v)| v == i as i32))
            }
            "expand_dims" | "squeeze" => {
                let axes = arg_name(&op, "axes").and_then(|n| i32_env.get(&n).cloned());
                matches!(axes, Some(a) if a.is_empty())
            }
            _ => false,
        };

        if noop {
            if let Some(src) = arg_name(&op, "x") {
                for out in &op.outputs {
                    alias.insert(out.name.clone(), src.clone());
                }
                removed += 1;
                continue;
            }
        }
        keep.push(op);
    }
    b.ops = keep;
    let renamed = apply_alias(b, &alias);
    PassReport {
        removed,
        added: 0,
        renamed,
    }
}

// ---------- pass 3: constant folding ----------

/// Evaluate ops whose data inputs are all inline consts and replace them
/// with a folded const. Folded today: `add`/`sub`/`mul` on fp16 tensors,
/// `reshape` (retag), `transpose` (permute), `concat` (strided copy).
/// Blob-backed consts are never folded — their bytes live in weight.bin
/// and we don't touch it.
/// Previously-emitted const carrying an inline value — foldable.
fn inline_const<'a>(ops: &'a [Op], name: &str) -> Option<&'a Op> {
    producer(ops, name).filter(|op| const_is_inline(op))
}

pub fn const_fold(b: &mut Block) -> PassReport {
    let mut alias: HashMap<String, String> = HashMap::new();
    let mut new_ops: Vec<Op> = Vec::with_capacity(b.ops.len());
    let mut removed = 0;
    let mut added = 0;
    let mut counter = 0usize;

    for op in std::mem::take(&mut b.ops) {
        if op.ty == "const" || op.ty == "write_state" {
            new_ops.push(op);
            continue;
        }
        let folded: Option<(Vec<f32>, Vec<i64>)> = match op.ty.as_str() {
            "add" | "sub" | "mul" => match (arg_name(&op, "x"), arg_name(&op, "y")) {
                (Some(xn), Some(yn)) => {
                    let xc = inline_const(&new_ops, &xn);
                    let yc = inline_const(&new_ops, &yn);
                    match (xc, yc) {
                        (Some(xc), Some(yc)) => {
                            let xs = const_out_shape(xc);
                            let xv = const_f32s(xc);
                            let ys = const_out_shape(yc);
                            let yv = const_f32s(yc);
                            match (xs, xv, ys, yv) {
                                (Some(xs), Some(xv), Some(ys), Some(yv)) => {
                                    elementwise_fold(&op.ty, &xs, &xv, &ys, &yv)
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    }
                }
                _ => None,
            },
            "reshape" => {
                let x = arg_name(&op, "x").and_then(|n| inline_const(&new_ops, &n));
                let s = arg_name(&op, "shape").and_then(|n| inline_const(&new_ops, &n));
                match (x, s) {
                    (Some(xc), Some(sc)) => {
                        let vs = const_f32s(xc);
                        let target = const_i32s(sc)
                            .map(|v| v.iter().map(|&d| d as i64).collect::<Vec<i64>>());
                        match (vs, target) {
                            (Some(vs), Some(target))
                                if target.iter().product::<i64>() == vs.len() as i64 =>
                            {
                                Some((vs, target))
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            "transpose" => {
                let x = arg_name(&op, "x").and_then(|n| inline_const(&new_ops, &n));
                let p = arg_name(&op, "perm").and_then(|n| inline_const(&new_ops, &n));
                match (x, p) {
                    (Some(xc), Some(pc)) => {
                        match (const_out_shape(xc), const_f32s(xc), const_i32s(pc)) {
                            (Some(xs), Some(xv), Some(perm)) => transpose_fold(&xs, &xv, &perm),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            "concat" => {
                let names = arg_list(&op, "values");
                let axis = arg_name(&op, "axis")
                    .and_then(|n| inline_const(&new_ops, &n))
                    .and_then(|c| const_i32s(c).and_then(|v| v.first().copied()));
                match (names, axis) {
                    (Some(names), Some(axis)) if !names.is_empty() => {
                        let mut parts = Vec::with_capacity(names.len());
                        let mut ok = true;
                        for n in &names {
                            match inline_const(&new_ops, n) {
                                Some(c) => match (const_out_shape(c), const_f32s(c)) {
                                    (Some(s), Some(v)) => parts.push((s, v)),
                                    _ => {
                                        ok = false;
                                        break;
                                    }
                                },
                                None => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if ok {
                            concat_fold(&parts, axis)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        };

        if let Some((vals, shape)) = folded {
            counter += 1;
            let name = format!("fold_{counter}");
            new_ops.push(make_f16_const(&name, &shape, &vals));
            for out in &op.outputs {
                alias.insert(out.name.clone(), name.clone());
            }
            removed += 1;
            added += 1;
        } else {
            new_ops.push(op);
        }
    }
    b.ops = new_ops;
    let renamed = apply_alias(b, &alias);
    PassReport {
        removed,
        added,
        renamed,
    }
}

/// Elementwise fold with scalar broadcast. Returns `(values, shape)`.
fn elementwise_fold(
    op: &str,
    xs: &[i64],
    xv: &[f32],
    ys: &[i64],
    yv: &[f32],
) -> Option<(Vec<f32>, Vec<i64>)> {
    let f = match op {
        "add" => |a: f32, b: f32| a + b,
        "sub" => |a: f32, b: f32| a - b,
        "mul" => |a: f32, b: f32| a * b,
        _ => return None,
    };
    if xv.len() == 1 {
        return Some((yv.iter().map(|&w| f(xv[0], w)).collect(), ys.to_vec()));
    }
    if yv.len() == 1 {
        return Some((xv.iter().map(|&v| f(v, yv[0])).collect(), xs.to_vec()));
    }
    if xs == ys && xv.len() == yv.len() {
        return Some((
            xv.iter().zip(yv.iter()).map(|(&a, &b)| f(a, b)).collect(),
            xs.to_vec(),
        ));
    }
    None
}

/// Permute `vals` laid out in row-major `shape` by `perm`.
fn transpose_fold(shape: &[i64], vals: &[f32], perm: &[i32]) -> Option<(Vec<f32>, Vec<i64>)> {
    let rank = shape.len();
    if perm.len() != rank || vals.is_empty() {
        return None;
    }
    let mut seen = vec![false; rank];
    for &p in perm {
        if p < 0 || p as usize >= rank || seen[p as usize] {
            return None;
        }
        seen[p as usize] = true;
    }
    let out_shape: Vec<i64> = perm.iter().map(|&p| shape[p as usize]).collect();
    let in_strides = strides(shape);
    let out_strides = strides(&out_shape);
    let mut out = vec![0f32; vals.len()];
    for (flat, o) in out.iter_mut().enumerate() {
        let mut rem = flat;
        let mut src = 0usize;
        for (i, &os) in out_strides.iter().enumerate() {
            let idx = rem / os as usize;
            rem %= os as usize;
            src += idx * in_strides[perm[i] as usize] as usize;
        }
        *o = vals[src];
    }
    Some((out, out_shape))
}

/// Concat along `axis` via strided row-major copy.
fn concat_fold(parts: &[(Vec<i64>, Vec<f32>)], axis: i32) -> Option<(Vec<f32>, Vec<i64>)> {
    if parts.is_empty() {
        return None;
    }
    let rank = parts[0].0.len();
    if rank == 0 {
        return None;
    }
    let ax = if axis < 0 { axis + rank as i32 } else { axis };
    if ax < 0 || ax as usize >= rank {
        return None;
    }
    let ax = ax as usize;
    let mut out_shape = parts[0].0.clone();
    let mut total_axis = 0i64;
    for (s, _) in parts {
        if s.len() != rank {
            return None;
        }
        for (d, (&a, &b)) in s.iter().zip(parts[0].0.iter()).enumerate() {
            if d != ax && a != b {
                return None;
            }
        }
        total_axis += s[ax];
    }
    out_shape[ax] = total_axis;

    let inner: i64 = parts[0].0[ax + 1..].iter().product();
    let outer: i64 = parts[0].0[..ax].iter().product();
    let out_axis_stride = inner;
    let out_row = total_axis * inner;
    let mut out = vec![0f32; (outer * out_row) as usize];

    let mut axis_base = 0i64;
    for (s, v) in parts {
        let part_row = s[ax] * inner;
        if v.len() != (outer * part_row) as usize {
            return None;
        }
        for o in 0..outer {
            for a in 0..s[ax] {
                let src = (o * part_row + a * inner) as usize;
                let dst = (o * out_row + (axis_base + a) * out_axis_stride) as usize;
                out[dst..dst + inner as usize].copy_from_slice(&v[src..src + inner as usize]);
            }
        }
        axis_base += s[ax];
    }
    Some((out, out_shape))
}

fn strides(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

// ---------- pass 4: activation fusion ----------

/// Fuse `mul(a, sigmoid(a))` (either operand order) into `silu(a)` —
/// the exact pattern every SwiGLU decoder emits per layer. The mul is
/// rewritten in place (its output name/type are preserved); the now
/// unused sigmoid dies in [`dead_code`].
pub fn fuse_silu(b: &mut Block) -> PassReport {
    // map: sigmoid output name → its input name
    let mut sigmoid_in: HashMap<String, String> = HashMap::new();
    for op in &b.ops {
        if op.ty == "sigmoid" {
            if let (Some(x), Some(out)) = (arg_name(op, "x"), op.outputs.first()) {
                sigmoid_in.insert(out.name.clone(), x);
            }
        }
    }
    if sigmoid_in.is_empty() {
        return PassReport::default();
    }
    let mut fused = 0;
    for op in &mut b.ops {
        if op.ty != "mul" {
            continue;
        }
        let (xn, yn) = match (arg_name(op, "x"), arg_name(op, "y")) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        // is one operand a sigmoid of the other?
        let base = if sigmoid_in.get(&xn) == Some(&yn) {
            yn.clone()
        } else if sigmoid_in.get(&yn) == Some(&xn) {
            xn.clone()
        } else {
            continue;
        };
        op.ty = "silu".into();
        op.inputs = vec![("x".into(), mil_spec::bind(&base).1)];
        fused += 1;
    }
    PassReport {
        removed: 0,
        added: 0,
        renamed: fused,
    }
}

// ---------- pass 5: dead code ----------

/// Remove ops whose outputs are consumed by nothing. Iterates to a
/// fixpoint so chains die leaf-to-root. `write_state` never dies — it
/// has no outputs but mutates a state, which is a side effect.
pub fn dead_code(b: &mut Block) -> PassReport {
    let mut removed = 0;
    loop {
        let mut used: HashSet<String> = b.outputs.iter().cloned().collect();
        for op in &b.ops {
            for n in consumed_names(op) {
                used.insert(n);
            }
        }
        let before = b.ops.len();
        b.ops.retain(|op| {
            if op.ty == "write_state" {
                return true;
            }
            op.outputs.iter().any(|o| used.contains(&o.name))
        });
        removed += before - b.ops.len();
        if before == b.ops.len() {
            break;
        }
    }
    PassReport {
        removed,
        added: 0,
        renamed: 0,
    }
}

// ---------- orchestration ----------

/// Run every pass to a fixpoint. Order: no-ops first (cheap aliases),
/// fusion, fold, dedup, then dead code; loop until nothing changes —
/// folding can expose new no-ops, dedup can expose new dead code.
pub fn optimize(b: &mut Block) -> PassReport {
    let mut total = PassReport::default();
    for _ in 0..16 {
        let a = noop_elim(b);
        let f = fuse_silu(b);
        let c = const_fold(b);
        let d = dedup_consts(b);
        let e = dead_code(b);
        let round = [a, f, c, d, e];
        let changed: usize = round.iter().map(|r| r.removed + r.added + r.renamed).sum();
        total.removed += round.iter().map(|r| r.removed).sum::<usize>();
        total.added += round.iter().map(|r| r.added).sum::<usize>();
        total.renamed += round.iter().map(|r| r.renamed).sum::<usize>();
        if changed == 0 {
            break;
        }
    }
    total
}

/// Convenience: optimize an owned block.
pub fn optimize_block(b: Block) -> (Block, PassReport) {
    let mut b = b;
    let r = optimize(&mut b);
    (b, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dup_const_block() -> Block {
        let mut b = Block::new();
        let s1 = b.konst_i32("s1", &[1, 1]);
        let s2 = b.konst_i32("s2", &[1, 1]);
        let a = b.mul("x", &s1, &[1, 2, 1, 1], "a");
        let c = b.mul("x", &s2, &[1, 2, 1, 1], "c");
        b.outputs = vec![a, c];
        b
    }

    #[test]
    fn dedup_merges_identical_consts() {
        let mut b = dup_const_block();
        let n_before = b.ops.len();
        let r = dedup_consts(&mut b);
        assert_eq!(r.removed, 1);
        assert_eq!(b.ops.len(), n_before - 1);
        let mul = b
            .ops
            .iter()
            .find(|o| o.ty == "mul" && o.outputs[0].name == "c")
            .unwrap();
        let bound = mul.inputs.iter().find(|(k, _)| k == "y").unwrap();
        match &bound.1 .0[0] {
            Binding::Name(n) => assert_eq!(n, "s1"),
            _ => panic!("expected name binding"),
        }
    }

    #[test]
    fn dead_code_removes_unfed_chains() {
        let mut b = Block::new();
        let k = b.konst_f16("k", 2.0);
        let _dead = b.mul("x", &k, &[1, 2, 1, 1], "dead");
        let live = b.add("x", "x", &[1, 2, 1, 1], "live");
        b.outputs = vec![live];
        let r = dead_code(&mut b);
        assert_eq!(r.removed, 2); // dead mul + its now-orphaned const
        assert!(b
            .ops
            .iter()
            .all(|o| o.outputs.iter().all(|o| o.name != "dead" && o.name != "k")));
    }

    #[test]
    fn write_state_survives_dead_code() {
        let mut b = Block::new();
        let k = b.konst_f16("k", 1.0);
        b.write_state("kv", &k);
        let y = b.add("x", "x", &[1, 2, 1, 1], "y");
        b.outputs = vec![y];
        dead_code(&mut b);
        assert!(b.ops.iter().any(|o| o.ty == "write_state"));
        // k feeds write_state → stays
        assert!(b
            .ops
            .iter()
            .any(|o| o.outputs.first().map(|o| o.name.as_str()) == Some("k")));
    }

    #[test]
    fn noop_identity_transpose_is_aliased() {
        let mut b = Block::new();
        let p = b.konst_i32("p", &[0, 1]);
        let t = {
            let vt = ValueType::Tensor(TensorType::f16(&[2, 3]));
            b.o1(
                "transpose",
                vec![
                    ("x".into(), mil_spec::bind("a").1),
                    ("perm".into(), mil_spec::bind(&p).1),
                ],
                "t",
                vt,
            )
        };
        let y = b.mul(&t, &t, &[2, 3], "y");
        b.outputs = vec![y];
        // 'a' is produced inside the block so the env knows it
        let mut b = b;
        let k = b.konst_f16("k", 0.0);
        let _ = k;
        let r = noop_elim(&mut b);
        assert_eq!(r.removed, 1);
        // y's x now binds 'a' directly
        let mul = b.ops.iter().find(|o| o.outputs[0].name == "y").unwrap();
        match &mul.inputs[0].1 .0[0] {
            Binding::Name(n) => assert_eq!(n, "a"),
            _ => panic!(),
        }
    }

    #[test]
    fn fuse_silu_pattern() {
        let mut b = Block::new();
        let sg = {
            let vt = ValueType::Tensor(TensorType::f16(&[1, 4, 1, 1]));
            b.o1(
                "sigmoid",
                vec![("x".into(), mil_spec::bind("gate").1)],
                "sg",
                vt,
            )
        };
        let s = b.mul("gate", &sg, &[1, 4, 1, 1], "swiglu");
        b.outputs = vec![s];
        let r = fuse_silu(&mut b);
        assert_eq!(r.renamed, 1);
        let op = b.ops.iter().find(|o| o.ty == "silu").unwrap();
        assert_eq!(op.outputs[0].name, "swiglu");
        // dead code reaps the orphaned sigmoid
        dead_code(&mut b);
        assert!(b.ops.iter().all(|o| o.ty != "sigmoid"));
    }

    #[test]
    fn fold_add_of_two_consts() {
        let mut b = Block::new();
        b.op(
            "const",
            vec![],
            vec![("a", ValueType::Tensor(TensorType::f16(&[2])))],
            vec![("val".into(), Value::f16s(&[2], &[1.0, 2.0]))],
        );
        b.op(
            "const",
            vec![],
            vec![("c", ValueType::Tensor(TensorType::f16(&[2])))],
            vec![("val".into(), Value::f16s(&[2], &[10.0, 20.0]))],
        );
        let y = b.add("a", "c", &[2], "y");
        b.outputs = vec![y];
        let r = const_fold(&mut b);
        assert_eq!(r.removed, 1);
        assert_eq!(r.added, 1);
        let out = b.outputs[0].clone();
        let folded = producer(&b.ops, &out).unwrap();
        assert_eq!(const_f32s(folded).unwrap(), vec![11.0, 22.0]);
    }

    #[test]
    fn fold_transpose_2x2() {
        let mut b = Block::new();
        b.op(
            "const",
            vec![],
            vec![("a", ValueType::Tensor(TensorType::f16(&[2, 2])))],
            vec![("val".into(), Value::f16s(&[2, 2], &[1.0, 2.0, 3.0, 4.0]))],
        );
        let p = b.konst_i32("p", &[1, 0]);
        let t = b.transpose("a", &[1, 0], &[2, 2], "t");
        let _ = p;
        b.outputs = vec![t];
        let r = const_fold(&mut b);
        assert_eq!(r.removed, 1);
        let folded = producer(&b.ops, &b.outputs[0]).unwrap();
        assert_eq!(const_f32s(folded).unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn fold_concat_axis0() {
        let mut b = Block::new();
        b.op(
            "const",
            vec![],
            vec![("a", ValueType::Tensor(TensorType::f16(&[1, 2])))],
            vec![("val".into(), Value::f16s(&[1, 2], &[1.0, 2.0]))],
        );
        b.op(
            "const",
            vec![],
            vec![("c", ValueType::Tensor(TensorType::f16(&[1, 2])))],
            vec![("val".into(), Value::f16s(&[1, 2], &[3.0, 4.0]))],
        );
        let cat = b.concat(&["a".into(), "c".into()], 0, &[2, 2], "cat");
        b.outputs = vec![cat];
        let r = const_fold(&mut b);
        assert_eq!(r.removed, 1);
        let folded = producer(&b.ops, &b.outputs[0]).unwrap();
        assert_eq!(const_f32s(folded).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn blob_consts_are_never_folded() {
        let mut b = Block::new();
        let w = b.konst_blob("w", "@model_path/weights/weight.bin", 64, DType::Fp16, &[4]);
        let y = b.mul("x", &w, &[1, 4, 1, 1], "y");
        b.outputs = vec![y];
        let r = const_fold(&mut b);
        assert_eq!(r.removed, 0);
        assert_eq!(r.added, 0);
    }

    #[test]
    fn optimize_fixpoint_shrinks_graph() {
        let mut b = dup_const_block();
        let n0 = b.ops.len();
        let _ = optimize(&mut b);
        assert!(b.ops.len() < n0);
    }
}
