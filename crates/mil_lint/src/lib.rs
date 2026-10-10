//! `mil_lint` — static ANE-placement analysis for MIL graphs.
//!
//! `coremltools` cannot tell you where a graph will run. You compile,
//! load, and profile — and find out the ANE rejected your model because
//! one op fell back to CPU, dragging the whole graph with it. `mil_lint`
//! predicts placement per-op *before* compile, with a stated reason for
//! every verdict, so you fix the graph instead of discovering it in
//! Instruments.
//!
//! The rules encode known CoreML/ANE behavior: fp16 conv ops in 4D
//! NCHW-ish layout are the ANE's native habitat; elementwise and small
//! matmuls join them; dynamic shapes, string ops, control flow, and
//! fp32-heavy regions push ops to GPU or CPU. `write_state`/`read_state`
//! are ANE-friendly (the ANE keeps state on-chip), which is what makes
//! fat KV-cache graphs viable.
//!
//! # Verdicts
//!
//! - [`Unit::Ane`] — the op matches ANE-native patterns
//! - [`Unit::Gpu`] — ANE-eligible shape but flagged dtype/size pattern
//! - [`Unit::Cpu`] — op class the ANE does not execute
//! - [`Unit::Unknown`] — no rule matched (treated as GPU by callers)
//!
//! # Usage
//!
//! ```
//! use mil_lint::{lint_block, Unit};
//! let mut b = mil_spec::Block::new();
//! let y = b.add("x", "x", &[1, 4, 1, 1], "y");
//! b.outputs = vec![y];
//! let report = lint_block(&b);
//! assert_eq!(report.verdicts[0].unit, Unit::Ane);
//! ```

#![forbid(unsafe_code)]

use mil_spec::{Block, Op, Value, ValueType};
use std::fmt;

/// Predicted execution unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Unit {
    /// Runs on the Apple Neural Engine.
    Ane,
    /// ANE-eligible shape but dtype/size/pattern flags push it to GPU.
    Gpu,
    /// Op class the ANE does not execute.
    Cpu,
    /// No rule matched — conservative callers treat this as GPU.
    Unknown,
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Unit::Ane => "ANE",
            Unit::Gpu => "GPU",
            Unit::Cpu => "CPU",
            Unit::Unknown => "???",
        };
        f.write_str(s)
    }
}

/// Severity of a lint flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// Likely harmless, worth knowing.
    Info,
    /// Probable placement problem — review.
    Warn,
    /// Guaranteed fallback or spec error.
    Error,
}

/// A flagged finding attached to an op (or the graph).
#[derive(Clone, Debug)]
pub struct Finding {
    /// Op output name or "<graph>".
    pub subject: String,
    /// Rule that fired.
    pub rule: String,
    /// What to do about it.
    pub message: String,
    /// How bad.
    pub severity: Severity,
}

/// One op's predicted unit plus why.
#[derive(Clone, Debug)]
pub struct Verdict {
    /// Op output name(s).
    pub name: String,
    /// MIL op type.
    pub op: String,
    /// Predicted unit.
    pub unit: Unit,
    /// Rule that produced the verdict.
    pub rule: String,
    /// Output shape (if tensor-typed).
    pub shape: Option<Vec<i64>>,
}

/// Full lint report for a block.
#[derive(Clone, Debug)]
pub struct LintReport {
    /// Per-op verdicts in program order.
    pub verdicts: Vec<Verdict>,
    /// Findings.
    pub findings: Vec<Finding>,
    /// Ops placed on ANE.
    pub ane_ops: usize,
    /// Ops placed on GPU.
    pub gpu_ops: usize,
    /// Ops placed on CPU.
    pub cpu_ops: usize,
    /// Ops with no rule.
    pub unknown_ops: usize,
    /// Estimated dispatch count — contiguous same-unit runs. This is the
    /// number that killed the 11-shard drafter: each boundary costs a
    /// context switch and a round-trip of activations off the ANE.
    pub estimated_dispatches: usize,
}

impl LintReport {
    /// Fraction of ops predicted to run on the ANE.
    pub fn ane_fraction(&self) -> f64 {
        if self.verdicts.is_empty() {
            0.0
        } else {
            self.ane_ops as f64 / self.verdicts.len() as f64
        }
    }

    /// True if every op is predicted ANE (the fat-graph goal).
    pub fn is_fat(&self) -> bool {
        self.ane_ops > 0 && self.cpu_ops == 0 && self.unknown_ops == 0
    }

    /// Render a per-op table plus summary — `milc lint` output.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "{:<32} {:<30} {:<5} {}\n",
            "name", "op", "unit", "rule"
        ));
        s.push_str(&format!("{}\n", "-".repeat(90)));
        for v in &self.verdicts {
            let name = if v.name.len() > 32 {
                format!("{}…", &v.name[..31])
            } else {
                v.name.clone()
            };
            s.push_str(&format!(
                "{:<32} {:<30} {:<5} {}\n",
                name, v.op, v.unit, v.rule
            ));
        }
        s.push_str(&format!("{}\n", "-".repeat(90)));
        s.push_str(&format!(
            "ops: {}  ANE {} ({:.1}%)  GPU {}  CPU {}  unknown {}\n",
            self.verdicts.len(),
            self.ane_ops,
            self.ane_fraction() * 100.0,
            self.gpu_ops,
            self.cpu_ops,
            self.unknown_ops
        ));
        s.push_str(&format!(
            "estimated dispatches: {}\n",
            self.estimated_dispatches
        ));
        if !self.findings.is_empty() {
            s.push_str("findings:\n");
            for f in &self.findings {
                let sev = match f.severity {
                    Severity::Info => "info ",
                    Severity::Warn => "warn ",
                    Severity::Error => "error",
                };
                s.push_str(&format!("  {} {}: {}\n", sev, f.subject, f.message));
            }
        }
        s
    }
}

/// Whether a const value is a string.
fn const_is_str(op: &Op) -> bool {
    op.ty == "const"
        && op.attrs.iter().any(|(_, v)| {
            matches!(v, Value::Str(_))
                || matches!(
                    v,
                    Value::Imm(ValueType::Tensor(t), _)
                        if t.dtype == mil_spec::DType::Str
                )
        })
}

/// dtype declared on an op's first output.
fn out_dtype(op: &Op) -> Option<mil_spec::DType> {
    op.outputs.first().and_then(|o| {
        if let ValueType::Tensor(t) = &o.ty {
            Some(t.dtype)
        } else {
            None
        }
    })
}

/// Rank of an op's output shape.
fn out_rank(op: &Op) -> usize {
    op.outputs
        .first()
        .and_then(|o| {
            if let ValueType::Tensor(t) = &o.ty {
                Some(t.shape.len())
            } else {
                None
            }
        })
        .unwrap_or(0)
}

/// Classify a single op by type, declared output dtype, output rank, and
/// whether it's a string const. This is the public face of the rule
/// table — `mil_verify` feeds spec-decoded ops through it so a lint
/// report works on packages, not just in-memory blocks.
pub fn classify_op(
    ty: &str,
    dtype: Option<mil_spec::DType>,
    rank: usize,
    str_const: bool,
) -> (Unit, &'static str) {
    // Rank guard first — the ANE's layout machinery is built around
    // rank ≤ 4 (5 works for some ops but is a fallback magnet).
    if rank > 4 {
        return (Unit::Gpu, "rank>4");
    }
    let f32 = dtype == Some(mil_spec::DType::Fp32);

    match ty {
        // --- Always-ANE core ---
        "conv" => {
            if f32 {
                (Unit::Gpu, "conv fp32")
            } else {
                (Unit::Ane, "conv")
            }
        }
        "batch_norm" | "activation" => (Unit::Ane, "norm/act"),

        // Nonlinearities — elementwise activations are ANE-native.
        "sigmoid" | "tanh" | "relu" | "relu6" | "leaky_relu" | "prelu" | "gelu" | "softplus"
        | "softsign" | "sigmoid_hard" | "silu" | "elu" => {
            if f32 {
                (Unit::Gpu, "activation fp32")
            } else {
                (Unit::Ane, "activation")
            }
        }

        // Elementwise fp16 — the ANE's bread and butter.
        "add" | "sub" | "mul" | "div" | "real_div" | "pow" | "maximum" | "minimum" | "floor"
        | "ceil" | "round" | "sqrt" | "rsqrt" | "exp" | "log" | "sin" | "cos" | "tan" | "abs"
        | "sign" | "neg" | "inverse" | "clip" | "threshold" | "scale" => {
            if f32 {
                (Unit::Gpu, "elementwise fp32")
            } else {
                (Unit::Ane, "elementwise fp16")
            }
        }

        // Reductions — ANE does these but they serialize on the channel.
        "reduce_sum" | "reduce_mean" | "reduce_max" | "reduce_min" | "reduce_prod"
        | "reduce_l2_norm" | "reduce_l1_norm" | "reduce_sumsquare" => {
            if f32 {
                (Unit::Gpu, "reduce fp32")
            } else {
                (Unit::Ane, "reduce fp16")
            }
        }

        // Shape gymnastics — free on ANE but each is a dispatch. Flag
        // heavy use so callers can fold them.
        "reshape" | "transpose" | "expand_dims" | "squeeze" | "flatten" | "reverse"
        | "slice_by_index" | "slice_by_size" | "split" => (Unit::Ane, "layout op"),

        "concat" | "stack" | "tile" | "pad" => (Unit::Ane, "layout op"),

        // slice_update is the KV-cache write path — the packed-state
        // pattern writes through it; it lives where the state lives.
        "slice_update" | "slice_update_dynamic" => (Unit::Ane, "state write"),

        // Matmul — ANE runs it; big vocab-side matmuls are the border.
        "matmul" | "batched_matmul" => {
            if f32 {
                (Unit::Gpu, "matmul fp32")
            } else {
                (Unit::Ane, "matmul")
            }
        }

        "conv_transpose" => (Unit::Gpu, "conv_transpose"),

        // Pooling.
        "pooling" | "global_pooling" | "l2_pooling" | "avg_pool" | "max_pool" => {
            if f32 {
                (Unit::Gpu, "pool fp32")
            } else {
                (Unit::Ane, "pool")
            }
        }

        "softmax" => {
            if f32 {
                (Unit::Gpu, "softmax fp32")
            } else {
                (Unit::Ane, "softmax fp16")
            }
        }

        "layer_norm" | "instance_norm" | "local_response_norm" => (Unit::Ane, "norm"),

        // Quantized path — constexpr_blockwise_shift_scale is the int8
        // residency pattern; dequant happens on-ANE.
        "constexpr_blockwise_shift_scale" | "constexpr_affine_dequantize" => (Unit::Ane, "dequant"),

        // State ops — the ANE keeps state on-chip. This is why the fat
        // KV-cache graph works; on GPU these bounce through unified mem.
        "read_state" | "write_state" => (Unit::Ane, "state"),

        "cast" => (Unit::Ane, "cast"),
        "gather" | "gather_nd" | "scatter" | "scatter_nd" => (Unit::Ane, "gather"),

        // Const ops are metadata — placement follows their consumer.
        "const" => {
            if str_const {
                (Unit::Cpu, "str const")
            } else {
                (Unit::Ane, "const")
            }
        }

        // --- GPU-only ---
        "gru" | "lstm" | "rnn" | "uni_directional_lstm" | "bi_directional_lstm" => {
            (Unit::Gpu, "recurrent")
        }
        "argmax" | "argmin" | "argsort" | "non_maximum_suppression" | "topk" => {
            (Unit::Gpu, "argmax-family")
        }
        "upsample" | "resize_bilinear" | "resample" | "crop_resize" | "crop" => {
            (Unit::Gpu, "resize")
        }
        "embedding" | "one_hot" => (Unit::Gpu, "embedding"),

        // --- CPU-only ---
        "while_loop" | "branch" | "if" | "loop" | "for" | "list" | "tuple" | "make_list"
        | "list_gather" | "list_scatter" | "select" => (Unit::Cpu, "control flow"),
        "random" | "random_bernoulli" | "random_categorical" | "random_normal"
        | "random_uniform" | "multinomial" => (Unit::Cpu, "random"),
        "get_shape" | "fill" | "fill_dynamic" | "range_1d" | "non_zero" | "cumsum" => {
            (Unit::Cpu, "dynamic shape")
        }

        // Custom ops land wherever the extension registers.
        t if t.starts_with("custom") || t.contains("custom") => (Unit::Unknown, "custom op"),

        _ => (Unit::Unknown, "no rule"),
    }
}

/// Lint a block: classify every op, then flag graph-level problems.
pub fn lint_block(b: &Block) -> LintReport {
    let mut verdicts = Vec::with_capacity(b.ops.len());

    for op in &b.ops {
        let name = op
            .outputs
            .first()
            .map(|o| o.name.clone())
            .unwrap_or_else(|| format!("<{}>", op.ty));
        let (unit, rule) = classify_op(&op.ty, out_dtype(op), out_rank(op), const_is_str(op));
        let shape = op.outputs.first().and_then(|o| {
            if let ValueType::Tensor(t) = &o.ty {
                Some(t.shape.clone())
            } else {
                None
            }
        });
        verdicts.push(Verdict {
            name,
            op: op.ty.clone(),
            unit,
            rule: rule.to_string(),
            shape,
        });
    }
    report_from_verdicts(verdicts)
}

/// Assemble a [`LintReport`] from verdicts: graph-level findings plus
/// the unit/dispatch counts. Shared by `lint_block` (in-memory graphs)
/// and `mil_verify::lint_spec` (decoded packages).
pub fn report_from_verdicts(verdicts: Vec<Verdict>) -> LintReport {
    let mut findings = Vec::new();

    // Graph-level findings.

    // 1. CPU ops in the middle of an ANE run — the real killer. A single
    //    CPU op between ANE ops costs two transfers.
    for (i, v) in verdicts.iter().enumerate() {
        if v.unit == Unit::Cpu {
            let left_ane = verdicts[..i]
                .iter()
                .rev()
                .find(|u| u.unit == Unit::Ane)
                .is_some();
            let right_ane = verdicts[i + 1..]
                .iter()
                .find(|u| u.unit == Unit::Ane)
                .is_some();
            if left_ane && right_ane {
                findings.push(Finding {
                    subject: v.name.clone(),
                    rule: "cpu-island".into(),
                    message: format!(
                        "{} on CPU between ANE ops — two off-ANE transfers per call",
                        v.op
                    ),
                    severity: Severity::Error,
                });
            } else {
                findings.push(Finding {
                    subject: v.name.clone(),
                    rule: "cpu-op".into(),
                    message: format!("{} cannot run on the ANE", v.op),
                    severity: Severity::Warn,
                });
            }
        }
    }

    // 2. fp32 islands — ANE is fp16-first; fp32 regions split the graph.
    for v in &verdicts {
        if v.rule.contains("fp32") {
            findings.push(Finding {
                subject: v.name.clone(),
                rule: "fp32".into(),
                message: format!(
                    "{} output is fp32 — fp16 runs on ANE, fp32 goes to GPU",
                    v.op
                ),
                severity: Severity::Warn,
            });
        }
    }

    // 3. Dispatch-count pressure. Every boundary between units is a
    //    context switch. 11 dispatches at ~10ms each is how the 0.6B
    //    drafter lost to a smaller GPU model.
    let mut dispatches = 0;
    let mut prev: Option<Unit> = None;
    for v in &verdicts {
        if v.unit == Unit::Unknown {
            continue; // const metadata joins its consumer's unit
        }
        if prev != Some(v.unit) {
            dispatches += 1;
            prev = Some(v.unit);
        }
    }
    if dispatches > 4 {
        findings.push(Finding {
            subject: "<graph>".into(),
            rule: "dispatch-pressure".into(),
            message: format!(
                "{} estimated dispatches — each boundary costs a unit switch",
                dispatches
            ),
            severity: Severity::Warn,
        });
    }

    // 4. Unknown ops — surface them so the rule table grows.
    for v in &verdicts {
        if v.unit == Unit::Unknown {
            findings.push(Finding {
                subject: v.name.clone(),
                rule: "unknown-op".into(),
                message: format!("{} has no placement rule", v.op),
                severity: Severity::Info,
            });
        }
    }

    let ane_ops = verdicts.iter().filter(|v| v.unit == Unit::Ane).count();
    let gpu_ops = verdicts.iter().filter(|v| v.unit == Unit::Gpu).count();
    let cpu_ops = verdicts.iter().filter(|v| v.unit == Unit::Cpu).count();
    let unknown_ops = verdicts.iter().filter(|v| v.unit == Unit::Unknown).count();

    LintReport {
        verdicts,
        findings,
        ane_ops,
        gpu_ops,
        cpu_ops,
        unknown_ops,
        estimated_dispatches: dispatches,
    }
}

/// Lint an IR program (compiles to a block first via `mil_spec::ir`).
pub fn lint_ir(source: &str) -> Result<LintReport, mil_spec::ir::IrError> {
    let prog = mil_spec::ir::compile(source)?;
    Ok(lint_block(&prog.block))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_and_conv_are_ane() {
        let mut b = Block::new();
        let w = b.konst_blob(
            "w",
            "@model_path/weights/weight.bin",
            64,
            mil_spec::DType::Fp16,
            &[4, 4, 1, 1],
        );
        let c = b.conv1x1("x", &w, None, 4, "c");
        let y = b.add(&c, &c, &[1, 4, 1, 1], "y");
        b.outputs = vec![y];
        let r = lint_block(&b);
        let conv = r.verdicts.iter().find(|v| v.op == "conv").unwrap();
        assert_eq!(conv.unit, Unit::Ane);
        let add = r.verdicts.iter().find(|v| v.op == "add").unwrap();
        assert_eq!(add.unit, Unit::Ane);
    }

    #[test]
    fn cpu_op_between_ane_ops_is_flagged() {
        let mut b = Block::new();
        let a = b.add("x", "x", &[1, 4, 1, 1], "a");
        // get_shape has no ANE rule → CPU
        let g = {
            let vt = ValueType::Tensor(mil_spec::TensorType {
                dtype: mil_spec::DType::Int32,
                shape: vec![4],
            });
            b.o1(
                "get_shape",
                vec![("x".into(), mil_spec::bind(&a).1)],
                "g",
                vt,
            )
        };
        let _ = g;
        let y = b.add(&a, &a, &[1, 4, 1, 1], "y");
        b.outputs = vec![y];
        let r = lint_block(&b);
        assert!(r.findings.iter().any(|f| f.rule == "cpu-island"));
    }

    #[test]
    fn fp32_softmax_goes_to_gpu() {
        let mut b = Block::new();
        let ax = b.konst_scalar_i32("ax", 1);
        let _ = ax;
        let s = {
            let vt = ValueType::Tensor(mil_spec::TensorType {
                dtype: mil_spec::DType::Fp32,
                shape: vec![1, 4, 1, 1],
            });
            b.op(
                "softmax",
                vec![("x".into(), mil_spec::bind("x").1)],
                vec![("s", vt)],
                vec![],
            );
            "s".to_string()
        };
        b.outputs = vec![s];
        let r = lint_block(&b);
        let sm = r.verdicts.iter().find(|v| v.op == "softmax").unwrap();
        assert_eq!(sm.unit, Unit::Gpu);
        assert!(r.findings.iter().any(|f| f.rule == "fp32"));
    }

    #[test]
    fn control_flow_is_cpu() {
        let mut b = Block::new();
        b.op("while_loop", vec![], vec![], vec![]);
        b.outputs = vec![];
        let r = lint_block(&b);
        assert_eq!(r.verdicts[0].unit, Unit::Cpu);
    }

    #[test]
    fn state_ops_are_ane() {
        let mut b = Block::new();
        let r = b.read_state("kv", &[1, 8, 64, 64], "rd");
        b.write_state("kv", &r);
        b.outputs = vec![r];
        let rep = lint_block(&b);
        assert!(rep
            .verdicts
            .iter()
            .filter(|v| v.op == "read_state" || v.op == "write_state")
            .all(|v| v.unit == Unit::Ane));
    }

    #[test]
    fn dispatch_count_tracks_boundaries() {
        let mut b = Block::new();
        let a = b.add("x", "x", &[1, 4, 1, 1], "a"); // ANE
        let g = {
            let vt = ValueType::Tensor(mil_spec::TensorType {
                dtype: mil_spec::DType::Int32,
                shape: vec![4],
            });
            b.o1(
                "get_shape",
                vec![("x".into(), mil_spec::bind(&a).1)],
                "g",
                vt,
            )
        };
        let _ = g;
        let y = b.add(&a, &a, &[1, 4, 1, 1], "y"); // ANE
        b.outputs = vec![y];
        let r = lint_block(&b);
        // ANE → CPU → ANE = 3 dispatches
        assert!(r.estimated_dispatches >= 3);
    }
}
