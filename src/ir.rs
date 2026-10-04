//! `ir` — a validated IR-to-MIL front-end for `mil_spec`.
//!
//! `Block` lets you write MIL by hand, one op at a time. This module adds
//! the layer above it: a small line-oriented IR that parses into
//! statements, lowers onto the typed `Block` helpers, and validates the
//! program *before* anything hits the wire — every referenced name must
//! exist, every shape must check out, and declared outputs must match
//! their inferred types.
//!
//! # Grammar
//!
//! ```text
//! # comment
//! input  x: fp16[1,64,1,1]
//! state  kv: fp16[1,8,64,64]
//! output y                          # inferred from the bound value
//!
//! w   = const_blob(file, offset, dtype, [shape])
//! q8  = const_q8(file, data_off, scale_off, [shape])
//! k   = const_f16(0.5) | const_i32([1,2]) | const_scalar_i32(1) | const_bool(true)
//! z   = mul(a, b) | add(a, b) | sub(a, b)
//! r   = reshape(x, [s,...])
//! t   = transpose(x, perm=[0,2,1])
//! e   = expand_dims(x, axes=[1]) | squeeze(x, axes=[1])
//! s   = slice(x, begin=[...], end=[...])
//! c   = concat([a, b, ...], axis)
//! m   = matmul(x, y[, transpose_y])
//! sm  = softmax(x, axis[, fp32])
//! cs  = cast(x, to=fp32|fp16|int32)
//! n   = rms_norm(x, w=gamma, d=64, eps=1e-5)
//! rd  = read_state(kv)
//!     write_state(kv, rd)
//! output z
//! ```
//!
//! Compile:
//!
//! ```no_run
//! let prog = mil_spec::ir::compile("input x: fp16[1,4,1,1]\ny = mul(x, x)\noutput y").unwrap();
//! let spec = mil_spec::encode_model(&prog.inputs, &prog.outputs, &prog.states,
//!                                   &prog.block, &prog.fn_inputs,
//!                                   &mil_spec::ModelMeta::new(8, "CoreML5"));
//! mil_spec::write_mlpackage(std::path::Path::new("out.mlpackage"), &spec, None).unwrap();
//! ```

use crate::{Block, DType, Feature, TensorType, ValueType, NVT};
use std::collections::BTreeMap;
use std::fmt;

/// A compile error with line context.
#[derive(Debug, Clone)]
pub struct IrError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for IrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            write!(f, "ir: {}", self.message)
        } else {
            write!(f, "ir line {}: {}", self.line, self.message)
        }
    }
}

impl std::error::Error for IrError {}

type Result<T> = std::result::Result<T, IrError>;

fn err<T>(line: usize, msg: impl Into<String>) -> Result<T> {
    Err(IrError {
        line,
        message: msg.into(),
    })
}

/// Inferred type of a named value in the program.
#[derive(Clone, Debug, PartialEq)]
pub struct Ty {
    pub dtype: DType,
    pub shape: Vec<i64>,
}

impl Ty {
    fn numel(&self) -> i64 {
        self.shape.iter().product()
    }
    fn value_type(&self) -> ValueType {
        ValueType::Tensor(TensorType {
            dtype: self.dtype,
            shape: self.shape.clone(),
        })
    }
}

/// A compiled program — everything `encode_model` needs.
pub struct Program {
    pub block: Block,
    pub inputs: Vec<Feature>,
    pub outputs: Vec<Feature>,
    pub states: Vec<Feature>,
    pub fn_inputs: Vec<NVT>,
    /// The inferred type of every named value — useful for inspection
    /// and for callers validating a follow-up compile.
    pub types: BTreeMap<String, Ty>,
}

/// Compile IR source into a `Program`. Parse, lower, and validate in one
/// pass — a returned `Program` is guaranteed: all value references
/// resolved, all op shapes consistent, all declared outputs defined.
pub fn compile(source: &str) -> Result<Program> {
    let mut c = Compiler::new();
    for (i, raw) in source.lines().enumerate() {
        c.line(raw, i + 1)?;
    }
    c.finish()
}

struct Compiler {
    block: Block,
    /// name → inferred type (inputs, consts, op outputs, read_state vals)
    env: BTreeMap<String, Ty>,
    inputs: Vec<Feature>,
    outputs: Vec<Feature>,
    states: Vec<Feature>,
}

impl Compiler {
    fn new() -> Self {
        Self {
            block: Block::new(),
            env: BTreeMap::new(),
            inputs: vec![],
            outputs: vec![],
            states: vec![],
        }
    }

    fn line(&mut self, raw: &str, n: usize) -> Result<()> {
        let text = raw.split('#').next().unwrap_or("").trim();
        if text.is_empty() {
            return Ok(());
        }
        if let Some(rest) = text.strip_prefix("input ") {
            return self.decl_input(rest, n);
        }
        if let Some(rest) = text.strip_prefix("state ") {
            return self.decl_state(rest, n);
        }
        if let Some(rest) = text.strip_prefix("output ") {
            return self.decl_output(rest, n);
        }
        if let Some((name, rhs)) = text.split_once('=') {
            return self.assign(name.trim(), rhs.trim(), n);
        }
        // bare call with no binding — currently only write_state produces
        // no value
        if text.starts_with("write_state") {
            return self.stmt_write_state(text, n);
        }
        err(n, format!("unrecognized statement: {text}"))
    }

    // ---------- declarations ----------

    fn parse_typed(&self, rest: &str, n: usize) -> Result<(String, DType, Vec<i64>)> {
        let Some((raw_name, ty)) = rest.split_once(':') else {
            return err(n, "expected `name: dtype[shape]`");
        };
        let name = raw_name.trim().to_string();
        let ty = ty.trim();
        let Some(lb) = ty.find('[') else {
            return err(n, format!("expected dtype[shape], got {ty}"));
        };
        let Some(rb) = ty.rfind(']') else {
            return err(n, format!("unterminated shape in {ty}"));
        };
        let Some(dtype) = parse_dtype(ty[..lb].trim()) else {
            return err(n, format!("unknown dtype {}", ty[..lb].trim()));
        };
        let shape: Result<Vec<i64>> = ty[lb + 1..rb]
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                s.trim().parse::<i64>().map_err(|_| IrError {
                    line: n,
                    message: format!("bad dim {s}"),
                })
            })
            .collect();
        Ok((name, dtype, shape?))
    }

    fn decl_input(&mut self, rest: &str, n: usize) -> Result<()> {
        let (name, dtype, shape) = self.parse_typed(rest, n)?;
        self.def(
            &name,
            Ty {
                dtype,
                shape: shape.clone(),
            },
            n,
        )?;
        self.inputs.push(Feature {
            name: name.clone(),
            shape,
            dtype,
            is_state: false,
        });
        Ok(())
    }

    fn decl_state(&mut self, rest: &str, n: usize) -> Result<()> {
        let (name, dtype, shape) = self.parse_typed(rest, n)?;
        // A state declares a feature but is not a readable tensor value —
        // read_state produces the tensor.
        self.states.push(Feature {
            name: name.clone(),
            shape,
            dtype,
            is_state: true,
        });
        if self.states.iter().filter(|s| s.name == name).count() > 1 {
            return err(n, format!("duplicate state {name}"));
        }
        Ok(())
    }

    fn decl_output(&mut self, rest: &str, n: usize) -> Result<()> {
        let name = rest.trim();
        let ty = self
            .env
            .get(name)
            .ok_or(())
            .or_else(|_| err::<&Ty>(n, format!("output {name} is not defined")))?
            .clone();
        self.outputs.push(Feature {
            name: name.to_string(),
            shape: ty.shape,
            dtype: ty.dtype,
            is_state: false,
        });
        self.block.outputs.push(name.to_string());
        Ok(())
    }

    // ---------- assignment ----------

    fn def(&mut self, name: &str, ty: Ty, n: usize) -> Result<()> {
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return err(n, format!("invalid value name {name:?}"));
        }
        if self.env.insert(name.to_string(), ty).is_some() {
            return err(n, format!("{name} is already defined"));
        }
        Ok(())
    }

    fn lookup(&self, name: &str, n: usize) -> Result<Ty> {
        self.env.get(name).cloned().ok_or_else(|| IrError {
            line: n,
            message: format!("undefined value {name}"),
        })
    }

    fn assign(&mut self, name: &str, rhs: &str, n: usize) -> Result<()> {
        let (op, argstr) = rhs
            .split_once('(')
            .ok_or(())
            .or_else(|_| err::<(&str, &str)>(n, format!("expected op(...), got {rhs}")))?;
        // Bracket-aware split — commas inside [...] belong to the list,
        // not the arg list.
        let mut args = Vec::new();
        let mut depth = 0i32;
        let mut cur = String::new();
        for ch in argstr.trim_end_matches(')').chars() {
            match ch {
                '[' => depth += 1,
                ']' => depth -= 1,
                ',' if depth == 0 => {
                    args.push(cur.trim().to_string());
                    cur.clear();
                    continue;
                }
                _ => {}
            }
            cur.push(ch);
        }
        if !cur.trim().is_empty() {
            args.push(cur.trim().to_string());
        }
        let args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let ty = self.lower(name, op.trim(), &args, n)?;
        self.def(name, ty, n)
    }

    /// Elementwise gate: both operands fp16, same shape, or one is a
    /// rank-0 scalar.
    fn elementwise(&mut self, name: &str, op: &str, args: &[&str], n: usize) -> Result<Ty> {
        if args.len() != 2 {
            return err(n, format!("{op} takes 2 args"));
        }
        let a = self.lookup(args[0], n)?;
        let b = self.lookup(args[1], n)?;
        for (i, t) in [&a, &b].iter().enumerate() {
            if t.dtype != DType::Fp16 {
                return err(n, format!("{op} arg {i} must be fp16, got {:?}", t.dtype));
            }
        }
        let shape = if a.shape == b.shape {
            a.shape.clone()
        } else if a.shape.is_empty() {
            b.shape.clone()
        } else if b.shape.is_empty() {
            a.shape.clone()
        } else {
            return err(
                n,
                format!("{op} shape mismatch {:?} vs {:?}", a.shape, b.shape),
            );
        };
        match op {
            "mul" => self.block.mul(args[0], args[1], &shape, name),
            "add" => self.block.add(args[0], args[1], &shape, name),
            "sub" => self.block.sub(args[0], args[1], &shape, name),
            _ => return err(n, format!("not elementwise: {op}")),
        };
        Ok(Ty {
            dtype: DType::Fp16,
            shape,
        })
    }

    fn lower(&mut self, name: &str, op: &str, args: &[&str], n: usize) -> Result<Ty> {
        match op {
            "mul" | "add" | "sub" => self.elementwise(name, op, args, n),

            "const_f16" => {
                if args.len() != 1 {
                    return err(n, "const_f16 takes 1 scalar arg");
                }
                let v: f32 = args[0].parse().map_err(|_| IrError {
                    line: n,
                    message: format!("bad f32 literal {}", args[0]),
                })?;
                self.block.konst_f16(name, v);
                Ok(Ty {
                    dtype: DType::Fp16,
                    shape: vec![],
                })
            }
            "const_i32" => {
                let vs = parse_i32_list(args.join(",").as_str(), n)?;
                self.block.konst_i32(name, &vs);
                Ok(Ty {
                    dtype: DType::Int32,
                    shape: vec![vs.len() as i64],
                })
            }
            "const_scalar_i32" => {
                if args.len() != 1 {
                    return err(n, "const_scalar_i32 takes 1 arg");
                }
                let v: i32 = args[0].parse().map_err(|_| IrError {
                    line: n,
                    message: format!("bad i32 literal {}", args[0]),
                })?;
                self.block.konst_scalar_i32(name, v);
                Ok(Ty {
                    dtype: DType::Int32,
                    shape: vec![],
                })
            }
            "const_bool" => {
                if args.len() != 1 {
                    return err(n, "const_bool takes 1 arg");
                }
                let v = match args[0] {
                    "true" => true,
                    "false" => false,
                    _ => return err(n, "const_bool takes true|false"),
                };
                self.block.konst_bool(name, v);
                Ok(Ty {
                    dtype: DType::Bool,
                    shape: vec![],
                })
            }
            "const_blob" => {
                // const_blob(file, offset, dtype, [shape])
                if args.len() != 4 {
                    return err(n, "const_blob(file, offset, dtype, [shape])");
                }
                let file = args[0];
                let off: u64 = args[1].parse().map_err(|_| IrError {
                    line: n,
                    message: "bad blob offset".into(),
                })?;
                let dtype = parse_dtype(args[2]).ok_or_else(|| IrError {
                    line: n,
                    message: format!("unknown dtype {}", args[2]),
                })?;
                let shape = parse_shape(args[3], n)?;
                self.block.konst_blob(name, file, off, dtype, &shape);
                Ok(Ty { dtype, shape })
            }
            "const_q8" => {
                // const_q8(file, data_off, scale_off, [shape]) — int8 weight
                // with per-channel fp16 scale; output is fp16.
                if args.len() != 4 {
                    return err(n, "const_q8(file, data_off, scale_off, [shape])");
                }
                let file = args[0];
                let data_off: u64 = args[1].parse().map_err(|_| IrError {
                    line: n,
                    message: "bad data offset".into(),
                })?;
                let scale_off: u64 = args[2].parse().map_err(|_| IrError {
                    line: n,
                    message: "bad scale offset".into(),
                })?;
                let shape = parse_shape(args[3], n)?;
                if shape.is_empty() {
                    return err(n, "const_q8 shape needs a channel dim");
                }
                self.block.konst_q8(name, file, data_off, scale_off, &shape);
                Ok(Ty {
                    dtype: DType::Fp16,
                    shape,
                })
            }

            "conv1x1" => {
                // conv1x1(x, w[, bias]) — x is (1,cin,1,1), w is (cout,cin,1,1);
                // bias, if given, is (cout,1,1).
                if args.len() < 2 || args.len() > 3 {
                    return err(n, "conv1x1(x, w[, bias])");
                }
                let x = self.lookup(args[0], n)?;
                let w = self.lookup(args[1], n)?;
                if x.shape.len() != 4 || w.shape.len() != 4 {
                    return err(n, "conv1x1 operands must be rank-4");
                }
                if w.shape[1] != x.shape[1] {
                    return err(
                        n,
                        format!(
                            "conv1x1 channel mismatch: x has cin {}, w has cin {}",
                            x.shape[1], w.shape[1]
                        ),
                    );
                }
                let cout = w.shape[0];
                if args.len() == 3 {
                    let b = self.lookup(args[2], n)?;
                    if b.shape.first() != Some(&cout) {
                        return err(n, "conv1x1 bias must have shape (cout,...)");
                    }
                }
                self.block
                    .conv1x1(args[0], args[1], args.get(2).copied(), cout, name);
                Ok(Ty {
                    dtype: DType::Fp16,
                    shape: vec![1, cout, 1, 1],
                })
            }

            "reshape" => {
                if args.len() != 2 {
                    return err(n, "reshape(x, [shape])");
                }
                let x = self.lookup(args[0], n)?;
                let shape = parse_shape(args[1], n)?;
                if x.numel() != shape.iter().product() {
                    return err(
                        n,
                        format!(
                            "reshape preserves elements: {} -> {:?} doesn't",
                            x.numel(),
                            shape
                        ),
                    );
                }
                self.block.reshape(args[0], &shape, name);
                Ok(Ty {
                    dtype: x.dtype,
                    shape,
                })
            }

            "transpose" => {
                if args.len() != 2 {
                    return err(n, "transpose(x, perm=[...])");
                }
                let x = self.lookup(args[0], n)?;
                let perm = parse_perm(args[1], n)?;
                if perm.len() != x.shape.len() {
                    return err(
                        n,
                        format!("perm {:?} doesn't match rank {}", perm, x.shape.len()),
                    );
                }
                let mut sorted = perm.clone();
                sorted.sort_unstable();
                if sorted != (0..x.shape.len() as i32).collect::<Vec<_>>() {
                    return err(n, format!("perm {perm:?} is not a permutation"));
                }
                let out: Vec<i64> = perm.iter().map(|&p| x.shape[p as usize]).collect();
                self.block.transpose(args[0], &perm, &out, name);
                Ok(Ty {
                    dtype: x.dtype,
                    shape: out,
                })
            }

            "expand_dims" => {
                if args.len() != 2 {
                    return err(n, "expand_dims(x, axes=[...])");
                }
                let x = self.lookup(args[0], n)?;
                let axes = parse_i32_list(args[1], n)?;
                let mut shape = x.shape.clone();
                let mut sorted = axes.clone();
                sorted.sort_unstable();
                for &ax in &sorted {
                    if ax < 0 || ax as usize > shape.len() {
                        return err(n, format!("expand_dims axis {ax} out of range"));
                    }
                }
                for &ax in sorted.iter().rev() {
                    shape.insert(ax as usize, 1);
                }
                self.block.expand_dims(args[0], &axes, &shape, name);
                Ok(Ty {
                    dtype: x.dtype,
                    shape,
                })
            }

            "squeeze" => {
                if args.len() != 2 {
                    return err(n, "squeeze(x, axes=[...])");
                }
                let x = self.lookup(args[0], n)?;
                let axes = parse_i32_list(args[1], n)?;
                let mut shape = x.shape.clone();
                let mut sorted = axes.clone();
                sorted.sort_unstable();
                sorted.dedup();
                for &ax in sorted.iter().rev() {
                    let rank = x.shape.len() as i32;
                    let real = if ax < 0 { ax + rank } else { ax };
                    if real < 0 || real as usize >= shape.len() {
                        return err(n, format!("squeeze axis {ax} out of range"));
                    }
                    if shape[real as usize] != 1 {
                        return err(
                            n,
                            format!("squeeze axis {ax} has dim {}, not 1", shape[real as usize]),
                        );
                    }
                    shape.remove(real as usize);
                }
                self.block.squeeze(args[0], &axes, &shape, name);
                Ok(Ty {
                    dtype: x.dtype,
                    shape,
                })
            }

            "slice" => {
                if args.len() != 3 {
                    return err(n, "slice(x, begin=[...], end=[...])");
                }
                let x = self.lookup(args[0], n)?;
                let begin = parse_i32_list(args[1], n)?;
                let end = parse_i32_list(args[2], n)?;
                if begin.len() != x.shape.len() || end.len() != x.shape.len() {
                    return err(n, "slice begin/end must match rank");
                }
                let mut shape = Vec::with_capacity(begin.len());
                for ((&b, &e), &dim) in begin.iter().zip(end.iter()).zip(x.shape.iter()) {
                    if b < 0 || e > dim as i32 || b >= e {
                        return err(n, format!("slice range {b}..{e} invalid for dim {dim}"));
                    }
                    shape.push((e - b) as i64);
                }
                self.block.slice(args[0], &begin, &end, &shape, name);
                Ok(Ty {
                    dtype: x.dtype,
                    shape,
                })
            }

            "concat" => {
                // concat([a, b, ...], axis)
                if args.len() != 2 {
                    return err(n, "concat([names], axis)");
                }
                let names: Vec<String> = args[0]
                    .trim()
                    .strip_prefix('[')
                    .and_then(|s| s.strip_suffix(']'))
                    .ok_or(())
                    .or_else(|_| err::<&str>(n, "concat inputs must be [name, name, ...]"))?
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if names.is_empty() {
                    return err(n, "concat needs at least one input");
                }
                let axis: i32 = args[1].parse().map_err(|_| IrError {
                    line: n,
                    message: format!("bad axis {}", args[1]),
                })?;
                let first = self.lookup(&names[0], n)?;
                let rank = first.shape.len() as i32;
                let real_axis = if axis < 0 { axis + rank } else { axis };
                if real_axis < 0 || real_axis >= rank {
                    return err(
                        n,
                        format!("concat axis {axis} out of range for rank {rank}"),
                    );
                }
                let mut shape = first.shape.clone();
                for name in &names[1..] {
                    let t = self.lookup(name, n)?;
                    if t.shape.len() != first.shape.len() {
                        return err(n, "concat rank mismatch");
                    }
                    for (d, (&a, &b)) in t.shape.iter().zip(first.shape.iter()).enumerate() {
                        if d != real_axis as usize && a != b {
                            return err(n, format!("concat dim {d} mismatch: {a} vs {b}"));
                        }
                    }
                    shape[real_axis as usize] += t.shape[real_axis as usize];
                }
                self.block.concat(&names, axis, &shape, name);
                Ok(Ty {
                    dtype: first.dtype,
                    shape,
                })
            }

            "matmul" => {
                // matmul(x, y[, transpose_y])
                if args.len() < 2 || args.len() > 3 {
                    return err(n, "matmul(x, y[, transpose_y])");
                }
                let x = self.lookup(args[0], n)?;
                let y = self.lookup(args[1], n)?;
                if x.shape.len() < 2 || y.shape.len() < 2 {
                    return err(n, "matmul operands must be rank >= 2");
                }
                let ty = args.len() == 3 && args[2] == "transpose_y";
                // x @ y contracts x's last dim with y's second-to-last;
                // transpose_y flips which of y's trailing dims is the
                // contracted one.
                let xk = *x.shape.last().unwrap();
                let (yk, ycols) = if ty {
                    (*y.shape.last().unwrap(), y.shape[y.shape.len() - 2])
                } else {
                    (y.shape[y.shape.len() - 2], *y.shape.last().unwrap())
                };
                if xk != yk {
                    return err(n, format!("matmul inner dims: {xk} vs {yk}"));
                }
                // out: x batch dims + x rows + y cols
                let mut shape = x.shape.clone();
                *shape.last_mut().unwrap() = ycols;
                self.block.matmul(args[0], args[1], ty, &shape, name);
                Ok(Ty {
                    dtype: DType::Fp16,
                    shape,
                })
            }

            "softmax" => {
                // softmax(x, axis[, fp32])
                if args.len() < 2 || args.len() > 3 {
                    return err(n, "softmax(x, axis[, fp32])");
                }
                let x = self.lookup(args[0], n)?;
                let axis: i32 = args[1].parse().map_err(|_| IrError {
                    line: n,
                    message: format!("bad axis {}", args[1]),
                })?;
                let rank = x.shape.len() as i32;
                let real = if axis < 0 { axis + rank } else { axis };
                if real < 0 || real >= rank {
                    return err(n, format!("softmax axis {axis} out of range"));
                }
                let fp32 = args.len() == 3 && args[2] == "fp32";
                self.block.softmax(args[0], axis, &x.shape, name, fp32);
                Ok(Ty {
                    dtype: if fp32 { DType::Fp32 } else { DType::Fp16 },
                    shape: x.shape.clone(),
                })
            }

            "cast" => {
                if args.len() != 2 {
                    return err(n, "cast(x, to=fp32|fp16|int32)");
                }
                let x = self.lookup(args[0], n)?;
                let to = args[1].strip_prefix("to=").unwrap_or(args[1]);
                let dtype = parse_dtype(to).ok_or_else(|| IrError {
                    line: n,
                    message: format!("unknown dtype {to}"),
                })?;
                if dtype != DType::Fp32 && dtype != DType::Fp16 && dtype != DType::Int32 {
                    return err(n, "cast supports fp32/fp16/int32");
                }
                self.block.cast(
                    args[0],
                    match dtype {
                        DType::Fp32 => "fp32",
                        DType::Int32 => "int32",
                        _ => "fp16",
                    },
                    &x.shape,
                    name,
                    dtype == DType::Fp32,
                );
                Ok(Ty {
                    dtype,
                    shape: x.shape.clone(),
                })
            }

            "rms_norm" => {
                // rms_norm(x, w=gamma_name, d=channels, eps=1e-5)
                if args.len() < 4 {
                    return err(n, "rms_norm(x, w, d, eps)");
                }
                let x = self.lookup(args[0], n)?;
                let w = self.lookup(args[1], n)?;
                let d: i64 = args[2].parse().map_err(|_| IrError {
                    line: n,
                    message: format!("bad channel count {}", args[2]),
                })?;
                let eps: f32 = args[3].parse().map_err(|_| IrError {
                    line: n,
                    message: format!("bad eps {}", args[3]),
                })?;
                if x.shape.len() != 4 {
                    return err(n, "rms_norm expects (n,d,1,1) input");
                }
                if x.shape[1] != d {
                    return err(
                        n,
                        format!("rms_norm d={d} but channel dim is {}", x.shape[1]),
                    );
                }
                if w.shape.first() != Some(&d) {
                    return err(n, "rms_norm weight must start with channel dim");
                }
                self.block
                    .rms_norm(args[0], args[1], d, eps, &x.shape, name);
                Ok(Ty {
                    dtype: DType::Fp16,
                    shape: x.shape.clone(),
                })
            }

            "read_state" => {
                if args.len() != 1 {
                    return err(n, "read_state(state)");
                }
                let st = self
                    .states
                    .iter()
                    .find(|s| s.name == args[0])
                    .ok_or_else(|| IrError {
                        line: n,
                        message: format!("undefined state {}", args[0]),
                    })?;
                let shape = st.shape.clone();
                self.block.read_state(args[0], &shape, name);
                Ok(Ty {
                    dtype: st.dtype,
                    shape,
                })
            }

            _ => err(n, format!("unknown op {op}")),
        }
    }

    fn stmt_write_state(&mut self, text: &str, n: usize) -> Result<()> {
        let inner = text
            .strip_prefix("write_state(")
            .and_then(|s| s.strip_suffix(')'))
            .ok_or(())
            .or_else(|_| err::<&str>(n, "write_state(state, value)"))?;
        let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return err(n, "write_state(state, value)");
        }
        let st = self
            .states
            .iter()
            .find(|s| s.name == parts[0])
            .ok_or_else(|| IrError {
                line: n,
                message: format!("undefined state {}", parts[0]),
            })?
            .clone();
        let v = self.lookup(parts[1], n)?;
        if v.dtype != st.dtype {
            return err(
                n,
                format!(
                    "write_state dtype mismatch: state {:?} vs value {:?}",
                    st.dtype, v.dtype
                ),
            );
        }
        self.block.write_state(parts[0], parts[1]);
        Ok(())
    }

    fn finish(self) -> Result<Program> {
        if self.outputs.is_empty() {
            return err(0, "program declares no outputs");
        }
        let fn_inputs = self
            .inputs
            .iter()
            .map(|f| NVT {
                name: f.name.clone(),
                ty: ValueType::Tensor(TensorType {
                    dtype: f.dtype,
                    shape: f.shape.clone(),
                }),
            })
            .chain(self.states.iter().map(|s| NVT {
                name: s.name.clone(),
                ty: ValueType::State(TensorType {
                    dtype: s.dtype,
                    shape: s.shape.clone(),
                }),
            }))
            .collect();
        Ok(Program {
            block: self.block,
            inputs: self.inputs,
            outputs: self.outputs,
            states: self.states,
            fn_inputs,
            types: self.env,
        })
    }
}

// ---------- small parsers ----------

fn parse_dtype(s: &str) -> Option<DType> {
    match s.trim() {
        "fp16" | "f16" => Some(DType::Fp16),
        "fp32" | "f32" => Some(DType::Fp32),
        "int8" | "i8" => Some(DType::Int8),
        "int32" | "i32" => Some(DType::Int32),
        "int64" | "i64" => Some(DType::Int64),
        "bool" => Some(DType::Bool),
        _ => None,
    }
}

fn parse_shape(s: &str, n: usize) -> Result<Vec<i64>> {
    let inner = s
        .trim()
        .strip_prefix('[')
        .and_then(|x| x.strip_suffix(']'))
        .ok_or(())
        .or_else(|_| err::<&str>(n, format!("expected [shape], got {s}")))?;
    inner
        .split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|x| {
            x.trim().parse::<i64>().map_err(|_| IrError {
                line: n,
                message: format!("bad dim {x}"),
            })
        })
        .collect()
}

fn parse_i32_list(s: &str, n: usize) -> Result<Vec<i32>> {
    let inner = s
        .trim()
        .strip_prefix('[')
        .and_then(|x| x.strip_suffix(']'))
        .ok_or(())
        .or_else(|_| err::<&str>(n, format!("expected [list], got {s}")))?;
    inner
        .split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|x| {
            x.trim().parse::<i32>().map_err(|_| IrError {
                line: n,
                message: format!("bad int {x}"),
            })
        })
        .collect()
}

fn parse_perm(s: &str, n: usize) -> Result<Vec<i32>> {
    let inner = s
        .trim()
        .strip_prefix("perm=")
        .unwrap_or(s.trim())
        .trim()
        .strip_prefix('[')
        .and_then(|x| x.strip_suffix(']'))
        .ok_or(())
        .or_else(|_| err::<&str>(n, format!("expected perm=[...], got {s}")))?;
    inner
        .split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|x| {
            x.trim().parse::<i32>().map_err(|_| IrError {
                line: n,
                message: format!("bad perm element {x}"),
            })
        })
        .collect()
}
