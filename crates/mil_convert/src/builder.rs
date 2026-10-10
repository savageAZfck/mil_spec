//! `builder` — transformer decoder graph generation.
//!
//! Emits a complete decoder as ONE `mil_spec` block — the "fat graph"
//! that replaces the 11-shard drafter. Layout follows the pattern that
//! measured on real hardware (`mil_kvpack` probe):
//!
//! - All activations are 4D `(1, C, S, 1)`-style conv-layout fp16 tensors
//!   — the ANE's native habitat.
//! - One **packed KV state** `(layers*2, kvh, max_kv, dh)`: layer `l`
//!   owns rows `2l` (K) and `2l+1` (V). Writes go through `slice_update`
//!   at runtime `pos`; the whole state is one feature, not `2*layers`.
//! - **GQA via query expansion**: q is reshaped to `(1, kvh, g, dh)`
//!   where `g = q_heads/kv_heads` — each kv head serves its query group
//!   without a tile op.
//! - RoPE applies per-head on `(1, h, S, dh)`; `cos`/`sin` arrive as
//!   graph inputs (caller computes per position — the ANE can't index
//!   a table at a runtime position).
//! - `mask` is a runtime `(1, 1, S, max_kv)` fp16 additive input
//!   (0 valid, large-negative invalid) — covers causality *and* the
//!   unwritten tail of the fixed state buffer.
//! - Weights are conv-packed `(out, in, 1, 1)` blob consts — int8
//!   per-channel via `constexpr_blockwise_shift_scale` or raw fp16.
//!
//! The builder writes `weight.bin` as it goes (streamed through
//! [`WeightEmitter`]), so a 4 B model converts without an extra copy
//! of its weights in RAM.

use crate::config::ModelConfig;
use crate::WeightSource;
use mil_spec::{bind, bind_many, Block, DType, Feature, TensorType, ValueType, NVT};

/// Weight quantization for the emitted graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    /// Raw fp16 weights.
    Fp16,
    /// Per-channel int8 with fp16 scale — `constexpr_blockwise_shift_scale`.
    Int8,
}

/// Graph-generation options.
#[derive(Clone, Debug)]
pub struct Options {
    /// Sequence length per call (1 for decode-step, >1 for chunked prefill).
    pub seq: i64,
    /// KV cache capacity in tokens.
    pub max_kv: i64,
    /// Weight quantization.
    pub quant: Quant,
    /// Emit the lm_head conv (vocab logits output).
    pub lm_head: bool,
    /// Emit the embedding gather at graph head (input becomes token ids).
    /// `false` → input is hidden states `(1, d, S, 1)`.
    pub embed: bool,
    /// `ModelMeta` spec/opset — `(10, "CoreML9")` for stateful graphs.
    pub spec_version: i32,
    /// `ModelMeta` opset name.
    pub opset: String,
    /// Optional LoRA adapter (dir with `adapter_config.json` +
    /// `adapters.safetensors`/`*.npz`, or a bare file). Fused in f32 as
    /// `W + scale·(B @ A)` before fp16/int8 emission.
    pub lora: Option<std::path::PathBuf>,
    /// Per-tensor precision plan (`--plan-file`) — overrides `quant`
    /// and `quant_policy` when set.
    pub plan: Option<crate::plan::Plan>,
    /// Planner policy applied at convert time when `plan` is unset
    /// (`--quant-policy`). `None` → `quant` uniformly — the historical
    /// behaviour.
    pub quant_policy: Option<crate::plan::QuantPolicy>,
    /// Enumerated sequence lengths (`--seq-lens a,b,c`). `len > 1`
    /// emits a flexible-shape program: sequence-dependent op output
    /// types become anonymous symbolic dims, every const that baked the
    /// sequence length is replaced by a runtime `seq` int32 input, and
    /// the inputs/outputs get `EnumeratedShapes` with `seq_lens[0]` as
    /// the default (the emitted type declarations use it as the shape
    /// value). Empty/`len == 1` → the fixed `seq` graph.
    pub seq_lens: Vec<i64>,
    /// Weight-const names the caller wants marked updatable
    /// (`--updatable l5_wq,l5_wk,...`).
    ///
    /// **Honest limitation:** CoreML's real updatable-model machinery
    /// (`NeuralNetwork.updatable`, the on-device update spec) exists
    /// only on the neural-network proto — `mlProgram` has no
    /// per-op/`isUpdatable` field for coremltools' `set_updatable` to
    /// map onto. So this emits a documented `mil.updatable` entry in
    /// `description.metadata.userDefined` naming the weights, nothing
    /// more. Empty → no marker.
    pub updatable: Vec<String>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            seq: 1,
            max_kv: 2048,
            quant: Quant::Int8,
            lm_head: true,
            embed: false,
            spec_version: 10,
            opset: "CoreML9".into(),
            lora: None,
            plan: None,
            quant_policy: None,
            seq_lens: Vec::new(),
            updatable: Vec::new(),
        }
    }
}

/// Everything `encode_model` needs once the graph is built.
pub struct Built {
    /// The graph.
    pub block: Block,
    /// Model inputs.
    pub inputs: Vec<Feature>,
    /// Model outputs.
    pub outputs: Vec<Feature>,
    /// Model states.
    pub states: Vec<Feature>,
    /// Function inputs (same declarations as `inputs` + `states`).
    pub fn_inputs: Vec<NVT>,
    /// `EnumeratedShapes` per feature name — `Some` only for
    /// flexible-seq builds.
    pub enum_shapes: Option<std::collections::BTreeMap<String, mil_spec::EnumeratedShapes>>,
    /// Symbolic dims per tensor name (fn inputs + op outputs) —
    /// `Some("s")` marks a seq-bound dim; empty unless flexible.
    pub syms: std::collections::BTreeMap<String, Vec<Option<String>>>,
}

fn tt(dt: DType, shape: &[i64]) -> ValueType {
    ValueType::Tensor(TensorType {
        dtype: dt,
        shape: shape.to_vec(),
    })
}
fn f16(shape: &[i64]) -> ValueType {
    tt(DType::Fp16, shape)
}

/// Streams weights into `weight.bin` and hands the builder blob refs.
pub struct WeightEmitter<'a> {
    /// File-backed writer.
    pub w: &'a mut mil_spec::BlobWriter,
    /// quantization mode
    pub quant: Quant,
    /// blob file name inside the package
    pub file: String,
    /// weight store — safetensors shards or a GGUF file
    pub src: &'a dyn WeightSource,
    /// Per-tensor precision map — `plan::Plan::uniform(quant)` is
    /// byte-identical to the `quant` field's old behaviour.
    pub plan: &'a crate::plan::Plan,
}

impl<'a> WeightEmitter<'a> {
    /// Emit a named HF tensor as a conv-packed `(out, in, 1, 1)` weight.
    /// Returns the bound const name for use as `conv1x1` weight.
    pub fn conv_weight(
        &mut self,
        b: &mut Block,
        hf_name: &str,
        name: &str,
    ) -> std::io::Result<String> {
        use crate::plan::Precision;
        match self.plan.precision(hf_name) {
            Precision::Native => {
                // Source already stores an exactly-transcodable block
                // format — emit the codes/scales as-is (lossless), or
                // fall back to fp16 when the source can't serve them.
                if let Some(nq) = self.src.native_qblocks(hf_name)? {
                    let shape = self.src.shape(hf_name)?;
                    if shape.len() != 2 {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("{hf_name}: expected 2D weight, got {shape:?}"),
                        ));
                    }
                    let (out_f, in_f) = (shape[0], shape[1]);
                    let shape4 = [out_f, in_f, 1, 1];
                    let d_off = self.w.append(nq.data_dtype, &nq.data)?;
                    let s_off = self.w.append(DType::Fp16, &nq.scales)?;
                    return Ok(match nq.offset {
                        None => {
                            b.konst_q8_blocks(name, &self.file, d_off, s_off, &shape4, nq.block)
                        }
                        Some((off_bytes, off_dt)) => {
                            let o_off = self.w.append(off_dt, &off_bytes)?;
                            b.konst_q4_blocks(
                                name, &self.file, d_off, s_off, o_off, &shape4, nq.block,
                            )
                        }
                    });
                }
                // plan asked for native but the source can't provide it
                // — fp16 is the honest fallback, not a silent requant
                let (shape, bytes) = self.src.tensor_f16(hf_name)?;
                if shape.len() != 2 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("{hf_name}: expected 2D weight, got {shape:?}"),
                    ));
                }
                let off = self.w.append(DType::Fp16, &bytes)?;
                Ok(b.konst_blob(
                    name,
                    &self.file,
                    off,
                    DType::Fp16,
                    &[shape[0], shape[1], 1, 1],
                ))
            }
            Precision::Fp16 => {
                let (shape, bytes) = self.src.tensor_f16(hf_name)?;
                if shape.len() != 2 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("{hf_name}: expected 2D weight, got {shape:?}"),
                    ));
                }
                let (out_f, in_f) = (shape[0], shape[1]);
                let off = self.w.append(DType::Fp16, &bytes)?;
                Ok(b.konst_blob(name, &self.file, off, DType::Fp16, &[out_f, in_f, 1, 1]))
            }
            Precision::Int8 => {
                let (shape, bytes) = self.src.tensor_f16(hf_name)?;
                if shape.len() != 2 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("{hf_name}: expected 2D weight, got {shape:?}"),
                    ));
                }
                let (out_f, in_f) = (shape[0], shape[1]);
                let (q, scales) = quantize_int8(&bytes, out_f, in_f);
                let q_off = self.w.append(DType::Int8, &q)?;
                let s_off = self.w.append(DType::Fp16, &scales)?;
                Ok(b.konst_q8(name, &self.file, q_off, s_off, &[out_f, in_f, 1, 1]))
            }
        }
    }

    /// Emit an optional conv bias `(cout,)` — `Ok(None)` when the
    /// checkpoint carries no bias tensor for this projection.
    pub fn bias_weight(
        &mut self,
        b: &mut Block,
        hf_name: &str,
        name: &str,
        cout: i64,
    ) -> std::io::Result<Option<String>> {
        if !self.src.has(hf_name) {
            return Ok(None);
        }
        let (_, bytes) = self.src.tensor_f16(hf_name)?;
        if bytes.len() != (cout * 2) as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{hf_name}: expected {cout} bias elems"),
            ));
        }
        let off = self.w.append(DType::Fp16, &bytes)?;
        Ok(Some(b.konst_blob(
            name,
            &self.file,
            off,
            DType::Fp16,
            &[cout],
        )))
    }

    /// Emit a 1D norm weight as fp16 blob `(1, d, 1, 1)` for channel-dim
    /// broadcast, or `(1,1,1,hd)` for last-dim broadcast (head norms).
    pub fn norm_weight(
        &mut self,
        b: &mut Block,
        hf_name: &str,
        name: &str,
        shape: &[i64],
    ) -> std::io::Result<String> {
        let (_, bytes) = self.src.tensor_f16(hf_name)?;
        let off = self.w.append(DType::Fp16, &bytes)?;
        Ok(b.konst_blob(name, &self.file, off, DType::Fp16, shape))
    }
}

/// Per-row int8 quantization: scale = absmax/127, q = round(w/scale).
/// Input is `(out, in)` fp16 row-major. Returns (int8 bytes, fp16 scales).
pub fn quantize_int8(w: &[u8], out_f: i64, in_f: i64) -> (Vec<u8>, Vec<u8>) {
    let n = (out_f * in_f) as usize;
    let row = in_f as usize;
    let mut q = Vec::with_capacity(n);
    let mut scales = Vec::with_capacity((out_f * 2) as usize);
    for r in 0..out_f as usize {
        let row_bytes = &w[r * row * 2..(r + 1) * row * 2];
        let mut amax = 0f32;
        for c in row_bytes.chunks_exact(2) {
            let v = half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs();
            if v > amax {
                amax = v;
            }
        }
        let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        scales.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        let inv = 1.0 / scale;
        for c in row_bytes.chunks_exact(2) {
            let v = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
            let qq = (v * inv).round().clamp(-127.0, 127.0) as i8;
            q.push(qq as u8);
        }
    }
    (q, scales)
}

/// Runtime-computed int32 shape tensor: `concat(parts, axis=0)` where
/// each part is a fresh `[1]` const or a bound runtime input.
/// Flexible-seq builds use these instead of baked `[…, s, …]` shape
/// consts — a const value cannot vary with the enumerated shape.
enum ShapePart<'a> {
    /// Literal `[v]` const.
    V(i32),
    /// Bound tensor name (the runtime `seq` input).
    N(&'a str),
}

fn dyn_shape(b: &mut Block, name: &str, parts: &[ShapePart]) -> String {
    let mut names = Vec::with_capacity(parts.len());
    for (i, p) in parts.iter().enumerate() {
        names.push(match p {
            ShapePart::V(v) => b.konst_i32(&format!("{name}_p{i}"), &[*v]),
            ShapePart::N(n) => n.to_string(),
        });
    }
    let ax = b.fresh("ax");
    let ax = b.konst_scalar_i32(&ax, 0);
    let il = b.fresh("il");
    let il = b.konst_bool(&il, false);
    b.o1(
        "concat",
        vec![
            (
                "values".into(),
                bind_many(&names.iter().map(String::as_str).collect::<Vec<_>>()),
            ),
            ("axis".into(), bind(&ax).1),
            ("interleave".into(), bind(&il).1),
        ],
        name,
        tt(DType::Int32, &[parts.len() as i64]),
    )
}

/// `reshape` whose target shape is a runtime tensor, not a const.
fn reshape_dyn(b: &mut Block, x: &str, shape: &str, out_shape: &[i64], name: &str) -> String {
    b.o1(
        "reshape",
        vec![("x".into(), bind(x).1), ("shape".into(), bind(shape).1)],
        name,
        f16(out_shape),
    )
}

/// `slice_by_index` with an open end on `open` dims — used by the
/// flexible build where a baked `end = s` would pin the sequence
/// length. End values at open dims are `i32::MAX` (clipped to the dim
/// even if the mask is ignored) plus a `true` end_mask bit.
fn slice_open_end(
    b: &mut Block,
    x: &str,
    begin: &[i32],
    end: &[i32],
    open: &[usize],
    out_shape: &[i64],
    name: &str,
) -> String {
    let n = begin.len();
    let b0 = b.fresh("begin");
    let b0 = b.konst_i32(&b0, begin);
    let e0 = b.fresh("end");
    let e0 = b.konst_i32(&e0, end);
    let st = b.fresh("stride");
    let st = b.konst_i32(&st, &vec![1; n]);
    let bm = b.fresh("bmask");
    let bm = b.op(
        "const",
        vec![],
        vec![(&bm, tt(DType::Bool, &[n as i64]))],
        vec![("val".into(), mil_spec::Value::bools(&vec![false; n]))],
    )[0]
    .clone();
    let mut emv = vec![false; n];
    for &d in open {
        emv[d] = true;
    }
    let em = b.fresh("emask");
    let em = b.op(
        "const",
        vec![],
        vec![(&em, tt(DType::Bool, &[n as i64]))],
        vec![("val".into(), mil_spec::Value::bools(&emv))],
    )[0]
    .clone();
    let sm = b.fresh("smask");
    let sm = b.op(
        "const",
        vec![],
        vec![(&sm, tt(DType::Bool, &[n as i64]))],
        vec![("val".into(), mil_spec::Value::bools(&vec![false; n]))],
    )[0]
    .clone();
    b.o1(
        "slice_by_index",
        vec![
            ("x".into(), bind(x).1),
            ("begin".into(), bind(&b0).1),
            ("end".into(), bind(&e0).1),
            ("stride".into(), bind(&st).1),
            ("begin_mask".into(), bind(&bm).1),
            ("end_mask".into(), bind(&em).1),
            ("squeeze_mask".into(), bind(&sm).1),
        ],
        name,
        f16(out_shape),
    )
}

/// Names of the runtime shape machinery emitted once at graph head in
/// flexible-seq builds.
struct FlexShapes {
    /// int32[1] runtime sequence length input.
    seq: String,
    /// `[1, seq, qh, hd]` reshape target for `seq_to_heads`.
    r_q: String,
    /// `[1, seq, kvh, hd]` reshape target for `seq_to_heads`.
    r_kv: String,
    /// `[1, kvh, g*seq, hd]` — the GQA fold.
    qg: String,
    /// `[1, qh, seq, hd]` — attention out un-fold.
    ah: String,
    /// `[1, seq, qh*hd, 1]` — back to conv channels.
    ac: String,
}

/// A generalized RMSNorm: `x / sqrt(mean(x^2, axes) + eps) * w` over an
/// arbitrary axis — fp16-safe via a dynamic max-abs prescale. With
/// `m = max(|x|)` (floored) and `xs = x/m` every element sits in
/// `[-1, 1]`, so `xs²` can never overflow, and when `xs²` underflows the
/// `eps/m²` term necessarily dominates the variance anyway, so
/// flush-to-zero can't change the result:
///
/// `out = xs·rsqrt(mean(xs²) + eps/m²)·w = x·rsqrt(mean(x²)+eps)·w`
///
/// `eps/m²` is computed as `(√eps/m)²` — eps ~1e-6 is an fp16 denormal,
/// but `√eps/m` stays in normal range for every m that matters (when it
/// does flush, `mean(xs²)` ≈ 1 dominates and eps is negligible anyway).
/// Division is used rather than reciprocal-multiply because `1/m` itself
/// is a fp16 denormal for `m > 16384`. An all-zero row yields exactly 0,
/// matching HF.
#[allow(clippy::too_many_arguments)]
fn rms_norm_axis(
    b: &mut Block,
    x: &str,
    w: &str,
    _d: i64,
    eps: f32,
    axes: &[i32],
    shape: &[i64],
    pfx: &str,
) -> String {
    // 2^-8: 1/floor = 256 fits fp16, and eps/floor² ≈ 0.065 (eps=1e-6)
    // stays a normal fp16 value so an all-zero row still gets r finite.
    const M_FLOOR: f32 = 3.90625e-3;
    let absx = b.o1(
        "abs",
        vec![("x".into(), bind(x).1)],
        &format!("{pfx}_abs"),
        f16(shape),
    );
    let ax = b.fresh("axes");
    let ax = b.konst_i32(&ax, axes);
    let kd = b.fresh("kd");
    let kd = b.konst_bool(&kd, true);
    let mut mshape = shape.to_vec();
    for &a in axes {
        mshape[a as usize] = 1;
    }
    let m = b.o1(
        "reduce_max",
        vec![
            ("x".into(), bind(&absx).1),
            ("axes".into(), bind(&ax).1),
            ("keep_dims".into(), bind(&kd).1),
        ],
        &format!("{pfx}_m"),
        f16(&mshape),
    );
    let fl = b.konst_f16(&format!("{pfx}_fl"), M_FLOOR);
    let mc = b.o1(
        "maximum",
        vec![("x".into(), bind(&m).1), ("y".into(), bind(&fl).1)],
        &format!("{pfx}_mc"),
        f16(&mshape),
    );
    let xs = b.o1(
        "real_div",
        vec![("x".into(), bind(x).1), ("y".into(), bind(&mc).1)],
        &format!("{pfx}_xs"),
        f16(shape),
    );
    let sq = b.mul(&xs, &xs, shape, &format!("{pfx}_sq"));
    let ax = b.fresh("axes");
    let ax = b.konst_i32(&ax, axes);
    let kd = b.fresh("kd");
    let kd = b.konst_bool(&kd, true);
    let mut mshape = shape.to_vec();
    for &a in axes {
        mshape[a as usize] = 1;
    }
    let mean = b.o1(
        "reduce_mean",
        vec![
            ("x".into(), bind(&sq).1),
            ("axes".into(), bind(&ax).1),
            ("keep_dims".into(), bind(&kd).1),
        ],
        &format!("{pfx}_mean"),
        f16(&mshape),
    );
    // eps/mc² as (√eps/mc)² — eps alone is an fp16 denormal.
    let esq = b.konst_f16(&format!("{pfx}_esq"), eps.sqrt());
    let t = b.o1(
        "real_div",
        vec![("x".into(), bind(&esq).1), ("y".into(), bind(&mc).1)],
        &format!("{pfx}_t"),
        f16(&mshape),
    );
    let e2 = b.mul(&t, &t, &mshape, &format!("{pfx}_e2"));
    let var = b.add(&mean, &e2, &mshape, &format!("{pfx}_var"));
    let eps_c = b.konst_f16(&format!("{pfx}_rse"), 0.0);
    let r = b.o1(
        "rsqrt",
        vec![
            ("x".into(), bind(&var).1),
            ("epsilon".into(), bind(&eps_c).1),
        ],
        &format!("{pfx}_rsq"),
        f16(&mshape),
    );
    let xn = b.mul(&xs, &r, shape, &format!("{pfx}_xn"));
    b.mul(&xn, w, shape, &format!("{pfx}_out"))
}

/// `(1, h*hd, S, 1)` conv layout → `(1, h, S, hd)` per-head layout:
/// transpose the sequence to the front, split the channel into
/// (head, dim), then move heads back — element (a,p,c) = channel
/// a*hd+c of position p.
fn seq_to_heads(
    b: &mut Block,
    x: &str,
    h: i64,
    s: i64,
    hd: i64,
    pfx: &str,
    rshape: Option<&str>,
) -> String {
    let t1 = b.transpose(x, &[0, 2, 1, 3], &[1, s, h * hd, 1], &format!("{pfx}_t1"));
    let r = match rshape {
        Some(sh) => reshape_dyn(b, &t1, sh, &[1, s, h, hd], &format!("{pfx}_r")),
        None => b.reshape(&t1, &[1, s, h, hd], &format!("{pfx}_r")),
    };
    b.transpose(&r, &[0, 2, 1, 3], &[1, h, s, hd], &format!("{pfx}_t2"))
}

/// `sigmoid` (raw op, not a helper).
fn sigmoid(b: &mut Block, x: &str, shape: &[i64], name: &str) -> String {
    b.o1("sigmoid", vec![("x".into(), bind(x).1)], name, f16(shape))
}

/// Elementwise negation — MIL has no `neg` op; `mul(x, -1)` is the idiom.
fn neg(b: &mut Block, x: &str, shape: &[i64], name: &str) -> String {
    let m1 = b.konst_f16(&format!("{name}_m1"), -1.0);
    b.mul(x, &m1, shape, name)
}

/// RoPE on `(1, h, S, hd)`: `x*cos + rotate_half(x)*sin`.
/// `rotate_half([x1|x2]) = [-x2|x1]` (non-interleaved, Llama-style).
#[allow(clippy::too_many_arguments)]
fn rope(
    b: &mut Block,
    x: &str,
    cos: &str,
    sin: &str,
    h: i64,
    s: i64,
    hd: i64,
    pfx: &str,
    flex: bool,
) -> String {
    let shape = &[1, h, s, hd];
    let half = (hd / 2) as i32;
    // slice halves of the last dim — the sequence dim's `end` can't be
    // a const in a flexible build, so it stays open (masked + i32::MAX).
    let (x1, x2) = if flex {
        (
            slice_open_end(
                b,
                x,
                &[0, 0, 0, 0],
                &[1, h as i32, i32::MAX, half],
                &[2],
                &[1, h, s, hd / 2],
                &format!("{pfx}_x1"),
            ),
            slice_open_end(
                b,
                x,
                &[0, 0, 0, half],
                &[1, h as i32, i32::MAX, hd as i32],
                &[2],
                &[1, h, s, hd / 2],
                &format!("{pfx}_x2"),
            ),
        )
    } else {
        (
            b.slice(
                x,
                &[0, 0, 0, 0],
                &[1, h as i32, s as i32, half],
                &[1, h, s, hd / 2],
                &format!("{pfx}_x1"),
            ),
            b.slice(
                x,
                &[0, 0, 0, half],
                &[1, h as i32, s as i32, hd as i32],
                &[1, h, s, hd / 2],
                &format!("{pfx}_x2"),
            ),
        )
    };
    let nx2 = neg(b, &x2, &[1, h, s, hd / 2], &format!("{pfx}_nx2"));
    let rot = b.concat(&[nx2, x1], 3, &[1, h, s, hd], &format!("{pfx}_rot"));
    let c = b.mul(x, cos, shape, &format!("{pfx}_c"));
    let sn = b.mul(&rot, sin, shape, &format!("{pfx}_sn"));
    b.add(&c, &sn, shape, &format!("{pfx}_rope"))
}

/// Build the packed-KV slice_update for one row of the state.
/// `row`/`row_end` select the state's row range; `pos` is the runtime
/// position input name; `update` is `(1, kvh, S, dh)`.
#[allow(clippy::too_many_arguments)]
fn slice_update(
    b: &mut Block,
    state_val: &str,
    update: &str,
    row: i32,
    row_end: i32,
    pos: &str,
    seq_in: Option<&str>,
    kvh: i64,
    s: i64,
    dh: i64,
    kv_shape: &[i64],
    pfx: &str,
) -> String {
    // begin = concat([row, 0, pos, 0]) ; end = concat([row_end, kvh, pos+S, dh])
    let b0 = b.fresh("b");
    let b0 = b.konst_i32(&b0, &[row]);
    let b1 = b.fresh("b");
    let b1 = b.konst_i32(&b1, &[0]);
    let b3 = b.fresh("b");
    let b3 = b.konst_i32(&b3, &[0]);
    let ax = b.fresh("ax");
    let ax = b.konst_scalar_i32(&ax, 0);
    let il = b.fresh("il");
    let il = b.konst_bool(&il, false);
    let beg = b.o1(
        "concat",
        vec![
            ("values".into(), bind_many(&[&b0, &b1, pos, &b3])),
            ("axis".into(), bind(&ax).1),
            ("interleave".into(), bind(&il).1),
        ],
        &format!("{pfx}_beg"),
        tt(DType::Int32, &[4]),
    );
    let p2 = match seq_in {
        // flexible build — `end[2] = pos + seq` with seq at runtime
        Some(seq) => b.o1(
            "add",
            vec![("x".into(), bind(pos).1), ("y".into(), bind(seq).1)],
            &format!("{pfx}_p2"),
            tt(DType::Int32, &[1]),
        ),
        None => {
            let sc = b.fresh("sc");
            let sc = b.konst_i32(&sc, &[s as i32]);
            b.o1(
                "add",
                vec![("x".into(), bind(pos).1), ("y".into(), bind(&sc).1)],
                &format!("{pfx}_p2"),
                tt(DType::Int32, &[1]),
            )
        }
    };
    let e0 = b.fresh("e");
    let e0 = b.konst_i32(&e0, &[row_end]);
    let e1 = b.fresh("e");
    let e1 = b.konst_i32(&e1, &[kvh as i32]);
    let e3 = b.fresh("e");
    let e3 = b.konst_i32(&e3, &[dh as i32]);
    let axe = b.fresh("ax");
    let axe = b.konst_scalar_i32(&axe, 0);
    let ile = b.fresh("il");
    let ile = b.konst_bool(&ile, false);
    let end = b.o1(
        "concat",
        vec![
            ("values".into(), bind_many(&[&e0, &e1, &p2, &e3])),
            ("axis".into(), bind(&axe).1),
            ("interleave".into(), bind(&ile).1),
        ],
        &format!("{pfx}_end"),
        tt(DType::Int32, &[4]),
    );
    let st = b.fresh("st");
    let st = b.konst_i32(&st, &[1, 1, 1, 1]);
    // all-false masks, rank 4
    let mk = |b: &mut Block, n: &str| -> String {
        b.op(
            "const",
            vec![],
            vec![(n, tt(DType::Bool, &[4]))],
            vec![(
                "val".into(),
                mil_spec::Value::bools(&[false, false, false, false]),
            )],
        )[0]
        .clone()
    };
    let bm = mk(b, &format!("{pfx}_bm"));
    let em = mk(b, &format!("{pfx}_em"));
    let sm = mk(b, &format!("{pfx}_sm"));
    b.o1(
        "slice_update",
        vec![
            ("x".into(), bind(state_val).1),
            ("update".into(), bind(update).1),
            ("begin".into(), bind(&beg).1),
            ("end".into(), bind(&end).1),
            ("stride".into(), bind(&st).1),
            ("begin_mask".into(), bind(&bm).1),
            ("end_mask".into(), bind(&em).1),
            ("squeeze_mask".into(), bind(&sm).1),
        ],
        &format!("{pfx}_su"),
        f16(kv_shape),
    )
}

/// Build the complete decoder block. Streams all weights through `em`.
///
/// Graph I/O:
/// - inputs: `x` `(1, d, S, 1)` hidden states, `cos`/`sin` `(1,1,S,hd)`
///   rope tables, `mask` `(1,1,S,max_kv)` additive fp16, `pos` `int32[1]`
/// - state: `kv` `(layers*2, kvh, max_kv, dh)` packed
/// - output: `logits` `(1, vocab, S, 1)` or `h` `(1, d, S, 1)` if no lm_head
pub fn build(cfg: &ModelConfig, opts: &Options, em: &mut WeightEmitter) -> std::io::Result<Built> {
    let mut b = Block::new();
    let d = cfg.hidden_size;
    // Flexible-seq builds declare their types at seq_lens[0] (the
    // default enumerated shape) and take the sequence length at
    // runtime through a `seq` int32 input — every const that used to
    // bake `s` is replaced by shape machinery the compiler can
    // re-parameterize.
    let flex = opts.seq_lens.len() > 1;
    let s = if flex { opts.seq_lens[0] } else { opts.seq };
    let hd = cfg.head_dim;
    let qh = cfg.num_heads;
    let kvh = cfg.num_kv_heads;
    let g = qh / kvh;
    let inter = cfg.intermediate_size;
    let max_kv = opts.max_kv;
    let kv_shape = vec![cfg.num_layers as i64 * 2, kvh, max_kv, hd];
    let act = vec![1, d, s, 1]; // conv layout: (1, C, S, 1)

    // ---- input features ----
    let inputs = vec![
        Feature {
            name: "x".into(),
            shape: act.clone(),
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "cos".into(),
            shape: vec![1, 1, s, hd],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "sin".into(),
            shape: vec![1, 1, s, hd],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "mask".into(),
            shape: vec![1, 1, s, max_kv],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "pos".into(),
            shape: vec![1],
            dtype: DType::Int32,
            is_state: false,
        },
    ];
    let mut inputs = inputs;
    if flex {
        inputs.push(Feature {
            name: "seq".into(),
            shape: vec![1],
            dtype: DType::Int32,
            is_state: false,
        });
    }
    let states = vec![Feature {
        name: "kv".into(),
        shape: kv_shape.clone(),
        dtype: DType::Fp16,
        is_state: true,
    }];

    // The state value flows through every layer's slice_updates —
    // a serial chain kv_0 → kv_1 → ... → kv_L written back once.
    let mut kv = b.read_state("kv", &kv_shape, "kv_0");
    let mut x = "x".to_string();

    // Flexible seq: the reshape/slice targets that bake `s` as a const
    // value in the fixed build become concat-computed tensors driven by
    // the `seq` input (enumerated-shape programs only flex op *types* —
    // const *values* are shared across shapes).
    let fx: Option<FlexShapes> = if flex {
        let gk = b.konst_i32("seq_g", &[g as i32]);
        let gseq = b.o1(
            "mul",
            vec![("x".into(), bind("seq").1), ("y".into(), bind(&gk).1)],
            "seq_gs",
            tt(DType::Int32, &[1]),
        );
        Some(FlexShapes {
            seq: "seq".into(),
            r_q: dyn_shape(
                &mut b,
                "seq_sh_rq",
                &[
                    ShapePart::V(1),
                    ShapePart::N("seq"),
                    ShapePart::V(qh as i32),
                    ShapePart::V(hd as i32),
                ],
            ),
            r_kv: dyn_shape(
                &mut b,
                "seq_sh_rkv",
                &[
                    ShapePart::V(1),
                    ShapePart::N("seq"),
                    ShapePart::V(kvh as i32),
                    ShapePart::V(hd as i32),
                ],
            ),
            qg: dyn_shape(
                &mut b,
                "seq_sh_qg",
                &[
                    ShapePart::V(1),
                    ShapePart::V(kvh as i32),
                    ShapePart::N(&gseq),
                    ShapePart::V(hd as i32),
                ],
            ),
            ah: dyn_shape(
                &mut b,
                "seq_sh_ah",
                &[
                    ShapePart::V(1),
                    ShapePart::V(qh as i32),
                    ShapePart::N("seq"),
                    ShapePart::V(hd as i32),
                ],
            ),
            ac: dyn_shape(
                &mut b,
                "seq_sh_ac",
                &[
                    ShapePart::V(1),
                    ShapePart::N("seq"),
                    ShapePart::V((qh * hd) as i32),
                    ShapePart::V(1),
                ],
            ),
        })
    } else {
        None
    };

    for l in 0..cfg.num_layers {
        let pfx = format!("l{l}");
        // ---- attention input norm ----
        let nw = em.norm_weight(
            &mut b,
            &format!("model.layers.{l}.input_layernorm.weight"),
            &format!("{pfx}_inw"),
            &[1, d, 1, 1],
        )?;
        let h = rms_norm_axis(&mut b, &x, &nw, d, cfg.rms_norm_eps, &[1], &act, &pfx);

        // ---- q/k/v projections (conv-packed; qwen2 carries biases) ----
        let wq = em.conv_weight(
            &mut b,
            &format!("model.layers.{l}.self_attn.q_proj.weight"),
            &format!("{pfx}_wq"),
        )?;
        let wk = em.conv_weight(
            &mut b,
            &format!("model.layers.{l}.self_attn.k_proj.weight"),
            &format!("{pfx}_wk"),
        )?;
        let wv = em.conv_weight(
            &mut b,
            &format!("model.layers.{l}.self_attn.v_proj.weight"),
            &format!("{pfx}_wv"),
        )?;
        // Optional biases — emitted when the checkpoint has them
        // (qwen2 q/k/v; llama/qwen3 have none). Shape (cout,) per the
        // conv op's bias signature.
        let bq = em.bias_weight(
            &mut b,
            &format!("model.layers.{l}.self_attn.q_proj.bias"),
            &format!("{pfx}_bq"),
            qh * hd,
        )?;
        let bk = em.bias_weight(
            &mut b,
            &format!("model.layers.{l}.self_attn.k_proj.bias"),
            &format!("{pfx}_bk"),
            kvh * hd,
        )?;
        let bv = em.bias_weight(
            &mut b,
            &format!("model.layers.{l}.self_attn.v_proj.bias"),
            &format!("{pfx}_bv"),
            kvh * hd,
        )?;
        let q = b.conv1x1_s(&h, &wq, bq.as_deref(), qh * hd, s, &format!("{pfx}_q"));
        let k = b.conv1x1_s(&h, &wk, bk.as_deref(), kvh * hd, s, &format!("{pfx}_k"));
        let v = b.conv1x1_s(&h, &wv, bv.as_deref(), kvh * hd, s, &format!("{pfx}_v"));

        // ---- to per-head layout (1, heads, S, hd) ----
        // conv out is (1, h*hd, S, 1); a bare reshape to (1,h,S,hd)
        // would interleave position into the channel index (only
        // correct at S==1). Transpose→reshape→transpose instead so
        // element (a,p,c) really is channel a*hd+c of position p.
        let q4 = seq_to_heads(
            &mut b,
            &q,
            qh,
            s,
            hd,
            &format!("{pfx}_q4"),
            fx.as_ref().map(|f| f.r_q.as_str()),
        );
        let k4 = seq_to_heads(
            &mut b,
            &k,
            kvh,
            s,
            hd,
            &format!("{pfx}_k4"),
            fx.as_ref().map(|f| f.r_kv.as_str()),
        );
        let v4 = seq_to_heads(
            &mut b,
            &v,
            kvh,
            s,
            hd,
            &format!("{pfx}_v4"),
            fx.as_ref().map(|f| f.r_kv.as_str()),
        );

        // ---- optional per-head q/k rms norms (Qwen3) ----
        let (q_n, k_n) = if cfg.qk_norm {
            let qw = em.norm_weight(
                &mut b,
                &format!("model.layers.{l}.self_attn.q_norm.weight"),
                &format!("{pfx}_qnw"),
                &[1, 1, 1, hd],
            )?;
            let kw = em.norm_weight(
                &mut b,
                &format!("model.layers.{l}.self_attn.k_norm.weight"),
                &format!("{pfx}_knw"),
                &[1, 1, 1, hd],
            )?;
            let qn = rms_norm_axis(
                &mut b,
                &q4,
                &qw,
                hd,
                cfg.rms_norm_eps,
                &[3],
                &[1, qh, s, hd],
                &format!("{pfx}_qn"),
            );
            let kn = rms_norm_axis(
                &mut b,
                &k4,
                &kw,
                hd,
                cfg.rms_norm_eps,
                &[3],
                &[1, kvh, s, hd],
                &format!("{pfx}_kn"),
            );
            (qn, kn)
        } else {
            (q4.clone(), k4.clone())
        };

        // ---- rope ----
        let qr = rope(
            &mut b,
            &q_n,
            "cos",
            "sin",
            qh,
            s,
            hd,
            &format!("{pfx}_rq"),
            flex,
        );
        let kr = rope(
            &mut b,
            &k_n,
            "cos",
            "sin",
            kvh,
            s,
            hd,
            &format!("{pfx}_rk"),
            flex,
        );

        // ---- packed-KV slice_updates: row 2l = K, row 2l+1 = V ----
        // updates land as (1, kvh, S, dh) — same layout the state stores.
        kv = slice_update(
            &mut b,
            &kv,
            &kr,
            (l * 2) as i32,
            (l * 2 + 1) as i32,
            "pos",
            fx.as_ref().map(|f| f.seq.as_str()),
            kvh,
            s,
            hd,
            &kv_shape,
            &format!("{pfx}_ku"),
        );
        kv = slice_update(
            &mut b,
            &kv,
            &v4,
            (l * 2 + 1) as i32,
            (l * 2 + 2) as i32,
            "pos",
            fx.as_ref().map(|f| f.seq.as_str()),
            kvh,
            s,
            hd,
            &kv_shape,
            &format!("{pfx}_vu"),
        );

        // ---- read this layer's K/V rows post-write ----
        let k_full = b.slice(
            &kv,
            &[(l * 2) as i32, 0, 0, 0],
            &[(l * 2 + 1) as i32, kvh as i32, max_kv as i32, hd as i32],
            &[1, kvh, max_kv, hd],
            &format!("{pfx}_kfull"),
        );
        let v_full = b.slice(
            &kv,
            &[(l * 2 + 1) as i32, 0, 0, 0],
            &[(l * 2 + 2) as i32, kvh as i32, max_kv as i32, hd as i32],
            &[1, kvh, max_kv, hd],
            &format!("{pfx}_vfull"),
        );

        // ---- attention: q expanded per kv-head (GQA) ----
        // (1,qh,S,hd) -> (1,kvh,g*S,hd)
        let qg = match &fx {
            Some(f) => reshape_dyn(
                &mut b,
                &qr,
                &f.qg,
                &[1, kvh, g * s, hd],
                &format!("{pfx}_qg"),
            ),
            None => b.reshape(&qr, &[1, kvh, g * s, hd], &format!("{pfx}_qg")),
        };
        let scores = {
            // matmul(qg, k_full^T): (1,kvh,g*S,hd) @ (1,kvh,hd,max_kv) -> (1,kvh,g*S,max_kv)
            let sc = b.matmul(
                &qg,
                &k_full,
                true,
                &[1, kvh, g * s, max_kv],
                &format!("{pfx}_sc"),
            );
            let inv = b.konst_f16(&format!("{pfx}_scale"), 1.0 / (hd as f32).sqrt());
            let scaled = b.mul(&sc, &inv, &[1, kvh, g * s, max_kv], &format!("{pfx}_scl"));
            // The GQA fold packs rows as (g_idx, p): mask row for score row j
            // is mask[j % s], i.e. the (1,1,s,mk) mask tiled g times.
            // In a flexible build the runtime seq can exceed the default —
            // gate on `flex` so the tile exists for every enumerated s > 1.
            let mask = if g > 1 && (s > 1 || flex) {
                let reps = b.konst_i32(&format!("{pfx}_mrep"), &[1, 1, g as i32, 1]);
                b.o1(
                    "tile",
                    vec![("x".into(), bind("mask").1), ("reps".into(), bind(&reps).1)],
                    &format!("{pfx}_maskt"),
                    f16(&[1, 1, g * s, max_kv]),
                )
            } else {
                "mask".to_string()
            };
            b.add(
                &scaled,
                &mask,
                &[1, kvh, g * s, max_kv],
                &format!("{pfx}_msk"),
            )
        };
        let probs = b.softmax(
            &scores,
            3,
            &[1, kvh, g * s, max_kv],
            &format!("{pfx}_p"),
            false,
        );
        // out = probs @ v: (1,kvh,g*S,max_kv) @ (1,kvh,max_kv,hd) -> (1,kvh,g*S,hd)
        let attn = b.matmul(
            &probs,
            &v_full,
            false,
            &[1, kvh, g * s, hd],
            &format!("{pfx}_att"),
        );
        // back to conv layout: (1,kvh,g*S,hd) -> (1,qh,S,hd) -> (1,qh*hd,S,1)
        let ah = match &fx {
            Some(f) => reshape_dyn(&mut b, &attn, &f.ah, &[1, qh, s, hd], &format!("{pfx}_ah")),
            None => b.reshape(&attn, &[1, qh, s, hd], &format!("{pfx}_ah")),
        };
        let at = b.transpose(&ah, &[0, 2, 1, 3], &[1, s, qh, hd], &format!("{pfx}_at"));
        let ac = match &fx {
            Some(f) => reshape_dyn(
                &mut b,
                &at,
                &f.ac,
                &[1, s, qh * hd, 1],
                &format!("{pfx}_ac"),
            ),
            None => b.reshape(&at, &[1, s, qh * hd, 1], &format!("{pfx}_ac")),
        };
        let ac4 = b.transpose(
            &ac,
            &[0, 2, 1, 3],
            &[1, qh * hd, s, 1],
            &format!("{pfx}_ac4"),
        );

        let wo = em.conv_weight(
            &mut b,
            &format!("model.layers.{l}.self_attn.o_proj.weight"),
            &format!("{pfx}_wo"),
        )?;
        let bo = em.bias_weight(
            &mut b,
            &format!("model.layers.{l}.self_attn.o_proj.bias"),
            &format!("{pfx}_bo"),
            d,
        )?;
        let o = b.conv1x1_s(&ac4, &wo, bo.as_deref(), d, s, &format!("{pfx}_o"));
        x = b.add(&x, &o, &act, &format!("{pfx}_ra"));

        // ---- MLP ----
        let mw = em.norm_weight(
            &mut b,
            &format!("model.layers.{l}.post_attention_layernorm.weight"),
            &format!("{pfx}_pw"),
            &[1, d, 1, 1],
        )?;
        let h2 = rms_norm_axis(
            &mut b,
            &x,
            &mw,
            d,
            cfg.rms_norm_eps,
            &[1],
            &act,
            &format!("{pfx}_m"),
        );
        let wg = em.conv_weight(
            &mut b,
            &format!("model.layers.{l}.mlp.gate_proj.weight"),
            &format!("{pfx}_wg"),
        )?;
        let wu = em.conv_weight(
            &mut b,
            &format!("model.layers.{l}.mlp.up_proj.weight"),
            &format!("{pfx}_wu"),
        )?;
        let wd = em.conv_weight(
            &mut b,
            &format!("model.layers.{l}.mlp.down_proj.weight"),
            &format!("{pfx}_wd"),
        )?;
        let gate = b.conv1x1_s(&h2, &wg, None, inter, s, &format!("{pfx}_gate"));
        let up = b.conv1x1_s(&h2, &wu, None, inter, s, &format!("{pfx}_up"));
        let sg = sigmoid(&mut b, &gate, &[1, inter, s, 1], &format!("{pfx}_sg"));
        let silu = b.mul(&gate, &sg, &[1, inter, s, 1], &format!("{pfx}_silu"));
        let act_mlp = b.mul(&silu, &up, &[1, inter, s, 1], &format!("{pfx}_swiglu"));
        let down = b.conv1x1_s(&act_mlp, &wd, None, d, s, &format!("{pfx}_down"));
        x = b.add(&x, &down, &act, &format!("{pfx}_rm"));
    }

    // write the accumulated state once
    b.write_state("kv", &kv);

    // ---- final norm + head ----
    let fw = em.norm_weight(&mut b, "model.norm.weight", "final_w", &[1, d, 1, 1])?;
    let xf = rms_norm_axis(&mut b, &x, &fw, d, cfg.rms_norm_eps, &[1], &act, "final");

    let (out_name, out_shape) = if opts.lm_head {
        // tied models reuse the embedding table as the head
        let head_w = if cfg.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        let wl = em.conv_weight(&mut b, head_w, "lm_w")?;
        let logits = b.conv1x1_s(&xf, &wl, None, cfg.vocab_size, s, "logits");
        (logits, vec![1, cfg.vocab_size, s, 1])
    } else {
        (xf, act.clone())
    };
    b.outputs = vec![out_name.clone()];

    let outputs = vec![Feature {
        name: out_name.clone(),
        shape: out_shape,
        dtype: DType::Fp16,
        is_state: false,
    }];
    let mut fn_inputs: Vec<NVT> = inputs
        .iter()
        .map(|f| NVT {
            name: f.name.clone(),
            ty: f16(&f.shape),
        })
        .collect();
    // pos/seq are int32 — fix the dtype the f16() helper set wrong
    for n in fn_inputs.iter_mut() {
        if n.name == "pos" || n.name == "seq" {
            n.ty = tt(DType::Int32, &[1]);
        }
    }
    fn_inputs.push(NVT {
        name: "kv".into(),
        ty: ValueType::State(TensorType::f16(&kv_shape)),
    });

    // ---- flexible-shape metadata ----
    // Op outputs whose shape *actually* tracks the runtime sequence
    // length need a symbolic (`unknown`) dim — a baked constant dim
    // clamps the buffer and CoreML silently truncates the result (seen
    // empirically). Marks are propagated input→output through the op
    // graph, NOT inferred by matching declared dims against `s`: with
    // s=1 a coincidental static dim (e.g. `kvh` on the state slices
    // `k_full`/`v_full`) would otherwise get marked, and a symbolic
    // batch dim on a `matmul` operand makes the execution-plan builder
    // fail at model load.
    let mut syms = std::collections::BTreeMap::new();
    let enum_shapes = if flex {
        for name in ["x", "cos", "sin", "mask"] {
            syms.insert(name.to_string(), vec![None, None, Some("s".into()), None]);
        }
        // Producer lookup: value name -> op that emits it.
        let mut producer: std::collections::BTreeMap<&str, &mil_spec::Op> =
            std::collections::BTreeMap::new();
        for op in &b.ops {
            for nvt in &op.outputs {
                producer.insert(nvt.name.as_str(), op);
            }
        }
        // Bound tensor names for an op input argument.
        let bound = |op: &mil_spec::Op, arg: &str| -> Vec<String> {
            op.inputs
                .iter()
                .find(|(a, _)| a == arg)
                .map(|(_, mil_spec::Argument(bs))| {
                    bs.iter()
                        .filter_map(|bnd| match bnd {
                            mil_spec::Binding::Name(n) => Some(n.clone()),
                            _ => None,
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        // Immediate payload of the const op producing `name`.
        let const_imm = |name: &str| -> Option<&mil_spec::Immediate> {
            let op = *producer.get(name)?;
            if op.ty != "const" {
                return None;
            }
            match op.attrs.first() {
                Some((_, mil_spec::Value::Imm(_, imm))) => Some(imm),
                _ => None,
            }
        };
        let ints_of = |name: &str| -> Option<Vec<i32>> {
            match const_imm(name) {
                Some(mil_spec::Immediate::Ints(v)) => Some(v.clone()),
                _ => None,
            }
        };
        let bool_of = |name: &str| -> Option<bool> {
            match const_imm(name) {
                Some(mil_spec::Immediate::Bools(v)) => v.first().copied(),
                _ => None,
            }
        };
        // Runtime-valued int32 names that carry the sequence length —
        // used as parts inside the `concat` dyn-shape tensors.
        let seq_parts = ["seq", "seq_gs"];
        let marks_of =
            |syms: &std::collections::BTreeMap<String, Vec<Option<String>>>,
             name: &str|
             -> Vec<Option<String>> { syms.get(name).cloned().unwrap_or_default() };
        for op in &b.ops {
            // const producers and state-chain ops must keep concrete
            // output types — `read_state`/`slice_update` validate
            // against the state feature's fixed shape (and a const's
            // type is pinned by its value anyway).
            if matches!(
                op.ty.as_str(),
                "const"
                    | "constexpr_blockwise_shift_scale"
                    | "constexpr_lut_to_dense"
                    | "constexpr_sparse_to_dense"
                    | "read_state"
                    | "write_state"
                    | "slice_update"
            ) {
                continue;
            }
            for nvt in &op.outputs {
                let ValueType::Tensor(t) = &nvt.ty else {
                    continue;
                };
                if t.dtype != DType::Fp16 || t.shape.len() != 4 {
                    continue;
                }
                let x_names = bound(op, "x");
                let xm = x_names
                    .first()
                    .map(|n| marks_of(&syms, n))
                    .unwrap_or_default();
                let mut marks = vec![None; 4];
                match op.ty.as_str() {
                    // 1x1 conv: output seq position follows the input's.
                    "conv" => {
                        if xm.len() == 4 {
                            marks[2] = xm[2].clone();
                        }
                    }
                    "transpose" => {
                        if xm.len() == 4 {
                            if let Some(perm) = bound(op, "perm").first().and_then(|n| ints_of(n)) {
                                for (j, &p) in perm.iter().enumerate() {
                                    marks[j] = xm[p as usize].clone();
                                }
                            }
                        }
                    }
                    // Runtime-shape reshape: the shape tensor is a
                    // `concat` of `[1]` parts; the parts bound to
                    // `seq`/`seq_gs` mark the corresponding output dims.
                    "reshape" => {
                        if let Some(sh) = bound(op, "shape").first() {
                            if producer.get(sh.as_str()).is_some_and(|p| p.ty == "concat") {
                                let sh_op = *producer.get(sh.as_str()).unwrap();
                                for (i, part) in bound(sh_op, "values").iter().enumerate() {
                                    if seq_parts.contains(&part.as_str()) && i < 4 {
                                        marks[i] = Some("s".into());
                                    }
                                }
                            }
                        }
                    }
                    // Slices and tiles preserve per-dim marks.
                    "slice_by_index" | "tile" => {
                        if xm.len() == 4 {
                            marks = xm.clone();
                        }
                    }
                    "concat" => {
                        for name in bound(op, "values") {
                            let im = marks_of(&syms, &name);
                            if im.len() == 4 {
                                for i in 0..4 {
                                    if im[i].is_some() {
                                        marks[i] = im[i].clone();
                                    }
                                }
                            }
                        }
                    }
                    // Rank-4 batched matmul: batch dims are the union of
                    // both operands'; the M dim comes from x's row dim
                    // (dim3 when transposed) and N from y's col dim
                    // (dim2 when transposed).
                    "matmul" => {
                        let ym = bound(op, "y")
                            .first()
                            .map(|n| marks_of(&syms, n))
                            .unwrap_or_default();
                        let tx = bound(op, "transpose_x")
                            .first()
                            .and_then(|n| bool_of(n))
                            .unwrap_or(false);
                        let ty = bound(op, "transpose_y")
                            .first()
                            .and_then(|n| bool_of(n))
                            .unwrap_or(false);
                        if xm.len() == 4 {
                            marks[0] = xm[0].clone();
                            marks[1] = xm[1].clone();
                            marks[2] = xm[if tx { 3 } else { 2 }].clone();
                        }
                        if ym.len() == 4 {
                            for i in 0..2 {
                                if marks[i].is_none() {
                                    marks[i] = ym[i].clone();
                                }
                            }
                            marks[3] = ym[if ty { 2 } else { 3 }].clone();
                        }
                    }
                    // Reduced dims collapse to 1 — their marks die.
                    "reduce_max" | "reduce_mean" | "reduce_sum" | "reduce_min" | "reduce_prod" => {
                        if xm.len() == 4 {
                            marks = xm.clone();
                            if let Some(axes) = bound(op, "axes").first().and_then(|n| ints_of(n)) {
                                for a in axes {
                                    let i = if a < 0 { a + 4 } else { a } as usize;
                                    if i < 4 {
                                        marks[i] = None;
                                    }
                                }
                            }
                        }
                    }
                    // softmax preserves all dims.
                    "softmax" => {
                        if xm.len() == 4 {
                            marks = xm.clone();
                        }
                    }
                    // Elementwise / broadcasting default: union the
                    // marks of every same-rank tensor input.
                    _ => {
                        for (_, arg) in &op.inputs {
                            for bnd in &arg.0 {
                                if let mil_spec::Binding::Name(n) = bnd {
                                    let im = marks_of(&syms, n);
                                    if im.len() == 4 {
                                        for i in 0..4 {
                                            if im[i].is_some() {
                                                marks[i] = im[i].clone();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if marks.iter().any(Option::is_some) {
                    syms.insert(nvt.name.clone(), marks);
                }
            }
        }
        let mut m = std::collections::BTreeMap::new();
        m.insert(
            "x".to_string(),
            mil_spec::EnumeratedShapes {
                shapes: opts.seq_lens.iter().map(|&v| vec![1, d, v, 1]).collect(),
            },
        );
        m.insert(
            "mask".to_string(),
            mil_spec::EnumeratedShapes {
                shapes: opts
                    .seq_lens
                    .iter()
                    .map(|&v| vec![1, 1, v, max_kv])
                    .collect(),
            },
        );
        for name in ["cos", "sin"] {
            m.insert(
                name.to_string(),
                mil_spec::EnumeratedShapes {
                    shapes: opts.seq_lens.iter().map(|&v| vec![1, 1, v, hd]).collect(),
                },
            );
        }
        m.insert(
            out_name.clone(),
            mil_spec::EnumeratedShapes {
                shapes: opts
                    .seq_lens
                    .iter()
                    .map(|&v| {
                        if opts.lm_head {
                            vec![1, cfg.vocab_size, v, 1]
                        } else {
                            vec![1, d, v, 1]
                        }
                    })
                    .collect(),
            },
        );
        Some(m)
    } else {
        None
    };

    Ok(Built {
        block: b,
        inputs,
        outputs,
        states,
        fn_inputs,
        enum_shapes,
        syms,
    })
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use mil_spec::{ModelMeta, Value};

    /// `x (1,d,s,1)` → `rms_norm_axis(x, w)` with const `w (1,d,1,1)` —
    /// the exact op chain the builder emits inside a layer.
    fn norm_pkg(d: i64, s: i64, wvals: &[f32], eps: f32) -> Block {
        let mut b = Block::new();
        let w = b.op(
            "const",
            vec![],
            vec![("w", ValueType::Tensor(TensorType::f16(&[1, d, 1, 1])))],
            vec![("val".into(), Value::f16s(&[1, d, 1, 1], wvals))],
        )[0]
        .clone();
        let out = rms_norm_axis(&mut b, "x", &w, d, eps, &[1], &[1, d, s, 1], "n");
        b.outputs = vec![out];
        b
    }

    fn compile_and_run(b: &Block, x: &[f32], d: i64, s: i64) -> Vec<f32> {
        if std::process::Command::new("xcrun")
            .args(["-f", "coremlc"])
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            panic!("coremlc not available — cannot run the compiled norm probe");
        }
        let dir = std::env::temp_dir().join(format!("rmsnorm_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let inputs = [Feature {
            name: "x".into(),
            shape: vec![1, d, s, 1],
            dtype: DType::Fp16,
            is_state: false,
        }];
        let outputs = [Feature {
            name: b.outputs[0].clone(),
            shape: vec![1, d, s, 1],
            dtype: DType::Fp16,
            is_state: false,
        }];
        let fin = [NVT {
            name: "x".into(),
            ty: ValueType::Tensor(TensorType::f16(&[1, d, s, 1])),
        }];
        let spec = mil_spec::encode_model(
            &inputs,
            &outputs,
            &[],
            b,
            &fin,
            &ModelMeta::new(10, "CoreML9"),
        );
        let pkg = dir.join("n.mlpackage");
        mil_spec::write_mlpackage(&pkg, &spec, None).unwrap();
        let comp = mil_compile::compile(&pkg, &dir.join("n.compiled")).unwrap();
        let data: Vec<u8> = x
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let m = mil_infer::Model::load(&comp.path, mil_infer::ComputeUnits::All).unwrap();
        let pr = m
            .predict(&[mil_infer::Input {
                name: "x",
                shape: &[1, d, s, 1],
                data: &data,
                dtype: DType::Fp16,
            }])
            .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        pr.outputs[0].values()
    }

    /// Op-sequence proof: the emitted norm must prescale by a dynamic
    /// max-abs (abs → reduce_max → maximum → real_div), not by a fixed
    /// const. A fixed `x * 1/sqrt(d)` prescale (the old impl) emits only
    /// mul/reduce_mean/add/rsqrt/mul/mul and fails this check outright.
    #[test]
    fn rms_norm_emits_maxabs_prescale() {
        let b = norm_pkg(8, 3, &[1.0; 8], 1e-6);
        let ops: Vec<&str> = b.ops.iter().map(|o| o.ty.as_str()).collect();
        for need in [
            "abs",
            "reduce_max",
            "maximum",
            "real_div",
            "reduce_mean",
            "rsqrt",
        ] {
            assert!(ops.contains(&need), "norm chain missing {need}: {ops:?}");
        }
        // fp16 only — an fp32 norm falls off the ANE.
        assert!(
            !ops.iter().any(|o| *o == "cast" || *o == "reduce_sum"),
            "norm chain must stay fp16 (no cast): {ops:?}"
        );
    }

    /// Numeric proof on the fp16 execution path (ANE under
    /// `ComputeUnits::All`). Three rows through one call:
    /// - p0: large activations (±2000) — `(x/√d)²` overflows fp16 in
    ///   the old impl → rsqrt(inf) = 0 → all-zero output. Correct
    ///   output is O(1).
    /// - p1: all-zero row — must produce exactly 0 (HF semantics).
    /// - p2: embedding-scale row (~0.01 RMS, the Qwen3 case) —
    ///   `(x/√d)² ~ 1e-5` underflows fp16 subnormals in the old impl.
    #[test]
    fn rms_norm_fp16_edge_rows() {
        let d: i64 = 8;
        let s: i64 = 3;
        let eps = 1e-6f32;
        let wvals = [1.0f32, 0.5, 2.0, 1.5, 1.0, 0.25, 1.0, 0.75];
        let rows = [
            // p0: large — (x/√8)² hits ~5e5, way past fp16 max 65504.
            [
                2000.0, -2000.0, 2000.0, -2000.0, 1000.0, -1000.0, 500.0, -500.0,
            ],
            // p1: all-zero row → exactly 0.
            [0.0; 8],
            // p2: embedding-scale — mean(x²) ≈ 7.6e-5.
            [0.012, -0.012, 0.008, -0.008, 0.004, -0.004, 0.002, -0.002],
        ];
        let b = norm_pkg(d, s, &wvals, eps);
        // x (1, d, s, 1): element (c, p) at index c*s + p.
        let mut x = vec![0f32; (d * s) as usize];
        for (p, row) in rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                x[c * s as usize + p] = v;
            }
        }
        let out = compile_and_run(&b, &x, d, s);
        assert_eq!(out.len(), (d * s) as usize);
        for (p, row) in rows.iter().enumerate() {
            let ms: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / d as f64;
            let inv = 1.0f64 / (ms + eps as f64).sqrt();
            for (c, &v) in row.iter().enumerate() {
                let want = (v as f64 * inv * wvals[c] as f64) as f32;
                let got = out[c * s as usize + p];
                if p == 1 {
                    assert_eq!(got, 0.0, "all-zero row must produce 0 (c={c})");
                } else {
                    assert!(
                        got.is_finite(),
                        "pos {p} c {c}: non-finite output {got} (want {want})"
                    );
                    assert!(
                        (got - want).abs() <= 0.02 + 0.03 * want.abs(),
                        "pos {p} c {c}: got {got}, want {want} (fp16 tol)"
                    );
                }
            }
        }
    }
}
