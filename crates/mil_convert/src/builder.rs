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
use crate::safetensors::{find, Safetensors};
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
    /// open safetensors readers
    pub shards: &'a [Safetensors],
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
        let st = find(self.shards, hf_name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing weight {hf_name}"),
            )
        })?;
        let (shape, bytes) = st.tensor_f16(hf_name)?;
        if shape.len() != 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{hf_name}: expected 2D weight, got {shape:?}"),
            ));
        }
        let (out_f, in_f) = (shape[0], shape[1]);
        match self.quant {
            Quant::Fp16 => {
                let off = self.w.append(DType::Fp16, &bytes)?;
                Ok(b.konst_blob(name, &self.file, off, DType::Fp16, &[out_f, in_f, 1, 1]))
            }
            Quant::Int8 => {
                let (q, scales) = quantize_int8(&bytes, out_f, in_f);
                let q_off = self.w.append(DType::Int8, &q)?;
                let s_off = self.w.append(DType::Fp16, &scales)?;
                Ok(b.konst_q8(name, &self.file, q_off, s_off, &[out_f, in_f, 1, 1]))
            }
        }
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
        let st = find(self.shards, hf_name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing weight {hf_name}"),
            )
        })?;
        let (_, bytes) = st.tensor_f16(hf_name)?;
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

/// A generalized RMSNorm: `x / sqrt(mean(x^2, axes) + eps) * w` over an
/// arbitrary axis, with the helper's prescale trick (scale x by 1/sqrt(d)
/// first so x² stays inside fp16 range on large activations).
fn rms_norm_axis(
    b: &mut Block,
    x: &str,
    w: &str,
    d: i64,
    eps: f32,
    axes: &[i32],
    shape: &[i64],
    pfx: &str,
) -> String {
    let k = (d as f32).sqrt();
    let inv_k = b.konst_f16(&format!("{pfx}_invk"), 1.0 / k);
    let xs = b.mul(x, &inv_k, shape, &format!("{pfx}_xs"));
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
    let eps2 = b.konst_f16(&format!("{pfx}_eps"), eps / (k * k));
    let var = b.add(&mean, &eps2, &mshape, &format!("{pfx}_var"));
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
fn rope(
    b: &mut Block,
    x: &str,
    cos: &str,
    sin: &str,
    h: i64,
    s: i64,
    hd: i64,
    pfx: &str,
) -> String {
    let shape = &[1, h, s, hd];
    let half = (hd / 2) as i32;
    // slice halves of the last dim
    let x1 = b.slice(
        x,
        &[0, 0, 0, 0],
        &[1, h as i32, s as i32, half],
        &[1, h, s, hd / 2],
        &format!("{pfx}_x1"),
    );
    let x2 = b.slice(
        x,
        &[0, 0, 0, half],
        &[1, h as i32, s as i32, hd as i32],
        &[1, h, s, hd / 2],
        &format!("{pfx}_x2"),
    );
    let nx2 = neg(b, &x2, &[1, h, s, hd / 2], &format!("{pfx}_nx2"));
    let rot = b.concat(&[nx2, x1], 3, &[1, h, s, hd], &format!("{pfx}_rot"));
    let c = b.mul(x, cos, shape, &format!("{pfx}_c"));
    let sn = b.mul(&rot, sin, shape, &format!("{pfx}_sn"));
    b.add(&c, &sn, shape, &format!("{pfx}_rope"))
}

/// Build the packed-KV slice_update for one row of the state.
/// `row`/`row_end` select the state's row range; `pos` is the runtime
/// position input name; `update` is `(1, kvh, S, dh)`.
fn slice_update(
    b: &mut Block,
    state_val: &str,
    update: &str,
    row: i32,
    row_end: i32,
    pos: &str,
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
    let sc = b.fresh("sc");
    let sc = b.konst_i32(&sc, &[s as i32]);
    let p2 = b.o1(
        "add",
        vec![("x".into(), bind(pos).1), ("y".into(), bind(&sc).1)],
        &format!("{pfx}_p2"),
        tt(DType::Int32, &[1]),
    );
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
    let s = opts.seq;
    let hd = cfg.head_dim;
    let qh = cfg.num_heads;
    let kvh = cfg.num_kv_heads;
    let g = qh / kvh;
    let inter = cfg.intermediate_size;
    let max_kv = opts.max_kv;
    let kv_shape = vec![cfg.num_layers as i64 * 2, kvh, max_kv, hd];
    let act = vec![1, d, s, 1]; // conv layout: (1, C, S, 1)

    // ---- input features ----
    let mut inputs = vec![
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

        // ---- q/k/v projections (conv-packed) ----
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
        let q = b.conv1x1(&h, &wq, None, qh * hd, &format!("{pfx}_q"));
        let k = b.conv1x1(&h, &wk, None, kvh * hd, &format!("{pfx}_k"));
        let v = b.conv1x1(&h, &wv, None, kvh * hd, &format!("{pfx}_v"));

        // ---- to per-head layout (1, heads, S, hd) ----
        let q4 = b.reshape(&q, &[1, qh, s, hd], &format!("{pfx}_q4"));
        let k4 = b.reshape(&k, &[1, kvh, s, hd], &format!("{pfx}_k4"));
        let v4 = b.reshape(&v, &[1, kvh, s, hd], &format!("{pfx}_v4"));

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
        let qr = rope(&mut b, &q_n, "cos", "sin", qh, s, hd, &format!("{pfx}_rq"));
        let kr = rope(&mut b, &k_n, "cos", "sin", kvh, s, hd, &format!("{pfx}_rk"));

        // ---- packed-KV slice_updates: row 2l = K, row 2l+1 = V ----
        // updates land as (1, kvh, S, dh) — same layout the state stores.
        kv = slice_update(
            &mut b,
            &kv,
            &kr,
            (l * 2) as i32,
            (l * 2 + 1) as i32,
            "pos",
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
        let qg = b.reshape(&qr, &[1, kvh, g * s, hd], &format!("{pfx}_qg"));
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
            b.add(
                &scaled,
                "mask",
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
        let ah = b.reshape(&attn, &[1, qh, s, hd], &format!("{pfx}_ah"));
        let at = b.transpose(&ah, &[0, 2, 1, 3], &[1, s, qh, hd], &format!("{pfx}_at"));
        let ac = b.reshape(&at, &[1, s, qh * hd, 1], &format!("{pfx}_ac"));
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
        let o = b.conv1x1(&ac4, &wo, None, d, &format!("{pfx}_o"));
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
        let gate = b.conv1x1(&h2, &wg, None, inter, &format!("{pfx}_gate"));
        let up = b.conv1x1(&h2, &wu, None, inter, &format!("{pfx}_up"));
        let sg = sigmoid(&mut b, &gate, &[1, inter, s, 1], &format!("{pfx}_sg"));
        let silu = b.mul(&gate, &sg, &[1, inter, s, 1], &format!("{pfx}_silu"));
        let act_mlp = b.mul(&silu, &up, &[1, inter, s, 1], &format!("{pfx}_swiglu"));
        let down = b.conv1x1(&act_mlp, &wd, None, d, &format!("{pfx}_down"));
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
        let logits = b.conv1x1(&xf, &wl, None, cfg.vocab_size, "logits");
        (logits, vec![1, cfg.vocab_size, s, 1])
    } else {
        (xf, act.clone())
    };
    b.outputs = vec![out_name.clone()];

    let outputs = vec![Feature {
        name: out_name,
        shape: out_shape,
        dtype: DType::Fp16,
        is_state: false,
    }];
    inputs.push(Feature {
        name: "".into(),
        shape: vec![],
        dtype: DType::Fp16,
        is_state: false,
    });
    inputs.pop(); // keep inputs clean — no-op placeholder for clarity

    let mut fn_inputs: Vec<NVT> = inputs
        .iter()
        .map(|f| NVT {
            name: f.name.clone(),
            ty: f16(&f.shape),
        })
        .collect();
    // pos is int32 — fix the dtype the f16() helper set wrong
    for n in fn_inputs.iter_mut() {
        if n.name == "pos" {
            n.ty = tt(DType::Int32, &[1]);
        }
    }
    fn_inputs.push(NVT {
        name: "kv".into(),
        ty: ValueType::State(TensorType::f16(&kv_shape)),
    });

    Ok(Built {
        block: b,
        inputs,
        outputs,
        states,
        fn_inputs,
    })
}
