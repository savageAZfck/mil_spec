//! Independent pure-Rust transformer forward — the correctness anchor
//! for converter tests.
//!
//! Written from HF modeling semantics (`modeling_qwen2.py`,
//! `modeling_qwen3.py`, `modeling_llama.py`), **not** from
//! `mil_convert::builder`: same math, independent implementation, so a
//! bug in one can't silently agree with the other. All dot products
//! accumulate in `f64`; storage and per-element math are `f32`.

use mil_convert::config::ModelConfig;
use mil_convert::WeightSource;

type Res<T> = std::io::Result<T>;

fn err(msg: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

/// f16 LE bytes → f32 (matches `half::f16::from_le_bytes`).
fn f16_bits(b: u16) -> f32 {
    let s = (b >> 15) as u32;
    let e = ((b >> 10) & 0x1f) as i32;
    let m = (b & 0x3ff) as f32;
    let sign = if s == 1 { -1.0 } else { 1.0 };
    match e {
        0 => sign * m * f32::from_bits(0x3380_0000), // 2^-24
        31 => {
            if m == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1024.0 + m) * 2f32.powi(e - 25),
    }
}

/// Fetch a tensor as f32 via `tensor_f16`.
fn w_f32(w: &dyn WeightSource, name: &str) -> Res<(Vec<i64>, Vec<f32>)> {
    let (shape, bytes) = w.tensor_f16(name)?;
    let vals = bytes
        .chunks_exact(2)
        .map(|c| f16_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    Ok((shape, vals))
}

/// Optional tensor (biases) — `None` when absent from the source.
fn w_f32_opt(w: &dyn WeightSource, name: &str) -> Res<Option<Vec<f32>>> {
    if !w.has(name) {
        return Ok(None);
    }
    Ok(Some(w_f32(w, name)?.1))
}

/// `x @ w^T` (+bias): `w` is `(out, in)` row-major; f64 accumulation,
/// or fp16 MACs when `macs_f16` (models the package's conv kernels).
fn matmul(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    n_out: usize,
    n_in: usize,
    macs_f16: bool,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len() / n_in * n_out);
    for xr in x.chunks_exact(n_in) {
        for r in 0..n_out {
            let wrow = &w[r * n_in..(r + 1) * n_in];
            let acc = if macs_f16 {
                let mut a = 0f32;
                for c in 0..n_in {
                    a = f16_rt(a + f16_rt(xr[c] * wrow[c]));
                }
                a as f64
            } else {
                let mut acc = 0f64;
                for c in 0..n_in {
                    acc += xr[c] as f64 * wrow[c] as f64;
                }
                acc
            };
            let mut v = acc as f32;
            if let Some(b) = bias {
                v += b[r];
                if macs_f16 {
                    v = f16_rt(v);
                }
            }
            out.push(v);
        }
    }
    out
}

/// HF `RMSNorm`: `x / sqrt(mean(x^2) + eps) * weight` per row.
fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(d) {
        let ms = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / d as f64;
        let scale = 1.0 / (ms + eps as f64).sqrt();
        for (c, &v) in row.iter().enumerate() {
            out.push((v as f64 * scale) as f32 * w[c]);
        }
    }
    out
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// f32 -> fp16 -> f32 round-trip.
fn f16_rt(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

/// Full HF forward for the dense decoder family (qwen2/qwen3/llama).
///
/// `ids` are token ids; returns `logits[position][vocab]` for every
/// position (causal mask applied, so position `p` predicts token `p+1`).
pub fn forward(cfg: &ModelConfig, w: &dyn WeightSource, ids: &[u32]) -> Res<Vec<Vec<f32>>> {
    forward_impl(cfg, w, ids, Fp16Attn::Off, None)
}

/// Probe variant of [`forward`]: additionally returns the max
/// `|q.k/sqrt(hd)|` attention score per layer over query positions
/// 0 and 1 (all heads, all allowed keys).
#[doc(hidden)]
pub fn forward_probe_qk(
    cfg: &ModelConfig,
    w: &dyn WeightSource,
    ids: &[u32],
) -> Res<(Vec<Vec<f32>>, Vec<f64>)> {
    let mut mq = Vec::new();
    let logits = forward_impl(cfg, w, ids, Fp16Attn::Off, Some(&mut mq))?;
    Ok((logits, mq))
}

/// Attention-precision variant of [`forward`]: q, k and the attention
/// scores are rounded through fp16, approximating the converted
/// package's fp16 attention path. Used for precision hypothesis
/// testing, not as a correctness anchor.
#[doc(hidden)]
pub fn forward_attn_fp16(
    cfg: &ModelConfig,
    w: &dyn WeightSource,
    ids: &[u32],
) -> Res<Vec<Vec<f32>>> {
    forward_impl(cfg, w, ids, Fp16Attn::Storage, None)
}

/// Stronger attention-precision variant: everything [`forward_attn_fp16`]
/// rounds, plus the q.k dot, the softmax probabilities and the
/// probability-weighted V accumulation are computed in fp16 — the
/// worst case for the package's attention arithmetic.
#[doc(hidden)]
pub fn forward_attn_fp16_mac(
    cfg: &ModelConfig,
    w: &dyn WeightSource,
    ids: &[u32],
) -> Res<Vec<Vec<f32>>> {
    forward_impl(cfg, w, ids, Fp16Attn::Mac, None)
}

/// Whole-graph fp16 variant: [`Fp16Attn::Mac`] plus every activation
/// tensor rounded to fp16 storage between ops (embeddings, normed
/// inputs, projections, attention output, residuals, MLP, final
/// logits). The most pessimistic model of the package's fp16 data
/// path.
#[doc(hidden)]
pub fn forward_fp16_all(
    cfg: &ModelConfig,
    w: &dyn WeightSource,
    ids: &[u32],
) -> Res<Vec<Vec<f32>>> {
    forward_impl(cfg, w, ids, Fp16Attn::All, None)
}

/// How much of the forward is computed in fp16.
#[derive(Clone, Copy, PartialEq)]
enum Fp16Attn {
    /// No rounding.
    Off,
    /// q, k and scores rounded to fp16; f64 math otherwise.
    Storage,
    /// Storage rounding plus fp16 q.k/PV accumulation and fp16
    /// softmax probabilities.
    Mac,
    /// `Mac` plus fp16 storage for every intermediate activation.
    All,
}

impl Fp16Attn {
    fn stores_f16(self) -> bool {
        self != Fp16Attn::Off
    }
    fn macs_f16(self) -> bool {
        matches!(self, Fp16Attn::Mac | Fp16Attn::All)
    }
    /// fp16 accumulation in the projection/MLP/head matmuls.
    fn conv_macs_f16(self) -> bool {
        self == Fp16Attn::All
    }
    fn activations_f16(self) -> bool {
        self == Fp16Attn::All
    }
}

fn forward_impl(
    cfg: &ModelConfig,
    w: &dyn WeightSource,
    ids: &[u32],
    fp16_attn: Fp16Attn,
    max_qk: Option<&mut Vec<f64>>,
) -> Res<Vec<Vec<f32>>> {
    let mut max_qk = max_qk;
    let d = cfg.hidden_size as usize;
    let nh = cfg.num_heads as usize;
    let kvh = cfg.num_kv_heads as usize;
    let hd = cfg.head_dim as usize;
    let inter = cfg.intermediate_size as usize;
    let s = ids.len();
    let g = nh / kvh;
    if let Some(mq) = max_qk.as_mut() {
        mq.resize(cfg.num_layers, 0.0);
    }

    // ---- embed ----
    let (eshape, emb) = w_f32(w, "model.embed_tokens.weight")?;
    if eshape.len() != 2 || eshape[1] as usize != d {
        return Err(err(format!("embed shape {eshape:?}")));
    }
    let mut x: Vec<f32> = Vec::with_capacity(s * d);
    for &id in ids {
        let r = &emb[id as usize * d..(id as usize + 1) * d];
        x.extend_from_slice(r);
    }

    // ---- rope tables (HF rotate-half): cos[p][c] = cos(p*inv[c % hd/2])
    let mut cos = vec![0f32; s * hd];
    let mut sin = vec![0f32; s * hd];
    for p in 0..s {
        for c in 0..hd {
            let inv = (cfg.rope_theta as f64).powf(-2.0 * (c % (hd / 2)) as f64 / hd as f64);
            let ang = p as f64 * inv;
            cos[p * hd + c] = ang.cos() as f32;
            sin[p * hd + c] = ang.sin() as f32;
        }
    }

    for l in 0..cfg.num_layers {
        let p = format!("model.layers.{l}");
        // ---- input norm ----
        let nw = w_f32(w, &format!("{p}.input_layernorm.weight"))?.1;
        let mut h = rmsnorm(&x, &nw, cfg.rms_norm_eps);
        if fp16_attn.activations_f16() {
            for v in h.iter_mut() {
                *v = f16_rt(*v);
            }
        }

        // ---- q/k/v ----
        let wq = w_f32(w, &format!("{p}.self_attn.q_proj.weight"))?.1;
        let wk = w_f32(w, &format!("{p}.self_attn.k_proj.weight"))?.1;
        let wv = w_f32(w, &format!("{p}.self_attn.v_proj.weight"))?.1;
        let bq = w_f32_opt(w, &format!("{p}.self_attn.q_proj.bias"))?;
        let bk = w_f32_opt(w, &format!("{p}.self_attn.k_proj.bias"))?;
        let bv = w_f32_opt(w, &format!("{p}.self_attn.v_proj.bias"))?;
        let mut q = matmul(
            &h,
            &wq,
            bq.as_deref(),
            nh * hd,
            d,
            fp16_attn.conv_macs_f16(),
        );
        let mut k = matmul(
            &h,
            &wk,
            bk.as_deref(),
            kvh * hd,
            d,
            fp16_attn.conv_macs_f16(),
        );
        let mut v = matmul(
            &h,
            &wv,
            bv.as_deref(),
            kvh * hd,
            d,
            fp16_attn.conv_macs_f16(),
        );

        // ---- qwen3 per-head norms (RMS over hd, no mean-subtract) ----
        if cfg.qk_norm {
            let qw = w_f32(w, &format!("{p}.self_attn.q_norm.weight"))?.1;
            let kw = w_f32(w, &format!("{p}.self_attn.k_norm.weight"))?.1;
            q = rmsnorm(&q, &qw, cfg.rms_norm_eps);
            k = rmsnorm(&k, &kw, cfg.rms_norm_eps);
        }

        // ---- rope on (S, heads, hd): rotate-half ----
        // Pair (c, c+hd/2) must be computed from the *original* values —
        // reading the partner in-place after writing c < hd/2 would
        // double-rotate the second half.
        let rope_head = |buf: &mut [f32], heads: usize| {
            for pos in 0..s {
                for hn in 0..heads {
                    let base = pos * heads * hd + hn * hd;
                    for c in 0..hd / 2 {
                        let lo = buf[base + c];
                        let hi = buf[base + c + hd / 2];
                        let (cs, sn) = (cos[pos * hd + c], sin[pos * hd + c]);
                        let (cs2, sn2) = (cos[pos * hd + c + hd / 2], sin[pos * hd + c + hd / 2]);
                        buf[base + c] = lo * cs - hi * sn;
                        buf[base + c + hd / 2] = hi * cs2 + lo * sn2;
                    }
                }
            }
        };
        rope_head(&mut q, nh);
        rope_head(&mut k, kvh);
        if fp16_attn.stores_f16() {
            for v in q.iter_mut().chain(k.iter_mut()) {
                *v = f16_rt(*v);
            }
        }
        if fp16_attn.activations_f16() {
            for v in v.iter_mut() {
                *v = f16_rt(*v);
            }
        }

        // ---- causal GQA attention ----
        let scale = 1.0 / (hd as f64).sqrt();
        let mut attn = vec![0f32; s * nh * hd];
        for pos in 0..s {
            for hn in 0..nh {
                let kh = hn / g;
                let qb = pos * nh * hd + hn * hd;
                // scores over allowed slots j (kv head kh)
                let mut scores = Vec::with_capacity(pos + 1);
                for j in 0..s {
                    if j > pos {
                        continue;
                    }
                    let kb = j * kvh * hd + kh * hd;
                    let mut acc = 0f64;
                    let mut acc16 = 0f32;
                    for c in 0..hd {
                        let pr = q[qb + c] * k[kb + c];
                        acc += pr as f64;
                        if fp16_attn.macs_f16() {
                            acc16 = f16_rt(acc16 + f16_rt(pr));
                        }
                    }
                    let mut sc = if fp16_attn.macs_f16() {
                        acc16 as f64 * scale
                    } else {
                        acc * scale
                    };
                    if fp16_attn.stores_f16() {
                        sc = f16_rt(sc as f32) as f64;
                    }
                    if pos < 2 {
                        if let Some(mq) = max_qk.as_deref_mut() {
                            mq[l] = mq[l].max(sc.abs());
                        }
                    }
                    scores.push((j, sc));
                }
                let mx = scores.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v.1));
                let den: f64 = scores.iter().map(|v| (v.1 - mx).exp()).sum();
                let ob = pos * nh * hd + hn * hd;
                for &(j, sc) in scores.iter() {
                    let mut pr = ((sc - mx).exp() / den) as f32;
                    if fp16_attn.macs_f16() {
                        pr = f16_rt(pr);
                    }
                    let vb = j * kvh * hd + kh * hd;
                    if fp16_attn.macs_f16() {
                        for c in 0..hd {
                            let acc = f16_rt(attn[ob + c] + f16_rt(pr * v[vb + c]));
                            attn[ob + c] = acc;
                        }
                    } else {
                        for c in 0..hd {
                            attn[ob + c] += pr * v[vb + c];
                        }
                    }
                }
            }
        }
        if fp16_attn.activations_f16() {
            for v in attn.iter_mut() {
                *v = f16_rt(*v);
            }
        }

        // ---- o_proj + residual ----
        let wo = w_f32(w, &format!("{p}.self_attn.o_proj.weight"))?.1;
        let bo = w_f32_opt(w, &format!("{p}.self_attn.o_proj.bias"))?;
        let o = matmul(
            &attn,
            &wo,
            bo.as_deref(),
            d,
            nh * hd,
            fp16_attn.conv_macs_f16(),
        );
        for (a, b) in x.iter_mut().zip(&o) {
            *a += b;
            if fp16_attn.activations_f16() {
                *a = f16_rt(*a);
            }
        }

        // ---- MLP ----
        let mw = w_f32(w, &format!("{p}.post_attention_layernorm.weight"))?.1;
        let mut h2 = rmsnorm(&x, &mw, cfg.rms_norm_eps);
        if fp16_attn.activations_f16() {
            for v in h2.iter_mut() {
                *v = f16_rt(*v);
            }
        }
        let wg = w_f32(w, &format!("{p}.mlp.gate_proj.weight"))?.1;
        let wu = w_f32(w, &format!("{p}.mlp.up_proj.weight"))?.1;
        let wd = w_f32(w, &format!("{p}.mlp.down_proj.weight"))?.1;
        let gate = matmul(&h2, &wg, None, inter, d, fp16_attn.conv_macs_f16());
        let up = matmul(&h2, &wu, None, inter, d, fp16_attn.conv_macs_f16());
        let sw: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&gg, &uu)| {
                let v = gg * sigmoid(gg) * uu;
                if fp16_attn.activations_f16() {
                    f16_rt(v)
                } else {
                    v
                }
            })
            .collect();
        let down = matmul(&sw, &wd, None, d, inter, fp16_attn.conv_macs_f16());
        for (a, b) in x.iter_mut().zip(&down) {
            *a += b;
            if fp16_attn.activations_f16() {
                *a = f16_rt(*a);
            }
        }
    }

    // ---- final norm + lm_head ----
    let fw = w_f32(w, "model.norm.weight")?.1;
    let mut xf = rmsnorm(&x, &fw, cfg.rms_norm_eps);
    if fp16_attn.activations_f16() {
        for v in xf.iter_mut() {
            *v = f16_rt(*v);
        }
    }
    let head_w = if cfg.tie_word_embeddings {
        "model.embed_tokens.weight"
    } else {
        "lm_head.weight"
    };
    let wl = w_f32(w, head_w)?.1;
    let vocab = cfg.vocab_size as usize;
    let mut logits = Vec::with_capacity(s);
    for row in xf.chunks_exact(d) {
        let mut lg = matmul(row, &wl, None, vocab, d, fp16_attn.conv_macs_f16());
        if fp16_attn.activations_f16() {
            for v in lg.iter_mut() {
                *v = f16_rt(*v);
            }
        }
        logits.push(lg);
    }
    Ok(logits)
}
