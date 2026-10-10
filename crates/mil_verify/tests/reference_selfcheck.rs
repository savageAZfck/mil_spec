//! Reference-forward self-check — pure numerics, no models on disk
//! and no `coremlc`, so this runs in the normal `cargo test` suite.
//!
//! The reference forward (`mil_verify::reference`) is the anchor every
//! package-vs-golden check ultimately trusts. If it drifts — a wrong
//! rope pairing, a transposed matmul, a missed residual — every golden
//! generated from it is wrong in the same way and the harness can't
//! notice. This test pins it from the *other* side:
//!
//! 1. `closed_form_identity` — a 2-layer tiny model built from raw
//!    tensor bytes (identity projections, unit norms) whose forward
//!    collapses to a closed form the test computes directly. Any
//!    wiring bug in embedding lookup, RMSNorm, attention, SwiGLU, the
//!    residual path, or the lm_head breaks exact agreement.
//! 2. `independent_forward_random` — the same model shape with
//!    pseudo-random weights at seq 3, forwarded a second time by an
//!    implementation written differently (f64 throughout, explicit
//!    per-head softmax, no shared helpers with `reference.rs`). The
//!    reference stores activations in f32 and accumulates in f64, so
//!    equality is approximate — the bound is measured, not guessed.

use mil_convert::config::ModelConfig;
use mil_convert::WeightSource;
use std::collections::HashMap;

/// In-memory `WeightSource` backed by f16 LE byte tensors — the same
/// bytes a safetensors file would hand back.
struct MapSource(HashMap<String, (Vec<i64>, Vec<u8>)>);

impl WeightSource for MapSource {
    fn has(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
    fn tensor_f16(&self, name: &str) -> std::io::Result<(Vec<i64>, Vec<u8>)> {
        self.0.get(name).cloned().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("missing {name}"))
        })
    }
}

fn f16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        b.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    b
}

fn f16_vals(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// Register an f32-valued tensor (stored as f16, like the real path).
fn put(src: &mut MapSource, name: &str, shape: &[i64], vals: &[f32]) {
    let n: i64 = shape.iter().product();
    assert_eq!(n as usize, vals.len(), "{name}");
    src.0
        .insert(name.to_string(), (shape.to_vec(), f16_bytes(vals)));
}

fn eye(n: usize) -> Vec<f32> {
    let mut m = vec![0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// `[I | 0]` — an `(rows, cols)` matrix with a leading identity.
fn eye_pad(rows: usize, cols: usize) -> Vec<f32> {
    let mut m = vec![0f32; rows * cols];
    for i in 0..rows.min(cols) {
        m[i * cols + i] = 1.0;
    }
    m
}

fn tiny_cfg() -> ModelConfig {
    ModelConfig::from_json(
        br#"{
            "model_type": "llama",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 16,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 64,
            "tie_word_embeddings": false
        }"#,
    )
    .unwrap()
}

/// Identity-weight 2-layer llama-family model: unit norm weights,
/// `q/o = I₈`, `k/v = [I₄|0]`, `gate/up = [I₈;0]`, `down = [I₈|0]`.
/// With these weights a single-token forward is closed form:
/// `attn = [h₀..h₄, h₀..h₄]` (one position ⇒ softmax = 1 ⇒ attention
/// returns v; kvh = 1 ⇒ both heads share it) and `mlp(h) = h²·σ(h)`.
/// `(shape, f32 values)` by tensor name — the test's own weight view.
type Tensors = HashMap<String, (Vec<i64>, Vec<f32>)>;

fn identity_weights() -> (MapSource, Tensors) {
    let mut src = MapSource(HashMap::new());
    let mut raw: Tensors = HashMap::new();
    let mut add = |name: &str, shape: &[i64], vals: Vec<f32>| {
        put(&mut src, name, shape, &vals);
        raw.insert(name.to_string(), (shape.to_vec(), vals));
    };
    // Distinct embedding rows so token ids can't be permuted silently.
    let emb: Vec<f32> = (0..32 * 8)
        .map(|i| ((i * 37) % 89) as f32 * 0.011 - 0.45)
        .collect();
    add("model.embed_tokens.weight", &[32, 8], emb);
    for l in 0..2 {
        let p = format!("model.layers.{l}");
        add(&format!("{p}.input_layernorm.weight"), &[8], vec![1.0; 8]);
        add(
            &format!("{p}.post_attention_layernorm.weight"),
            &[8],
            vec![1.0; 8],
        );
        add(&format!("{p}.self_attn.q_proj.weight"), &[8, 8], eye(8));
        add(
            &format!("{p}.self_attn.k_proj.weight"),
            &[4, 8],
            eye_pad(4, 8),
        );
        add(
            &format!("{p}.self_attn.v_proj.weight"),
            &[4, 8],
            eye_pad(4, 8),
        );
        add(&format!("{p}.self_attn.o_proj.weight"), &[8, 8], eye(8));
        // gate/up: rows 0..8 = I, rows 8..16 = 0 → sw = h²σ(h) on d=8.
        let mut g = vec![0f32; 16 * 8];
        g[..8 * 8].copy_from_slice(&eye(8));
        add(&format!("{p}.mlp.gate_proj.weight"), &[16, 8], g.clone());
        add(&format!("{p}.mlp.up_proj.weight"), &[16, 8], g);
        // down: columns 0..8 = I → only sw[0..8] contribute.
        add(
            &format!("{p}.mlp.down_proj.weight"),
            &[8, 16],
            eye_pad(8, 16),
        );
    }
    add("model.norm.weight", &[8], vec![1.0; 8]);
    // lm_head rows point at different dims so argmax can't tie.
    let lm: Vec<f32> = (0..32 * 8)
        .map(|i| {
            let (v, c) = (i / 8, i % 8);
            if c == v % 8 {
                1.0
            } else {
                ((v * 11 + c * 5) % 13) as f32 * 0.01 - 0.06
            }
        })
        .collect();
    add("lm_head.weight", &[32, 8], lm);
    (src, raw)
}

fn rms_f64(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let s = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * s * w).collect()
}

fn sigmoid64(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// Closed-form forward for the identity-weight model, computed in f64
/// straight from the raw f16 tensor bytes (the test decodes them
/// itself — nothing is shared with `reference.rs`).
#[test]
fn closed_form_identity() {
    let cfg = tiny_cfg();
    let (src, _raw) = identity_weights();
    let eps = cfg.rms_norm_eps as f64;
    let ones = vec![1.0f64; 8];

    // Decode the embedding table the way the test owns it: raw bytes.
    let emb = f16_vals(&src.0["model.embed_tokens.weight"].1);
    for &id in &[0u32, 7, 19, 31] {
        let mut x: Vec<f64> = emb[id as usize * 8..id as usize * 8 + 8]
            .iter()
            .map(|&v| v as f64)
            .collect();
        for _layer in 0..2 {
            let h = rms_f64(&x, &ones, eps);
            // attention: q=k=v=h for kv dims; both heads share kv →
            // out = [h0..4, h0..4]; o = I → residual adds it.
            for c in 0..8 {
                x[c] += h[c % 4];
            }
            let h2 = rms_f64(&x, &ones, eps);
            for c in 0..8 {
                x[c] += h2[c] * h2[c] * sigmoid64(h2[c]);
            }
        }
        let xf = rms_f64(&x, &ones, eps);
        let lm = f16_vals(&src.0["lm_head.weight"].1);
        let want: Vec<f64> = (0..32)
            .map(|v| (0..8).map(|c| xf[c] * lm[v * 8 + c] as f64).sum::<f64>())
            .collect();

        let got = mil_verify::reference::forward(&cfg, &src, &[id]).unwrap();
        assert_eq!(got.len(), 1);
        let d = got[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0f64, f64::max);
        // f32 activation storage in the reference vs f64 here: the
        // observed gap is ~1e-6; 1e-4 is a hard wall, not a fit.
        assert!(d < 1e-4, "id {id}: closed-form gap {d}");
        let arg = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        let want_arg = want
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(arg(&got[0]), want_arg, "id {id}: argmax mismatch");
    }
}

/// Deterministic pseudo-random f32 vector (SplitMix-ish LCG), scaled
/// so a 2-layer stack stays in a sane activation range.
fn lcg_fill(n: usize, seed: u64, scale: f32, center: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) & 0xFFFF) as f32 / 65536.0; // [0,1)
            center + (u - 0.5) * 2.0 * scale
        })
        .collect()
}

/// Random-weight model — every projection distinct, so nothing is
/// analytically degenerate. Returns the source and its f32 decode.
fn random_weights() -> (MapSource, HashMap<String, Vec<f32>>) {
    let mut src = MapSource(HashMap::new());
    let mut raw = HashMap::new();
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut add = |name: &str, shape: &[i64], scale: f32, center: f32| {
        seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        let n: i64 = shape.iter().product();
        let vals = lcg_fill(n as usize, seed, scale, center);
        put(&mut src, name, shape, &vals);
        raw.insert(name.to_string(), vals);
    };
    add("model.embed_tokens.weight", &[32, 8], 0.4, 0.0);
    for l in 0..2 {
        let p = format!("model.layers.{l}");
        add(&format!("{p}.input_layernorm.weight"), &[8], 0.15, 1.0);
        add(
            &format!("{p}.post_attention_layernorm.weight"),
            &[8],
            0.15,
            1.0,
        );
        add(&format!("{p}.self_attn.q_proj.weight"), &[8, 8], 0.25, 0.0);
        add(&format!("{p}.self_attn.k_proj.weight"), &[4, 8], 0.25, 0.0);
        add(&format!("{p}.self_attn.v_proj.weight"), &[4, 8], 0.25, 0.0);
        add(&format!("{p}.self_attn.o_proj.weight"), &[8, 8], 0.25, 0.0);
        add(&format!("{p}.mlp.gate_proj.weight"), &[16, 8], 0.25, 0.0);
        add(&format!("{p}.mlp.up_proj.weight"), &[16, 8], 0.25, 0.0);
        add(&format!("{p}.mlp.down_proj.weight"), &[8, 16], 0.25, 0.0);
    }
    add("model.norm.weight", &[8], 0.15, 1.0);
    add("lm_head.weight", &[32, 8], 0.3, 0.0);
    (src, raw)
}

/// A second implementation of the same forward, written differently:
/// f64 end to end, per-position vectors, explicit softmax, and rope
/// applied through explicit pair rotation. Shares no code with
/// `reference.rs` — agreement is evidence, not a tautology.
fn indep_forward(cfg: &ModelConfig, t: &HashMap<String, Vec<f32>>, ids: &[u32]) -> Vec<Vec<f64>> {
    let d = cfg.hidden_size as usize;
    let nh = cfg.num_heads as usize;
    let kvh = cfg.num_kv_heads as usize;
    let hd = cfg.head_dim as usize;
    let inter = cfg.intermediate_size as usize;
    let s = ids.len();
    let eps = cfg.rms_norm_eps as f64;

    let get = |name: &str| t[name].iter().map(|&v| v as f64).collect::<Vec<f64>>();
    // out[r] = Σ_c w[r·in + c]·x[c] for each row of x (x is s·in flat)
    let matmul = |x: &[f64], w: &[f64], rows: usize, cols: usize| -> Vec<f64> {
        x.chunks_exact(cols)
            .flat_map(|xr| {
                (0..rows).map(move |r| (0..cols).map(|c| w[r * cols + c] * xr[c]).sum::<f64>())
            })
            .collect()
    };
    let norm = |x: &[f64], w: &[f64]| -> Vec<f64> {
        x.chunks_exact(d).flat_map(|r| rms_f64(r, w, eps)).collect()
    };

    let emb = get("model.embed_tokens.weight");
    let mut x: Vec<f64> = ids
        .iter()
        .flat_map(|&id| emb[id as usize * d..id as usize * d + d].iter().copied())
        .collect();

    for l in 0..cfg.num_layers {
        let p = format!("model.layers.{l}");
        let h = norm(&x, &get(&format!("{p}.input_layernorm.weight")));
        let mut q = matmul(
            &h,
            &get(&format!("{p}.self_attn.q_proj.weight")),
            nh * hd,
            d,
        );
        let mut k = matmul(
            &h,
            &get(&format!("{p}.self_attn.k_proj.weight")),
            kvh * hd,
            d,
        );
        let v = matmul(
            &h,
            &get(&format!("{p}.self_attn.v_proj.weight")),
            kvh * hd,
            d,
        );

        // rope (rotate-half): pair (c, c+hd/2), angle = pos·θ^{−2(c%hd/2)/hd}
        let rope = |buf: &mut [f64], heads: usize| {
            for pos in 0..s {
                for hn in 0..heads {
                    let base = pos * heads * hd + hn * hd;
                    for c in 0..hd / 2 {
                        let ang =
                            pos as f64 * (cfg.rope_theta as f64).powf(-2.0 * c as f64 / hd as f64);
                        let (cs, sn) = (ang.cos(), ang.sin());
                        let lo = buf[base + c];
                        let hi = buf[base + c + hd / 2];
                        buf[base + c] = lo * cs - hi * sn;
                        buf[base + c + hd / 2] = hi * cs + lo * sn;
                    }
                }
            }
        };
        rope(&mut q, nh);
        rope(&mut k, kvh);

        // causal GQA attention, explicit per-head softmax
        let g = nh / kvh;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut attn = vec![0f64; s * nh * hd];
        for pos in 0..s {
            for hn in 0..nh {
                let kh = hn / g;
                let qb = pos * nh * hd + hn * hd;
                let mut scores: Vec<f64> = (0..=pos)
                    .map(|j| {
                        let kb = j * kvh * hd + kh * hd;
                        (0..hd).map(|c| q[qb + c] * k[kb + c]).sum::<f64>() * scale
                    })
                    .collect();
                let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let den: f64 = scores.iter().map(|v| (v - mx).exp()).sum();
                for v in scores.iter_mut() {
                    *v = (*v - mx).exp() / den;
                }
                let ob = pos * nh * hd + hn * hd;
                for (j, &pr) in scores.iter().enumerate() {
                    let vb = j * kvh * hd + kh * hd;
                    for c in 0..hd {
                        attn[ob + c] += pr * v[vb + c];
                    }
                }
            }
        }

        let o = matmul(
            &attn,
            &get(&format!("{p}.self_attn.o_proj.weight")),
            d,
            nh * hd,
        );
        for c in 0..x.len() {
            x[c] += o[c];
        }

        let h2 = norm(&x, &get(&format!("{p}.post_attention_layernorm.weight")));
        let gate = matmul(&h2, &get(&format!("{p}.mlp.gate_proj.weight")), inter, d);
        let up = matmul(&h2, &get(&format!("{p}.mlp.up_proj.weight")), inter, d);
        let sw: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g * sigmoid64(g) * u)
            .collect();
        let down = matmul(&sw, &get(&format!("{p}.mlp.down_proj.weight")), d, inter);
        for c in 0..x.len() {
            x[c] += down[c];
        }
    }

    let xf = norm(&x, &get("model.norm.weight"));
    let lm = get("lm_head.weight");
    let vocab = cfg.vocab_size as usize;
    xf.chunks_exact(d)
        .map(|row| matmul(row, &lm, vocab, d))
        .collect()
}

#[test]
fn independent_forward_random() {
    let cfg = tiny_cfg();
    let (src, raw) = random_weights();
    let ids = [3u32, 11, 27];
    let got = mil_verify::reference::forward(&cfg, &src, &ids).unwrap();
    let want = indep_forward(&cfg, &raw, &ids);

    assert_eq!(got.len(), want.len());
    let mut worst = 0f64;
    for (pos, (g, w)) in got.iter().zip(&want).enumerate() {
        let d = g
            .iter()
            .zip(w)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0f64, f64::max);
        worst = worst.max(d);
        let arg = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        let want_arg = w
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(arg(g), want_arg, "pos {pos}: argmax mismatch");
        let dot: f64 = g.iter().zip(w).map(|(a, b)| *a as f64 * b).sum();
        let na: f64 = g.iter().map(|v| *v as f64 * *v as f64).sum::<f64>().sqrt();
        let nb: f64 = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        let cos = dot / (na * nb);
        assert!(cos > 0.9999, "pos {pos}: cosine {cos}");
        eprintln!("selfcheck pos {pos}: max|d|={d:.2e} cosine={cos:.8}");
    }
    // The gap is the reference's f32 activation storage vs this f64
    // path: measured ≈ 1e-4..1e-3 on logits of O(10). 0.05 is ~50×
    // the measurement — a wall, not a fit.
    assert!(worst < 0.05, "independent forward gap {worst}");
}
