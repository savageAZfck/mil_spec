//! qwen_decode — full-depth Qwen3-0.6B decode on self-emitted HWX
//! binaries. No ANECompiler, no CoreML, no Apple codegen: every
//! on-engine program comes out of `mil_infer::hwx` emitters, real
//! BF16 safetensors weights are converted to fp16 and baked into the
//! Mach-O kernel sections, and stages are chained by IOSurface
//! adoption exactly like `layer_probe`, scaled to 28 layers and
//! Qwen3's grouped-query attention (16 q heads / 8 kv heads, hd=128).
//!
//! Work split:
//!   on-ANE (self-emitted): all matmuls (proj/up/gate/down/o/head),
//!     sum-reduce matvec for RMS, elementwise mul/add, LUT sigmoid,
//!     avw score matvec, causal mask add, channel softmax, attn·V.
//!   host (our runtime): bf16->f16 weight prep + norm-gamma folding,
//!     embed lookup, per-head q/k RMS-norm + RoPE + attn scale on the
//!     128-d vectors we already move, KV row placement, strided<->
//!     contiguous repacks where no skeleton exists, argmax.
//!   A CPU f32 reference forward pass verifies token agreement.
//!
//! Layout notes (same currencies as layer_probe):
//!   - residual stream: strided [1,1024,1,1], element c at c*64.
//!   - matmul_ns input: contiguous fp16 [K]; output strided [1,N,1,1].
//!   - K frame for avw(r=T,k=HD): packed rows [T,HD], row t at t*HD*2.
//!   - V frame for attnv(T,HD): same packed-rows layout.
//!   - scores: strided [1,T,1,1], t at t*64 (avw z-plane layout).
//!
//! Run: cargo run --release --example qwen_decode -- \
//!        /Users/savag3/models/qwen3-0.6b/model.safetensors [steps]

use mil_infer::espresso::{EspressoProgram, SharedSurface};
use mil_infer::hwx::{emit_attn_v, emit_avw, emit_elementwise2, emit_matmul_ns, emit_repack, EwOp};
use std::collections::HashMap;
use std::path::Path;

// ---- Qwen3-0.6B dims -------------------------------------------------
const D: usize = 1024; // hidden
const Q_DIM: usize = 2048; // 16 q heads x 128
const KV_DIM: usize = 1024; // 8 kv heads x 128
const NQH: usize = 16;
const NKH: usize = 8;
const HD: usize = 128;
const FF: usize = 3072;
const T_MAX: usize = 64; // decode horizon (skeleton-verified sizes)
const LAYERS: usize = 28;
const VOCAB: usize = 151936;
const EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;
const SCALE: f32 = 0.088_388_35; // 1/sqrt(128)
                                 // resident-program budget: 28 qkv + 28 upgate + N down + ~8 shared <= 85
const DOWN_BAKED: usize = 15;

// ---- fp16 / bf16 helpers --------------------------------------------
fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
fn bf16_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
fn rd16(b: &[u8], o: usize) -> f32 {
    f32_of(u16::from_le_bytes([b[o], b[o + 1]]))
}
fn wr16(b: &mut [u8], o: usize, v: f32) {
    b[o..o + 2].copy_from_slice(&f16(v).to_le_bytes());
}

// ---- minimal safetensors reader --------------------------------------
struct SafeTensors {
    buf: Vec<u8>,
    index: HashMap<String, (usize, usize, Vec<usize>, String)>,
}

impl SafeTensors {
    fn load(path: &str) -> Self {
        let buf = std::fs::read(path).expect("safetensors read");
        let hlen = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&buf[8..8 + hlen]).expect("header json");
        let mut index = HashMap::new();
        for (name, v) in header.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            let dt = v["dtype"].as_str().unwrap().to_string();
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let off = v["data_offsets"].as_array().unwrap();
            let (a, b) = (
                off[0].as_u64().unwrap() as usize + 8 + hlen,
                off[1].as_u64().unwrap() as usize + 8 + hlen,
            );
            index.insert(name.clone(), (a, b, shape, dt));
        }
        SafeTensors { buf, index }
    }

    fn f32s(&self, name: &str) -> Vec<f32> {
        let (a, b, _, dt) = self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(dt, "BF16", "{name} dtype");
        self.buf[*a..*b]
            .chunks_exact(2)
            .map(|c| bf16_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }
}

/// W'[j,i] = W[j,i]*g[i] — fold a per-input-element norm gain into a
/// projection weight [N,K] row-major (f32, converted on surface write).
#[allow(dead_code)]
fn fold_gamma(w: &[f32], g: &[f32], n: usize, k: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n * k);
    for j in 0..n {
        for i in 0..k {
            out.push(f16(w[j * k + i] * g[i]));
        }
    }
    out
}

fn to_f16(w: &[f32]) -> Vec<u16> {
    w.iter().map(|&v| f16(v)).collect()
}

// ---- host-side vector ops --------------------------------------------
fn rms_norm(x: &mut [f32], g: &[f32], eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = (ss + eps).sqrt().recip();
    for (v, &gi) in x.iter_mut().zip(g) {
        *v *= r * gi;
    }
}

/// Qwen3 RoPE: pair (i, i+HD/2) inside each 128-d head.
fn rope_head(h: &mut [f32], pos: usize) {
    for i in 0..HD / 2 {
        let freq = 1.0 / ROPE_THETA.powf(2.0 * i as f32 / HD as f32);
        let ang = pos as f32 * freq;
        let (c, s) = (ang.cos(), ang.sin());
        let (a, b) = (h[i], h[i + HD / 2]);
        h[i] = a * c - b * s;
        h[i + HD / 2] = b * c + a * s;
    }
}

fn matvec_f32(w: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|j| (0..k).map(|i| x[i] * w[j * k + i]).sum::<f32>())
        .collect()
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn top5(v: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    idx[..5].to_vec()
}

// ---- program loading -------------------------------------------------
fn emit_load(img: &[u8], path: &Path) -> EspressoProgram {
    std::fs::write(path, img).unwrap();
    let p = EspressoProgram::load_precompiled(path, &[])
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    // the daemon has consumed the image by now; drop the file so the
    // emit dir doesn't eat the disk (verified: evals still succeed)
    let _ = std::fs::remove_file(path);
    p
}

fn mil() -> &'static str {
    "/tmp/qd.mil"
}

/// Chunk of a projection weight for the shared `avw_mm` matvec:
/// 128 surface rows of `k` fp16 elements — row 0 is a dead lane at
/// r>64, so rows 1..=127 carry 127 real weight rows.
struct WChunks {
    surf: Vec<SharedSurface>, // each holds 128 x K fp16 (row-stride K*2 aligned 256)
    rows_per: usize,          // useful lanes per chunk (127)
    n: usize,
}

impl WChunks {
    fn new(w: &[f32], n: usize, k: usize) -> Self {
        let rows_per = 127usize;
        let n_chunks = n.div_ceil(rows_per);
        let stride = (k * 2 + 255) & !255;
        let surf: Vec<SharedSurface> = (0..n_chunks)
            .map(|c| {
                let s = SharedSurface::new(128 * stride).unwrap();
                let b = s.bytes_mut();
                for lane in 1..128usize {
                    let j = c * rows_per + (lane - 1);
                    if j >= n {
                        break;
                    }
                    for i in 0..k {
                        wr16(b, lane * stride + i * 2, w[j * k + i]);
                    }
                }
                s
            })
            .collect();
        WChunks { surf, rows_per, n }
    }
}

struct Layer {
    // baked fused projections (one eval each; distinct binaries/layer)
    qkv: EspressoProgram,          // 1024 -> 4096 [q|k|v], ln1 folded
    upgate: EspressoProgram,       // 1024 -> 6144 [up|gate], ln2 folded
    down: Option<EspressoProgram>, // 3072 -> 1024, baked when slots allow
    // dynamic-weight leftovers on the shared matvec
    wo: Vec<WChunks>,  // k=2048 -> 2 slices of 1024
    wdn: Vec<WChunks>, // k=3072 -> 3 slices of 1024
    // KV frames: [T_MAX,HD] rows at 256B stride, one per kv head
    kf: Vec<SharedSurface>,
    vf: Vec<SharedSurface>,
    // host-side tensors
    qn: Vec<f32>,
    kn: Vec<f32>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let st_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/Users/savag3/models/qwen3-0.6b/model.safetensors");
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed: Vec<usize> = args
        .get(3)
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![151643, 9707]);

    let dir = format!("/tmp/qd_{}", std::process::id());
    std::fs::create_dir_all(&dir).unwrap();
    let dir = Path::new(&dir);
    let t0 = std::time::Instant::now();
    println!("qwen_decode: loading {st_path}");
    let st = SafeTensors::load(st_path);
    let embed = st.f32s("model.embed_tokens.weight");
    println!(
        "st: {} tensors, embed {} elems ({:?})",
        st.index.len(),
        embed.len(),
        t0.elapsed()
    );

    // ---- shared (shape-only) programs --------------------------------
    let mut xn = emit_load(
        &emit_elementwise2(
            EwOp::Mul,
            D as u32,
            mil(),
            &dir.join("xn.macho").to_string_lossy(),
        ),
        &dir.join("xn.macho"),
    );
    let mut xnc = emit_load(
        &emit_repack(D as u32, mil(), &dir.join("xnc.macho").to_string_lossy()).unwrap(),
        &dir.join("xnc.macho"),
    );
    // attention contractions — shared across every layer/head
    let mut avw = emit_load(
        &emit_avw(
            T_MAX as u32,
            HD as u32,
            mil(),
            &dir.join("avw.macho").to_string_lossy(),
        ),
        &dir.join("avw.macho"),
    );
    let mut attnv = emit_load(
        &emit_attn_v(
            T_MAX as u32,
            HD as u32,
            mil(),
            &dir.join("atv.macho").to_string_lossy(),
        )
        .unwrap(),
        &dir.join("atv.macho"),
    );
    // KV frame row strides from the programs' expected v-surface sizes
    let kstride = avw.input_mut(1).len() / T_MAX;
    let vstride = attnv.input_mut(1).len() / T_MAX;
    // the general dynamic-weight matvec — projections without a baked
    // binary run here with weight surfaces adopted per eval.
    // r=128, k=1024; lane 0 is dead at r>64 so chunks carry 127 rows.
    let mut mmv = emit_load(
        &emit_avw(128, 1024, mil(), &dir.join("mmv.macho").to_string_lossy()),
        &dir.join("mmv.macho"),
    );
    let down_s = SharedSurface::new(D * 64).expect("down surface");
    let po_s = SharedSurface::new(D * 64).expect("po surface");
    // per-head attention-prob surface (host softmax writes here; the
    // layout matches attnv's expected score input)
    let prob_s = SharedSurface::new(attnv.input_mut(0).len()).expect("prob surface");
    let dprob = attnv.input_descs[0].clone();
    let dsc = avw.output_descs[0].clone();
    let mut res = emit_load(
        &emit_elementwise2(
            EwOp::Add,
            D as u32,
            mil(),
            &dir.join("res.macho").to_string_lossy(),
        ),
        &dir.join("res.macho"),
    );
    // strided [1,D] surface layout + embedding-row donor surface
    let dxs = res.input_descs[0].clone();
    let embed_s = SharedSurface::new(dxs.bytes()).expect("embed surface");
    let drvec = xn.input_descs[1].clone();
    println!(
        "shared programs up ({:?}); kstride={kstride} vstride={vstride}",
        t0.elapsed()
    );

    // ---- per-layer weight surfaces (no program loads!) ---------------
    let mut layers: Vec<Layer> = Vec::with_capacity(LAYERS);
    for l in 0..LAYERS {
        let get = |s: &str| st.f32s(&format!("model.layers.{l}.{s}"));
        let ln1 = get("input_layernorm.weight");
        let ln2 = get("post_attention_layernorm.weight");
        // fold the preceding RMSNorm gain into each projection's input
        let fold = |w: &[f32]| -> Vec<f32> {
            w.iter()
                .enumerate()
                .map(|(idx, &v)| v * ln1[idx % D])
                .collect()
        };
        let fold2 = |w: &[f32]| -> Vec<f32> {
            w.iter()
                .enumerate()
                .map(|(idx, &v)| v * ln2[idx % D])
                .collect()
        };
        // fused qkv: one baked matmul 1024->4096, output [q|k|v]
        let mut wqkv = Vec::with_capacity((Q_DIM + 2 * KV_DIM) * D);
        wqkv.extend_from_slice(&fold(&get("self_attn.q_proj.weight")));
        wqkv.extend_from_slice(&fold(&get("self_attn.k_proj.weight")));
        wqkv.extend_from_slice(&fold(&get("self_attn.v_proj.weight")));
        let name = format!("l{l}_qkv");
        let qkv = emit_load(
            &emit_matmul_ns(
                D,
                Q_DIM + 2 * KV_DIM,
                &to_f16(&wqkv),
                mil(),
                &dir.join(format!("{name}.macho")).to_string_lossy(),
            )
            .unwrap(),
            &dir.join(format!("{name}.macho")),
        );
        // fused upgate: one baked matmul 1024->6144, output [up|gate]
        let mut wug = Vec::with_capacity(2 * FF * D);
        wug.extend_from_slice(&fold2(&get("mlp.up_proj.weight")));
        wug.extend_from_slice(&fold2(&get("mlp.gate_proj.weight")));
        let name = format!("l{l}_ug");
        let upgate = emit_load(
            &emit_matmul_ns(
                D,
                2 * FF,
                &to_f16(&wug),
                mil(),
                &dir.join(format!("{name}.macho")).to_string_lossy(),
            )
            .unwrap(),
            &dir.join(format!("{name}.macho")),
        );
        // down baked where the resident budget allows
        let wdn_full = get("mlp.down_proj.weight");
        let down = if l < DOWN_BAKED {
            let name = format!("l{l}_dn");
            Some(emit_load(
                &emit_matmul_ns(
                    FF,
                    D,
                    &to_f16(&wdn_full),
                    mil(),
                    &dir.join(format!("{name}.macho")).to_string_lossy(),
                )
                .unwrap(),
                &dir.join(format!("{name}.macho")),
            ))
        } else {
            None
        };
        // o/down leftovers on the shared matvec
        let wo_full = get("self_attn.o_proj.weight");
        let wo: Vec<WChunks> = (0..Q_DIM / D)
            .map(|s| {
                let mut w = vec![0f32; D * D];
                for j in 0..D {
                    for i in 0..D {
                        w[j * D + i] = wo_full[j * Q_DIM + s * D + i];
                    }
                }
                WChunks::new(&w, D, D)
            })
            .collect();
        let wdn: Vec<WChunks> = if down.is_none() {
            (0..FF / D)
                .map(|s| {
                    let mut w = vec![0f32; D * D];
                    for j in 0..D {
                        for i in 0..D {
                            w[j * D + i] = wdn_full[j * FF + s * D + i];
                        }
                    }
                    WChunks::new(&w, D, D)
                })
                .collect()
        } else {
            Vec::new()
        };
        let kf: Vec<SharedSurface> = (0..NKH)
            .map(|_| SharedSurface::new(T_MAX * kstride).unwrap())
            .collect();
        let vf: Vec<SharedSurface> = (0..NKH)
            .map(|_| SharedSurface::new(T_MAX * vstride).unwrap())
            .collect();
        if l % 7 == 0 {
            println!("layer {l} weights staged ({:?})", t0.elapsed());
        }
        layers.push(Layer {
            qkv,
            upgate,
            down,
            wo,
            wdn,
            kf,
            vf,
            qn: get("self_attn.q_norm.weight"),
            kn: get("self_attn.k_norm.weight"),
        });
    }

    // ---- lm_head, sliced (vocab is 152k wide) -------------------------
    let lm = st.f32s("lm_head.weight");
    let nf = st.f32s("model.norm.weight");
    let chunk = 20480usize.min(VOCAB);
    let n_chunks = VOCAB.div_ceil(chunk);
    let mut head: Vec<EspressoProgram> = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        let n = (VOCAB - c * chunk).min(chunk);
        let mut w = vec![0f32; n * D];
        for j in 0..n {
            for i in 0..D {
                w[j * D + i] = lm[(c * chunk + j) * D + i] * nf[i];
            }
        }
        let wh = to_f16(&w);
        let name = format!("head{c}");
        head.push(emit_load(
            &emit_matmul_ns(
                D,
                n,
                &wh,
                mil(),
                &dir.join(format!("{name}.macho")).to_string_lossy(),
            )
            .unwrap(),
            &dir.join(format!("{name}.macho")),
        ));
    }
    println!("head: {n_chunks} slices ({:?})", t0.elapsed());

    // ---- helpers ------------------------------------------------------
    let read_strided = |p: &EspressoProgram,
                        out: usize,
                        d: &mil_infer::espresso::TensorDesc,
                        n: usize|
     -> Vec<f32> {
        let b = p.output(out);
        (0..n)
            .map(|i| rd16(b, d.offset(0, 0, i as u64, 0, 0)))
            .collect()
    };

    // strided [1,N,1,1] write through a surface
    let write_strided = |b: &mut [u8], d: &mil_infer::espresso::TensorDesc, v: &[f32]| {
        for (i, &x) in v.iter().enumerate() {
            wr16(b, d.offset(0, 0, i as u64, 0, 0), x);
        }
    };

    // dynamic-weight matvec: out[0..n] += x * W_chunks over every
    // 128-row chunk (lane 0 dead; lanes 1..=127 carry weight rows).
    // k is fixed at 1024 by the shared program; caller splits k>1024
    // into slices and accumulates.
    let zmm = mmv.output_descs[0].clone();
    let mm_run = |mmv: &mut EspressoProgram, x: &[f32], wc: &WChunks, out: &mut Vec<f32>| {
        {
            let a = mmv.input_mut(0);
            for (i, &v) in x.iter().enumerate() {
                wr16(a, i * 2, v);
            }
        }
        for (c, s) in wc.surf.iter().enumerate() {
            mmv.adopt_input(1, s.as_ptr()).unwrap();
            mmv.eval().unwrap();
            let z = mmv.output(0);
            for lane in 1..128usize {
                let j = c * wc.rows_per + (lane - 1);
                if j >= wc.n {
                    break;
                }
                out[j] += rd16(z, zmm.offset(0, 0, lane as u64, 0, 0));
            }
        }
    };

    // ---- decode --------------------------------------------------------
    let mut tokens = seed.clone();
    let mut pos = 0usize;
    let mut agree = 0usize;
    let mut checked = 0usize;
    println!("decoding {} steps from seed {seed:?}", steps);
    for step in 0..steps {
        let st_t = std::time::Instant::now();
        // embed current token into the strided residual surface
        let x_v: Vec<f32> = (0..D).map(|i| embed[tokens[pos] * D + i]).collect();
        write_strided(embed_s.bytes_mut(), &dxs, &x_v);
        let mut xs = embed_s.as_ptr();

        for (_l, layer) in layers.iter_mut().enumerate() {
            // norm1: ss on host in f32 (residual magnitudes exceed f16
            // range when squared — ANE-side reduction saturates to inf)
            let res_v: Vec<f32> = if _l == 0 {
                x_v.clone()
            } else {
                read_strided(&res, 0, &res.output_descs[0].clone(), D)
            };
            let ss: f32 = res_v.iter().map(|v| v * v).sum();
            let r = (ss / D as f32 + EPS).sqrt().recip();
            if std::env::var_os("QD_DEBUG").is_some() && pos == 0 && _l == 0 {
                println!("  dbg l0: ss={ss} r={r}");
            }
            {
                let b = xn.input_mut(1);
                for i in 0..D {
                    wr16(b, drvec.offset(0, 0, i as u64, 0, 0), r);
                }
            }
            xn.adopt_input(0, xs).unwrap();
            xn.eval().unwrap();
            xnc.adopt_input(0, xn.output_surf(0)).unwrap();
            xnc.eval().unwrap();
            // normed input, contiguous — read once for all projections
            let xc_v: Vec<f32> = {
                let b = xnc.output(0);
                (0..D).map(|i| rd16(b, i * 2)).collect()
            };

            // fused qkv: one baked matmul, output [q|k|v] strided
            if std::env::var_os("QD_DEBUG").is_some() && pos == 0 && _l == 0 {
                let dx = layer.qkv.input_descs[0].clone();
                let dz = layer.qkv.output_descs[0].clone();
                let inlen = layer.qkv.input_mut(0).len();
                let outlen = layer.qkv.output(0).len();
                println!("  dbg qkv in n{} k{} h{} w{} ps{} rs{} inlen{} | out n{} k{} h{} w{} ps{} rs{} outlen{}",
                    dx.n, dx.k, dx.h, dx.w, dx.plane_stride, dx.row_stride, inlen,
                    dz.n, dz.k, dz.h, dz.w, dz.plane_stride, dz.row_stride, outlen);
                let dxv = layer.upgate.input_descs[0].clone();
                let uglen = layer.upgate.input_mut(0).len();
                println!(
                    "  dbg ug-in n{} k{} h{} w{} ps{} rs{} len{}",
                    dxv.n, dxv.k, dxv.h, dxv.w, dxv.plane_stride, dxv.row_stride, uglen
                );
                println!(
                    "  dbg dsc n{} k{} h{} w{} ps{} rs{}",
                    dsc.n, dsc.k, dsc.h, dsc.w, dsc.plane_stride, dsc.row_stride
                );
            }
            {
                let dx = layer.qkv.input_descs[0].clone();
                let b = layer.qkv.input_mut(0);
                for i in 0..D {
                    wr16(b, dx.offset(0, 0, 0, 0, i as u64), xc_v[i]);
                }
            }
            layer.qkv.eval().unwrap();
            let zq: Vec<f32> = {
                let dz = layer.qkv.output_descs[0].clone();
                let b = layer.qkv.output(0);
                (0..(Q_DIM + 2 * KV_DIM))
                    .map(|i| rd16(b, dz.offset(0, 0, i as u64, 0, 0)))
                    .collect()
            };
            let q = &zq[..Q_DIM];
            let k = &zq[Q_DIM..Q_DIM + KV_DIM];
            let v = &zq[Q_DIM + KV_DIM..];
            if std::env::var_os("QD_DEBUG").is_some() && pos == 0 && _l == 0 {
                // host recompute of qkv row j: xc . (ln1-folded) W[j]
                let ln1 = st.f32s("model.layers.0.input_layernorm.weight");
                let mut host_q = [0f32; 4];
                for j in 0..4 {
                    let mut acc = 0f32;
                    for i in 0..D {
                        acc += xc_v[i]
                            * st.f32s("model.layers.0.self_attn.q_proj.weight")[j * D + i]
                            * ln1[i];
                    }
                    host_q[j] = acc;
                }
                println!(
                    "  dbg l0: xc={:?} q={:?} k={:?} hostq={:?}",
                    &xc_v[..4],
                    &q[..4],
                    &k[..4],
                    host_q
                );
            }
            let mut qh: Vec<Vec<f32>> =
                (0..NQH).map(|h| q[h * HD..(h + 1) * HD].to_vec()).collect();
            let mut kh: Vec<Vec<f32>> =
                (0..NKH).map(|h| k[h * HD..(h + 1) * HD].to_vec()).collect();
            for h in 0..NQH {
                rms_norm(&mut qh[h], &layer.qn, EPS);
                rope_head(&mut qh[h], pos);
                for x in &mut qh[h] {
                    *x *= SCALE;
                }
            }
            for h in 0..NKH {
                rms_norm(&mut kh[h], &layer.kn, EPS);
                rope_head(&mut kh[h], pos);
            }
            // write K/V row pos into each group's frames (256B row stride)
            for g in 0..NKH {
                let kfb = layer.kf[g].bytes_mut();
                for i in 0..HD {
                    wr16(kfb, pos * kstride + i * 2, kh[g][i]);
                }
                let vfb = layer.vf[g].bytes_mut();
                for i in 0..HD {
                    wr16(vfb, pos * vstride + i * 2, v[g * HD + i]);
                }
            }
            let mut attn_out = vec![0f32; Q_DIM];
            for h in 0..NQH {
                let g = h / 2;
                // q into avw input0 (contiguous [HD]); K frame adopted
                {
                    let b = avw.input_mut(0);
                    for i in 0..HD {
                        wr16(b, i * 2, qh[h][i]);
                    }
                }
                avw.adopt_input(1, layer.kf[g].as_ptr()).unwrap();
                avw.eval().unwrap();
                // host mask + softmax over the score row (drops mkm/smc
                // evals — two fewer XPC round trips per head). scores
                // sit on the h-axis of the c=0 output plane.
                let mut sc: Vec<f32> = {
                    let b = avw.output(0);
                    (0..T_MAX)
                        .map(|t| rd16(b, dsc.offset(0, 0, t as u64, 0, 0)))
                        .collect()
                };
                for t in (pos + 1)..T_MAX {
                    sc[t] = f32::NEG_INFINITY;
                }
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut ssum = 0f32;
                for x in &mut sc {
                    *x = (*x - mx).exp();
                    ssum += *x;
                }
                for x in &mut sc {
                    *x /= ssum;
                }
                {
                    // probs in the same layout smc produced: lane 0 of the
                    // t-th 64B plane (that's where attnv reads them)
                    let b = prob_s.bytes_mut();
                    for t in 0..T_MAX {
                        wr16(b, t * 64, sc[t]);
                    }
                }
                attnv.adopt_input(0, prob_s.as_ptr()).unwrap();
                attnv.adopt_input(1, layer.vf[g].as_ptr()).unwrap();
                attnv.eval().unwrap();
                {
                    let b = attnv.output(0);
                    for i in 0..HD {
                        attn_out[h * HD + i] = rd16(b, i * 2);
                    }
                }
            }
            // o proj: k=2048 -> two 1024 slices accumulated on host
            let mut po = vec![0f32; D];
            for s in 0..2 {
                mm_run(
                    &mut mmv,
                    &attn_out[s * D..(s + 1) * D],
                    &layer.wo[s],
                    &mut po,
                );
            }
            if std::env::var_os("QD_DEBUG").is_some() && pos == 0 && _l == 0 {
                let wo_h = st.f32s("model.layers.0.self_attn.o_proj.weight");
                let mut hpo = [0f32; 4];
                for j in 0..4 {
                    let mut acc = 0f32;
                    for i in 0..Q_DIM {
                        acc += attn_out[i] * wo_h[j * Q_DIM + i];
                    }
                    hpo[j] = acc;
                }
                println!(
                    "  dbg l0: attn={:?} v0={:?} po={:?} hostpo={:?} xv={:?}",
                    &attn_out[..4],
                    &v[..4],
                    &po[..4],
                    hpo,
                    &x_v[..4]
                );
            }
            // po -> strided surface for the residual add
            write_strided(po_s.bytes_mut(), &dxs, &po);
            res.adopt_input(0, xs).unwrap();
            res.adopt_input(1, po_s.as_ptr()).unwrap();
            res.eval().unwrap();
            let r1 = res.output_surf(0);

            // norm2 (host ss, same f16-overflow reason)
            let res2_v = read_strided(&res, 0, &res.output_descs[0].clone(), D);
            let ss2: f32 = res2_v.iter().map(|v| v * v).sum();
            let r2 = (ss2 / D as f32 + EPS).sqrt().recip();
            {
                let b = xn.input_mut(1);
                for i in 0..D {
                    wr16(b, drvec.offset(0, 0, i as u64, 0, 0), r2);
                }
            }
            xn.adopt_input(0, r1).unwrap();
            xn.eval().unwrap();
            xnc.adopt_input(0, xn.output_surf(0)).unwrap();
            xnc.eval().unwrap();
            let xc2_v: Vec<f32> = {
                let b = xnc.output(0);
                (0..D).map(|i| rd16(b, i * 2)).collect()
            };

            // fused upgate: one baked matmul, output [up|gate] strided
            {
                let dx = layer.upgate.input_descs[0].clone();
                let b = layer.upgate.input_mut(0);
                for i in 0..D {
                    wr16(b, dx.offset(0, 0, 0, 0, i as u64), xc2_v[i]);
                }
            }
            layer.upgate.eval().unwrap();
            let zug: Vec<f32> = {
                let dz = layer.upgate.output_descs[0].clone();
                let b = layer.upgate.output(0);
                (0..2 * FF)
                    .map(|i| rd16(b, dz.offset(0, 0, i as u64, 0, 0)))
                    .collect()
            };
            // swiglu on host: up * gate * sigmoid(gate) = up * silu(gate)
            let act_v: Vec<f32> = (0..FF)
                .map(|i| zug[i] * zug[FF + i] * sigmoid(zug[FF + i]))
                .collect();
            if std::env::var_os("QD_DEBUG").is_some() && pos == 0 && _l == 0 {
                let ln2 = st.f32s("model.layers.0.post_attention_layernorm.weight");
                let wup = st.f32s("model.layers.0.mlp.up_proj.weight");
                let mut hug = [0f32; 4];
                for j in 0..4 {
                    let mut acc = 0f32;
                    for i in 0..D {
                        acc += xc2_v[i] * wup[j * D + i] * ln2[i];
                    }
                    hug[j] = acc;
                }
                println!(
                    "  dbg l0: xc2={:?} up={:?} hostup={:?}",
                    &xc2_v[..4],
                    &zug[..4],
                    hug
                );
            }
            // down proj: baked binary where loaded, else shared matvec
            let mut dn = vec![0f32; D];
            if let Some(down) = &mut layer.down {
                {
                    let dx = down.input_descs[0].clone();
                    let b = down.input_mut(0);
                    for i in 0..FF {
                        wr16(b, dx.offset(0, 0, 0, 0, i as u64), act_v[i]);
                    }
                }
                down.eval().unwrap();
                let dz = down.output_descs[0].clone();
                let b = down.output(0);
                for i in 0..D {
                    dn[i] = rd16(b, dz.offset(0, 0, i as u64, 0, 0));
                }
            } else {
                for s in 0..3 {
                    mm_run(&mut mmv, &act_v[s * D..(s + 1) * D], &layer.wdn[s], &mut dn);
                }
            }
            if std::env::var_os("QD_DEBUG").is_some() && pos == 0 && _l == 0 {
                let wdn_h = st.f32s("model.layers.0.mlp.down_proj.weight");
                let mut hdn = [0f32; 4];
                for j in 0..4 {
                    let mut acc = 0f32;
                    for i in 0..FF {
                        acc += act_v[i] * wdn_h[j * FF + i];
                    }
                    hdn[j] = acc;
                }
                println!(
                    "  dbg l0: dn={:?} hostdn={:?} act={:?}",
                    &dn[..4],
                    hdn,
                    &act_v[..4]
                );
            }
            write_strided(down_s.bytes_mut(), &dxs, &dn);
            res.adopt_input(0, r1).unwrap();
            res.adopt_input(1, down_s.as_ptr()).unwrap();
            res.eval().unwrap();
            // residual for next layer / final norm
            xs = res.output_surf(0);
            if std::env::var_os("QD_DEBUG").is_some()
                && pos == 0
                && (_l == 0 || _l == 13 || _l == 27)
            {
                let b = res.output(0);
                let cur: Vec<f32> = (0..4)
                    .map(|i| rd16(b, dxs.offset(0, 0, i as u64, 0, 0)))
                    .collect();
                let dn4: Vec<f32> = dn[..4].to_vec();
                println!("  dbg l{_l}: res={cur:?} dn={dn4:?}");
            }
        }
        // final norm on last residual (host ss)
        let last = xs;
        let resf_v = read_strided(&res, 0, &res.output_descs[0].clone(), D);
        let ssf: f32 = resf_v.iter().map(|v| v * v).sum();
        let rf = (ssf / D as f32 + EPS).sqrt().recip();
        if std::env::var_os("QD_DEBUG").is_some() && pos == 0 {
            println!("  dbg ssf={ssf} rf={rf}");
        }
        {
            let b = xn.input_mut(1);
            for i in 0..D {
                wr16(b, drvec.offset(0, 0, i as u64, 0, 0), rf);
            }
        }
        xn.adopt_input(0, last).unwrap();
        xn.eval().unwrap();
        xnc.adopt_input(0, xn.output_surf(0)).unwrap();
        xnc.eval().unwrap();
        let xf = xnc.output_surf(0);
        // lm_head slices (gamma folded into weights)
        if std::env::var_os("QD_DEBUG").is_some() && pos == 0 {
            let b = xnc.output(0);
            let xv: Vec<f32> = (0..8).map(|i| rd16(b, i * 2)).collect();
            println!("  dbg xf={xv:?}");
        }
        let mut logits = vec![0f32; VOCAB];
        for (c, m) in head.iter_mut().enumerate() {
            m.adopt_input(0, xf).unwrap();
            m.eval().unwrap();
            let dh = m.output_descs[0].clone();
            if std::env::var_os("QD_DEBUG").is_some() && pos == 0 && c == 0 {
                println!(
                    "  dbg head desc: n{} k{} h{} w{} ps{} outlen{}",
                    dh.n,
                    dh.k,
                    dh.h,
                    dh.w,
                    dh.plane_stride,
                    m.output(0).len()
                );
                let b = m.output(0);
                let raw: Vec<f32> = (0..8).map(|i| rd16(b, i * 64)).collect();
                println!("  dbg head z strided[0..8]={raw:?}");
            }
            let n = (VOCAB - c * chunk).min(chunk);
            let part = read_strided(m, 0, &dh, n);
            logits[c * chunk..c * chunk + n].copy_from_slice(&part);
        }
        let next = argmax(&logits);
        if std::env::var_os("QD_DEBUG").is_some() && pos == 0 {
            println!("  dbg logits[0..8]={:?}", &logits[..8]);
            println!("  dbg logits[75776..75784]={:?}", &logits[75776..75784]);
            let mn = logits.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("  dbg logits min={mn} max={mx}");
        }
        // CPU f32 reference forward — verifies the ANE stack agrees with
        // an exact implementation of the same weights. pos==0 needs no
        // KV history, so it stays cheap.
        if pos == 0 {
            let ref_logits = cpu_forward(&st, &embed, &tokens, pos);
            let ref_next = argmax(&ref_logits);
            let rtop = top5(&ref_logits);
            let atop = top5(&logits);
            let hit = ref_next == next;
            checked += 1;
            agree += usize::from(hit);
            println!("  ref: argmax={ref_next} top5={rtop:?} | ane top5={atop:?} | agree={hit}");
        }
        tokens.push(next);
        pos += 1;
        println!(
            "step {step}: next={next} top5={:?} ({:?}/step)",
            top5(&logits),
            st_t.elapsed()
        );
        if pos >= T_MAX - 1 {
            println!("T_MAX reached");
            break;
        }
    }
    println!("tokens: {tokens:?}");
    println!("argmax agreement with CPU ref: {agree}/{checked}");
}

// ---- CPU f32 reference forward (pos==0 only: no KV history) ----------
fn cpu_forward(st: &SafeTensors, embed: &[f32], tokens: &[usize], pos: usize) -> Vec<f32> {
    assert_eq!(pos, 0, "lean ref handles pos==0 only");
    let mut x: Vec<f32> = (0..D).map(|i| embed[tokens[0] * D + i]).collect();
    for l in 0..LAYERS {
        let get = |s: &str| st.f32s(&format!("model.layers.{l}.{s}"));
        let ln1 = get("input_layernorm.weight");
        let ln2 = get("post_attention_layernorm.weight");
        let mut xn = x.clone();
        rms_norm(&mut xn, &ln1, EPS);
        let q = matvec_f32(&get("self_attn.q_proj.weight"), &xn, Q_DIM, D);
        let k = matvec_f32(&get("self_attn.k_proj.weight"), &xn, KV_DIM, D);
        let v = matvec_f32(&get("self_attn.v_proj.weight"), &xn, KV_DIM, D);
        let qn = get("self_attn.q_norm.weight");
        let kn = get("self_attn.k_norm.weight");
        let mut qh: Vec<Vec<f32>> = (0..NQH).map(|h| q[h * HD..(h + 1) * HD].to_vec()).collect();
        let mut kh: Vec<Vec<f32>> = (0..NKH).map(|h| k[h * HD..(h + 1) * HD].to_vec()).collect();
        for h in 0..NQH {
            rms_norm(&mut qh[h], &qn, EPS);
            rope_head(&mut qh[h], pos);
        }
        for h in 0..NKH {
            rms_norm(&mut kh[h], &kn, EPS);
            rope_head(&mut kh[h], pos);
        }
        // pos==0: softmax over a single score = 1.0, so attn = V rows
        let mut attn = vec![0f32; Q_DIM];
        for h in 0..NQH {
            let g = h / 2;
            for i in 0..HD {
                attn[h * HD + i] = v[g * HD + i];
            }
        }
        let po = matvec_f32(&get("self_attn.o_proj.weight"), &attn, D, Q_DIM);
        for i in 0..D {
            x[i] += po[i];
        }
        let mut xn2 = x.clone();
        rms_norm(&mut xn2, &ln2, EPS);
        let u = matvec_f32(&get("mlp.up_proj.weight"), &xn2, FF, D);
        let g = matvec_f32(&get("mlp.gate_proj.weight"), &xn2, FF, D);
        let act: Vec<f32> = (0..FF).map(|i| u[i] * g[i] * sigmoid(g[i])).collect();
        let dn = matvec_f32(&get("mlp.down_proj.weight"), &act, D, FF);
        for i in 0..D {
            x[i] += dn[i];
        }
        if std::env::var_os("QD_DEBUG").is_some() {
            let mx = x.iter().cloned().fold(0f32, |a, b| a.max(b.abs()));
            println!("  ref l{l}: |x|max={mx} x[..4]={:?}", &x[..4]);
        }
    }
    let mut xf = x.clone();
    rms_norm(&mut xf, &st.f32s("model.norm.weight"), EPS);
    matvec_f32(&st.f32s("lm_head.weight"), &xf, VOCAB, D)
}
