//! cy4b_decode — fused native ANE decode for CY4B (36L, d=2560,
//! 32q/8kv GQA hd=128, FF=10240, vocab=3103) on self-emitted HWX.
//! No ANECompiler, no CoreML, no e5: every program comes out of
//! `mil_infer::hwx` emitters; weights are per-channel int8 + fp16
//! scales (cy4b_q8 output) dequantized to fp16 — baked into Mach-O
//! const pools where the resident budget allows, dynamic weight
//! surfaces through the shared avw matvec otherwise.
//!
//! Run: cargo run --release -p mil_infer --example cy4b_decode -- \
//!        /Users/savag3/cy4b/cy4b_q8.safetensors [steps] [tok,tok,..]
//!
//! Env: CY4B_BAKE_BYTES — resident weight budget for baked matmuls
//!      (default 4.5e9); layers past the budget run on the shared
//!      dynamic-weight matvec.

use mil_infer::espresso::{EspressoProgram, SharedSurface};
use mil_infer::hwx::{emit_attn_v, emit_avw, emit_elementwise2, emit_matmul_ns, EwOp};
use std::collections::HashMap;
use std::path::Path;

// ---- CY4B dims --------------------------------------------------------
const D: usize = 2560;
const Q_DIM: usize = 4096; // 32 q heads x 128
const KV_DIM: usize = 1024; // 8 kv heads x 128
const NQH: usize = 32;
const NKH: usize = 8;
const HD: usize = 128;
const HALF: usize = 64;
const FF: usize = 10240;
const T_MAX: usize = 256;
const LAYERS: usize = 36;
const VOCAB: usize = 3103;
const EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 1_000_000.0;
const SCORE_SCALE: f32 = 0.125;

// ---- fp16 helpers ------------------------------------------------------
fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
fn rd16(b: &[u8], o: usize) -> f32 {
    f32_of(u16::from_le_bytes([b[o], b[o + 1]]))
}
fn wr16(b: &mut [u8], o: usize, v: f32) {
    b[o..o + 2].copy_from_slice(&f16(v).to_le_bytes());
}

// ---- safetensors reader: F16/F32 passthrough, I8+qscale dequant ------
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
        let (a, b, shape, dt) = self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        let (a, b, shape, dt) = (*a, *b, shape.clone(), dt.clone());
        let numel: usize = shape.iter().product();
        match dt.as_str() {
            "I8" => {
                let (sa, _, _, sdt) = self
                    .index
                    .get(&format!("{name}.qscale"))
                    .unwrap_or_else(|| panic!("missing {name}.qscale"));
                assert_eq!(sdt, "F16");
                let n = shape[0];
                let k = numel / n;
                let mut out = vec![0f32; numel];
                for j in 0..n {
                    let s = f32_of(u16::from_le_bytes([
                        self.buf[*sa + j * 2],
                        self.buf[*sa + j * 2 + 1],
                    ]));
                    for i in 0..k {
                        out[j * k + i] = self.buf[a + j * k + i] as i8 as f32 * s;
                    }
                }
                out
            }
            "F16" => self.buf[a..b]
                .chunks_exact(2)
                .map(|c| f32_of(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            "BF16" => self.buf[a..b]
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            "F32" => self.buf[a..b]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            other => panic!("{name}: dtype {other}"),
        }
    }

    fn f16s(&self, name: &str) -> Vec<u16> {
        self.f32s(name).iter().map(|&v| f16(v)).collect()
    }
}

fn to_f16(w: &[f32]) -> Vec<u16> {
    w.iter().map(|&v| f16(v)).collect()
}

fn fold_gamma(w: &[f32], g: &[f32], k: usize) -> Vec<f32> {
    w.iter()
        .enumerate()
        .map(|(idx, &v)| v * g[idx % k])
        .collect()
}

// ---- host vector ops ---------------------------------------------------
fn rms_norm(x: &mut [f32], g: &[f32], eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = (ss + eps).sqrt().recip();
    for (v, &gi) in x.iter_mut().zip(g) {
        *v *= r * gi;
    }
}

fn rope_head(h: &mut [f32], pos: usize) {
    for i in 0..HALF {
        let freq = 1.0 / ROPE_THETA.powf(2.0 * i as f32 / HD as f32);
        let ang = pos as f32 * freq;
        let (c, s) = (ang.cos(), ang.sin());
        let (a, b) = (h[i], h[i + HALF]);
        h[i] = a * c - b * s;
        h[i + HALF] = b * c + a * s;
    }
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

// ---- program loading ---------------------------------------------------
fn emit_load(img: &[u8], path: &Path) -> EspressoProgram {
    std::fs::write(path, img).unwrap();
    let p = EspressoProgram::load_precompiled(path, &[])
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let _ = std::fs::remove_file(path);
    p
}

fn mil() -> &'static str {
    "/tmp/cyd.mil"
}

/// Chunk of a [N,K] projection for the shared `avw(128,1024)` matvec:
/// 128 surface rows per (n-chunk, k-slice); lanes 1..=127 carry 127
/// real weight rows, lane 0 dead. K is sliced into 1024-wide slabs.
struct WChunks {
    surf: Vec<SharedSurface>, // idx = n_chunk * k_slices + k_slice
    rows_per: usize,          // 127
    n: usize,
    k: usize,
    k_slices: usize,
}

impl WChunks {
    fn new(w: &[f32], n: usize, k: usize) -> Self {
        const RK: usize = 1024; // avw k extent
        const ROWS: usize = 128; // avw r extent
        let rows_per = ROWS - 1; // lane 0 dead
        let n_chunks = n.div_ceil(rows_per);
        let k_slices = k.div_ceil(RK);
        let stride = (RK * 2 + 255) & !255;
        let mut surf = Vec::with_capacity(n_chunks * k_slices);
        for c in 0..n_chunks {
            for s in 0..k_slices {
                let sf = SharedSurface::new(ROWS * stride).unwrap();
                let b = sf.bytes_mut();
                for lane in 1..ROWS {
                    let j = c * rows_per + (lane - 1);
                    if j >= n {
                        break;
                    }
                    for i in 0..RK {
                        let src = s * RK + i;
                        if src < k {
                            wr16(b, lane * stride + i * 2, w[j * k + src]);
                        }
                    }
                }
                surf.push(sf);
            }
        }
        WChunks { surf, rows_per, n, k, k_slices }
    }
}

struct Layer {
    qkv: Option<EspressoProgram>,    // 2560 -> 6144 [q|k|v], ln1 folded
    upgate: Option<EspressoProgram>, // 2560 -> 20480 [up|gate], ln2 folded
    o: Option<EspressoProgram>,      // 4096 -> 2560
    down: Option<EspressoProgram>,   // 10240 -> 2560
    // dynamic-weight fallbacks (present iff the baked side is None)
    wqkv: Option<WChunks>,
    wug: Option<WChunks>,
    wo: Option<WChunks>,
    wdn: Option<WChunks>,
    kf: Vec<SharedSurface>,
    vf: Vec<SharedSurface>,
    qn: Vec<f32>,
    kn: Vec<f32>,
}

/// out = x @ W^T style matvec: out[j] = sum_i x[i] * W[j,i], W [N,K].
/// Baked path: one matmul_ns eval. Dynamic path: shared avw matvec
/// over 127-row x 1024-k chunks, accumulated on host.
fn linear(
    prog: &mut Option<EspressoProgram>,
    mmv: &mut EspressoProgram,
    wc: &Option<WChunks>,
    x: &[f32],
    k: usize,
    n: usize,
) -> Vec<f32> {
    if let Some(p) = prog.as_mut() {
        let dx = p.input_descs[0].clone();
        let b = p.input_mut(0);
        for i in 0..k {
            wr16(b, dx.offset(0, 0, 0, 0, i as u64), x[i]);
        }
        p.eval().unwrap();
        let dz = p.output_descs[0].clone();
        let b = p.output(0);
        (0..n)
            .map(|i| rd16(b, dz.offset(0, 0, i as u64, 0, 0)))
            .collect()
    } else {
        let wc = wc.as_ref().unwrap();
        let mut out = vec![0f32; n];
        let zmm = mmv.output_descs[0].clone();
        for s in 0..wc.k_slices {
            {
                let a = mmv.input_mut(0);
                for i in 0..1024usize {
                    let src = s * 1024 + i;
                    wr16(a, i * 2, if src < k { x[src] } else { 0.0 });
                }
            }
            let n_chunks = n.div_ceil(wc.rows_per);
            for c in 0..n_chunks {
                mmv.adopt_input(1, wc.surf[c * wc.k_slices + s].as_ptr())
                    .unwrap();
                mmv.eval().unwrap();
                let z = mmv.output(0);
                for lane in 1..128usize {
                    let j = c * wc.rows_per + (lane - 1);
                    if j >= n {
                        break;
                    }
                    out[j] += rd16(z, zmm.offset(0, 0, lane as u64, 0, 0));
                }
            }
        }
        out
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let st_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/Users/savag3/cy4b/cy4b_q8.safetensors");
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let seed: Vec<usize> = args
        .get(3)
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![9707, 11]);
    let bake_budget: u64 = std::env::var("CY4B_BAKE_BYTES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4_500_000_000);

    let dir = format!("/tmp/cyd_{}", std::process::id());
    std::fs::create_dir_all(&dir).unwrap();
    let dir = Path::new(&dir);
    let t0 = std::time::Instant::now();
    println!("cy4b_decode: loading {st_path} (bake budget {bake_budget}B)");
    let st = SafeTensors::load(st_path);
    println!("st: {} tensors ({:?})", st.index.len(), t0.elapsed());

    // ---- shared (shape-only) programs ---------------------------------
    let mut avw = emit_load(
        &emit_avw(T_MAX as u32, HD as u32, mil(), &dir.join("avw.macho").to_string_lossy()),
        &dir.join("avw.macho"),
    );
    let mut attnv = emit_load(
        &emit_attn_v(T_MAX as u32, HD as u32, mil(), &dir.join("atv.macho").to_string_lossy())
            .unwrap(),
        &dir.join("atv.macho"),
    );
    // shared dynamic-weight matvec: avw(128 rows, k=1024) — r=256
    // breaks lane->row mapping (parity gate catches it); k-slice
    // count is the remaining lever (avw k=1792 fits the lane mask).
    let mut mmv = emit_load(
        &emit_avw(128, 1024, mil(), &dir.join("mmv.macho").to_string_lossy()),
        &dir.join("mmv.macho"),
    );
    let kstride = avw.input_mut(1).len() / T_MAX;
    let vstride = attnv.input_mut(1).len() / T_MAX;
    let dsc = avw.output_descs[0].clone();
    let prob_s = SharedSurface::new(attnv.input_mut(0).len()).expect("prob surface");
    let down_s = SharedSurface::new(D * 64).expect("down surface");
    let po_s = SharedSurface::new(D * 64).expect("po surface");
    let mut res = emit_load(
        &emit_elementwise2(EwOp::Add, D as u32, mil(), &dir.join("res.macho").to_string_lossy()),
        &dir.join("res.macho"),
    );
    let dxs = res.input_descs[0].clone();
    let embed_s = SharedSurface::new(dxs.bytes()).expect("embed surface");
    println!("shared programs up; kstride={kstride} vstride={vstride} ({:?})", t0.elapsed());

    // ---- per-layer weights: bake while the resident budget lasts ------
    let layer_fp16_bytes =
        ((Q_DIM + 2 * KV_DIM) + 2 * FF + Q_DIM + FF) as u64 * D as u64 * 2; // ~qkv+ug+o+dn per layer, K varies — approx
    let _ = layer_fp16_bytes;
    let mut baked_bytes = 0u64;
    let mut layers: Vec<Layer> = Vec::with_capacity(LAYERS);
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}.");
        let ln1 = st.f32s(&format!("{p}input_layernorm.weight"));
        let ln2 = st.f32s(&format!("{p}post_attention_layernorm.weight"));
        // weight tensors (dequantized f32; gamma folded where a norm precedes)
        let wqkv_f: Vec<f32> = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]
            .iter()
            .flat_map(|n| fold_gamma(&st.f32s(&format!("{p}{n}")), &ln1, D))
            .collect();
        let wug_f: Vec<f32> = ["mlp.up_proj.weight", "mlp.gate_proj.weight"]
            .iter()
            .flat_map(|n| fold_gamma(&st.f32s(&format!("{p}{n}")), &ln2, D))
            .collect();
        let wo_f = st.f32s(&format!("{p}self_attn.o_proj.weight"));
        let wd_f = st.f32s(&format!("{p}mlp.down_proj.weight"));
        let wbytes = (wqkv_f.len() + wug_f.len() + wo_f.len() + wd_f.len()) as u64 * 2;
        let bake = baked_bytes + wbytes <= bake_budget;
        if bake {
            baked_bytes += wbytes;
        }

        let (qkv, wqkv) = if bake {
            (
                Some(emit_load(
                    &emit_matmul_ns(D, Q_DIM + 2 * KV_DIM, &to_f16(&wqkv_f), mil(),
                        &dir.join(format!("l{l}_qkv.macho")).to_string_lossy()).unwrap(),
                    &dir.join(format!("l{l}_qkv.macho")),
                )),
                None,
            )
        } else {
            (None, Some(WChunks::new(&wqkv_f, Q_DIM + 2 * KV_DIM, D)))
        };
        let (upgate, wug) = if bake {
            (
                Some(emit_load(
                    &emit_matmul_ns(D, 2 * FF, &to_f16(&wug_f), mil(),
                        &dir.join(format!("l{l}_ug.macho")).to_string_lossy()).unwrap(),
                    &dir.join(format!("l{l}_ug.macho")),
                )),
                None,
            )
        } else {
            (None, Some(WChunks::new(&wug_f, 2 * FF, D)))
        };
        let (o, wo) = if bake {
            (
                Some(emit_load(
                    &emit_matmul_ns(Q_DIM, D, &st.f16s(&format!("{p}self_attn.o_proj.weight")), mil(),
                        &dir.join(format!("l{l}_o.macho")).to_string_lossy()).unwrap(),
                    &dir.join(format!("l{l}_o.macho")),
                )),
                None,
            )
        } else {
            // o_proj input is attn output — no norm fold applies
            (None, Some(WChunks::new(&wo_f, D, Q_DIM)))
        };
        let (down, wdn) = if bake {
            (
                Some(emit_load(
                    &emit_matmul_ns(FF, D, &st.f16s(&format!("{p}mlp.down_proj.weight")), mil(),
                        &dir.join(format!("l{l}_dn.macho")).to_string_lossy()).unwrap(),
                    &dir.join(format!("l{l}_dn.macho")),
                )),
                None,
            )
        } else {
            (None, Some(WChunks::new(&wd_f, D, FF)))
        };
        let kf: Vec<SharedSurface> = (0..NKH)
            .map(|_| SharedSurface::new(T_MAX * kstride).unwrap())
            .collect();
        let vf: Vec<SharedSurface> = (0..NKH)
            .map(|_| SharedSurface::new(T_MAX * vstride).unwrap())
            .collect();
        if l % 6 == 0 {
            println!("layer {l} {} ({:?}, baked {:.1}GB)", if bake { "baked" } else { "mmv" },
                t0.elapsed(), baked_bytes as f64 / 1e9);
        }
        layers.push(Layer {
            qkv, upgate, o, down, wqkv, wug, wo, wdn, kf, vf,
            qn: st.f32s(&format!("{p}self_attn.q_norm.weight")),
            kn: st.f32s(&format!("{p}self_attn.k_norm.weight")),
        });
    }

    // ---- lm_head + final norm (gamma folded) ----------------------------
    let nf = st.f32s("model.norm.weight");
    let lm = st.f32s("lm_head.weight");
    let wh: Vec<u16> = lm
        .iter()
        .enumerate()
        .map(|(idx, &v)| f16(v * nf[idx % D]))
        .collect();
    let mut head = emit_load(
        &emit_matmul_ns(D, VOCAB, &wh, mil(), &dir.join("head.macho").to_string_lossy()).unwrap(),
        &dir.join("head.macho"),
    );
    println!("all programs resident ({:?})", t0.elapsed());

    // ---- helpers ---------------------------------------------------------
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
    let write_strided = |b: &mut [u8], d: &mil_infer::espresso::TensorDesc, v: &[f32]| {
        for (i, &x) in v.iter().enumerate() {
            wr16(b, d.offset(0, 0, i as u64, 0, 0), x);
        }
    };

    // ---- decode ------------------------------------------------------------
    let embed = st.f32s("model.embed_tokens.weight");
    let mut tokens = seed.clone();
    let mut pos = 0usize;
    for step in 0..steps {
        let st_t = std::time::Instant::now();
        let x_v: Vec<f32> = (0..D).map(|i| embed[tokens[pos] * D + i]).collect();
        write_strided(embed_s.bytes_mut(), &dxs, &x_v);
        let mut xs = embed_s.as_ptr();

        for (_l, layer) in layers.iter_mut().enumerate() {
            // ---- norm1 on host: res_v * rsqrt(mean(x^2)+eps); gamma lives
            //      folded inside the qkv weights.
            let res_v: Vec<f32> = if _l == 0 {
                x_v.clone()
            } else {
                read_strided(&res, 0, &res.output_descs[0].clone(), D)
            };
            let ss: f32 = res_v.iter().map(|v| v * v).sum();
            let r = (ss / D as f32 + EPS).sqrt().recip();
            let xc_v: Vec<f32> = res_v.iter().map(|&v| v * r).collect();

            // ---- fused qkv -> [q|k|v]
            let zq = linear(&mut layer.qkv, &mut mmv, &layer.wqkv, &xc_v, D, Q_DIM + 2 * KV_DIM);
            let q = &zq[..Q_DIM];
            let k = &zq[Q_DIM..Q_DIM + KV_DIM];
            let v = &zq[Q_DIM + KV_DIM..];
            let mut qh: Vec<Vec<f32>> = Vec::with_capacity(NQH);
            for h in 0..NQH {
                let mut head = q[h * HD..(h + 1) * HD].to_vec();
                rms_norm(&mut head[..HALF], &layer.qn[..HALF], EPS);
                rms_norm(&mut head[HALF..], &layer.qn[HALF..], EPS);
                rope_head(&mut head, pos);
                qh.push(head);
            }
            for g in 0..NKH {
                let mut kh = k[g * HD..(g + 1) * HD].to_vec();
                rms_norm(&mut kh[..HALF], &layer.kn[..HALF], EPS);
                rms_norm(&mut kh[HALF..], &layer.kn[HALF..], EPS);
                rope_head(&mut kh, pos);
                let kfb = layer.kf[g].bytes_mut();
                let vfb = layer.vf[g].bytes_mut();
                for i in 0..HD {
                    wr16(kfb, pos * kstride + i * 2, kh[i]);
                    wr16(vfb, pos * vstride + i * 2, v[g * HD + i]);
                }
            }
            // ---- attention per q head
            let mut attn_out = vec![0f32; Q_DIM];
            for h in 0..NQH {
                let g = h / 4;
                {
                    let b = avw.input_mut(0);
                    for i in 0..HD {
                        wr16(b, i * 2, qh[h][i]);
                    }
                }
                avw.adopt_input(1, layer.kf[g].as_ptr()).unwrap();
                avw.eval().unwrap();
                let mut sc: Vec<f32> = {
                    let b = avw.output(0);
                    (0..T_MAX)
                        .map(|t| rd16(b, dsc.offset(0, 0, t as u64, 0, 0)))
                        .collect()
                };
                for t in (pos + 1)..T_MAX {
                    sc[t] = f32::NEG_INFINITY;
                }
                for x in &mut sc {
                    *x *= SCORE_SCALE;
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
            // ---- o proj + residual
            let po = linear(&mut layer.o, &mut mmv, &layer.wo, &attn_out, Q_DIM, D);
            write_strided(po_s.bytes_mut(), &dxs, &po);
            res.adopt_input(0, xs).unwrap();
            res.adopt_input(1, po_s.as_ptr()).unwrap();
            res.eval().unwrap();
            let r1 = res.output_surf(0);

            // ---- norm2 + upgate + swiglu + down + residual
            let res2_v = read_strided(&res, 0, &res.output_descs[0].clone(), D);
            let ss2: f32 = res2_v.iter().map(|v| v * v).sum();
            let r2 = (ss2 / D as f32 + EPS).sqrt().recip();
            let xc2_v: Vec<f32> = res2_v.iter().map(|&v| v * r2).collect();
            let zug = linear(&mut layer.upgate, &mut mmv, &layer.wug, &xc2_v, D, 2 * FF);
            let act_v: Vec<f32> = (0..FF)
                .map(|i| {
                    let (u, g) = (zug[i], zug[FF + i]);
                    u * g / (1.0 + (-g).exp())
                })
                .collect();
            let dn = linear(&mut layer.down, &mut mmv, &layer.wdn, &act_v, FF, D);
            write_strided(down_s.bytes_mut(), &dxs, &dn);
            res.adopt_input(0, r1).unwrap();
            res.adopt_input(1, down_s.as_ptr()).unwrap();
            res.eval().unwrap();
            xs = res.output_surf(0);
        }

        // ---- final norm + lm_head
        let resf_v = read_strided(&res, 0, &res.output_descs[0].clone(), D);
        let ssf: f32 = resf_v.iter().map(|v| v * v).sum();
        let rf = (ssf / D as f32 + EPS).sqrt().recip();
        let xf_v: Vec<f32> = resf_v.iter().map(|&v| v * rf).collect();
        {
            let dx = head.input_descs[0].clone();
            let b = head.input_mut(0);
            for i in 0..D {
                wr16(b, dx.offset(0, 0, 0, 0, i as u64), xf_v[i]);
            }
        }
        head.eval().unwrap();
        let dh = head.output_descs[0].clone();
        let logits: Vec<f32> = {
            let b = head.output(0);
            (0..VOCAB)
                .map(|i| rd16(b, dh.offset(0, 0, i as u64, 0, 0)))
                .collect()
        };
        let next = argmax(&logits);
        if pos == 0 {
            // CPU f32 reference at pos==0: rope is identity (angle 0)
            // and causal softmax over one position is 1.0, so each
            // attention head emits its V slice directly.
            let ref_logits = cpu_forward(&st, &embed, tokens[0]);
            let rn = argmax(&ref_logits);
            println!(
                "  ref: argmax={rn} top5={:?} | ane top5={:?} agree={}",
                top5(&ref_logits),
                top5(&logits),
                rn == next
            );
        }
        println!(
            "step {step}: pos={pos} next={next} top5={:?} ({:?}/step)",
            top5(&logits),
            st_t.elapsed()
        );
        tokens.push(next);
        pos += 1;
        if pos >= T_MAX {
            println!("T_MAX reached");
            break;
        }
    }
    println!("tokens: {tokens:?}");
}

// ---- CPU f32 reference forward (pos==0 only: rope=identity, attn=V) --
fn matvec_f32(w: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    for j in 0..n {
        let row = &w[j * k..(j + 1) * k];
        let mut acc = 0f32;
        for i in 0..k {
            acc += x[i] * row[i];
        }
        out[j] = acc;
    }
    out
}

fn cpu_forward(st: &SafeTensors, embed: &[f32], tok: usize) -> Vec<f32> {
    let mut x: Vec<f32> = (0..D).map(|i| embed[tok * D + i]).collect();
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}.");
        let mut h = x.clone();
        rms_norm(&mut h, &st.f32s(&format!("{p}input_layernorm.weight")), EPS);
        // v only — q/k irrelevant at pos==0
        let v = matvec_f32(&st.f32s(&format!("{p}self_attn.v_proj.weight")), &h, KV_DIM, D);
        let mut attn = vec![0f32; Q_DIM];
        for hq in 0..NQH {
            let g = hq / 4;
            attn[hq * HD..(hq + 1) * HD].copy_from_slice(&v[g * HD..(g + 1) * HD]);
        }
        let po = matvec_f32(&st.f32s(&format!("{p}self_attn.o_proj.weight")), &attn, D, Q_DIM);
        for i in 0..D {
            x[i] += po[i];
        }
        let mut h2 = x.clone();
        rms_norm(&mut h2, &st.f32s(&format!("{p}post_attention_layernorm.weight")), EPS);
        let mut ug = st.f32s(&format!("{p}mlp.up_proj.weight"));
        let g = st.f32s(&format!("{p}mlp.gate_proj.weight"));
        for i in 0..FF {
            ug[i] = ug[i] * g[i] / (1.0 + (-g[i]).exp());
        }
        let dn = matvec_f32(&st.f32s(&format!("{p}mlp.down_proj.weight")), &ug, D, FF);
        for i in 0..D {
            x[i] += dn[i];
        }
    }
    let mut xf = x;
    rms_norm(&mut xf, &st.f32s("model.norm.weight"), EPS);
    matvec_f32(&st.f32s("lm_head.weight"), &xf, VOCAB, D)
}
