//! graph_probe — compile a full transformer-layer MIL through
//! `compile_graph` (the scheduler) and execute via `GraphRuntime`,
//! verifying against a CPU reference over 3 masked decode steps.
//!
//! The MIL carries only the logical graph; repacks, the rmsnorm
//! expansion, and the host-rsqrt bridge are inserted by the compiler.

use half::f16;
use mil_infer::graph::{compile_graph, GraphRuntime};

const T_MAX: usize = 256;
const D: usize = 64;
const FF: usize = 256;
const VOCAB: usize = 256;
const LAYERS: usize = 2;
const EPS: f32 = 1e-5;

fn f32_of(h: u16) -> f32 {
    f16::from_bits(h).to_f32()
}
fn wload(tag: u8, i: usize, j: usize) -> f32 {
    f16::from_f32(((i * 7 + j * 3 + tag as usize * 11) % 17) as f32 / 64.0 - 0.13).to_f32()
}
fn blob(path: &str, vals: &[u16]) {
    let b: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, b).unwrap();
}

fn main() {
    let pid = std::process::id();
    let dir = format!("/tmp/gp_{pid}");
    std::fs::create_dir_all(&dir).unwrap();

    // weights -> blob files (per-layer: tag base = b'A' + layer*8)
    for l in 0..LAYERS {
        let tag = (b'A' + (l * 8) as u8) as u32;
        let mkw = |name: &str, rows: usize, cols: usize, off: u8| {
            blob(
                &format!("{dir}/{name}{l}.bin"),
                &(0..rows * cols)
                    .map(|i| {
                        f16::from_f32(wload((tag + off as u32) as u8, i / cols, i % cols)).to_bits()
                    })
                    .collect::<Vec<_>>(),
            );
        };
        mkw("wq", D, D, 0);
        mkw("wk", D, D, 1);
        mkw("wv", D, D, 2);
        mkw("wo", D, D, 3);
        mkw("wup", FF, D, 4);
        mkw("wg", FF, D, 5);
        mkw("wdn", D, FF, 6);
    }
    blob(
        &format!("{dir}/gam.bin"),
        &[f16::from_f32(1.0).to_bits(); D],
    );
    // half-rotation permutation matrix: Perm[j][k] = delta(j, k ^ 32)
    blob(
        &format!("{dir}/perm.bin"),
        &(0..D * D)
            .map(|i| f16::from_f32(if (i / D) == ((i % D) ^ 32) { 1.0 } else { 0.0 }).to_bits())
            .collect::<Vec<_>>(),
    );
    // Q8_0 quantized vocab head W_lm [VOCAB, D]: int8 codes + blockwise fp16 scales (block=32)
    let wlm_q8: Vec<i8> = (0..VOCAB * D)
        .map(|i| (((i * 41 + 19) % 199) as i32 - 99) as i8)
        .collect();
    let wlm_sc: Vec<f32> = (0..VOCAB * (D / 32))
        .map(|i| f16::from_f32(0.0015 * ((i * 13 % 37) as f32 + 1.0)).to_f32())
        .collect();
    std::fs::write(
        format!("{dir}/wlm_q8.bin"),
        wlm_q8.iter().map(|&c| c as u8).collect::<Vec<u8>>(),
    )
    .unwrap();
    blob(
        &format!("{dir}/wlm_sc.bin"),
        &wlm_sc
            .iter()
            .map(|&s| f16::from_f32(s).to_bits())
            .collect::<Vec<_>>(),
    );

    // ---- generate the MIL: LAYERS stacked transformer layers --------
    let mut mil = String::from("program(1.3)\n{\n    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x,\n                     tensor<fp16, [1, 256, 1, 1]> e_t,\n                     tensor<fp16, [1, 256, 1, 1]> msk,\n");
    for l in 0..LAYERS {
        mil.push_str(&format!(
            "                     tensor<fp16, [1, 256, 1, 64]> kcache{l},\n                     tensor<fp16, [1, 256, 1, 64]> vcache{l},\n"
        ));
    }
    mil.push_str("                     tensor<fp16, [1, 64, 1, 1]> cos_t,\n                     tensor<fp16, [1, 64, 1, 1]> sin_t) {\n");
    for l in 0..LAYERS {
        mil.push_str(&format!(
            "        tensor<fp16, [64, 64]> Wq{l} = const()[val = tensor<fp16, [64, 64]>(BLOBFILE(path = string(\"{dir}/wq{l}.bin\"), offset = uint64(0)))];\n        tensor<fp16, [64, 64]> Wk{l} = const()[val = tensor<fp16, [64, 64]>(BLOBFILE(path = string(\"{dir}/wk{l}.bin\"), offset = uint64(0)))];\n        tensor<fp16, [64, 64]> Wv{l} = const()[val = tensor<fp16, [64, 64]>(BLOBFILE(path = string(\"{dir}/wv{l}.bin\"), offset = uint64(0)))];\n        tensor<fp16, [64, 64]> Wo{l} = const()[val = tensor<fp16, [64, 64]>(BLOBFILE(path = string(\"{dir}/wo{l}.bin\"), offset = uint64(0)))];\n        tensor<fp16, [256, 64]> Wu{l} = const()[val = tensor<fp16, [256, 64]>(BLOBFILE(path = string(\"{dir}/wup{l}.bin\"), offset = uint64(0)))];\n        tensor<fp16, [256, 64]> Wg{l} = const()[val = tensor<fp16, [256, 64]>(BLOBFILE(path = string(\"{dir}/wg{l}.bin\"), offset = uint64(0)))];\n        tensor<fp16, [64, 256]> Wd{l} = const()[val = tensor<fp16, [64, 256]>(BLOBFILE(path = string(\"{dir}/wdn{l}.bin\"), offset = uint64(0)))];\n        tensor<fp16, [64]> gam1_{l} = const()[val = tensor<fp16, [64]>(BLOBFILE(path = string(\"{dir}/gam.bin\"), offset = uint64(0)))];\n        tensor<fp16, [64]> gam2_{l} = const()[val = tensor<fp16, [64]>(BLOBFILE(path = string(\"{dir}/gam.bin\"), offset = uint64(0)))];\n"
        ));
    }
    mil.push_str(&format!("        tensor<int32, []> ax1 = const()[val = int32(1)];\n        tensor<bool, []> bt = const()[val = bool(true)];\n        tensor<fp16, [64, 64]> Perm = const()[val = tensor<fp16, [64, 64]>(BLOBFILE(path = string(\"{dir}/perm.bin\"), offset = uint64(0)))];\n"));
    for l in 0..LAYERS {
        let xin = if l == 0 {
            "x".to_string()
        } else {
            format!("y{}", l - 1)
        };
        mil.push_str(&format!(
            "        tensor<fp16, [1, 64, 1, 1]> xn1_{l} = rms_norm(x = {xin}, w = gam1_{l}, eps = fp16(0.00001))[name = string(\"xn1_{l}\")];\n        tensor<fp16, [1, 1, 1, 64]> q{l} = matmul(x = xn1_{l}, y = Wq{l}, transpose_y = bt)[name = string(\"q{l}\")];\n        tensor<fp16, [1, 1, 1, 64]> kk{l} = matmul(x = xn1_{l}, y = Wk{l}, transpose_y = bt)[name = string(\"kk{l}\")];\n        tensor<fp16, [1, 1, 1, 64]> vv{l} = matmul(x = xn1_{l}, y = Wv{l}, transpose_y = bt)[name = string(\"vv{l}\")];\n        tensor<fp16, [1, 1, 1, 64]> qp{l} = matmul(x = q{l}, y = Perm, transpose_y = bt)[name = string(\"qp{l}\")];\n        tensor<fp16, [1, 64, 1, 1]> qc{l} = mul(x = q{l}, y = cos_t)[name = string(\"qc{l}\")];\n        tensor<fp16, [1, 64, 1, 1]> qps{l} = mul(x = qp{l}, y = sin_t)[name = string(\"qps{l}\")];\n        tensor<fp16, [1, 64, 1, 1]> qr{l} = add(x = qc{l}, y = qps{l})[name = string(\"qr{l}\")];\n        tensor<fp16, [1, 1, 1, 64]> kp{l} = matmul(x = kk{l}, y = Perm, transpose_y = bt)[name = string(\"kp{l}\")];\n        tensor<fp16, [1, 64, 1, 1]> kc{l} = mul(x = kk{l}, y = cos_t)[name = string(\"kc{l}\")];\n        tensor<fp16, [1, 64, 1, 1]> kps{l} = mul(x = kp{l}, y = sin_t)[name = string(\"kps{l}\")];\n        tensor<fp16, [1, 64, 1, 1]> kr{l} = add(x = kc{l}, y = kps{l})[name = string(\"kr{l}\")];\n        tensor<fp16, [1, 256, 1, 64]> ek{l} = matmul(x = e_t, y = kr{l}, transpose_x = bt)[name = string(\"ek{l}\")];\n        tensor<fp16, [1, 256, 1, 64]> ev{l} = matmul(x = e_t, y = vv{l}, transpose_x = bt)[name = string(\"ev{l}\")];\n        tensor<fp16, [1, 256, 1, 64]> knew{l} = add(x = kcache{l}, y = ek{l})[name = string(\"knew{l}\")];\n        tensor<fp16, [1, 256, 1, 64]> vnew{l} = add(x = vcache{l}, y = ev{l})[name = string(\"vnew{l}\")];\n        tensor<fp16, [1, 256, 1, 1]> s{l} = matmul(x = qr{l}, y = knew{l}, transpose_y = bt)[name = string(\"s{l}\")];\n        tensor<fp16, [1, 256, 1, 1]> sm2_{l} = add(x = s{l}, y = msk)[name = string(\"sm2_{l}\")];\n        tensor<fp16, [1, 256, 1, 1]> p{l} = softmax(x = sm2_{l}, axis = ax1)[name = string(\"p{l}\")];\n        tensor<fp16, [1, 1, 1, 64]> aout{l} = matmul(x = p{l}, y = vnew{l}, transpose_x = bt)[name = string(\"aout{l}\")];\n        tensor<fp16, [1, 1, 1, 64]> o{l} = matmul(x = aout{l}, y = Wo{l}, transpose_y = bt)[name = string(\"o{l}\")];\n        tensor<fp16, [1, 64, 1, 1]> r1_{l} = add(x = {xin}, y = o{l})[name = string(\"r1_{l}\")];\n        tensor<fp16, [1, 64, 1, 1]> xn2_{l} = rms_norm(x = r1_{l}, w = gam2_{l}, eps = fp16(0.00001))[name = string(\"xn2_{l}\")];\n        tensor<fp16, [1, 1, 1, 256]> u{l} = matmul(x = xn2_{l}, y = Wu{l}, transpose_y = bt)[name = string(\"u{l}\")];\n        tensor<fp16, [1, 1, 1, 256]> g{l} = matmul(x = xn2_{l}, y = Wg{l}, transpose_y = bt)[name = string(\"g{l}\")];\n        tensor<fp16, [1, 256, 1, 1]> gs{l} = sigmoid(x = g{l})[name = string(\"gs{l}\")];\n        tensor<fp16, [1, 256, 1, 1]> gsl{l} = mul(x = g{l}, y = gs{l})[name = string(\"gsl{l}\")];\n        tensor<fp16, [1, 256, 1, 1]> fu{l} = mul(x = u{l}, y = gsl{l})[name = string(\"fu{l}\")];\n        tensor<fp16, [1, 1, 1, 64]> dn{l} = matmul(x = fu{l}, y = Wd{l}, transpose_y = bt)[name = string(\"dn{l}\")];\n        tensor<fp16, [1, 64, 1, 1]> y{l} = add(x = r1_{l}, y = dn{l})[name = string(\"y{l}\")];\n"
        ));
    }
    mil.push_str(&format!(
        "        tensor<fp16, [64]> gam_f = const()[val = tensor<fp16, [64]>(BLOBFILE(path = string(\"{dir}/gam.bin\"), offset = uint64(0)))];\n        tensor<int8, [256, 64]> Wlm_q = const()[val = tensor<int8, [256, 64]>(BLOBFILE(path = string(\"{dir}/wlm_q8.bin\"), offset = uint64(0)))];\n        tensor<fp16, [256, 2]> Wlm_s = const()[val = tensor<fp16, [256, 2]>(BLOBFILE(path = string(\"{dir}/wlm_sc.bin\"), offset = uint64(0)))];\n        tensor<fp16, [256, 64]> Wlm = constexpr_blockwise_shift_scale(data = Wlm_q, scale = Wlm_s)[name = string(\"Wlm\")];\n        tensor<fp16, [1, 64, 1, 1]> xn_f = rms_norm(x = y{}, w = gam_f, eps = fp16(0.00001))[name = string(\"xn_f\")];\n        tensor<fp16, [1, 1, 1, 256]> logits = matmul(x = xn_f, y = Wlm, transpose_y = bt)[name = string(\"logits\")];\n    }} -> (y{}, logits);\n}}\n",
        LAYERS - 1,
        LAYERS - 1
    ));
    let mil_path = format!("{dir}/layer.mil");
    std::fs::write(&mil_path, &mil).unwrap();

    let plan = compile_graph(&mil_path, &format!("{dir}/out"))
        .unwrap_or_else(|e| panic!("compile_graph: {e}"));
    println!(
        "graph: {} nodes ({} prog, {} host), {} externals, order {:?}",
        plan.nodes.len(),
        plan.nodes
            .iter()
            .filter(|n| matches!(n, mil_infer::graph::Node::Prog { .. }))
            .count(),
        plan.nodes
            .iter()
            .filter(|n| matches!(n, mil_infer::graph::Node::Host(_)))
            .count(),
        plan.externals.len(),
        plan.order
    );
    let mut rt = GraphRuntime::load(plan).unwrap_or_else(|e| panic!("load: {e}"));

    // --- autoregressive decode loop (prompt tok=7 -> greedy decode) ----
    let embed = |tok: usize| -> Vec<f32> {
        (0..D)
            .map(|i| f16::from_f32(((tok * 13 + i * 5 + 3) % 19) as f32 / 16.0 - 0.55).to_f32())
            .collect()
    };
    let mut kref: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; D]; T_MAX]; LAYERS];
    let mut vref: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; D]; T_MAX]; LAYERS];
    let mut bad = 0usize;
    let mut tok_hw = 7usize;
    let mut tok_ref = 7usize;
    let mut seq_hw = vec![tok_hw];
    let mut seq_ref = vec![tok_ref];
    let mut step0_top1 = 0usize;
    for step in 0..4usize {
        let x = embed(tok_hw);
        let xs = x.clone();
        rt.write_ext("x", move |b, d| {
            for i in 0..D {
                let o = d.offset(0, 0, i as u64, 0, 0);
                if o + 2 <= b.len() {
                    b[o..o + 2].copy_from_slice(&f16::from_f32(xs[i]).to_le_bytes());
                }
            }
        });
        rt.write_ext("e_t", move |b, d| {
            for i in 0..T_MAX {
                let o = d.offset(0, 0, i as u64, 0, 0);
                if o + 2 <= b.len() {
                    b[o..o + 2].copy_from_slice(
                        &f16::from_f32(if i == step { 1.0 } else { 0.0 }).to_le_bytes(),
                    );
                }
            }
        });
        // RoPE tables for this position: theta_i = base^{-2i/d}
        let (half, base) = (D / 2, 10000f32);
        let ang = |i: usize| (step as f32) * base.powf(-2.0 * (i % half) as f32 / D as f32);
        let cosv: Vec<f32> = (0..D).map(|i| ang(i).cos()).collect();
        let sinv: Vec<f32> = (0..D)
            .map(|i| if i < half { -1.0 } else { 1.0 } * ang(i).sin())
            .collect();
        let cv = cosv.clone();
        rt.write_ext("cos_t", move |b, d| {
            for i in 0..D {
                let o = d.offset(0, 0, i as u64, 0, 0);
                if o + 2 <= b.len() {
                    b[o..o + 2].copy_from_slice(&f16::from_f32(cv[i]).to_le_bytes());
                }
            }
        });
        let sv = sinv.clone();
        rt.write_ext("sin_t", move |b, d| {
            for i in 0..D {
                let o = d.offset(0, 0, i as u64, 0, 0);
                if o + 2 <= b.len() {
                    b[o..o + 2].copy_from_slice(&f16::from_f32(sv[i]).to_le_bytes());
                }
            }
        });
        rt.write_ext("msk", move |b, d| {
            for i in 0..T_MAX {
                let o = d.offset(0, 0, i as u64, 0, 0);
                if o + 2 <= b.len() {
                    b[o..o + 2].copy_from_slice(
                        &f16::from_f32(if i <= step { 0.0 } else { -30000.0 }).to_le_bytes(),
                    );
                }
            }
        });
        let t0 = std::time::Instant::now();
        rt.run().expect("run");
        if std::env::var("GP_DIAG").is_ok() {
            for nm in [
                "q0", "qp0", "qr0", "kk0", "kp0", "kr0", "s0", "p0", "y0", "q1", "kr1", "p1",
            ] {
                if let Some(ni) = rt.node_named(nm) {
                    let d = rt.out_desc(ni);
                    let b = rt.output(ni);
                    let mut v = Vec::new();
                    for c in 0..6u64 {
                        let o = if d.k > 1 {
                            d.offset(0, 0, c, 0, 0)
                        } else {
                            d.offset(0, 0, 0, 0, c)
                        };
                        if o + 2 <= b.len() {
                            v.push(f32_of(u16::from_le_bytes([b[o], b[o + 1]])));
                        }
                    }
                    println!("  {nm}: {:?} (desc k{}h{}w{})", v, d.k, d.h, d.w);
                }
            }
        }
        if step == 2 {
            println!("run() took {:?}", t0.elapsed());
        }

        // ---- CPU reference ----
        let mm = |wtag: u8, v: &[f32], n: usize, k: usize| -> Vec<f32> {
            (0..n)
                .map(|j| (0..k).map(|i| v[i] * wload(wtag, j, i)).sum::<f32>())
                .collect()
        };
        let rms = |v: &[f32]| {
            (v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32 + EPS)
                .sqrt()
                .recip()
        };
        let mut x_cur = x.clone();
        for l in 0..LAYERS {
            let tag = |off: u8| (b'A' + (l * 8) as u8).wrapping_add(off);
            let xnr: Vec<f32> = x_cur.iter().map(|a| a * rms(&x_cur)).collect();
            let mut qr = mm(tag(0), &xnr, D, D);
            let mut kr = mm(tag(1), &xnr, D, D);
            let vr = mm(tag(2), &xnr, D, D);
            // rope: r[i] = v[i]*cos + v[i^half]*sin_signed
            let rope = |v: &mut Vec<f32>| {
                let orig = v.clone();
                for i in 0..D {
                    v[i] = orig[i] * cosv[i] + orig[i ^ half] * sinv[i];
                }
            };
            rope(&mut qr);
            rope(&mut kr);
            kref[l][step] = kr;
            vref[l][step] = vr;
            let sc_r: Vec<f32> = (0..T_MAX)
                .map(|t| (0..D).map(|i| qr[i] * kref[l][t][i]).sum())
                .collect();
            let mx = sc_r[..=step]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let den: f32 = sc_r[..=step].iter().map(|v| (v - mx).exp()).sum();
            let p: Vec<f32> = (0..T_MAX)
                .map(|t| {
                    if t <= step {
                        (sc_r[t] - mx).exp() / den
                    } else {
                        0.0
                    }
                })
                .collect();
            let out: Vec<f32> = (0..D)
                .map(|j| (0..T_MAX).map(|t| p[t] * vref[l][t][j]).sum())
                .collect();
            let attn = mm(tag(3), &out, D, D);
            let r1r: Vec<f32> = x_cur.iter().zip(&attn).map(|(a, b)| a + b).collect();
            let xn2r: Vec<f32> = r1r.iter().map(|a| a * rms(&r1r)).collect();
            let upr = mm(tag(4), &xn2r, FF, D);
            let gr = mm(tag(5), &xn2r, FF, D);
            let slr: Vec<f32> = upr
                .iter()
                .zip(&gr)
                .map(|(u, g)| u * g / (1.0 + (-g).exp()))
                .collect();
            let dr = mm(tag(6), &slr, D, FF);
            x_cur = r1r.iter().zip(&dr).map(|(a, b)| a + b).collect();
        }
        let yr = x_cur;
        let xnf_r: Vec<f32> = yr.iter().map(|a| a * rms(&yr)).collect();
        let logits_r: Vec<f32> = (0..VOCAB)
            .map(|j| {
                (0..D)
                    .map(|i| {
                        let w =
                            f16::from_f32(wlm_q8[j * D + i] as f32 * wlm_sc[j * (D / 32) + i / 32])
                                .to_f32();
                        xnf_r[i] * w
                    })
                    .sum()
            })
            .collect();

        let yn = rt.out_node(&format!("y{}", LAYERS - 1)).expect("y node");
        let dy = rt.out_desc(yn);
        let zb = rt.output(yn);
        let ymax = yr.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for j in 0..D {
            let o = dy.offset(0, 0, j as u64, 0, 0);
            let hw = f32_of(u16::from_le_bytes([zb[o], zb[o + 1]]));
            if (hw - yr[j]).abs() > yr[j].abs() * 0.05 + ymax * 0.015 {
                bad += 1;
                if bad < 8 {
                    println!("step{step} y[{j}]: hw={hw:.4} ref={:.4}", yr[j]);
                }
            }
        }
        let ln = rt.out_node("logits").expect("logits node");
        let dl = rt.out_desc(ln);
        let lb = rt.output(ln);
        let lmax = logits_r.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let mut best_hw = (0usize, f32::NEG_INFINITY);
        let mut best_ref = (0usize, f32::NEG_INFINITY);
        for j in 0..VOCAB {
            let o = dl.offset(0, 0, j as u64, 0, 0);
            let hw = f32_of(u16::from_le_bytes([lb[o], lb[o + 1]]));
            if hw > best_hw.1 {
                best_hw = (j, hw);
            }
            if logits_r[j] > best_ref.1 {
                best_ref = (j, logits_r[j]);
            }
            if (hw - logits_r[j]).abs() > logits_r[j].abs() * 0.05 + lmax * 0.015 {
                bad += 1;
                if bad < 8 {
                    println!("step{step} logit[{j}]: hw={hw:.4} ref={:.4}", logits_r[j]);
                }
            }
        }
        if best_hw.0 != best_ref.0 {
            bad += 1;
            println!(
                "step{step} top1 mismatch: hw={} ({:.4}) ref={} ({:.4})",
                best_hw.0, best_hw.1, best_ref.0, best_ref.1
            );
        }
        if step == 0 {
            step0_top1 = best_hw.0;
        }
        tok_hw = best_hw.0;
        tok_ref = best_ref.0;
        seq_hw.push(tok_hw);
        seq_ref.push(tok_ref);
        println!(
            "step {step}: next_tok={} (logit={:.4} ref={:.4}) y0..3 hw=[{:.4},{:.4},{:.4}] ref=[{:.4},{:.4},{:.4}]",
            best_hw.0,
            best_hw.1,
            best_ref.1,
            f32_of(u16::from_le_bytes([
                zb[dy.offset(0, 0, 0, 0, 0)],
                zb[dy.offset(0, 0, 0, 0, 0) + 1]
            ])),
            f32_of(u16::from_le_bytes([
                zb[dy.offset(0, 0, 1, 0, 0)],
                zb[dy.offset(0, 0, 1, 0, 0) + 1]
            ])),
            f32_of(u16::from_le_bytes([
                zb[dy.offset(0, 0, 2, 0, 0)],
                zb[dy.offset(0, 0, 2, 0, 0) + 1]
            ])),
            yr[0],
            yr[1],
            yr[2]
        );
    }
    println!("generated tokens: hw={seq_hw:?} ref={seq_ref:?}");
    // Verify reset_state zeroes the KV caches and reproduces step 0's exact token
    rt.reset_state();
    let x0 = embed(7);
    rt.write_ext("x", move |b, d| {
        for i in 0..D {
            let o = d.offset(0, 0, i as u64, 0, 0);
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(&f16::from_f32(x0[i]).to_le_bytes());
            }
        }
    });
    rt.write_ext("e_t", |b, d| {
        for i in 0..T_MAX {
            let o = d.offset(0, 0, i as u64, 0, 0);
            if o + 2 <= b.len() {
                b[o..o + 2]
                    .copy_from_slice(&f16::from_f32(if i == 0 { 1.0 } else { 0.0 }).to_le_bytes());
            }
        }
    });
    rt.write_ext("cos_t", |b, d| {
        for i in 0..D {
            let o = d.offset(0, 0, i as u64, 0, 0);
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
            }
        }
    });
    rt.write_ext("sin_t", |b, d| {
        for i in 0..D {
            let o = d.offset(0, 0, i as u64, 0, 0);
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(&f16::from_f32(0.0).to_le_bytes());
            }
        }
    });
    rt.write_ext("msk", |b, d| {
        for i in 0..T_MAX {
            let o = d.offset(0, 0, i as u64, 0, 0);
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(
                    &f16::from_f32(if i == 0 { 0.0 } else { -30000.0 }).to_le_bytes(),
                );
            }
        }
    });
    rt.run().expect("run after reset");
    let ln = rt.out_node("logits").expect("logits node");
    let dl = rt.out_desc(ln);
    let lb = rt.output(ln);
    let mut reset_top1 = (0usize, f32::NEG_INFINITY);
    for j in 0..VOCAB {
        let o = dl.offset(0, 0, j as u64, 0, 0);
        let hw = f32_of(u16::from_le_bytes([lb[o], lb[o + 1]]));
        if hw > reset_top1.1 {
            reset_top1 = (j, hw);
        }
    }
    if reset_top1.0 != step0_top1 {
        bad += 1;
        println!(
            "reset_state top1 mismatch: {} vs {}",
            reset_top1.0, step0_top1
        );
    }
    println!("GRAPH-LAYER mismatches: {bad}");
}
