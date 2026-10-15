//! graph_probe — compile a full transformer-layer MIL through
//! `compile_graph` (the scheduler) and execute via `GraphRuntime`,
//! verifying against a CPU reference over 3 masked decode steps.
//!
//! The MIL carries only the logical graph; repacks, the rmsnorm
//! expansion, and the host-rsqrt bridge are inserted by the compiler.

use half::f16;
use mil_infer::graph::{compile_graph, GraphRuntime};
use std::path::Path;

const T_MAX: usize = 256;
const D: usize = 64;
const FF: usize = 256;
const EPS: f32 = 1e-5;

fn f32_of(h: u16) -> f32 {
    f16::from_bits(h).to_f32()
}
fn wload(tag: u8, i: usize, j: usize) -> f32 {
    ((i * 7 + j * 3 + tag as usize * 11) % 17) as f32 / 64.0 - 0.13
}
fn blob(path: &str, vals: &[u16]) {
    let b: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, b).unwrap();
}

fn main() {
    let pid = std::process::id();
    let dir = format!("/tmp/gp_{pid}");
    std::fs::create_dir_all(&dir).unwrap();

    // weights -> blob files
    blob(
        &format!("{dir}/wq.bin"),
        &(0..D * D)
            .map(|i| f16::from_f32(wload(b'q', i / D, i % D)).to_bits())
            .collect::<Vec<_>>(),
    );
    blob(
        &format!("{dir}/wk.bin"),
        &(0..D * D)
            .map(|i| f16::from_f32(wload(b'k', i / D, i % D)).to_bits())
            .collect::<Vec<_>>(),
    );
    blob(
        &format!("{dir}/wv.bin"),
        &(0..D * D)
            .map(|i| f16::from_f32(wload(b'v', i / D, i % D)).to_bits())
            .collect::<Vec<_>>(),
    );
    blob(
        &format!("{dir}/wo.bin"),
        &(0..D * D)
            .map(|i| f16::from_f32(wload(b'o', i / D, i % D)).to_bits())
            .collect::<Vec<_>>(),
    );
    blob(
        &format!("{dir}/wup.bin"),
        &(0..FF * D)
            .map(|i| f16::from_f32(wload(b'u', i / D, i % D)).to_bits())
            .collect::<Vec<_>>(),
    );
    blob(
        &format!("{dir}/wg.bin"),
        &(0..FF * D)
            .map(|i| f16::from_f32(wload(b'g', i / D, i % D)).to_bits())
            .collect::<Vec<_>>(),
    );
    blob(
        &format!("{dir}/wdn.bin"),
        &(0..D * FF)
            .map(|i| f16::from_f32(wload(b'd', i / FF, i % FF)).to_bits())
            .collect::<Vec<_>>(),
    );
    blob(
        &format!("{dir}/gam.bin"),
        &vec![f16::from_f32(1.0).to_bits(); D],
    );

    let mil = format!(
        r#"program(1.3)
{{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x,
                     tensor<fp16, [1, 256, 1, 1]> e_t,
                     tensor<fp16, [1, 256, 1, 1]> msk,
                     tensor<fp16, [1, 256, 1, 64]> kcache,
                     tensor<fp16, [1, 256, 1, 64]> vcache) {{
        tensor<fp16, [64, 64]> Wq = const()[val = tensor<fp16, [64, 64]>(BLOBFILE(path = string("{dir}/wq.bin"), offset = uint64(0)))];
        tensor<fp16, [64, 64]> Wk = const()[val = tensor<fp16, [64, 64]>(BLOBFILE(path = string("{dir}/wk.bin"), offset = uint64(0)))];
        tensor<fp16, [64, 64]> Wv = const()[val = tensor<fp16, [64, 64]>(BLOBFILE(path = string("{dir}/wv.bin"), offset = uint64(0)))];
        tensor<fp16, [64, 64]> Wo = const()[val = tensor<fp16, [64, 64]>(BLOBFILE(path = string("{dir}/wo.bin"), offset = uint64(0)))];
        tensor<fp16, [256, 64]> Wu = const()[val = tensor<fp16, [128, 64]>(BLOBFILE(path = string("{dir}/wup.bin"), offset = uint64(0)))];
        tensor<fp16, [256, 64]> Wg = const()[val = tensor<fp16, [128, 64]>(BLOBFILE(path = string("{dir}/wg.bin"), offset = uint64(0)))];
        tensor<fp16, [64, 256]> Wd = const()[val = tensor<fp16, [64, 128]>(BLOBFILE(path = string("{dir}/wdn.bin"), offset = uint64(0)))];
        tensor<fp16, [64]> gam1 = const()[val = tensor<fp16, [64]>(BLOBFILE(path = string("{dir}/gam.bin"), offset = uint64(0)))];
        tensor<fp16, [64]> gam2 = const()[val = tensor<fp16, [64]>(BLOBFILE(path = string("{dir}/gam.bin"), offset = uint64(0)))];
        tensor<int32, []> ax1 = const()[val = int32(1)];
        tensor<bool, []> bt = const()[val = bool(true)];

        tensor<fp16, [1, 64, 1, 1]> xn1 = rms_norm(x = x, w = gam1, eps = fp16(0.00001))[name = string("xn1")];
        tensor<fp16, [1, 1, 1, 64]> q = matmul(x = xn1, y = Wq, transpose_y = bt)[name = string("q")];
        tensor<fp16, [1, 1, 1, 64]> kk = matmul(x = xn1, y = Wk, transpose_y = bt)[name = string("kk")];
        tensor<fp16, [1, 1, 1, 64]> vv = matmul(x = xn1, y = Wv, transpose_y = bt)[name = string("vv")];
        tensor<fp16, [1, 256, 1, 64]> ek = matmul(x = e_t, y = kk, transpose_x = bt)[name = string("ek")];
        tensor<fp16, [1, 256, 1, 64]> ev = matmul(x = e_t, y = vv, transpose_x = bt)[name = string("ev")];
        tensor<fp16, [1, 256, 1, 64]> knew = add(x = kcache, y = ek)[name = string("knew")];
        tensor<fp16, [1, 256, 1, 64]> vnew = add(x = vcache, y = ev)[name = string("vnew")];
        tensor<fp16, [1, 256, 1, 1]> s = matmul(x = q, y = knew, transpose_y = bt)[name = string("s")];
        tensor<fp16, [1, 256, 1, 1]> sm2 = add(x = s, y = msk)[name = string("sm2")];
        tensor<fp16, [1, 256, 1, 1]> p = softmax(x = sm2, axis = ax1)[name = string("p")];
        tensor<fp16, [1, 1, 1, 64]> aout = matmul(x = p, y = vnew, transpose_x = bt)[name = string("aout")];
        tensor<fp16, [1, 1, 1, 64]> o = matmul(x = aout, y = Wo, transpose_y = bt)[name = string("o")];
        tensor<fp16, [1, 64, 1, 1]> r1 = add(x = x, y = o)[name = string("r1")];
        tensor<fp16, [1, 64, 1, 1]> xn2 = rms_norm(x = r1, w = gam2, eps = fp16(0.00001))[name = string("xn2")];
        tensor<fp16, [1, 1, 1, 256]> u = matmul(x = xn2, y = Wu, transpose_y = bt)[name = string("u")];
        tensor<fp16, [1, 1, 1, 256]> g = matmul(x = xn2, y = Wg, transpose_y = bt)[name = string("g")];
        tensor<fp16, [1, 256, 1, 1]> gs = sigmoid(x = g)[name = string("gs")];
        tensor<fp16, [1, 256, 1, 1]> gsl = mul(x = g, y = gs)[name = string("gsl")];
        tensor<fp16, [1, 256, 1, 1]> fu = mul(x = u, y = gsl)[name = string("fu")];
        tensor<fp16, [1, 1, 1, 64]> dn = matmul(x = fu, y = Wd, transpose_y = bt)[name = string("dn")];
        tensor<fp16, [1, 64, 1, 1]> y = add(x = r1, y = dn)[name = string("y")];
    }} -> (y);
}}
"#
    );
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

    // --- decode loop ---------------------------------------------------
    let mut kref: Vec<Vec<f32>> = vec![vec![0.0; D]; T_MAX];
    let mut vref: Vec<Vec<f32>> = vec![vec![0.0; D]; T_MAX];
    let mut bad = 0usize;
    for step in 0..3usize {
        let x: Vec<f32> = (0..D)
            .map(|i| ((step * 13 + i * 5) % 11) as f32 / 16.0 - 0.3)
            .collect();
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
        rt.run().expect("run");

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
        let xnr: Vec<f32> = x.iter().map(|a| a * rms(&x)).collect();
        let qr = mm(b'q', &xnr, D, D);
        let kr = mm(b'k', &xnr, D, D);
        let vr = mm(b'v', &xnr, D, D);
        kref[step] = kr;
        vref[step] = vr;
        let sc_r: Vec<f32> = (0..T_MAX)
            .map(|t| (0..D).map(|i| qr[i] * kref[t][i]).sum())
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
            .map(|j| (0..T_MAX).map(|t| p[t] * vref[t][j]).sum())
            .collect();
        let attn = mm(b'o', &out, D, D);
        let r1r: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
        let xn2r: Vec<f32> = r1r.iter().map(|a| a * rms(&r1r)).collect();
        let upr = mm(b'u', &xn2r, FF, D);
        let gr = mm(b'g', &xn2r, FF, D);
        let slr: Vec<f32> = upr
            .iter()
            .zip(&gr)
            .map(|(u, g)| u * g / (1.0 + (-g).exp()))
            .collect();
        let dr = mm(b'd', &slr, D, FF);
        let yr: Vec<f32> = r1r.iter().zip(&dr).map(|(a, b)| a + b).collect();

        let yn = rt.out_node("y").expect("y node");
        let dy = rt.out_desc(yn);
        let zb = rt.output(yn);
        for j in 0..D {
            let o = dy.offset(0, 0, j as u64, 0, 0);
            let hw = f32_of(u16::from_le_bytes([zb[o], zb[o + 1]]));
            if (hw - yr[j]).abs() > yr[j].abs() * 0.05 + 0.03 {
                bad += 1;
                if bad < 8 {
                    println!("step{step} y[{j}]: hw={hw:.4} ref={:.4}", yr[j]);
                }
            }
        }
        println!(
            "step {step}: y0..3 hw=[{:.4},{:.4},{:.4}] ref=[{:.4},{:.4},{:.4}]",
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
    println!("GRAPH-LAYER mismatches: {bad}");
}
