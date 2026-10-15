//! layer_probe — a full transformer-layer decode step on ANE, all
//! self-emitted programs chained by IOSurface adoption:
//!
//!   norm1: sq=x*x -> repack -> matvec(ones) -> host rsqrt -> xn=x*rvec
//!          -> repack -> xn_c        (gamma folded into Wq/Wk/Wv)
//!   attn : q,k,v = xn_c*W   -> repack -> K/V += outer(e_t,row)
//!          -> scores=q*K^T -> +mask -> softmax -> attn*V -> *Wo
//!   res1 : r1 = x + attnout                 (strided add)
//!   norm2: same rmsnorm chain on r1
//!   ffn  : up,gate = xn2*Wu,*Wg -> silu=gate*sigmoid(gate)... actually
//!          silu = up * (gate*sigmoid(gate)) -> repack128 -> *Wd
//!   res2 : y = r1 + down
//!
//! Verified against a CPU reference over sequential decode steps.

use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::{
    emit_add4d64, emit_attn_v, emit_avw, emit_elementwise2, emit_maskadd64, emit_matmul_ns,
    emit_mul128, emit_outer, emit_repack128, emit_repack64, emit_sigmoid128, emit_softmax_ch, EwOp,
};
use std::path::Path;

fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
fn wload(tag: u8, i: usize, j: usize) -> f32 {
    ((i * 7 + j * 3 + tag as usize * 11) % 17) as f32 / 64.0 - 0.13
}
fn load(img: Vec<u8>, path: &String) -> EspressoProgram {
    std::fs::write(path, &img).unwrap();
    EspressoProgram::load_precompiled(Path::new(path), &[]).expect(path)
}
fn rd16(b: &[u8], o: usize) -> f32 {
    f32_of(u16::from_le_bytes([b[o], b[o + 1]]))
}
fn wr16(b: &mut [u8], o: usize, v: f32) {
    if o + 2 <= b.len() {
        b[o..o + 2].copy_from_slice(&f16(v).to_le_bytes());
    }
}

fn main() {
    let pid = std::process::id();
    let (t_max, d, ff) = (64usize, 64usize, 128usize);
    let eps = 1e-5f32;

    // weights baked into artifacts (fp16 bits, [N,K] row-major)
    let mut wsum = vec![0u16; 32 * d];
    for k in 0..d {
        wsum[k] = f16(1.0); // row 0 = ones -> z[0] = sum
    }
    let wq: Vec<u16> = (0..d * d).map(|i| f16(wload(b'q', i / d, i % d))).collect();
    let wk: Vec<u16> = (0..d * d).map(|i| f16(wload(b'k', i / d, i % d))).collect();
    let wv: Vec<u16> = (0..d * d).map(|i| f16(wload(b'v', i / d, i % d))).collect();
    let wo: Vec<u16> = (0..d * d).map(|i| f16(wload(b'o', i / d, i % d))).collect();
    let wup: Vec<u16> = (0..ff * d)
        .map(|i| f16(wload(b'u', i / d, i % d)))
        .collect();
    let wg: Vec<u16> = (0..ff * d)
        .map(|i| f16(wload(b'g', i / d, i % d)))
        .collect();
    let wdn: Vec<u16> = (0..d * ff)
        .map(|i| f16(wload(b'd', i / ff, i % ff)))
        .collect();

    // --- emit + load all programs ---
    let mk = |tag: &str, img: Vec<u8>| {
        let p = format!("/tmp/lp_{tag}_{pid}.macho");
        load(img, &p)
    };
    // norm chain (x2 for the two norms — programs own their adopted surfaces)
    let mut sq1 = mk(
        "sq1",
        emit_elementwise2(
            EwOp::Mul,
            64,
            "/tmp/lp.mil",
            &format!("/tmp/lp_sq1_{pid}.macho"),
        ),
    );
    let mut sqc1 = mk(
        "sqc1",
        emit_repack64("/tmp/lp.mil", &format!("/tmp/lp_sqc1_{pid}.macho")),
    );
    let mut sum1 = mk(
        "sum1",
        emit_matmul_ns(
            d,
            32,
            &wsum,
            "/tmp/lp.mil",
            &format!("/tmp/lp_sum1_{pid}.macho"),
        )
        .unwrap(),
    );
    let mut xn1 = mk(
        "xn1",
        emit_elementwise2(
            EwOp::Mul,
            64,
            "/tmp/lp.mil",
            &format!("/tmp/lp_xn1_{pid}.macho"),
        ),
    );
    let mut xnc1 = mk(
        "xnc1",
        emit_repack64("/tmp/lp.mil", &format!("/tmp/lp_xnc1_{pid}.macho")),
    );
    // attention
    let mut pq = mk(
        "pq",
        emit_matmul_ns(d, d, &wq, "/tmp/lp.mil", &format!("/tmp/lp_pq_{pid}.macho")).unwrap(),
    );
    let mut pk = mk(
        "pk",
        emit_matmul_ns(d, d, &wk, "/tmp/lp.mil", &format!("/tmp/lp_pk_{pid}.macho")).unwrap(),
    );
    let mut pv = mk(
        "pv",
        emit_matmul_ns(d, d, &wv, "/tmp/lp.mil", &format!("/tmp/lp_pv_{pid}.macho")).unwrap(),
    );
    let mut rq = mk(
        "rq",
        emit_repack64("/tmp/lp.mil", &format!("/tmp/lp_rq_{pid}.macho")),
    );
    let mut rk = mk(
        "rk",
        emit_repack64("/tmp/lp.mil", &format!("/tmp/lp_rk_{pid}.macho")),
    );
    let mut rv = mk(
        "rv",
        emit_repack64("/tmp/lp.mil", &format!("/tmp/lp_rv_{pid}.macho")),
    );
    let mut ok = mk(
        "ok",
        emit_outer("/tmp/lp.mil", &format!("/tmp/lp_ok_{pid}.macho")),
    );
    let mut ov = mk(
        "ov",
        emit_outer("/tmp/lp.mil", &format!("/tmp/lp_ov_{pid}.macho")),
    );
    let mut ak = mk(
        "ak",
        emit_add4d64("/tmp/lp.mil", &format!("/tmp/lp_ak_{pid}.macho")),
    );
    let mut av = mk(
        "av",
        emit_add4d64("/tmp/lp.mil", &format!("/tmp/lp_av_{pid}.macho")),
    );
    let mut sc = mk(
        "sc",
        emit_avw(
            t_max as u32,
            d as u32,
            "/tmp/lp.mil",
            &format!("/tmp/lp_sc_{pid}.macho"),
        ),
    );
    let mut mkm = mk(
        "mk",
        emit_maskadd64("/tmp/lp.mil", &format!("/tmp/lp_mk_{pid}.macho")),
    );
    let mut sm = mk(
        "sm",
        emit_softmax_ch(
            t_max as u32,
            "/tmp/lp.mil",
            &format!("/tmp/lp_sm_{pid}.macho"),
        )
        .unwrap(),
    );
    let mut ao = mk(
        "ao",
        emit_attn_v(
            t_max as u32,
            d as u32,
            "/tmp/lp.mil",
            &format!("/tmp/lp_ao_{pid}.macho"),
        )
        .unwrap(),
    );
    let mut po = mk(
        "po",
        emit_matmul_ns(d, d, &wo, "/tmp/lp.mil", &format!("/tmp/lp_po_{pid}.macho")).unwrap(),
    );
    let mut r1p = mk(
        "r1",
        emit_maskadd64("/tmp/lp.mil", &format!("/tmp/lp_r1_{pid}.macho")),
    );
    // norm2
    let mut sq2 = mk(
        "sq2",
        emit_elementwise2(
            EwOp::Mul,
            64,
            "/tmp/lp.mil",
            &format!("/tmp/lp_sq2_{pid}.macho"),
        ),
    );
    let mut sqc2 = mk(
        "sqc2",
        emit_repack64("/tmp/lp.mil", &format!("/tmp/lp_sqc2_{pid}.macho")),
    );
    let mut sum2 = mk(
        "sum2",
        emit_matmul_ns(
            d,
            32,
            &wsum,
            "/tmp/lp.mil",
            &format!("/tmp/lp_sum2_{pid}.macho"),
        )
        .unwrap(),
    );
    let mut xn2 = mk(
        "xn2",
        emit_elementwise2(
            EwOp::Mul,
            64,
            "/tmp/lp.mil",
            &format!("/tmp/lp_xn2_{pid}.macho"),
        ),
    );
    let mut xnc2 = mk(
        "xnc2",
        emit_repack64("/tmp/lp.mil", &format!("/tmp/lp_xnc2_{pid}.macho")),
    );
    // ffn
    let mut up = mk(
        "up",
        emit_matmul_ns(
            d,
            ff,
            &wup,
            "/tmp/lp.mil",
            &format!("/tmp/lp_up_{pid}.macho"),
        )
        .unwrap(),
    );
    let mut gate = mk(
        "gate",
        emit_matmul_ns(
            d,
            ff,
            &wg,
            "/tmp/lp.mil",
            &format!("/tmp/lp_gt_{pid}.macho"),
        )
        .unwrap(),
    );
    let mut sig = mk(
        "sig",
        emit_sigmoid128("/tmp/lp.mil", &format!("/tmp/lp_sig_{pid}.macho")),
    );
    let mut slu = mk(
        "slu",
        emit_mul128("/tmp/lp.mil", &format!("/tmp/lp_slu_{pid}.macho")),
    );
    let mut upd = mk(
        "upd",
        emit_mul128("/tmp/lp.mil", &format!("/tmp/lp_upd_{pid}.macho")),
    );
    let mut slc = mk(
        "slc",
        emit_repack128("/tmp/lp.mil", &format!("/tmp/lp_slc_{pid}.macho")),
    );
    let mut down = mk(
        "down",
        emit_matmul_ns(
            ff,
            d,
            &wdn,
            "/tmp/lp.mil",
            &format!("/tmp/lp_dn_{pid}.macho"),
        )
        .unwrap(),
    );
    let mut yy = mk(
        "yy",
        emit_maskadd64("/tmp/lp.mil", &format!("/tmp/lp_yy_{pid}.macho")),
    );

    // print layouts once
    for (n, m) in [
        ("sq", &sq1),
        ("sqc", &sqc1),
        ("sum", &sum1),
        ("sig", &sig),
        ("slu", &slu),
        ("slc", &slc),
        ("down", &down),
        ("r1", &r1p),
    ] {
        let ins: Vec<String> = m
            .input_descs
            .iter()
            .map(|x| {
                format!(
                    "{}:n{}k{}h{}w{}ps{}",
                    x.name, x.n, x.k, x.h, x.w, x.plane_stride
                )
            })
            .collect();
        let outs: Vec<String> = m
            .output_descs
            .iter()
            .map(|x| {
                format!(
                    "{}:n{}k{}h{}w{}ps{}",
                    x.name, x.n, x.k, x.h, x.w, x.plane_stride
                )
            })
            .collect();
        println!("{n}: in {ins:?} out {outs:?}");
    }

    // --- wire surfaces ---
    // x residual stream lives on sq1's input0 surface (strided [1,64,1,1])
    let xs = sq1.input_surf(0);
    sq1.adopt_input(1, xs).expect("sq y");
    sqc1.adopt_input(0, sq1.output_surf(0)).expect("sqc");
    sum1.adopt_input(0, sqc1.output_surf(0)).expect("sum x");
    xn1.adopt_input(0, xs).expect("xn x");
    xnc1.adopt_input(0, xn1.output_surf(0)).expect("xnc");
    let xnc1_s = xnc1.output_surf(0);
    pq.adopt_input(0, xnc1_s).expect("pq x");
    pk.adopt_input(0, xnc1_s).expect("pk x");
    pv.adopt_input(0, xnc1_s).expect("pv x");
    rq.adopt_input(0, pq.output_surf(0)).expect("rq");
    rk.adopt_input(0, pk.output_surf(0)).expect("rk");
    rv.adopt_input(0, pv.output_surf(0)).expect("rv");
    ok.adopt_input(1, rk.output_surf(0)).expect("k row");
    ov.adopt_input(1, rv.output_surf(0)).expect("v row");
    ak.alias_out_to_in(0, 0);
    av.alias_out_to_in(0, 0);
    for b in ak.input_mut(0).iter_mut() {
        *b = 0;
    }
    for b in av.input_mut(0).iter_mut() {
        *b = 0;
    }
    ak.adopt_input(1, ok.output_surf(0)).expect("k tmp");
    av.adopt_input(1, ov.output_surf(0)).expect("v tmp");
    let scores = sc.output_surf(0);
    sc.adopt_input(0, rq.output_surf(0)).expect("q");
    sc.adopt_input(1, ak.output_surf(0)).expect("K");
    mkm.adopt_input(0, scores).expect("mk in");
    mkm.adopt_output(0, scores).expect("mk out");
    sm.adopt_input(0, scores).expect("sm in");
    sm.adopt_output(0, scores).expect("sm out");
    ao.adopt_input(0, scores).expect("attn s");
    ao.adopt_input(1, av.output_surf(0)).expect("V");
    po.adopt_input(0, ao.output_surf(0)).expect("Wo in");
    // residual1: r1 = x + po_z (both strided)
    r1p.adopt_input(0, xs).expect("r1 x");
    r1p.adopt_input(1, po.output_surf(0)).expect("r1 attn");
    let r1s = r1p.output_surf(0);
    // norm2
    sq2.adopt_input(0, r1s).expect("sq2 x");
    sq2.adopt_input(1, r1s).expect("sq2 y");
    sqc2.adopt_input(0, sq2.output_surf(0)).expect("sqc2");
    sum2.adopt_input(0, sqc2.output_surf(0)).expect("sum2 x");
    xn2.adopt_input(0, r1s).expect("xn2 x");
    xnc2.adopt_input(0, xn2.output_surf(0)).expect("xnc2");
    let xnc2_s = xnc2.output_surf(0);
    // ffn
    up.adopt_input(0, xnc2_s).expect("up x");
    gate.adopt_input(0, xnc2_s).expect("gate x");
    sig.adopt_input(0, gate.output_surf(0)).expect("sig x");
    slu.adopt_input(0, gate.output_surf(0)).expect("silu g");
    slu.adopt_input(1, sig.output_surf(0)).expect("silu s");
    upd.adopt_input(0, up.output_surf(0)).expect("upd u");
    upd.adopt_input(1, slu.output_surf(0)).expect("upd s");
    slc.adopt_input(0, upd.output_surf(0)).expect("slc");
    down.adopt_input(0, slc.output_surf(0)).expect("down x");
    yy.adopt_input(0, r1s).expect("y r1");
    yy.adopt_input(1, down.output_surf(0)).expect("y d");

    // descs for host access
    let dxs = sq1.input_descs[0].clone();
    let drvec = xn1.input_descs[1].clone();
    let dsum = sum1.output_descs[0].clone();
    let de = ok.input_descs[0].clone();
    let dmk = mkm.input_descs[1].clone();
    let dy = yy.output_descs[0].clone();
    println!(
        "xs: k{} ps{} | rvec: k{}h{}w{} ps{} | sum z: k{}h{}w{} ps{} | e: k{}h{} ps{} | y: k{} ps{}",
        dxs.k, dxs.plane_stride, drvec.k, drvec.h, drvec.w, drvec.plane_stride,
        dsum.k, dsum.h, dsum.w, dsum.plane_stride, de.k, de.h, de.plane_stride,
        dy.k, dy.plane_stride
    );

    let mut kref: Vec<Vec<f32>> = vec![vec![0.0; d]; t_max];
    let mut vref: Vec<Vec<f32>> = vec![vec![0.0; d]; t_max];
    let mut bad = 0usize;
    for step in 0..3usize {
        let x: Vec<f32> = (0..d)
            .map(|i| ((step * 13 + i * 5) % 11) as f32 / 16.0 - 0.3)
            .collect();
        // write x strided onto the shared residual surface
        {
            let b = sq1.input_mut(0);
            for i in 0..d {
                let o = dxs.offset(0, 0, i as u64, 0, 0);
                wr16(b, o, x[i]);
            }
        }
        // one-hot e_t and causal mask
        for m in [&mut ok, &mut ov] {
            let b = m.input_mut(0);
            for i in 0..t_max {
                let o = de.offset(0, 0, i as u64, 0, 0);
                wr16(b, o, if i == step { 1.0 } else { 0.0 });
            }
        }
        {
            let b = mkm.input_mut(1);
            for i in 0..t_max {
                let o = dmk.offset(0, 0, i as u64, 0, 0);
                wr16(b, o, if i <= step { 0.0 } else { -30000.0 });
            }
        }

        // norm1: sq -> repack -> sum -> host rsqrt -> xn -> repack
        sq1.eval().unwrap();
        sqc1.eval().unwrap();
        sum1.eval().unwrap();
        let ss = rd16(sum1.output(0), dsum.offset(0, 0, 0, 0, 0));
        let r = (ss / d as f32 + eps).sqrt().recip();
        {
            let b = xn1.input_mut(1);
            for i in 0..d {
                let o = drvec.offset(0, 0, i as u64, 0, 0);
                wr16(b, o, r);
            }
        }
        xn1.eval().unwrap();
        xnc1.eval().unwrap();

        // attention
        pq.eval().unwrap();
        pk.eval().unwrap();
        pv.eval().unwrap();
        rq.eval().unwrap();
        rk.eval().unwrap();
        rv.eval().unwrap();
        ok.eval().unwrap();
        ov.eval().unwrap();
        ak.eval().unwrap();
        av.eval().unwrap();
        sc.eval().unwrap();
        mkm.eval().unwrap();
        sm.eval().unwrap();
        ao.eval().unwrap();
        po.eval().unwrap();
        r1p.eval().unwrap();

        // norm2
        sq2.eval().unwrap();
        sqc2.eval().unwrap();
        sum2.eval().unwrap();
        let ss2 = rd16(sum2.output(0), dsum.offset(0, 0, 0, 0, 0));
        let r2 = (ss2 / d as f32 + eps).sqrt().recip();
        {
            let b = xn2.input_mut(1);
            for i in 0..d {
                let o = drvec.offset(0, 0, i as u64, 0, 0);
                wr16(b, o, r2);
            }
        }
        xn2.eval().unwrap();
        xnc2.eval().unwrap();

        // ffn
        up.eval().unwrap();
        gate.eval().unwrap();
        sig.eval().unwrap();
        slu.eval().unwrap();
        upd.eval().unwrap();
        slc.eval().unwrap();
        down.eval().unwrap();
        yy.eval().unwrap();

        // --- CPU reference ---
        let mm = |wtag: u8, v: &[f32], n: usize, k: usize| -> Vec<f32> {
            (0..n)
                .map(|j| (0..k).map(|i| v[i] * wload(wtag, j, i)).sum::<f32>())
                .collect()
        };
        let rms = |v: &[f32]| {
            (v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32 + eps)
                .sqrt()
                .recip()
        };
        let xnr: Vec<f32> = x.iter().map(|a| a * rms(&x)).collect();
        let qr = mm(b'q', &xnr, d, d);
        let kr = mm(b'k', &xnr, d, d);
        let vr = mm(b'v', &xnr, d, d);
        kref[step] = kr;
        vref[step] = vr;
        let sc_r: Vec<f32> = (0..t_max)
            .map(|t| (0..d).map(|i| qr[i] * kref[t][i]).sum())
            .collect();
        let mx = sc_r[..=step]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let den: f32 = sc_r[..=step].iter().map(|v| (v - mx).exp()).sum();
        let p: Vec<f32> = (0..t_max)
            .map(|t| {
                if t <= step {
                    (sc_r[t] - mx).exp() / den
                } else {
                    0.0
                }
            })
            .collect();
        let out: Vec<f32> = (0..d)
            .map(|j| (0..t_max).map(|t| p[t] * vref[t][j]).sum())
            .collect();
        let attn = mm(b'o', &out, d, d);
        let r1r: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
        let xn2r: Vec<f32> = r1r.iter().map(|a| a * rms(&r1r)).collect();
        let upr = mm(b'u', &xn2r, ff, d);
        let gr = mm(b'g', &xn2r, ff, d);
        let slr: Vec<f32> = upr
            .iter()
            .zip(&gr)
            .map(|(u, g)| u * g / (1.0 + (-g).exp()))
            .collect();
        let dr = mm(b'd', &slr, d, ff);
        let yr: Vec<f32> = r1r.iter().zip(&dr).map(|(a, b)| a + b).collect();

        // verify final y (strided) + the two scalar norm outputs
        let z = yy.output(0);
        for j in 0..d {
            let o = dy.offset(0, 0, j as u64, 0, 0);
            let hw = rd16(z, o);
            if (hw - yr[j]).abs() > yr[j].abs() * 0.05 + 0.03 {
                bad += 1;
                if bad < 8 {
                    println!("step{step} y[{j}]: hw={hw:.4} ref={:.4}", yr[j]);
                }
            }
        }
        let ssr: f32 = x.iter().map(|a| a * a).sum();
        if (ss - ssr).abs() > 0.02 {
            bad += 1;
            println!("step{step} sumsq: hw={ss:.5} ref={ssr:.5}");
        }
        println!(
            "step {step}: ss={ss:.4} ss2={ss2:.4} y0..3 hw=[{:.4},{:.4},{:.4}] ref=[{:.4},{:.4},{:.4}]",
            rd16(z, dy.offset(0, 0, 0, 0, 0)),
            rd16(z, dy.offset(0, 0, 1, 0, 0)),
            rd16(z, dy.offset(0, 0, 2, 0, 0)),
            yr[0], yr[1], yr[2]
        );
    }
    println!("LAYER-SLICE mismatches: {bad}");
}
