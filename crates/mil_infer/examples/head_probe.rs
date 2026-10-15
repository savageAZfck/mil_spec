//! head_probe — a full transformer attention-head decode step on ANE.
//! Every program self-emitted; all dataflow via IOSurface adoption,
//! zero host copies once x is written:
//!   q=x·Wq, k=x·Wk, v=x·Wv        matmul_ns (const weights)
//!   K += outer(e_t,k)             emit_outer + ew add (in-place alias)
//!   V += outer(e_t,v)
//!   s = q·K[t]                    emit_avw
//!   p = softmax(s)                emit_softmax_ch (in-place)
//!   out[j] = Σ p[t]·V[t][j]       emit_attn_v
//!   o = out·Wo                    matmul_ns
use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::{
    emit_add4d64, emit_attn_v, emit_avw, emit_matmul_ns, emit_outer, emit_softmax_ch,
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
fn main() {
    let pid = std::process::id();
    let (t_max, d) = (64usize, 64usize);
    let wq: Vec<u16> = (0..d * d).map(|i| f16(wload(b'q', i / d, i % d))).collect();
    let wk: Vec<u16> = (0..d * d).map(|i| f16(wload(b'k', i / d, i % d))).collect();
    let wv: Vec<u16> = (0..d * d).map(|i| f16(wload(b'v', i / d, i % d))).collect();
    let wo: Vec<u16> = (0..d * d).map(|i| f16(wload(b'o', i / d, i % d))).collect();
    let mk = |tag: &str, w: &[u16]| {
        let p = format!("/tmp/hp_{tag}_{pid}.macho");
        load(emit_matmul_ns(d, d, w, "/tmp/hp.mil", &p).unwrap(), &p)
    };
    let (mut pq, mut pk, mut pv, mut po) = (mk("q", &wq), mk("k", &wk), mk("v", &wv), mk("o", &wo));
    let p = format!("/tmp/hp_ok_{pid}.macho");
    let mut ok = load(emit_outer("/tmp/ho.mil", &p), &p);
    let p = format!("/tmp/hp_ov_{pid}.macho");
    let mut ov = load(emit_outer("/tmp/ho.mil", &p), &p);
    // packed 4-D adds over the T*d cache frames (in-place via alias)
    let p = format!("/tmp/hp_ak_{pid}.macho");
    let mut ak = load(emit_add4d64("/tmp/ha.mil", &p), &p);
    let p = format!("/tmp/hp_av_{pid}.macho");
    let mut av = load(emit_add4d64("/tmp/ha.mil", &p), &p);
    let p = format!("/tmp/hp_s_{pid}.macho");
    let mut sc = load(emit_avw(t_max as u32, d as u32, "/tmp/hs.mil", &p), &p);
    let p = format!("/tmp/hp_sm_{pid}.macho");
    let mut sm = load(
        emit_softmax_ch(t_max as u32, "/tmp/hm.mil", &p).unwrap(),
        &p,
    );
    let p = format!("/tmp/hp_avv_{pid}.macho");
    let mut ao = load(
        emit_attn_v(t_max as u32, d as u32, "/tmp/hv.mil", &p).unwrap(),
        &p,
    );

    // print layouts once
    for (n, m) in [
        ("proj", &pq),
        ("outer", &ok),
        ("add", &ak),
        ("scores", &sc),
        ("sm", &sm),
        ("attnv", &ao),
    ] {
        let ins: Vec<String> = m
            .input_descs
            .iter()
            .map(|x| {
                format!(
                    "{}:k{}h{}w{}ps{}rs{}",
                    x.name, x.k, x.h, x.w, x.plane_stride, x.row_stride
                )
            })
            .collect();
        let outs: Vec<String> = m
            .output_descs
            .iter()
            .map(|x| format!("{}:k{}h{}w{}ps{}", x.name, x.k, x.h, x.w, x.plane_stride))
            .collect();
        println!("{n}: in {:?} out {:?}", ins, outs);
    }

    // --- wire surfaces ---
    // K/V cache = the ADD programs' input0/output0 aliased (in-place).
    ak.alias_out_to_in(0, 0);
    av.alias_out_to_in(0, 0);
    for b in ak.input_mut(0).iter_mut() {
        *b = 0;
    }
    for b in av.input_mut(0).iter_mut() {
        *b = 0;
    }
    // outer outputs -> add input1
    ak.adopt_input(1, ok.output_surf(0)).expect("k tmp");
    av.adopt_input(1, ov.output_surf(0)).expect("v tmp");
    // scores avw: a=q proj output adopted; v=K cache frame adopted
    sc.adopt_input(0, pq.output_surf(0)).expect("q");
    sc.adopt_input(1, ak.output_surf(0)).expect("K");
    // softmax in-place on scores surface
    let scores = sc.output_surf(0);
    sm.adopt_input(0, scores).expect("sm in");
    sm.adopt_output(0, scores).expect("sm out");
    // attn_v: s=softmax out, v=V cache
    ao.adopt_input(0, scores).expect("attn s");
    ao.adopt_input(1, av.output_surf(0)).expect("V");
    // out-proj: x = attn output
    po.adopt_input(0, ao.output_surf(0)).expect("Wo in");
    // e_t one-hot mask = outer input0; kv vectors = outer input1 adopted
    // from proj outputs
    ok.adopt_input(1, pk.output_surf(0)).expect("k row");
    ov.adopt_input(1, pv.output_surf(0)).expect("v row");

    // --- run a few decode steps ---
    let (dq, de, dr, dkc, dvc, dz) = (
        pq.input_descs[0].clone(),
        ok.input_descs[0].clone(),
        ok.input_descs[1].clone(),
        ak.input_descs[0].clone(),
        av.input_descs[0].clone(),
        po.output_descs[0].clone(),
    );
    println!(
        "e desc k{} w{} ps{} | r k{} w{} ps{} | add x n{} ps{}",
        de.k, de.w, de.plane_stride, dr.k, dr.w, dr.plane_stride, dkc.n, dkc.plane_stride
    );
    let mut kref: Vec<Vec<f32>> = vec![vec![0.0; d]; t_max];
    let mut vref: Vec<Vec<f32>> = vec![vec![0.0; d]; t_max];
    let mut bad = 0usize;
    for step in 0..3usize {
        let x: Vec<f32> = (0..d)
            .map(|i| ((step * 13 + i * 5) % 11) as f32 / 16.0 - 0.3)
            .collect();
        // same x to all three projections
        for m in [&mut pq, &mut pk, &mut pv] {
            let b = m.input_mut(0);
            for i in 0..d {
                let o = dq.offset(0, 0, 0, 0, i as u64);
                if o + 2 <= b.len() {
                    b[o..o + 2].copy_from_slice(&f16(x[i]).to_le_bytes());
                }
            }
        }
        // one-hot e_t at channel t (strided input desc)
        {
            let b = ok.input_mut(0);
            for i in 0..t_max {
                let o = de.offset(0, 0, i as u64, 0, 0);
                if o + 2 <= b.len() {
                    b[o..o + 2]
                        .copy_from_slice(&f16(if i == step { 1.0 } else { 0.0 }).to_le_bytes());
                }
            }
        }
        {
            let b = ov.input_mut(0);
            for i in 0..t_max {
                let o = de.offset(0, 0, i as u64, 0, 0);
                if o + 2 <= b.len() {
                    b[o..o + 2]
                        .copy_from_slice(&f16(if i == step { 1.0 } else { 0.0 }).to_le_bytes());
                }
            }
        }
        pq.eval().unwrap();
        pk.eval().unwrap();
        pv.eval().unwrap();
        ok.eval().unwrap();
        ov.eval().unwrap();
        ak.eval().unwrap();
        av.eval().unwrap();
        sc.eval().unwrap();
        sm.eval().unwrap();
        ao.eval().unwrap();
        po.eval().unwrap();
        // CPU ref: update cache rows, then attention over rows 0..=step
        let mm = |w: u8| {
            (0..d)
                .map(|j| (0..d).map(|i| x[i] * wload(w, i, j)).sum::<f32>())
                .collect::<Vec<f32>>()
        };
        let (qr, kr, vr) = (mm(b'q'), mm(b'k'), mm(b'v'));
        kref[step] = kr.clone();
        vref[step] = vr.clone();
        let sc_r: Vec<f32> = (0..t_max)
            .map(|t| (0..d).map(|i| qr[i] * kref[t][i]).sum())
            .collect();
        let mx = sc_r[..=step]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        // NOTE: softmax runs over all t_max lanes incl. unwritten zero rows
        let den: f32 = sc_r.iter().map(|v| (v - mx).exp()).sum();
        let p: Vec<f32> = sc_r.iter().map(|v| (v - mx).exp() / den).collect();
        let out: Vec<f32> = (0..d)
            .map(|j| (0..t_max).map(|t| p[t] * vref[t][j]).sum())
            .collect();
        let oref: Vec<f32> = (0..d)
            .map(|j| (0..d).map(|i| out[i] * wload(b'o', i, j)).sum())
            .collect();
        let z = po.output(0);
        for j in 0..d {
            let o = dz.offset(0, 0, 0, 0, j as u64);
            if o + 2 > z.len() {
                bad += 1;
                continue;
            }
            let v = f32_of(u16::from_le_bytes([z[o], z[o + 1]]));
            if (v - oref[j]).abs() > oref[j].abs() * 0.03 + 0.02 {
                bad += 1;
                if bad < 6 {
                    println!("step{step} j{j}: hw={:.4} ref={:.4}", v, oref[j]);
                }
            }
        }
        println!("step {step}: done");
    }
    println!("HEAD-SLICE mismatches: {}", bad);
}
