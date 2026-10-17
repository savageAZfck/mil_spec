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
    emit_add4d64, emit_attn_v, emit_avw, emit_maskadd64, emit_matmul_ns, emit_outer, emit_repack64,
    emit_softmax_ch,
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
    // repacks: strided proj outputs -> contiguous vector operands
    let p = format!("/tmp/hp_rq_{pid}.macho");
    let mut rq = load(emit_repack64("/tmp/hpq.mil", &p), &p);
    let p = format!("/tmp/hp_rk_{pid}.macho");
    let mut rk = load(emit_repack64("/tmp/hpk.mil", &p), &p);
    let p = format!("/tmp/hp_rv_{pid}.macho");
    let mut rv = load(emit_repack64("/tmp/hpv.mil", &p), &p);
    let p = format!("/tmp/hp_mk_{pid}.macho");
    let mut mk_ = load(emit_maskadd64("/tmp/hk.mil", &p), &p);
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
    // repack wiring: proj z (strided) -> repack in; repack z (contig)
    rq.adopt_input(0, pq.output_surf(0)).expect("q->rq");
    rk.adopt_input(0, pk.output_surf(0)).expect("k->rk");
    rv.adopt_input(0, pv.output_surf(0)).expect("v->rv");
    // scores avw: a=q repacked (contiguous); v=K cache frame
    sc.adopt_input(0, rq.output_surf(0)).expect("q");
    sc.adopt_input(1, ak.output_surf(0)).expect("K");
    // scores += mask in-place, then softmax in-place on scores surface
    let scores = sc.output_surf(0);
    mk_.adopt_input(0, scores).expect("mk in");
    mk_.adopt_output(0, scores).expect("mk out");
    sm.adopt_input(0, scores).expect("sm in");
    sm.adopt_output(0, scores).expect("sm out");
    // attn_v: s=softmax out, v=V cache
    ao.adopt_input(0, scores).expect("attn s");
    ao.adopt_input(1, av.output_surf(0)).expect("V");
    // out-proj: x = attn output
    po.adopt_input(0, ao.output_surf(0)).expect("Wo in");
    // e_t one-hot mask = outer input0; kv rows = repacked proj outs
    ok.adopt_input(1, rk.output_surf(0)).expect("k row");
    ov.adopt_input(1, rv.output_surf(0)).expect("v row");

    // --- run a few decode steps ---
    let (dq, de, dr, dkc, _dvc, dz) = (
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
        {
            let dm = mk_.input_descs[1].clone();
            let b = mk_.input_mut(1);
            for i in 0..t_max {
                let o = dm.offset(0, 0, i as u64, 0, 0);
                if o + 2 <= b.len() {
                    b[o..o + 2].copy_from_slice(
                        &f16(if i <= step { 0.0 } else { -30000.0 }).to_le_bytes(),
                    );
                }
            }
        }
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
        if std::env::var("HP_DIAG").is_ok() {
            let ss = sc.output(0);
            let dss = sc.output_descs[0].clone();
            let mut sv = Vec::new();
            for c in 0..8u64 {
                let o = dss.offset(0, 0, c, 0, 0);
                sv.push(f32_of(u16::from_le_bytes([ss[o], ss[o + 1]])));
            }
            println!("  raw scores[0..8]: {:?}", sv);
            let kk = ak.input_bytes(0);
            let mut kv = Vec::new();
            for t in 0..4usize {
                let o = t * 128;
                kv.push(f32_of(u16::from_le_bytes([kk[o], kk[o + 1]])));
            }
            println!("  Kcache row heads: {:?}", kv);
            let oz = ok.output(0);
            let doz = ok.output_descs[0].clone();
            let mut ov_ = Vec::new();
            for c in 0..6u64 {
                let o = doz.offset(0, 0, c, 0, 0);
                ov_.push(f32_of(u16::from_le_bytes([oz[o], oz[o + 1]])));
            }
            println!("  outer z row heads: {:?}", ov_);
            let rkout = rk.output(0);
            let drk = rk.output_descs[0].clone();
            let mut rv_ = Vec::new();
            for w in 0..6u64 {
                let o = drk.offset(0, 0, 0, 0, w);
                rv_.push(f32_of(u16::from_le_bytes([rkout[o], rkout[o + 1]])));
            }
            println!("  rk out[0..6]: {:?}", rv_);
        }
        mk_.eval().unwrap();
        sm.eval().unwrap();
        ao.eval().unwrap();
        po.eval().unwrap();
        // CPU ref: update cache rows, then attention over rows 0..=step
        // z[j] = sum_k v[k]·W[j,k] — W is [N,K] row-major
        let mm = |w: u8, v: &[f32]| {
            (0..d)
                .map(|j| (0..d).map(|k| v[k] * wload(w, j, k)).sum::<f32>())
                .collect::<Vec<f32>>()
        };
        let (qr, kr, vr) = (mm(b'q', &x), mm(b'k', &x), mm(b'v', &x));
        kref[step] = kr.clone();
        vref[step] = vr.clone();
        let sc_r: Vec<f32> = (0..t_max)
            .map(|t| (0..d).map(|i| qr[i] * kref[t][i]).sum())
            .collect();
        let mx = sc_r[..=step]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        // causal: only written rows 0..=step contribute (masked lanes → 0)
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
        let oref = mm(b'o', &out);
        if std::env::var("HP_DIAG").is_ok() {
            // p surface (post-softmax, strided) and V cache row 0
            let sp = sm.output(0);
            let mut pv_ = Vec::new();
            for t in 0..6usize {
                let o = t * 64;
                pv_.push(f32_of(u16::from_le_bytes([sp[o], sp[o + 1]])));
            }
            let kk2 = av.input_bytes(0);
            let mut vv_ = Vec::new();
            for j in 0..6usize {
                let o = j * 2;
                vv_.push(f32_of(u16::from_le_bytes([kk2[o], kk2[o + 1]])));
            }
            println!("  p[0..6]: {:?} | V row0: {:?}", pv_, vv_);
            let zp = pv.output(0);
            let dzp = pv.output_descs[0].clone();
            let mut pvz = Vec::new();
            for c in 0..6u64 {
                let o = dzp.offset(0, 0, c, 0, 0);
                pvz.push(f32_of(u16::from_le_bytes([zp[o], zp[o + 1]])));
            }
            println!(
                "  pv z strided[0..6]: {:?} | vr ref[0..6]: {:?}",
                pvz,
                &vr[..6]
            );
            let zr = rv.output(0);
            let dzr = rv.output_descs[0].clone();
            let mut rvz = Vec::new();
            for w in 0..6u64 {
                let o = dzr.offset(0, 0, 0, 0, w);
                rvz.push(f32_of(u16::from_le_bytes([zr[o], zr[o + 1]])));
            }
            println!("  rv out contig[0..6]: {:?}", rvz);
            let za = ao.output(0);
            let dza = ao.output_descs[0].clone();
            let mut hv = Vec::new();
            for j in 0..6u64 {
                let o = dza.offset(0, 0, 0, 0, j);
                hv.push(f32_of(u16::from_le_bytes([za[o], za[o + 1]])));
            }
            println!("  attnV out[0..6] hw: {:?} ref: {:?}", hv, &out[..6]);
            let sq = pq.output(0);
            let dzq = pq.output_descs[0].clone();
            let mut v = Vec::new();
            for c in 0..8u64 {
                let o = dzq.offset(0, 0, c, 0, 0);
                v.push(f32_of(u16::from_le_bytes([sq[o], sq[o + 1]])));
            }
            println!("  q proj z[0..8] strided: {:?}", v);
            let mut v2 = Vec::new();
            for w in 0..8u64 {
                let o = (w * 2) as usize;
                v2.push(f32_of(u16::from_le_bytes([sq[o], sq[o + 1]])));
            }
            println!("  q proj z[0..8] contig:  {:?}", v2);
            println!("  q ref[0..8]: {:?}", &qr[..8]);
            let ss = sc.output(0);
            let dss = sc.output_descs[0].clone();
            let mut sv = Vec::new();
            for c in 0..8u64 {
                let o = dss.offset(0, 0, c, 0, 0);
                sv.push(f32_of(u16::from_le_bytes([ss[o], ss[o + 1]])));
            }
            println!("  scores[0..8]: {:?} ref: {:?}", sv, &sc_r[..8]);
        }
        let z = po.output(0);
        if std::env::var("HP_DIAG").is_ok() {
            let mut a = Vec::new();
            let mut b = Vec::new();
            for j in 0..8u64 {
                let o = dz.offset(0, 0, j, 0, 0);
                a.push(f32_of(u16::from_le_bytes([z[o], z[o + 1]])));
            }
            for j in 0..8u64 {
                let o = dz.offset(0, 0, 0, 0, j);
                b.push(f32_of(u16::from_le_bytes([z[o], z[o + 1]])));
            }
            println!(
                "  po z strided: {:?} | contig: {:?} | oref: {:?}",
                a,
                b,
                &oref[..8]
            );
        }
        for j in 0..d {
            let o = dz.offset(0, 0, j as u64, 0, 0);
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
// diag appended below main via flag
