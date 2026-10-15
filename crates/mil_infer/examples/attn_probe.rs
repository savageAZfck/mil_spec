//! attn_probe — full decode attention step on ANE, all self-emitted:
//!   scores[t] = q·K[t]          (emit_avw)
//!   p = softmax(scores)         (emit_softmax_ch, channel-axis)
//!   out[j]  = Σ_t p[t]·V[t][j]  (emit_attn_v)
//! Programs chained via IOSurface adoption — zero host copies.
use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::{emit_attn_v, emit_avw, emit_softmax_ch};
use std::path::Path;
fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
fn main() {
    let pid = std::process::id();
    let (t, k, d) = (64u64, 64u64, 64u64);
    let p1 = format!("/tmp/at1_{pid}.macho");
    let p2 = format!("/tmp/at2_{pid}.macho");
    let p3 = format!("/tmp/at3_{pid}.macho");
    std::fs::write(&p1, emit_avw(t as u32, k as u32, "/tmp/at1.mil", &p1)).unwrap();
    std::fs::write(&p2, emit_softmax_ch(t as u32, "/tmp/at2.mil", &p2).unwrap()).unwrap();
    std::fs::write(
        &p3,
        emit_attn_v(t as u32, d as u32, "/tmp/at3.mil", &p3).unwrap(),
    )
    .unwrap();
    let mut m1 = EspressoProgram::load_precompiled(Path::new(&p1), &[]).expect("avw");
    let mut m2 = EspressoProgram::load_precompiled(Path::new(&p2), &[]).expect("smc");
    let mut m3 = EspressoProgram::load_precompiled(Path::new(&p3), &[]).expect("attnv");
    let (dq, dk) = (m1.input_descs[0].clone(), m1.input_descs[1].clone());
    let (dz1, dz2) = (m1.output_descs[0].clone(), m2.output_descs[0].clone());
    let (dv, dz3) = (m3.input_descs[1].clone(), m3.output_descs[0].clone());
    println!(
        "avw z: k{} w{} ps{} | smc z: k{} w{} ps{} | attnv s: k{} h{} w{} rs{} | z: w{}",
        dz1.k,
        dz1.w,
        dz1.plane_stride,
        dz2.k,
        dz2.w,
        dz2.plane_stride,
        m3.input_descs[0].k,
        m3.input_descs[0].h,
        m3.input_descs[0].w,
        m3.input_descs[0].row_stride,
        dz3.w
    );
    // chain: scores surf -> smc in+out (in-place) -> attnv s
    let scores = m1.output_surf(0);
    m2.adopt_input(0, scores).expect("smc in");
    m2.adopt_output(0, scores).expect("smc out");
    m3.adopt_input(0, scores).expect("attnv s");
    // fill q[w]=w/64, K[t][w]=(t+w)/128, V[t][j]=((t*3+j)%9)/8
    {
        let b = m1.input_mut(0);
        for w in 0..k {
            let o = dq.offset(0, 0, 0, 0, w);
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(&f16(w as f32 / 64.0).to_le_bytes());
            }
        }
    }
    {
        let b = m1.input_mut(1);
        for i in 0..t {
            for w in 0..k {
                let o = dk.offset(0, 0, i, 0, w);
                if o + 2 <= b.len() {
                    b[o..o + 2].copy_from_slice(&f16((i + w) as f32 / 128.0).to_le_bytes());
                }
            }
        }
    }
    {
        let b = m3.input_mut(1);
        for i in 0..t {
            for j in 0..d {
                let o = dv.offset(0, 0, 0, i, j);
                if o + 2 <= b.len() {
                    b[o..o + 2].copy_from_slice(&f16(((i * 3 + j) % 9) as f32 / 8.0).to_le_bytes());
                }
            }
        }
    }
    m1.eval().expect("scores");
    m2.eval().expect("softmax");
    m3.eval().expect("attnv");
    // CPU reference
    let qs: Vec<f32> = (0..k).map(|w| w as f32 / 64.0).collect();
    let sc: Vec<f32> = (0..t)
        .map(|i| {
            (0..k)
                .map(|w| qs[w as usize] * ((i + w) as f32 / 128.0))
                .sum()
        })
        .collect();
    let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let den: f32 = sc.iter().map(|v| (v - mx).exp()).sum();
    let p: Vec<f32> = sc.iter().map(|v| (v - mx).exp() / den).collect();
    let z = m3.output(0);
    let mut mm = 0;
    for j in 0..d {
        let o = dz3.offset(0, 0, 0, 0, j);
        if o + 2 > z.len() {
            mm += 1;
            continue;
        }
        let v = f32_of(u16::from_le_bytes([z[o], z[o + 1]]));
        let exp: f32 = (0..t)
            .map(|i| p[i as usize] * (((i * 3 + j) % 9) as f32 / 8.0))
            .sum();
        if (v - exp).abs() > exp * 0.02 + 0.02 {
            mm += 1;
            if mm < 4 {
                println!("j={} z={} exp={:.4}", j, v, exp);
            }
        }
    }
    println!("ATTENTION decode step mismatches: {}/{}", mm, d);
}
