//! sm_run — exec-verify an emitted standalone softmax: z = softmax(x).
use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::emit_softmax;
use std::path::Path;
fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
fn main() {
    let n: u64 = std::env::args().nth(1).unwrap().parse().unwrap();
    let out = format!("/tmp/sm_em_{}_{}.macho", n, std::process::id());
    let img = emit_softmax(n as u32, "/tmp/sm_em.mil", &out).unwrap();
    std::fs::write(&out, &img).unwrap();
    let mut m = EspressoProgram::load_precompiled(Path::new(&out), &[]).unwrap();
    let dx = m.input_descs[0].clone();
    let dz = m.output_descs[0].clone();
    println!("x:k{} w{} | z:k{} w{}", dx.k, dx.w, dz.k, dz.w);
    // x[w] = (w%13)-6 — mixed sign values
    {
        let b = m.input_mut(0);
        for w in 0..n {
            let o = dx.offset(0, 0, 0, 0, w);
            if o + 2 <= b.len() {
                b[o..o + 2].copy_from_slice(&f16((w % 13) as f32 - 6.0).to_le_bytes());
            }
        }
    }
    m.eval().unwrap();
    let z = m.output(0);
    let xs: Vec<f32> = (0..n).map(|w| (w % 13) as f32 - 6.0).collect();
    let mx = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let den: f32 = xs.iter().map(|v| (v - mx).exp()).sum();
    let mut mm = 0;
    for w in 0..n {
        let o = dz.offset(0, 0, 0, 0, w);
        if o + 2 > z.len() {
            mm += 1;
            continue;
        }
        let v = f32_of(u16::from_le_bytes([z[o], z[o + 1]]));
        let exp = ((w % 13) as f32 - 6.0 - mx).exp() / den;
        if (v - exp).abs() > 0.002 {
            mm += 1;
            if mm < 4 {
                println!("w={} z={} exp={}", w, v, exp);
            }
        }
    }
    println!("softmax mismatches: {}/{}", mm, n);
}
