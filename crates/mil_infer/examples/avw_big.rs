//! avw_big — does the dynamic matvec scale past r=128?
//! For r in {256,512,1024,2048}: emit avw(r,64), write a=delta(0),
//! V[lane][0] = lane id, read z[lane] -> learned row map.
use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::emit_avw;

fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

fn main() {
    let dir = std::env::temp_dir().join("avwbig");
    std::fs::create_dir_all(&dir).unwrap();
    let mil = dir.join("x.mil");
    std::fs::write(&mil, b"x").unwrap();
    for r in [256usize, 512, 1024, 2048] {
        let p = dir.join(format!("avw{r}.macho"));
        let img = emit_avw(r as u32, 64, &mil.to_string_lossy(), &p.to_string_lossy());
        let mut m = match EspressoProgram::load_precompiled(&p, &[]) {
            Ok(m) => m,
            Err(e) => {
                println!("r={r}: emit {}B, load FAIL {e}", img.len());
                continue;
            }
        };
        let (ds, dv, dz) = (
            m.input_descs[0].clone(),
            m.input_descs[1].clone(),
            m.output_descs[0].clone(),
        );
        println!(
            "r={r}: loaded img {}B | a n{}k{}h{}w{} | v n{}k{}h{}w{} | z n{}k{}h{}w{} inlen{} outlen{}",
            img.len(), ds.n, ds.k, ds.h, ds.w, dv.n, dv.k, dv.h, dv.w, dz.n, dz.k, dz.h, dz.w,
            m.input_mut(1).len(), m.output(0).len()
        );
        {
            let b = m.input_mut(0);
            b.iter_mut().for_each(|x| *x = 0);
            let o = ds.offset(0, 0, 0, 0, 0);
            b[o..o + 2].copy_from_slice(&f16(1.0).to_le_bytes());
        }
        {
            let b = m.input_mut(1);
            b.iter_mut().for_each(|x| *x = 0);
            for lane in 0..r.min(dv.k.max(dv.n) as usize).max(r) {
                // try both axes: row index may be k or n
                let o1 = dv.offset(0, 0, 0, lane as u64, 0);
                if o1 + 2 <= b.len() {
                    b[o1..o1 + 2].copy_from_slice(&f16(lane as f32).to_le_bytes());
                }
            }
        }
        match m.eval() {
            Ok(()) => {
                let z = m.output(0);
                let mut got = Vec::new();
                for lane in 0..r.min(dz.k.max(dz.n) as usize).max(r) {
                    let o = dz.offset(0, 0, 0, lane as u64, 0);
                    if o + 2 <= z.len() {
                        got.push(f32_of(u16::from_le_bytes([z[o], z[o + 1]])));
                    }
                }
                let mapped = got.iter().filter(|&&v| v > 0.0).count();
                println!(
                    "  lanes read {} nonzero of {}: first16={:?}",
                    mapped,
                    got.len(),
                    &got[..16.min(got.len())]
                );
            }
            Err(e) => println!("  eval FAIL {e}"),
        }
    }
}
