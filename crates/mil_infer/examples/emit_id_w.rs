//! emit_id_w <w> — emit conv1x1(64,64,W) with identity weights, load it
//! through the precompiled path, eval a per-position ramp, and verify
//! every W position round-trips (proves spatial/sequence emission).
use mil_infer::espresso::EspressoProgram;
use std::path::Path;

fn f16(v: f32) -> u16 {
    let f = v.to_bits();
    let s = ((f >> 16) & 0x8000) as u16;
    let e = ((f >> 23) & 0xff) as i32 - 127 + 15;
    let m = (f >> 13) & 0x3ff;
    if e <= 0 {
        return s;
    }
    if e >= 31 {
        return s | 0x7c00;
    }
    s | ((e as u16) << 10) | m as u16
}
fn f32_of(h: u16) -> f32 {
    let s = (h & 0x8000) != 0;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    let v = if e == 0 {
        m / 1024.0 * 2f32.powi(-14)
    } else if e == 31 {
        f32::INFINITY
    } else {
        (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    };
    if s {
        -v
    } else {
        v
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let w_dim: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(4);
    let cin: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(64);
    let cout: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(64);
    // w[o,i] = 1 iff i == o % cin -> z[o,m] = x[o % cin, m]
    let w: Vec<u16> = (0..cout * cin)
        .map(|i| {
            let (o, i2) = (i / cin, i % cin);
            if i2 == o % cin {
                0x3c00
            } else {
                0
            }
        })
        .collect();
    let img =
        mil_infer::hwx::emit_conv1x1_w(cin, cout, w_dim, &w, "/tmp/idw.mil", "/tmp/idw.macho")
            .expect("emit");
    let p = format!("/tmp/emit_id_w{w_dim}.hwx");
    std::fs::write(&p, &img).unwrap();
    println!("emitted {} bytes", img.len());

    let mut m = EspressoProgram::load_precompiled(Path::new(&p), &[]).expect("load");
    println!(
        "loaded: ph={} in={:?} out={:?} procs={:?}",
        m.program_handle,
        m.input_descs
            .iter()
            .map(|d| (d.name.clone(), d.k, d.w))
            .collect::<Vec<_>>(),
        m.output_descs
            .iter()
            .map(|d| (d.name.clone(), d.k, d.w))
            .collect::<Vec<_>>(),
        m.procedures
    );
    // fill x[c,pos] = c + pos*100 — distinct per (channel, position)
    {
        let d = m.input_descs[0].clone();
        let b = m.input_mut(0);
        for c in 0..d.k {
            for pos in 0..d.w {
                let off = d.offset(0, 0, c, 0, pos);
                b[off..off + 2].copy_from_slice(&f16(c as f32 * 16.0 + pos as f32).to_le_bytes());
            }
        }
    }
    m.eval().expect("eval");
    let d = m.output_descs[0].clone();
    let out = m.output(0);
    let mut bad = 0;
    let mut first = Vec::new();
    for c in 0..d.k {
        for pos in 0..d.w {
            let off = d.offset(0, 0, c, 0, pos);
            let got = f32_of(u16::from_le_bytes([out[off], out[off + 1]]));
            let want = (c % cin as u64) as f32 * 16.0 + pos as f32;
            if (got - want).abs() > 0.01 {
                bad += 1;
                if first.len() < 8 {
                    first.push(format!("z[{c},{pos}]={got} want {want}"));
                }
            }
        }
    }
    if std::env::var("DUMP").is_ok() {
        for c in 0..3u64 {
            let row: Vec<String> = (0..d.w)
                .map(|pos| {
                    let off = d.offset(0, 0, c, 0, pos);
                    format!(
                        "{:.0}",
                        f32_of(u16::from_le_bytes([out[off], out[off + 1]]))
                    )
                })
                .collect();
            println!("z row c={c} (my stride): {row:?}");
        }
        // also dump raw first 256 bytes as fp16 values
        let raw: Vec<String> = (0..128)
            .map(|i| {
                format!(
                    "{:.0}",
                    f32_of(u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]))
                )
            })
            .collect();
        println!("raw[0..128]: {raw:?}");
    }
    if bad == 0 {
        println!(
            "PASS: all {} positions x {} channels bit-match (identity)",
            d.w, d.k
        );
    } else {
        println!("FAIL: {bad} mismatches — {first:?}");
    }
}
