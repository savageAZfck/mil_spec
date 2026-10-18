//! wb_probe — does `_ANERequest.weightsBuffer` override baked weights?
//! matmul_ns(64,64) baked with identity -> z=x. Re-eval with a
//! weightsBuffer carrying 2*I in __KERN_0 layout: z=2x means honored.
use mil_infer::espresso::{EspressoProgram, SharedSurface};
use mil_infer::hwx::{emit_matmul_ns, weight_layout_ns};

fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f32_of(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

fn main() {
    let dir = std::env::temp_dir().join("wbprobe");
    std::fs::create_dir_all(&dir).unwrap();
    let mil = dir.join("x.mil");
    std::fs::write(&mil, b"x").unwrap();
    let (k, n) = (1024usize, 256usize);

    // baked identity
    let mut w = vec![0u16; k * n];
    for i in 0..n {
        w[i * k + i] = f16(1.0);
    }
    let p = dir.join("mm.macho");
    let img = emit_matmul_ns(k, n, &w, &mil.to_string_lossy(), &p.to_string_lossy()).unwrap();
    std::fs::write(&p, &img).unwrap();
    let mut prog = EspressoProgram::load_precompiled(&p, &[]).unwrap();
    let dx = prog.input_descs[0].clone();
    let dz = prog.output_descs[0].clone();
    println!(
        "x: n{} k{} h{} w{} ps{} | z: n{} k{} h{} w{} ps{} inlen{}",
        dx.n,
        dx.k,
        dx.h,
        dx.w,
        dx.plane_stride,
        dz.n,
        dz.k,
        dz.h,
        dz.w,
        dz.plane_stride,
        prog.input_mut(0).len()
    );
    {
        let b = prog.input_mut(0);
        for i in 0..k {
            let o = dx.offset(0, 0, 0, 0, i as u64); // w axis: K is seq on input
            if o + 2 > b.len() {
                if i < 4 {
                    println!("input desc offset overflow: {o} > {}", b.len());
                }
                continue;
            }
            b[o..o + 2].copy_from_slice(&f16(i as f32 + 1.0).to_le_bytes());
        }
    }
    prog.eval().unwrap();
    let z0: Vec<f32> = (0..4)
        .map(|i| {
            f32_of(u16::from_le_bytes([
                prog.output(0)[dz.offset(0, 0, i as u64, 0, 0)],
                prog.output(0)[dz.offset(0, 0, i as u64, 0, 0) + 1],
            ]))
        })
        .collect();
    println!("baked eval z[0..4]={z0:?} (want 1,2,3,4)");

    // weightsBuffer: 2*I in __KERN_0 layout (same transform the
    // emitter applies, but doubled on the diagonal before layout)
    let mut w2 = vec![0u16; k * n];
    for i in 0..n {
        w2[i * k + i] = f16(2.0);
    }
    let (stream, _, _) = weight_layout_ns(&w2, k, n);
    let wb = SharedSurface::new(stream.len() * 2 + 0x4000).unwrap();
    {
        let b = wb.bytes_mut();
        for (i, v) in stream.iter().enumerate() {
            b[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    match prog.eval_wb(wb.as_ptr()) {
        Ok(()) => {
            let z: Vec<f32> = (0..4)
                .map(|i| {
                    f32_of(u16::from_le_bytes([
                        prog.output(0)[dz.offset(0, 0, i as u64, 0, 0)],
                        prog.output(0)[dz.offset(0, 0, i as u64, 0, 0) + 1],
                    ]))
                })
                .collect();
            println!("wb eval z[0..4]={z:?} (2,4,6,8 => honored; 1,2,3,4 => ignored)");
        }
        Err(e) => println!("eval_wb failed: {e}"),
    }
}
