//! Does the ANE honor fp32 io surfaces? Emit an fp16 elementwise add,
//! patch the layout descriptors `s2w:5` (stride 2, type 5=float16) to
//! `s4w:6` (stride 4, type 6=float), bind fp32 surfaces, eval, check.
use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::{emit_elementwise2, EwOp};

fn patch_dtype(img: &[u8], from: &str, to: &str) -> Vec<u8> {
    let mut out = img.to_vec();
    let f = from.as_bytes();
    let t = to.as_bytes();
    assert_eq!(f.len(), t.len());
    let mut n = 0;
    let mut i = 0;
    while i + f.len() <= out.len() {
        if &out[i..i + f.len()] == f {
            out[i..i + f.len()].copy_from_slice(t);
            i += f.len();
            n += 1;
        } else {
            i += 1;
        }
    }
    println!("patched {n} x `{from}` -> `{to}`");
    out
}

fn main() {
    let dir = std::env::temp_dir().join("f32probe");
    std::fs::create_dir_all(&dir).unwrap();
    let mil = dir.join("add.mil");
    std::fs::write(&mil, b"placeholder").unwrap();
    let p16 = dir.join("add16.macho");
    let img = emit_elementwise2(
        EwOp::Add,
        64,
        &mil.to_string_lossy(),
        &p16.to_string_lossy(),
    );
    let p32 = dir.join("add32.macho");
    // dtype-only patch: w=1 descs put each element at a 64B plane
    // start, so plane stride stays s64; only elem stride/type flips.
    let img32 = patch_dtype(&img, "s2w:5", "s4w:6");
    std::fs::write(&p32, &img32).unwrap();

    let prog = match EspressoProgram::load_precompiled(&p32, &[]) {
        Ok(p) => p,
        Err(e) => {
            println!("load failed: {e}");
            // control: load the fp16 original
            let c = EspressoProgram::load_precompiled(&p16, &[]).unwrap();
            println!("fp16 control loads ok ({} in descs)", c.input_descs.len());
            return;
        }
    };
    println!(
        "fp32-desc program loaded; descs: {:?}",
        prog.input_descs
            .iter()
            .map(|d| (d.name.clone(), d.n, d.k, d.plane_stride))
            .collect::<Vec<_>>()
    );
    let mut prog = prog;
    {
        // fill whole surfaces: fp16 lanes read 1.0, fp32 lanes read ~0.0117
        let pat = [0x00u8, 0x3C, 0x00, 0x3C];
        let a = prog.input_mut(0);
        for c in a.chunks_exact_mut(4) {
            c.copy_from_slice(&pat);
        }
        let b = prog.input_mut(1);
        for c in b.chunks_exact_mut(4) {
            c.copy_from_slice(&pat);
        }
    }
    match prog.eval() {
        Ok(()) => {
            let z = prog.output(0);
            let nz: Vec<usize> = (0..z.len()).filter(|&i| z[i] != 0).collect();
            println!("nonzero output bytes: {} (len {})", nz.len(), z.len());
            for &i in nz.iter().take(16) {
                print!("{i:#x}:{:#04x} ", z[i]);
            }
            println!();
            // maybe kernel wrote fp16 results at stride 2 instead
            let h: Vec<f32> = (0..4)
                .map(|i| half::f16::from_le_bytes([z[i * 2], z[i * 2 + 1]]).to_f32())
                .collect();
            println!("as packed fp16 z[0..4]={h:?}");
            let h2: Vec<f32> = (0..4)
                .map(|i| half::f16::from_le_bytes([z[i * 64], z[i * 64 + 1]]).to_f32())
                .collect();
            println!("as strided fp16 z[0..4]={h2:?}");
        }
        Err(e) => println!("eval failed: {e}"),
    }
}
