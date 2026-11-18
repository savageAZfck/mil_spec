use mil_infer::espresso::EspressoProgram;
use std::path::Path;
fn f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let mut m = EspressoProgram::load_precompiled(Path::new(&p), &[]).unwrap();
    for (i, d) in m.input_descs.iter().enumerate() {
        println!(
            "in{i} {} n{} k{} h{} w{} ps{} rs{} bs{} ds{} e{}",
            d.name, d.n, d.k, d.h, d.w, d.plane_stride, d.row_stride,
            d.batch_stride, d.depth_stride, d.elem
        );
    }
    for (i, d) in m.output_descs.iter().enumerate() {
        println!(
            "out{i} {} n{} k{} h{} w{} ps{} rs{} bs{} ds{} e{}",
            d.name, d.n, d.k, d.h, d.w, d.plane_stride, d.row_stride,
            d.batch_stride, d.depth_stride, d.elem
        );
    }
    // write 1.0 into every element slot of every input
    for i in 0..m.inputs.len() {
        let d = m.input_descs[i].clone();
        let b = m.input_mut(i);
        let n = (d.n * d.k.max(1) * d.h.max(1) * d.w.max(1)).max(1);
        for e in 0..n {
            let o = d.offset(e % d.n.max(1), 0, (e / d.n.max(1)) % d.k.max(1), 0, e % d.w.max(1));
            let o = o.min(b.len().saturating_sub(2));
            b[o..o + 2].copy_from_slice(&f16(1.0).to_le_bytes());
        }
    }
    m.eval().unwrap();
    for i in 0..m.outputs.len() {
        let o = m.output(i);
        let nz = o.iter().filter(|v| **v != 0).count();
        println!("out{i}: {}B {nz}nz first={:?}", o.len(), &o[..8.min(o.len())]);
    }
    for i in 0..m.inputs.len() {
        let b = m.input_bytes(i);
        let nz = b.iter().filter(|v| **v != 0).count();
        println!("in{i} post: {}B {nz}nz", b.len());
    }
}
