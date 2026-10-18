use mil_infer::espresso::{EspressoProgram, SharedSurface};
use mil_infer::hwx::emit_avw;
use std::path::Path;
fn main() {
    for (r, k) in [(128u32, 1024u32), (256, 1024), (64, 1024)] {
        let img = emit_avw(r, k, "/tmp/a.mil", "/tmp/avw_t.macho");
        std::fs::write("/tmp/avw_t.macho", &img).unwrap();
        let mut m = EspressoProgram::load_precompiled(Path::new("/tmp/avw_t.macho"), &[]).unwrap();
        let vlen = m.input_mut(1).len();
        let stride = vlen / r as usize;
        let (rb, kb) = (r as usize, k as usize);
        let ws = SharedSurface::new(vlen).unwrap();
        {
            let wb = ws.bytes_mut();
            for j in 0..rb {
                let i = 0usize;
                let v = 100.0 + j as f32;
                wb[j * stride + i * 2..j * stride + i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
            }
        }
        {
            let ab = m.input_mut(0);
            for b in ab.iter_mut() {
                *b = 0;
            }
            ab[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        }
        m.adopt_input(1, ws.as_ptr()).unwrap();
        m.eval().unwrap();
        let d = m.output_descs[0].clone();
        let b = m.output(0);
        let rows: Vec<f32> = [0usize, 1, 2, rb / 2, rb - 1]
            .iter()
            .map(|&j| {
                let o = d.offset(0, 0, j as u64, 0, 0);
                half::f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32() - 100.0
            })
            .collect();
        println!(
            "{r}x{k} stride{stride}: lanes [0,1,2,{},{ }] read V rows {rows:?}",
            rb / 2,
            rb - 1
        );
    }
}
