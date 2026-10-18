use mil_infer::espresso::EspressoProgram;
use mil_infer::hwx::emit_matmul_ns;
use std::path::Path;
fn main() {
    let mut n = 0usize;
    for i in 0..300usize {
        let w = vec![0u16; 1024 * (1024 + i)];
        let p = format!("/tmp/lc_d{i}.macho");
        let img = emit_matmul_ns(1024, 1024 + i, &w, "/tmp/lc.mil", &p).unwrap();
        std::fs::write(&p, &img).unwrap();
        match EspressoProgram::load_precompiled(Path::new(&p), &[]) {
            Ok(pr) => {
                n += 1;
                std::mem::forget(pr);
            }
            Err(e) => {
                eprintln!("failed at {i}: {e}");
                break;
            }
        }
        if i % 50 == 0 {
            eprintln!("{i} loaded");
        }
    }
    eprintln!("total: {n}");
}
