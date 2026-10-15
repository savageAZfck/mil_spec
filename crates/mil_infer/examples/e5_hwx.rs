//! `e5_hwx` — try loading a bare .hwx/.macho via `e5rt_program_library_create`,
//! then bind+exec to prove the artifact is consumed.

use mil_infer::e5rt::Program;
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
    let p = Path::new(&a[1]);
    let (prog, map) = match Program::from_e5_bundle(p) {
        Ok(v) => v,
        Err(e) => {
            println!("library_create failed: {e:?}");
            return;
        }
    };
    println!("loaded: {map:?}");
    for (n, e) in &map.inputs {
        let data: Vec<u16> = (0..*e).map(|k| f16(((k % 16) + 1) as f32)).collect();
        prog.set_input_f16(n, &data).unwrap();
    }
    match prog.execute() {
        Ok(()) => {
            for (n, _) in &map.outputs {
                let v = prog.output_f16(n).unwrap();
                let head: Vec<f32> = v.iter().take(8).map(|h| f32_of(*h)).collect();
                println!("{n}[0..8] = {head:?}");
            }
        }
        Err(e) => println!("exec failed: {e:?}"),
    }
}
