//! id_mut — compile a conv-chain mil, eval; then patch the cached
//! bundle's descriptor ids and re-eval to test id-sensitivity.
//!
//!   id_mut <net.mil> <in_name> <out_name> <cin> <cout>
use mil_infer::e5rt::{DeviceMask, Program};
use std::path::Path;

fn newest_bundle() -> Option<std::path::PathBuf> {
    let root = Path::new("/tmp/e5_cache/com.apple.e5rt.e5bundlecache");
    let mut best: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    fn walk(d: &Path, out: &mut Vec<std::path::PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    walk(&e.path(), out);
                } else if e.file_name() == "net.macho" {
                    out.push(e.path());
                }
            }
        }
    }
    let mut v = vec![];
    walk(root, &mut v);
    for p in v {
        if let Ok(m) = p.metadata().and_then(|m| m.modified()) {
            if best.as_ref().map(|(t, _)| m > *t).unwrap_or(true) {
                best = Some((m, p));
            }
        }
    }
    best.map(|(_, p)| p.parent().unwrap().to_path_buf())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mil = std::path::PathBuf::from(&a[1]);
    let (inp, outp) = (a[2].clone(), a[3].clone());
    let cin: usize = a[4].parse().unwrap();
    let _cout: usize = a[5].parse().unwrap();

    let _before = newest_bundle();
    let (prog, map) = Program::compile_auto(&mil, DeviceMask::Ane).unwrap();
    println!("ports: in={:?} out={:?}", map.inputs, map.outputs);
    let x: Vec<u16> = (0..cin)
        .map(|i| half::f16::from_f32(((i * 7) % 13) as f32 * 0.25 - 1.0).to_bits())
        .collect();
    prog.set_input_f16(&inp, &x).unwrap();
    prog.execute().unwrap();
    let y0 = prog.output_f16(&outp).unwrap();
    println!("ref out[0..8]: {:?}", &y0[..8.min(y0.len())]);

    let b = newest_bundle().unwrap();
    println!("bundle: {}", b.display());
    let macho = b.join("net.macho");
    if !macho.exists() {
        // find it inside
        let mut v = vec![];
        fn walk(d: &Path, out: &mut Vec<std::path::PathBuf>) {
            if let Ok(rd) = std::fs::read_dir(d) {
                for e in rd.flatten() {
                    if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        walk(&e.path(), out);
                    } else if e.file_name().to_string_lossy().contains("macho") {
                        out.push(e.path());
                    }
                }
            }
        }
        walk(&b, &mut v);
        println!("macho candidates: {:?}", v);
        return;
    }
    let mut img = std::fs::read(&macho).unwrap();
    // patch every 00{id}00{idx} word: remap each desc id to a shifted value
    let mut n = 0;
    for i in (0..img.len() - 4).step_by(4) {
        let w = u32::from_le_bytes(img[i..i + 4].try_into().unwrap());
        if w >> 24 == 0 && (0x54..=0x56).contains(&(w >> 16 & 0xff)) && (w >> 8 & 0xff) == 0 {
            let nw = (w & 0xffff) | (((w >> 16 & 0xff) + 0x20) << 16);
            img[i..i + 4].copy_from_slice(&nw.to_le_bytes());
            n += 1;
        }
    }
    std::fs::write(&macho, &img).unwrap();
    println!("patched {n} desc-id words (ids+0x20)");

    match Program::from_e5_bundle(&b) {
        Ok((p2, _)) => {
            p2.set_input_f16(&inp, &x).unwrap();
            match p2.execute() {
                Ok(()) => {
                    let y1 = p2.output_f16(&outp).unwrap();
                    println!("mut out[0..8]: {:?}", &y1[..8.min(y1.len())]);
                    println!("identical: {}", y0 == y1);
                }
                Err(e) => println!("exec failed: {e}"),
            }
        }
        Err(e) => println!("reload failed: {e}"),
    }
}
