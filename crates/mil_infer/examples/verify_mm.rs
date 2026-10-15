//! verify_mm — emit `emit_matmul_ns` for every `m1k<K>n<N>` ANEC
//! artifact under the given roots and compare byte-for-byte.
//!   verify_mm /tmp/mmx5 /tmp/mmvar /tmp/mmvar2 /tmp/mmcap
use mil_infer::hwx;
use std::fs;
use std::path::Path;

fn main() {
    let (mut ok, mut bad) = (0u32, 0u32);
    for root in std::env::args().skip(1) {
        let root = Path::new(&root);
        let mut dirs: Vec<_> = fs::read_dir(root)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.join("out/net.macho").exists())
            .collect();
        dirs.sort();
        for d in dirs {
            let name = d.file_name().unwrap().to_string_lossy().to_string();
            // m<M>k<K>n<N>
            let Some(kp) = name.find('k') else { continue };
            let Some(np) = name.rfind('n') else { continue };
            if np <= kp + 1 {
                continue;
            }
            let (Ok(k), Ok(n)) = (
                name[kp + 1..np].parse::<usize>(),
                name[np + 1..].parse::<usize>(),
            ) else {
                continue;
            };
            // weights = ramp (probe's pattern)
            let w: Vec<u16> = (0..k * n).map(|i| i as u16).collect();
            let mil = d.join("net.mil");
            let out = d.join("out/net.macho");
            std::env::set_var("HWX_SIG_ANEC", "1");
            match hwx::emit_matmul_ns(k, n, &w, mil.to_str().unwrap(), out.to_str().unwrap()) {
                Ok(img) => {
                    let exp = fs::read(&out).unwrap();
                    if img == exp {
                        ok += 1;
                    } else {
                        bad += 1;
                        let i = img.iter().zip(&exp).position(|(a, b)| a != b).unwrap_or(0);
                        println!("{name}: DIFF @{i:#x} len {} vs {}", img.len(), exp.len());
                    }
                }
                Err(e) => {
                    bad += 1;
                    println!("{name}: ERR {e}");
                }
            }
        }
    }
    println!("{ok}/{} identical", ok + bad);
}
