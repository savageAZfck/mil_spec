fn main() {
    unsafe { std::env::set_var("HWX_SIG_ANEC", "1") };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cases: Vec<(String, String)> = Vec::new();
    // every net.mil under the named roots → its own out/net.macho
    for root in &args {
        let rd = std::fs::read_dir(root).unwrap();
        for e in rd.flatten() {
            let mil = e.path().join("net.mil");
            if !mil.exists() {
                continue;
            }
            for cand in [
                e.path().join("out/net.macho"),
                e.path().join("out/net.macho/net.macho"),
            ] {
                if cand.is_file() {
                    cases.push((mil.display().to_string(), cand.display().to_string()));
                }
            }
        }
    }
    let mut bad = 0;
    for (mil, out) in &cases {
        match mil_infer::hwx::emit_from_mil(mil, out) {
            Ok(img) => {
                let r = std::fs::read(out).unwrap();
                let d = (0..img.len().max(r.len()))
                    .filter(|&i| img.get(i) != r.get(i))
                    .count();
                let mark = if d == 0 {
                    "OK "
                } else {
                    bad += 1;
                    "DIFF"
                };
                println!("{mark} {mil}: {d} diffs");
            }
            Err(e) => {
                bad += 1;
                println!("ERR  {mil}: {e}")
            }
        }
    }
    println!("--- {} cases, {} failures", cases.len(), bad);
}
