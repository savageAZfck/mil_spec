// Does aned's unload return surface-charge? Load programs with retry
// (mimic ensure's backoff), drop them, then watch how fast
// surface-create capacity recovers.
use mil_infer::espresso::{EspressoProgram, SharedSurface, LIVE_SURFACES};
use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;
fn main() {
    let dir = std::env::args().nth(1).unwrap_or("/tmp/cy4b_full/l0".into());
    let mut machos: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            p.extension().map(|x| x == "macho").unwrap_or(false).then_some(p)
        })
        .collect();
    machos.sort();
    machos.truncate(1200);
    let t = Instant::now();
    let mut progs: Vec<EspressoProgram> = Vec::new();
    let mut total = 0usize;
    // bounded residency ~32: drop oldest as we go, keep churning loads
    for p in machos.iter().cycle() {
        let mut tries = 0;
        loop {
            match EspressoProgram::load_precompiled(Path::new(p), &[]) {
                Ok(mut pr) => {
                    // mimic the real path: adopt a foreign SharedSurface
                    // onto every port, then eval, then drop.
                    let adopt = std::env::var("ADOPT").is_ok();
                    let mut adopted = Vec::new();
                    if adopt {
                        let ni = pr.num_inputs as usize;
                        let no = pr.num_outputs as usize;
                        for i in 0..ni {
                            let b = pr.input_descs.get(i).map(|d| d.bytes()).unwrap_or(1 << 20);
                            if let Ok(s) = SharedSurface::new(b) {
                                let _ = pr.adopt_input(i, s.as_ptr());
                                adopted.push(s);
                            }
                        }
                        for o in 0..no {
                            let b = pr.output_descs.get(o).map(|d| d.bytes()).unwrap_or(1 << 20);
                            if let Ok(s) = SharedSurface::new(b) {
                                let _ = pr.adopt_output(o, s.as_ptr());
                                adopted.push(s);
                            }
                        }
                    }
                    if let Err(e) = pr.eval() {
                        if total % 500 == 0 {
                            eprintln!("eval failed total={total}: {e:?}");
                        }
                    }
                    drop(adopted);
                    progs.push(pr);
                    total += 1;
                    break;
                }
                Err(_) => {
                    tries += 1;
                    if tries > 50 {
                        eprintln!("load stuck at total={total}");
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
        if tries > 50 {
            break;
        }
        if progs.len() > 32 {
            progs.remove(0); // drop oldest — unload queued in aned
        }
        if total % 200 == 0 {
            eprintln!(
                "t={:?} loaded {total} (resident {}, live_surf={})",
                t.elapsed(),
                progs.len(),
                LIVE_SURFACES.load(Relaxed)
            );
        }
        if total >= 8000 {
            break;
        }
    }
    eprintln!("total loaded {total}, dropping {}", progs.len());
    drop(progs);
    for round in 0..10 {
        let mut held = Vec::new();
        let mut got = 0;
        while got < 2000 {
            match SharedSurface::new(64 * 1024) {
                Ok(s) => {
                    held.push(s);
                    got += 1;
                }
                Err(_) => break,
            }
        }
        eprintln!(
            "t={:?}: create burst {} (live_surf={})",
            t.elapsed(),
            got,
            LIVE_SURFACES.load(Relaxed)
        );
        drop(held);
        std::thread::sleep(std::time::Duration::from_secs(6));
    }
}
