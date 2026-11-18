// Isolate the IOSurface creation limit: how many SharedSurface::new
// calls succeed in one process?
//   mode "drop": create+free each iteration -> tests cumulative cap
//   mode "keep": hold every surface alive     -> tests live-count cap
use mil_infer::espresso::SharedSurface;
fn main() {
    let keep = std::env::args().nth(1).as_deref() == Some("keep");
    let mut held: Vec<SharedSurface> = Vec::new();
    let mut n = 0u64;
    let bytes = 8 * 1024; // ~4K fp16 elems — typical small node surface
    loop {
        match SharedSurface::new(bytes) {
            Ok(s) => {
                n += 1;
                if keep {
                    held.push(s);
                }
                if n % 2000 == 0 {
                    eprintln!("{n} created");
                }
                if n >= 200_000 {
                    eprintln!("done: {n}");
                    break;
                }
            }
            Err(e) => {
                eprintln!("FAILED at {n}: {e:?}");
                break;
            }
        }
    }
}
