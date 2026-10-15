//! Compile a MIL twice in-process so lldb can watchpoint ANEC's image buffer
//! (first run discovers the buffer VA; second run hits the watchpoint).
use mil_infer::Anec;
use std::path::Path;

fn main() {
    let anec = Anec::open().expect("ANEC unavailable");
    let mil = std::env::args().nth(1).unwrap();
    let out = std::env::args().nth(2).unwrap();
    for round in 0..3 {
        let v = anec
            .compile(Path::new(&mil), Path::new(&out), "h16")
            .unwrap();
        println!("round {round}: status={}", v.status);
    }
}
